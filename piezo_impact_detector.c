/*
 * piezo_impact_detector.c
 *
 * PC-side simulation of the LPC4370 piezoelectric impact detection firmware.
 *
 * This code is structured to be directly analogous to how the algorithm would
 * run on the LPC4370 in MCUXpresso. The core logic (circular buffer, running
 * power, adaptive noise floor, state machine) is identical. The only
 * differences from real firmware are:
 *
 *   - AD7685 SPI reads replaced by loading samples from a CSV file
 *   - SD card writes replaced by writing event CSV files to disk
 *   - LPC4370 timer peripheral replaced by sample counter for timestamps
 *   - DMA transfer replaced by direct array access
 *
 * Hardware context:
 *   AD7685  — 16-bit external ADC, 0V–VREF, straight binary, SPI interface
 *   LPC4370 — ARM Cortex-M4, 282 kB SRAM, SD card via SPI for event storage
 *   Sample rate: 100 kSPS (well within AD7685's 250 kSPS max)
 *
 * Algorithm:
 *   1. CALIBRATION  — build initial noise floor estimate (2 seconds)
 *   2. IDLE         — circular buffer + running power + adaptive noise floor
 *   3. TRIGGERED    — save capture window to SD card, start cooldown
 *   4. COOLDOWN     — dead time (300 ms) to avoid re-triggering on ringdown
 *
 * Detection parameters (derived from PLB experimental data):
 *   BUFFER_SAMPLES  = 30,000  (300 ms @ 100 kSPS, 60 KB — fits in 282 kB SRAM)
 *   POWER_WINDOW    = 1,000   (10 ms sliding window for running power)
 *   NOISE_ALPHA     = 0.001   (slow exponential average for noise floor)
 *   TRIGGER_K       = 8.0     (trigger at 8x noise floor power)
 *                             (worst-case PLB ratio was 25x, K=8 gives margin)
 *   COOLDOWN_SAMPLES= 30,000  (300 ms refractory period)
 *   CALIBRATION_SAMPLES= 20,000 (200 ms — PC-side value; hardware uses 200,000)
 *
 * Compile:  gcc piezo_impact_detector.c -o detector -lm
 * Run:      ./detector <adc_output.csv>
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

/* ── Hardware / algorithm parameters ───────────────────────────────────── */
#define SAMPLE_RATE_HZ       100000
#define VREF                 5.0f
#define ADC_CODES            65536
#define MIDSCALE_CODE        32768       /* 0x8000 — DC bias point          */

/* Circular capture buffer: 300 ms @ 100 kSPS = 30,000 samples = 60 KB     */
/* On LPC4370 this sits in SRAM (282 kB total), allocated with __DATA(RAM1) */
#define BUFFER_SAMPLES       30000

/* Sliding power window: 10 ms = 1,000 samples                              */
/* Shorter than buffer so power reacts quickly to event onset               */
#define POWER_WINDOW         1000

/* Noise floor: exponential moving average, only updated during quiet       */
/* Alpha = 0.001 gives a ~1000-sample (10 ms) time constant                */
#define NOISE_ALPHA          0.001f

/* Trigger: fire when running power > K * noise_floor_power                 */
/* K=8 sits well above noise (1x) and below worst-case PLB event (25x)     */
#define TRIGGER_K            8.0f

/* Cooldown: 300 ms dead time after trigger to suppress ringdown            */
#define COOLDOWN_SAMPLES     30000

/* Startup calibration period.
 * Hardware (LPC4370): 200,000 samples = 2 s — allows EMA to converge on
 *   the true quiescent noise floor before monitoring begins.
 * PC-side (this file): reduced to 20,000 samples = 200 ms, because the
 *   ADC output CSV files are 120,000 samples long; using the hardware value
 *   would exhaust the file before calibration completes, preventing detection.
 * All other parameters are identical between the two versions. */
#define CALIBRATION_SAMPLES  20000

/* Max events to save per run (SD card limit proxy)                         */
#define MAX_EVENTS           64

/* ── Detector state machine ─────────────────────────────────────────────── */
typedef enum {
    STATE_CALIBRATION,
    STATE_IDLE,
    STATE_TRIGGERED,
    STATE_COOLDOWN
} DetectorState;

/* ── Circular buffer ────────────────────────────────────────────────────── */
/*
 * On LPC4370 this would be declared as:
 *   static uint16_t circ_buf[BUFFER_SAMPLES] __DATA(RAM1);
 * and filled via DMA from the SPI peripheral connected to the AD7685.
 * Here we manage it manually in the sample loop.
 */
static uint16_t circ_buf[BUFFER_SAMPLES];
static uint32_t buf_head = 0;       /* next write position                  */
static uint32_t buf_count = 0;      /* samples currently in buffer          */

/* ── Running power sum ──────────────────────────────────────────────────── */
/*
 * Maintains sum of squared AC-coupled samples over POWER_WINDOW samples.
 * Updated as: power_sum += new² - oldest²
 * On LPC4370 this runs in the M4 core between DMA completion interrupts.
 */
static double power_sum = 0.0;

/* Separate small window to track oldest sample for the power calculation   */
static float  power_window_buf[POWER_WINDOW];
static uint32_t pw_head = 0;
static uint32_t pw_count = 0;

/* ── Noise floor estimate ───────────────────────────────────────────────── */
static double noise_floor_power = 0.0;  /* exponential moving average       */
static int    noise_initialised  = 0;   /* flag: has EMA been seeded?       */

/* ── State and counters ─────────────────────────────────────────────────── */
static DetectorState state          = STATE_CALIBRATION;
static uint32_t      state_counter  = 0;    /* samples in current state     */
static uint32_t      total_samples  = 0;    /* global sample counter        */
static uint32_t      event_count    = 0;

/* ── Utility: convert ADC code to AC-coupled voltage ────────────────────── */
/*
 * Remove DC bias (midscale) before power calculation.
 * On hardware the signal is biased at VREF/2 = 2.5V by the signal chain.
 * We subtract midscale so the power calculation reflects signal excursions
 * rather than DC level.
 */
static inline float code_to_ac_volts(uint16_t code)
{
    return ((float)code - (float)MIDSCALE_CODE) / (float)ADC_CODES * VREF;
}

/* ── Circular buffer operations ─────────────────────────────────────────── */
static void circ_buf_write(uint16_t sample)
{
    circ_buf[buf_head] = sample;
    buf_head = (buf_head + 1) % BUFFER_SAMPLES;
    if (buf_count < BUFFER_SAMPLES) buf_count++;
}

/* Read sample at position [index] samples back from current head           */
static uint16_t circ_buf_read_back(uint32_t index)
{
    uint32_t pos = (buf_head + BUFFER_SAMPLES - 1 - index) % BUFFER_SAMPLES;
    return circ_buf[pos];
}

/* ── Running power update ───────────────────────────────────────────────── */
static void update_running_power(float new_sample_v)
{
    float oldest = 0.0f;

    if (pw_count == POWER_WINDOW) {
        /* Window is full — subtract oldest sample before overwriting        */
        oldest = power_window_buf[pw_head];
        power_sum -= (double)(oldest * oldest);
    }

    /* Write new sample into power window                                    */
    power_window_buf[pw_head] = new_sample_v;
    pw_head = (pw_head + 1) % POWER_WINDOW;
    if (pw_count < POWER_WINDOW) pw_count++;

    /* Add new sample squared                                                */
    power_sum += (double)(new_sample_v * new_sample_v);
}

/* Current mean power over the window                                       */
static double get_current_power(void)
{
    if (pw_count == 0) return 0.0;
    return power_sum / (double)pw_count;
}

/* ── Noise floor update (only called during quiet periods) ──────────────── */
static void update_noise_floor(double current_power)
{
    if (!noise_initialised) {
        noise_floor_power = current_power;
        noise_initialised = 1;
    } else {
        /* Exponential moving average — slow to react, immune to transients  */
        noise_floor_power = (1.0 - NOISE_ALPHA) * noise_floor_power
                          + NOISE_ALPHA * current_power;
    }
}

/* ── Save event to disk (replaces SD card write on LPC4370) ─────────────── */
/*
 * On LPC4370 this would:
 *   1. Assert SD CS GPIO pin
 *   2. Write header block (timestamp, event_id, sample_count) via SPI
 *   3. DMA the circular buffer contents to SD card in 512-byte sectors
 *   4. Deassert CS
 *
 * Here we write a CSV file per event for analysis.
 */
static void save_event(uint32_t trigger_sample, double trigger_power,
                       double noise_at_trigger, const char *output_dir)
{
    char filename[256];
    snprintf(filename, sizeof(filename),
             "%s/event_%03u_sample_%u.csv", output_dir, event_count, trigger_sample);

    FILE *f = fopen(filename, "w");
    if (!f) {
        fprintf(stderr, "ERROR: Could not write event file %s\n", filename);
        return;
    }

    /* Header — mirrors what would be stored as metadata block on SD card   */
    fprintf(f, "# Event %u\n", event_count);
    fprintf(f, "# Trigger sample:       %u\n",   trigger_sample);
    fprintf(f, "# Trigger time (s):     %.6f\n",
            (double)trigger_sample / SAMPLE_RATE_HZ);
    fprintf(f, "# Trigger power (V2):   %.6e\n",  trigger_power);
    fprintf(f, "# Noise floor (V2):     %.6e\n",  noise_at_trigger);
    fprintf(f, "# SNR ratio:            %.1fx\n",
            (noise_at_trigger > 0) ? trigger_power / noise_at_trigger : 0.0);
    fprintf(f, "# Buffer samples:       %u\n",    BUFFER_SAMPLES);
    fprintf(f, "# Sample rate (SPS):    %u\n",    SAMPLE_RATE_HZ);
    fprintf(f, "sample_index,time_from_trigger_us,adc_code,voltage_V\n");

    /* Dump circular buffer — oldest sample first                           */
    /* Buffer contains BUFFER_SAMPLES leading up to and including trigger   */
    uint32_t n = (buf_count < BUFFER_SAMPLES) ? buf_count : BUFFER_SAMPLES;
    int32_t  trigger_offset = (int32_t)n - 1;  /* trigger is at end of buf */

    for (uint32_t i = 0; i < n; i++) {
        uint16_t code = circ_buf_read_back(n - 1 - i);
        float    v    = code_to_ac_volts(code) + (VREF / 2.0f);  /* restore DC */
        int32_t  rel  = (int32_t)i - trigger_offset;
        float    t_us = (float)rel / (float)SAMPLE_RATE_HZ * 1e6f;
        fprintf(f, "%d,%.2f,%u,%.6f\n", (int32_t)i - trigger_offset,
                t_us, code, v);
    }

    fclose(f);
    printf("  [EVENT %03u] t=%.4f s  power=%.3e V²  SNR=%.1fx  → %s\n",
           event_count,
           (double)trigger_sample / SAMPLE_RATE_HZ,
           trigger_power,
           (noise_at_trigger > 0) ? trigger_power / noise_at_trigger : 0.0,
           filename);
}

/* ── Process one ADC sample — core of the main loop ────────────────────── */
/*
 * On LPC4370 this function body would execute inside the main while() loop,
 * called after the DMA signals that a new sample is ready in the SPI FIFO.
 * The LPC4370 M4 core at 204 MHz can execute this in ~1 µs, well within
 * the 10 µs budget at 100 kSPS.
 */
static void process_sample(uint16_t adc_code, const char *output_dir)
{
    float v_ac = code_to_ac_volts(adc_code);

    /* Always: write to circular buffer and update running power             */
    circ_buf_write(adc_code);
    update_running_power(v_ac);
    double current_power = get_current_power();

    total_samples++;
    state_counter++;

    switch (state) {

        /* ── CALIBRATION ── build initial noise floor before enabling trigger */
        case STATE_CALIBRATION:
            update_noise_floor(current_power);
            if (state_counter >= CALIBRATION_SAMPLES) {
                state         = STATE_IDLE;
                state_counter = 0;
                printf("  Calibration complete. Noise floor = %.4e V²  "
                       "(threshold = %.4e V²)\n",
                       noise_floor_power,
                       noise_floor_power * TRIGGER_K);
            }
            break;

        /* ── IDLE ── continuous monitoring, adaptive noise floor update      */
        case STATE_IDLE:
            /* Only update noise floor when signal is quiet                  */
            if (current_power < noise_floor_power * TRIGGER_K) {
                update_noise_floor(current_power);
            }

            /* Trigger condition: power exceeds K * noise floor              */
            if (pw_count == POWER_WINDOW &&
                noise_initialised &&
                current_power > noise_floor_power * TRIGGER_K)
            {
                if (event_count < MAX_EVENTS) {
                    save_event(total_samples, current_power,
                               noise_floor_power, output_dir);
                    event_count++;
                }
                state         = STATE_COOLDOWN;
                state_counter = 0;
            }
            break;

        /* ── TRIGGERED ── transition state, immediately enter cooldown       */
        /* (save_event called above; this state kept for LPC4370 analogy     */
        /*  where SD write takes non-zero time)                              */
        case STATE_TRIGGERED:
            state         = STATE_COOLDOWN;
            state_counter = 0;
            break;

        /* ── COOLDOWN ── dead time, suppress retriggering on ringdown        */
        case STATE_COOLDOWN:
            /* Keep sampling into buffer but don't trigger or update noise   */
            if (state_counter >= COOLDOWN_SAMPLES) {
                state         = STATE_IDLE;
                state_counter = 0;
            }
            break;
    }
}

/* ── Load ADC CSV and run detector ─────────────────────────────────────── */
static int run_detector(const char *input_csv, const char *output_dir)
{
    FILE *f = fopen(input_csv, "r");
    if (!f) {
        fprintf(stderr, "ERROR: Cannot open %s\n", input_csv);
        return -1;
    }

    /* Skip header */
    char line[256];
    fgets(line, sizeof(line), f);

    uint32_t loaded = 0;
    uint32_t sample_col, code_col;
    int      col_found = 0;

    /* Parse column indices from header */
    /* Expected: sample,time_s,voltage_in_V,adc_code,adc_code_hex,... */
    /* adc_code is column index 3 (0-based)                           */
    code_col = 3;

    printf("\nRunning detector on: %s\n", input_csv);
    printf("Parameters: BUFFER=%d samples (%.0f ms)  "
           "POWER_WINDOW=%d samples (%.0f ms)  K=%.1f\n\n",
           BUFFER_SAMPLES, (float)BUFFER_SAMPLES / SAMPLE_RATE_HZ * 1000.0f,
           POWER_WINDOW,   (float)POWER_WINDOW   / SAMPLE_RATE_HZ * 1000.0f,
           TRIGGER_K);

    while (fgets(line, sizeof(line), f)) {
        /* Skip comment lines (event metadata if re-processing)             */
        if (line[0] == '#') continue;

        /* Parse CSV columns                                                 */
        char *tok = strtok(line, ",");
        int   col = 0;
        uint16_t code = 0;
        int   got_code = 0;

        while (tok != NULL) {
            if (col == (int)code_col) {
                code = (uint16_t)atoi(tok);
                got_code = 1;
            }
            tok = strtok(NULL, ",");
            col++;
        }

        if (!got_code) continue;

        process_sample(code, output_dir);
        loaded++;
    }

    fclose(f);

    printf("\n── Detector Summary ────────────────────────────────────\n");
    printf("  Input file     : %s\n",   input_csv);
    printf("  Samples loaded : %u\n",   loaded);
    printf("  Duration       : %.3f s\n", (double)loaded / SAMPLE_RATE_HZ);
    printf("  Events detected: %u\n",   event_count);
    printf("  Noise floor    : %.4e V²\n", noise_floor_power);
    printf("  Threshold      : %.4e V²  (K=%.1f)\n",
           noise_floor_power * TRIGGER_K, TRIGGER_K);
    printf("────────────────────────────────────────────────────────\n\n");

    return 0;
}

/* ── Entry point ─────────────────────────────────────────────────────────── */
/*
 * On LPC4370 main() is a dummy that calls real_main() from SRAM.
 * Here we just run directly.
 */
int main(int argc, char *argv[])
{
    if (argc < 2) {
        fprintf(stderr,
            "Usage: %s <adc_output.csv> [output_dir]\n"
            "  adc_output.csv  — output from ad7685_sim.py\n"
            "  output_dir      — folder for event CSVs (default: ./events)\n",
            argv[0]);
        return 1;
    }

    const char *input_csv  = argv[1];
    const char *output_dir = (argc >= 3) ? argv[2] : "events";

    /* Create output directory (POSIX) — on LPC4370 this is SD card init    */
    char mkdir_cmd[512];
    snprintf(mkdir_cmd, sizeof(mkdir_cmd), "mkdir -p \"%s\"", output_dir);
    system(mkdir_cmd);

    return run_detector(input_csv, output_dir) == 0 ? 0 : 1;
}

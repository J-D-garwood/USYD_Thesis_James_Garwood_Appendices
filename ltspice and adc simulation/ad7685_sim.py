"""
ad7685_sim.py  —  AD7685 ADC simulation from LTSpice CSV outputs

Automatically finds all files matching processed_*mm_*.csv in the same
directory as this script and writes results to an "ADC outputs" folder.

Expected input format (LTSpice export):
    time,V(filterout),V(sdtin)
    0.000000000000000e+00,2.514045e+00,2.501774e+00
    ...

AD7685 datasheet parameters modelled:
  - 16-bit straight binary, 0V to VREF
  - Transfer function: code = floor(V_IN / VREF * 65536), clamped [0, 65535]
  - Transition noise:  0.5 LSB rms  (Table 2, VREF=5V)
  - Offset error:      0.1 mV typ   (Table 2)
  - Gain error:        2 LSB typ    (Table 2)
  - No pipeline delay

Usage:
  python ad7685_sim.py [--vref 5.0] [--rate 100000] [--ideal]
"""

import argparse
import numpy as np
import csv
import sys
import os
import glob
from pathlib import Path

# ── AD7685 constants (from datasheet Table 2) ───────────────────────────────
ADC_BITS             = 16
ADC_CODES            = 65536        # 2^16
ADC_MAX_CODE         = 65535        # 0xFFFF
TRANSITION_NOISE_LSB = 0.5          # rms, VREF=5V
OFFSET_ERROR_V       = 0.0001       # 0.1 mV typical
GAIN_ERROR_LSB       = 2            # 2 LSB typical


def find_input_files(script_dir):
    """
    Find all processed_*mm_*.csv files in the same directory as the script.
    Returns a sorted list of Path objects.
    """
    pattern = os.path.join(script_dir, 'processed_*mm_*.csv')
    files = sorted(glob.glob(pattern))
    if not files:
        sys.exit(
            f"ERROR: No files matching 'processed_*mm_*.csv' found in:\n  {script_dir}\n"
            f"Place this script in the same folder as your CSV files and run again."
        )
    return [Path(f) for f in files]


def load_ltspice_csv(filepath):
    """
    Load LTSpice CSV with format: time,V(filterout),V(sdtin)
    Non-uniform adaptive timesteps handled — np.interp resamples later.
    Returns (times, v_filterout, v_sdtin) as float64 numpy arrays.
    """
    times, v_filter, v_sdt = [], [], []

    with open(filepath, 'r') as f:
        header = f.readline().strip()
        cols = [c.strip() for c in header.split(',')]

        try:
            idx_t      = cols.index('time')
            idx_filter = next(i for i, c in enumerate(cols) if 'filterout' in c.lower())
            idx_sdt    = next(i for i, c in enumerate(cols) if 'sdt' in c.lower())
        except (ValueError, StopIteration):
            sys.exit(
                f"ERROR: Could not find expected columns in {filepath.name}\n"
                f"  Found: {cols}\n"
                f"  Expected: time, V(filterout), V(sdtin)"
            )

        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) < 3:
                continue
            try:
                times.append(float(parts[idx_t]))
                v_filter.append(float(parts[idx_filter]))
                v_sdt.append(float(parts[idx_sdt]))
            except ValueError:
                continue

    return np.array(times), np.array(v_filter), np.array(v_sdt)


def simulate_ad7685(times, v_filterout, vref, sample_rate_hz, ideal=False):
    """
    Resample V(filterout) at uniform CNV trigger intervals and quantise
    to AD7685 16-bit straight binary codes.
    """
    if sample_rate_hz > 250_000:
        sys.exit("ERROR: AD7685 maximum throughput is 250 kSPS")

    lsb_v = vref / ADC_CODES

    sample_times = np.arange(times[0], times[-1], 1.0 / sample_rate_hz)
    v_sampled = np.interp(sample_times, times, v_filterout)

    if not ideal:
        v_sampled = v_sampled + OFFSET_ERROR_V
        gain_correction = 1.0 + (GAIN_ERROR_LSB * lsb_v) / vref
        v_sampled = v_sampled * gain_correction
        v_sampled = v_sampled + np.random.normal(0.0, 0.5 * lsb_v, len(v_sampled))

    v_sampled = np.clip(v_sampled, 0.0, vref)

    codes = np.clip(
        np.floor(v_sampled / vref * ADC_CODES),
        0, ADC_MAX_CODE
    ).astype(np.uint16)

    return sample_times, codes, v_sampled


def write_output(filepath, sample_times, codes, v_sampled, vref):
    lsb_v = vref / ADC_CODES
    v_reconstructed = codes.astype(np.float64) / ADC_CODES * vref

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'sample', 'time_s', 'voltage_in_V',
            'adc_code', 'adc_code_hex',
            'reconstructed_V', 'quantisation_error_V'
        ])
        for i, (t, v_in, code, v_rec) in enumerate(
                zip(sample_times, v_sampled, codes, v_reconstructed)):
            writer.writerow([
                i,
                f'{t:.9e}',
                f'{v_in:.6f}',
                int(code),
                f'0x{int(code):04X}',
                f'{v_rec:.6f}',
                f'{(v_in - v_rec):.2e}'
            ])


def summarise(name, sample_times, codes, v_sampled, vref, sample_rate_hz):
    lsb_v    = vref / ADC_CODES
    v_rec    = codes.astype(np.float64) / ADC_CODES * vref
    midscale = ADC_CODES // 2
    peak_dev = max(abs(int(np.max(codes)) - midscale),
                   abs(int(np.min(codes)) - midscale))

    clipped_high = int(np.sum(codes == ADC_MAX_CODE))
    clipped_low  = int(np.sum(codes == 0))
    clip_str = (f"⚠  HIGH={clipped_high} LOW={clipped_low}"
                if (clipped_high or clipped_low) else "✓  none")

    print(f"  {name:<22}  "
          f"samples={len(codes):>7,}  "
          f"peak=0x{np.max(codes):04X}  "
          f"min=0x{np.min(codes):04X}  "
          f"dev={peak_dev:>5} LSB  "
          f"clip={clip_str}")


def process_file(input_path, output_dir, vref, sample_rate_hz, ideal):
    times, v_filterout, v_sdtin = load_ltspice_csv(input_path)

    sample_times, codes, v_sampled = simulate_ad7685(
        times, v_filterout, vref, sample_rate_hz, ideal=ideal
    )

    # Output filename mirrors input name
    out_path = output_dir / f"adc_{input_path.stem}.csv"
    write_output(out_path, sample_times, codes, v_sampled, vref)

    summarise(input_path.name, sample_times, codes, v_sampled, vref, sample_rate_hz)
    return out_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='AD7685 ADC simulation — batch processes all processed_*mm_*.csv files'
    )
    parser.add_argument('--vref',  type=float, default=5.0,
                        help='AD7685 REF pin voltage in Volts (default: 5.0)')
    parser.add_argument('--rate',  type=int,   default=100_000,
                        help='CNV sample rate in Hz, max 250000 (default: 100000)')
    parser.add_argument('--ideal', action='store_true',
                        help='Ideal ADC — suppress noise and static errors')
    args = parser.parse_args()

    # Locate input files relative to this script
    script_dir = Path(__file__).parent.resolve()
    input_files = find_input_files(script_dir)

    # Create output folder
    output_dir = script_dir / 'ADC outputs'
    output_dir.mkdir(exist_ok=True)

    mode_str = 'ideal' if args.ideal else 'with datasheet non-idealities'
    print(f"\n── AD7685 Batch Simulation ─────────────────────────────────────")
    print(f"  Found {len(input_files)} input files in: {script_dir}")
    print(f"  Output folder : {output_dir}")
    print(f"  VREF          : {args.vref} V  |  1 LSB = {args.vref/ADC_CODES*1e6:.1f} µV")
    print(f"  Sample rate   : {args.rate:,} SPS  ({mode_str})")
    print(f"────────────────────────────────────────────────────────────────\n")

    for i, input_path in enumerate(input_files, 1):
        print(f"[{i}/{len(input_files)}] Processing {input_path.name} ...")
        out_path = process_file(input_path, output_dir, args.vref, args.rate, args.ideal)

    print(f"\n✓  All done — {len(input_files)} files written to '{output_dir.name}/'")
    print(f"────────────────────────────────────────────────────────────────\n")

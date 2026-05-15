"""
RTB2000 CSV -> LTSpice PWL converter
=====================================
Batch-converts R&S RTB2000 waveform CSV files to LTSpice PWL source files.

Input format  (comma-separated, RTB2000 export):
    in s,C1 in V
    -5.9999832E-01,1.7738342E-03
    ...

Output format (LTSpice PWL – two space-separated columns):
    0.000000000000e+00  1.773834200000e-03
    9.160000000000e-06  1.850128200000e-03
    ...

Usage
-----
1. Put this script in the same folder as your .CSV files (or set INPUT_FOLDER).
2. Run:  python converting_to_LTSpice_format.py
3. .pwl files appear in pwl_output/, ready for LTSpice.
   Right-click voltage source -> Advanced -> PWL file -> browse to the file.

Settings
--------
DECIMATE        – keep every Nth sample to reduce file size (5 is a good start)
TRIM_WINDOW_S   – seconds of data to keep, centred on the peak impact signal
                  (set to None to export the entire waveform)
PEAK_PRETRIGGER – fraction of the trim window to include BEFORE the peak
TIME_ZERO       – if True, the first exported sample is shifted to t = 0
"""

import os
import glob

# ── SETTINGS ──────────────────────────────────────────────────────────────────
INPUT_FOLDER    = "."           # folder containing .CSV files
OUTPUT_FOLDER   = "pwl_output"  # output folder (auto-created)
FILE_PATTERN    = "*.CSV"       # glob pattern; also catches *.csv automatically

DECIMATE        = 5             # keep every 5th sample  (adjust as needed)
TRIM_WINDOW_S   = 0.002         # keep 2 ms around the peak  (None = full waveform)
PEAK_PRETRIGGER = 0.2           # 20 % of the window before the peak
TIME_ZERO       = True          # re-zero time so first sample is t = 0
# ──────────────────────────────────────────────────────────────────────────────


def read_csv(filepath):
    """Parse RTB2000 CSV (comma-separated) into (times, voltages) lists."""
    times, voltages = [], []
    with open(filepath, "r") as fh:
        for i, line in enumerate(fh):
            if i == 0:
                continue          # skip "in s,C1 in V" header
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 2:
                continue
            try:
                times.append(float(parts[0]))
                voltages.append(float(parts[1]))
            except ValueError:
                continue
    return times, voltages


def convert_file(csv_path):
    basename = os.path.splitext(os.path.basename(csv_path))[0]
    pwl_path = os.path.join(OUTPUT_FOLDER, basename + ".pwl")

    print(f"  Reading   : {os.path.basename(csv_path)}")
    times, voltages = read_csv(csv_path)
    if not times:
        print("  WARNING   : no data found - skipping.\n")
        return
    print(f"  Total pts : {len(times)}")

    # 1. Trim around the peak ---------------------------------------------------
    if TRIM_WINDOW_S is not None:
        peak_i    = max(range(len(voltages)), key=lambda i: abs(voltages[i]))
        peak_time = times[peak_i]
        t_start   = peak_time - TRIM_WINDOW_S * PEAK_PRETRIGGER
        t_end     = peak_time + TRIM_WINDOW_S * (1.0 - PEAK_PRETRIGGER)
        si = next((i for i, t in enumerate(times) if t >= t_start), 0)
        ei = next((i for i, t in enumerate(times) if t >  t_end),   len(times))
        times    = times[si:ei]
        voltages = voltages[si:ei]
        print(f"  Peak      : {peak_time:.6f} s  |  "
              f"window {t_start:.6f} -> {t_end:.6f} s  ->  {len(times)} pts")

    # 2. Decimate ---------------------------------------------------------------
    if DECIMATE > 1:
        times    = times[::DECIMATE]
        voltages = voltages[::DECIMATE]
        print(f"  Decimated : every {DECIMATE}th pt  ->  {len(times)} pts")

    # 3. Shift to t = 0 ---------------------------------------------------------
    if TIME_ZERO and times:
        t0    = times[0]
        times = [t - t0 for t in times]

    # 4. Write PWL --------------------------------------------------------------
    with open(pwl_path, "w") as fh:
        for t, v in zip(times, voltages):
            fh.write(f"{t:.12e} {v:.12e}\n")
    print(f"  Written   : {pwl_path}  ({len(times)} points)\n")


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    found = sorted(set(
        glob.glob(os.path.join(INPUT_FOLDER, FILE_PATTERN)) +
        glob.glob(os.path.join(INPUT_FOLDER, FILE_PATTERN.lower()))
    ))
    if not found:
        print(f"No CSV files found in '{INPUT_FOLDER}' matching '{FILE_PATTERN}'.")
        return

    print(f"Found {len(found)} CSV file(s).  Output -> '{OUTPUT_FOLDER}'\n")
    for p in found:
        convert_file(p)
    print("Done!")
    print("Copy the .pwl files into your LTSpice project folder.")
    print("In LTSpice: right-click voltage source -> Advanced -> PWL file -> browse.")

if __name__ == "__main__":
    main()
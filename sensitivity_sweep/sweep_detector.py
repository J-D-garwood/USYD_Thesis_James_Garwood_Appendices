#!/usr/bin/env python3
"""
sweep_detector.py

Runs the piezo detector across a grid of (K, alpha) parameters against all
9 ADC CSV files (10/30/50 mm × 3 runs), for two conditions:
    - baseline   (no freeze mitigation; freeze = K)
    - mitigated  (freeze = 2.0, per §6.3.3 proposal)

Produces sweep_results.csv with one row per (condition, K, alpha, file).
"""

import subprocess
import csv
import itertools
import time
from pathlib import Path

DETECTOR = "./detector"
ADC_DIR  = Path("/mnt/project")
OUTPUT   = Path("sweep_results.csv")

DISTANCES = ["10mm", "30mm", "50mm"]
RUNS      = [1, 2, 3]
FILES = [(d, r, ADC_DIR / f"adc_processed_{d}_{r}.csv")
         for d in DISTANCES for r in RUNS]

# Parameter grid
K_VALUES     = [2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0]
ALPHA_VALUES = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5]

# Conditions: baseline (no freeze) and §6.3.3 mitigation (freeze at 2σ²)
CONDITIONS = [
    ("baseline",  None),   # freeze = K (no-op)
    ("mitigated", 2.0),    # freeze when P > 2 * noise_floor
]

CAL_SAMPLES = 20000  # matches thesis §4.3.4 PC-side calibration


def run_one(input_csv: Path, k: float, alpha: float, freeze) -> dict:
    cmd = [DETECTOR, "--k", f"{k}", "--alpha", f"{alpha}",
           "--cal", f"{CAL_SAMPLES}", "--quiet"]
    if freeze is not None:
        cmd += ["--freeze", f"{freeze}"]
    cmd += [str(input_csv)]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    # Expect a single "RESULT,..." line
    for line in out.stdout.strip().splitlines():
        if line.startswith("RESULT,"):
            parts = line.split(",")
            return {
                "file": parts[1],
                "k": float(parts[2]),
                "alpha": float(parts[3]),
                "events": int(parts[4]),
                "freeze": float(parts[5]),
                "trigger_snr": float(parts[6]),
                "noise_floor": float(parts[7]),
                "peak_ratio": float(parts[8]),
                "duration_s": float(parts[9]),
            }
    raise RuntimeError(f"No RESULT line from detector for {input_csv}")


def main() -> None:
    total = len(CONDITIONS) * len(K_VALUES) * len(ALPHA_VALUES) * len(FILES)
    print(f"Sweep: {total} runs "
          f"({len(CONDITIONS)} conditions × {len(K_VALUES)} K × "
          f"{len(ALPHA_VALUES)} alpha × {len(FILES)} files)")
    t0 = time.time()

    with OUTPUT.open("w", newline="") as fout:
        w = csv.writer(fout)
        w.writerow(["condition", "distance", "run", "k", "alpha", "freeze",
                    "events", "detected", "trigger_snr",
                    "peak_ratio", "noise_floor"])
        done = 0
        for cond_name, freeze in CONDITIONS:
            for k, alpha in itertools.product(K_VALUES, ALPHA_VALUES):
                for d, r, fpath in FILES:
                    res = run_one(fpath, k, alpha, freeze)
                    w.writerow([
                        cond_name, d, r, k, alpha,
                        res["freeze"], res["events"],
                        1 if res["events"] >= 1 else 0,
                        res["trigger_snr"], res["peak_ratio"],
                        res["noise_floor"],
                    ])
                    done += 1
                    if done % 100 == 0:
                        elapsed = time.time() - t0
                        rate = done / elapsed
                        eta = (total - done) / rate
                        print(f"  [{done:>4}/{total}]  "
                              f"{elapsed:5.1f}s elapsed, "
                              f"{eta:5.1f}s remaining  "
                              f"({rate:.1f} runs/s)")

    print(f"\nDone in {time.time() - t0:.1f}s → {OUTPUT}")


if __name__ == "__main__":
    main()

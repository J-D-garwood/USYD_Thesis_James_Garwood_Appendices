"""
plot_adc.py  —  Plot LTSpice V(filterout) input vs AD7685 ADC output

Place in the same folder as your processed_*mm_*.csv files and ADC outputs/ folder.

Usage:
  python plot_adc.py              # plots all 9 captures, saves PNGs to "plots/" folder
  python plot_adc.py --show       # also opens interactive windows
  python plot_adc.py --file processed_10mm_1  # single capture only
"""

import argparse
import numpy as np
import csv
import glob
import os
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — change to 'TkAgg' if --show is used
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Config ───────────────────────────────────────────────────────────────────
VREF          = 5.0
ADC_CODES     = 65536
MIDSCALE_CODE = 32768   # 0x8000 = VREF/2
MIDSCALE_V    = VREF / 2

COLOUR_FILTER = '#C0392B'   # red  — V(filterout)
COLOUR_SDT    = '#7F8C8D'   # grey — V(sdtin) reference
COLOUR_ADC    = '#2980B9'   # blue — ADC codes
COLOUR_RECON  = '#27AE60'   # green — reconstructed voltage


def load_ltspice_csv(filepath):
    times, v_filter, v_sdt = [], [], []
    with open(filepath, 'r') as f:
        cols = [c.strip() for c in f.readline().strip().split(',')]
        idx_t      = cols.index('time')
        idx_filter = next(i for i, c in enumerate(cols) if 'filterout' in c.lower())
        idx_sdt    = next(i for i, c in enumerate(cols) if 'sdt' in c.lower())
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 3:
                continue
            try:
                times.append(float(parts[idx_t]))
                v_filter.append(float(parts[idx_filter]))
                v_sdt.append(float(parts[idx_sdt]))
            except ValueError:
                continue
    return np.array(times), np.array(v_filter), np.array(v_sdt)


def load_adc_csv(filepath):
    samples, times, v_in, codes, v_rec = [], [], [], [], []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append(int(row['sample']))
            times.append(float(row['time_s']))
            v_in.append(float(row['voltage_in_V']))
            codes.append(int(row['adc_code']))
            v_rec.append(float(row['reconstructed_V']))
    return (np.array(times), np.array(v_in),
            np.array(codes), np.array(v_rec))


def to_us(t):
    """Convert seconds array to microseconds, zeroed at start."""
    return (t - t[0]) * 1e6


def plot_capture(stem, ltspice_path, adc_path, output_dir, show=False):
    """
    Four-panel plot for one capture:
      Panel 1 — Raw SDT sensor output vs filtered chain output (LTSpice)
      Panel 2 — ADC codes over time (bar/step)
      Panel 3 — Reconstructed voltage overlaid on V(filterout)
      Panel 4 — Quantisation error (V_in - V_reconstructed)
    """
    print(f"  Plotting {stem} ...")

    # Load data
    lt_t, v_filter, v_sdt = load_ltspice_csv(ltspice_path)
    adc_t, adc_vin, codes, v_rec = load_adc_csv(adc_path)

    lt_us  = to_us(lt_t)
    adc_us = to_us(adc_t)

    # Find the event window — where signal deviates > 5 mV from baseline
    baseline = np.median(v_filter)
    deviation = np.abs(v_filter - baseline)
    event_mask = deviation > 0.005
    if event_mask.any():
        event_start = lt_us[np.argmax(event_mask)]
        event_end   = lt_us[len(lt_us) - 1 - np.argmax(event_mask[::-1])]
        # Add 20% padding
        span    = event_end - event_start
        t_lo    = max(0, event_start - span * 0.2)
        t_hi    = min(lt_us[-1], event_end + span * 0.2)
    else:
        t_lo, t_hi = lt_us[0], lt_us[-1]

    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor('#FAFAFA')
    title_dist = stem.replace('processed_', '').replace('_', ' — trial ').replace('mm', ' mm')
    fig.suptitle(f"AD7685 ADC Simulation  |  {title_dist}",
                 fontsize=14, fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(4, 1, hspace=0.55, top=0.93, bottom=0.07,
                           left=0.09, right=0.97)

    # ── Panel 1: LTSpice signals ──────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(lt_us, v_sdt,    color=COLOUR_SDT,    lw=0.8, label='V(sdtin) — raw sensor', alpha=0.7)
    ax1.plot(lt_us, v_filter, color=COLOUR_FILTER,  lw=1.0, label='V(filterout) — ADC input')
    ax1.axhline(MIDSCALE_V, color='#BDC3C7', lw=0.7, ls='--', label=f'Midscale ({MIDSCALE_V} V)')
    ax1.set_xlim(t_lo, t_hi)
    ax1.set_ylabel('Voltage (V)', fontsize=9)
    ax1.set_title('Signal Chain Output  (LTSpice)', fontsize=9, loc='left', pad=3)
    ax1.legend(fontsize=8, loc='upper right', framealpha=0.8)
    ax1.grid(True, lw=0.4, alpha=0.5)
    ax1.tick_params(labelsize=8)
    _shade_event(ax1, event_start if event_mask.any() else None,
                      event_end   if event_mask.any() else None)

    # ── Panel 2: ADC codes ────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.step(adc_us, codes, where='post', color=COLOUR_ADC, lw=0.9, label='ADC code')
    ax2.axhline(MIDSCALE_CODE, color='#BDC3C7', lw=0.7, ls='--',
                label=f'Midscale (0x8000 = {MIDSCALE_CODE})')
    ax2.set_ylabel('ADC Code (16-bit)', fontsize=9)
    ax2.set_title('AD7685 Output Codes  (straight binary)', fontsize=9, loc='left', pad=3)
    ax2.legend(fontsize=8, loc='upper right', framealpha=0.8)
    ax2.grid(True, lw=0.4, alpha=0.5)
    ax2.tick_params(labelsize=8)
    # Secondary y-axis in hex
    ax2r = ax2.twinx()
    ax2r.set_ylim(ax2.get_ylim())
    hex_ticks = np.linspace(ax2.get_ylim()[0], ax2.get_ylim()[1], 5)
    ax2r.set_yticks(hex_ticks)
    ax2r.set_yticklabels([f'0x{int(v):04X}' for v in hex_ticks], fontsize=7)
    _shade_event(ax2, event_start if event_mask.any() else None,
                      event_end   if event_mask.any() else None)

    # ── Panel 3: Reconstructed voltage vs V(filterout) ───────────────────
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(lt_us,  v_filter, color=COLOUR_FILTER, lw=1.0, alpha=0.6,
             label='V(filterout) — continuous')
    ax3.step(adc_us, v_rec,    color=COLOUR_RECON,  lw=0.9, where='post',
             label='Reconstructed (code → V)')
    ax3.axhline(MIDSCALE_V, color='#BDC3C7', lw=0.7, ls='--')
    ax3.set_ylabel('Voltage (V)', fontsize=9)
    ax3.set_title('Reconstructed ADC Voltage vs Filter Output', fontsize=9, loc='left', pad=3)
    ax3.legend(fontsize=8, loc='upper right', framealpha=0.8)
    ax3.grid(True, lw=0.4, alpha=0.5)
    ax3.tick_params(labelsize=8)
    _shade_event(ax3, event_start if event_mask.any() else None,
                      event_end   if event_mask.any() else None)

    # ── Panel 4: Quantisation error ───────────────────────────────────────
    q_error_mv = (adc_vin - v_rec) * 1000
    lsb_mv     = VREF / ADC_CODES * 1000
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax4.step(adc_us, q_error_mv, color='#8E44AD', lw=0.8, where='post',
             label='Quantisation error')
    ax4.axhline( lsb_mv / 2, color='#E74C3C', lw=0.7, ls='--', label=f'+0.5 LSB ({lsb_mv/2:.3f} mV)')
    ax4.axhline(-lsb_mv / 2, color='#E74C3C', lw=0.7, ls='--', label=f'−0.5 LSB')
    ax4.axhline(0, color='#BDC3C7', lw=0.5)
    ax4.set_xlabel('Time (µs)', fontsize=9)
    ax4.set_ylabel('Error (mV)', fontsize=9)
    ax4.set_title('Quantisation Error  (V_in − V_reconstructed)', fontsize=9, loc='left', pad=3)
    ax4.legend(fontsize=8, loc='upper right', framealpha=0.8)
    ax4.grid(True, lw=0.4, alpha=0.5)
    ax4.tick_params(labelsize=8)
    _shade_event(ax4, event_start if event_mask.any() else None,
                      event_end   if event_mask.any() else None)

    # Save
    out_path = output_dir / f"{stem}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"    Saved → {out_path.name}")

    if show:
        matplotlib.use('TkAgg')
        plt.show()

    plt.close(fig)


def _shade_event(ax, t_start, t_end):
    """Light yellow shading over the detected impact event window."""
    if t_start is not None:
        ax.axvspan(t_start, t_end, alpha=0.08, color='#F1C40F', zorder=0)


def find_pairs(script_dir):
    """
    Match processed_*mm_*.csv  ↔  ADC outputs/adc_processed_*mm_*.csv
    Returns list of (stem, ltspice_path, adc_path) tuples.
    """
    ltspice_files = sorted(glob.glob(str(script_dir / 'processed_*mm_*.csv')))
    pairs = []
    for lp in ltspice_files:
        stem     = Path(lp).stem
        adc_path = script_dir / 'ADC outputs' / f'adc_{stem}.csv'
        if adc_path.exists():
            pairs.append((stem, Path(lp), adc_path))
        else:
            print(f"  WARNING: No ADC output found for {stem}, skipping.")
    return pairs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot LTSpice input vs AD7685 ADC output')
    parser.add_argument('--show', action='store_true',
                        help='Open interactive plot windows (in addition to saving PNGs)')
    parser.add_argument('--file', default=None,
                        help='Process a single capture by stem name, e.g. processed_10mm_1')
    args = parser.parse_args()

    script_dir = Path(__file__).parent.resolve()
    output_dir = script_dir / 'plots'
    output_dir.mkdir(exist_ok=True)

    pairs = find_pairs(script_dir)
    if not pairs:
        sys.exit("ERROR: No matching LTSpice + ADC output pairs found.\n"
                 "Run ad7685_sim.py first to generate the ADC outputs/ folder.")

    if args.file:
        pairs = [(s, lp, ap) for s, lp, ap in pairs if s == args.file]
        if not pairs:
            sys.exit(f"ERROR: '{args.file}' not found in matched pairs.")

    print(f"\n── Plotting {len(pairs)} capture(s) → plots/ ─────────────────────")
    for stem, ltspice_path, adc_path in pairs:
        plot_capture(stem, ltspice_path, adc_path, output_dir, show=args.show)

    print(f"\n✓  Done — {len(pairs)} plot(s) saved to '{output_dir}/'")

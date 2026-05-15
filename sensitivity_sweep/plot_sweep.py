#!/usr/bin/env python3
"""
plot_sweep.py

Generates the sensitivity-analysis figure for thesis §6.3:
  - Left:    baseline detection rate over (K, alpha)
  - Middle:  mitigated (freeze=2σ²) detection rate over (K, alpha)
  - Right:   delta (mitigated − baseline)

Each cell shows detection rate (n/9) for the 9 PLB captures.
The current firmware operating point (K=8, alpha=0.001) is highlighted.
"""

import csv
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})

ROWS = list(csv.DictReader(open("sweep_results.csv")))

# Build (condition, K, alpha) -> detection count
agg = defaultdict(int)
total = defaultdict(int)
for r in ROWS:
    key = (r["condition"], float(r["k"]), float(r["alpha"]))
    agg[key]   += int(r["detected"])
    total[key] += 1

K_VALUES     = sorted({float(r["k"])     for r in ROWS})
ALPHA_VALUES = sorted({float(r["alpha"]) for r in ROWS}, reverse=True)  # large→small

def grid(condition):
    g = np.full((len(ALPHA_VALUES), len(K_VALUES)), np.nan)
    for i, a in enumerate(ALPHA_VALUES):
        for j, k in enumerate(K_VALUES):
            n = agg[(condition, k, a)]
            t = total[(condition, k, a)]
            if t > 0:
                g[i, j] = 100.0 * n / t
    return g

G_base = grid("baseline")
G_mit  = grid("mitigated")
G_del  = G_mit - G_base

fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6),
                          constrained_layout=True)

def draw(ax, data, title, vmin, vmax, cmap, fmt, label):
    im = ax.imshow(data, aspect="auto", origin="upper",
                    vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xticks(range(len(K_VALUES)))
    ax.set_xticklabels([f"{k:g}" for k in K_VALUES])
    ax.set_yticks(range(len(ALPHA_VALUES)))
    ax.set_yticklabels([f"{a:.0e}".replace("e-0", "e-") for a in ALPHA_VALUES])
    ax.set_xlabel(r"Trigger multiplier $K$")
    ax.set_ylabel(r"EMA smoothing $\alpha$")
    ax.set_title(title)

    # Numeric overlay
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isnan(val):
                continue
            mid = (vmin + vmax) / 2.0
            colour = "white" if (cmap.name == "RdBu_r" and abs(val) > 30) \
                    or (cmap.name == "viridis" and val < mid) else "black"
            ax.text(j, i, fmt.format(val),
                    ha="center", va="center", color=colour, fontsize=7)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(label)
    cbar.ax.tick_params(labelsize=7)

    # Highlight current firmware operating point: K=8, alpha=1e-3
    try:
        j_op = K_VALUES.index(8.0)
        i_op = ALPHA_VALUES.index(1e-3)
        ax.add_patch(plt.Rectangle((j_op - 0.5, i_op - 0.5), 1, 1,
                                    fill=False, edgecolor="black",
                                    linewidth=1.8))
    except ValueError:
        pass

draw(axes[0], G_base,
     r"(a) Baseline: no freeze",
     0, 100, plt.cm.viridis, "{:.0f}", "Detection rate (%)")
draw(axes[1], G_mit,
     r"(b) Mitigated: freeze at $P > 2\sigma_n^2$",
     0, 100, plt.cm.viridis, "{:.0f}", "Detection rate (%)")
draw(axes[2], G_del,
     r"(c) $\Delta$ = mitigated $-$ baseline",
     -50, 50, plt.cm.RdBu_r, "{:+.0f}", "Recovery (pp)")

fig.suptitle(
    r"Detection rate sensitivity over the parameter grid "
    r"($n = 9$ PLB captures per cell). "
    r"Black square: current firmware operating point.",
    fontsize=10, y=1.06,
)

fig.savefig("sensitivity_heatmaps.pdf", bbox_inches="tight", dpi=300)
fig.savefig("sensitivity_heatmaps.png", bbox_inches="tight", dpi=200)
print("Wrote sensitivity_heatmaps.pdf and .png")
print()
print(f"Operating point (K=8, alpha=1e-3):")
print(f"  baseline   : {G_base[ALPHA_VALUES.index(1e-3), K_VALUES.index(8.0)]:.0f}%")
print(f"  mitigated  : {G_mit [ALPHA_VALUES.index(1e-3), K_VALUES.index(8.0)]:.0f}%")
print(f"  recovery   : +{G_del [ALPHA_VALUES.index(1e-3), K_VALUES.index(8.0)]:.0f} percentage points")

#!/usr/bin/env python3
"""
plot_08_side_by_side.py

Side-by-side comparison: Cell vs Nuclei segmentation.
Uses fixed colors for each algorithm defined in config.py.
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# Import shared configuration
from config import (
    RESULTS_DIR, PLOTS_DIR, PNG_SUBDIR_NAME, FIGURE_DPI, TRANSPARENT_BG,
    ALGORITHM_COLORS, ALGORITHM_LINESTYLES, ALGORITHM_MARKERS,
    FONT_SIZES, get_algorithm_display_name, save_figure_with_no_legend
)

# Dedicated legend font size for the precision-IoU panels
LEGEND_FONT_SIZE = 7

# Ensure output directory exists
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
PNG_DIR = PLOTS_DIR / PNG_SUBDIR_NAME
PNG_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# STYLE
# ============================================================================

try:
    plt.style.use(['science', 'no-latex'])
except Exception:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.size": FONT_SIZES["label"],
        "axes.titlesize": FONT_SIZES["title"],
        "axes.labelsize": FONT_SIZES["label"],
        "legend.fontsize": 8,
        "xtick.labelsize": FONT_SIZES["tick"],
        "ytick.labelsize": FONT_SIZES["tick"],
        "xtick.direction": "in",
        "ytick.direction": "in",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "savefig.bbox": "tight",
        "figure.dpi": 200,
    })

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def infer_ap_cols(df):
    ap = [c for c in df.columns if re.match(r"^AP@\d\.\d{2}$", c)]
    if not ap:
        raise ValueError("No AP@xx.xx columns found")
    thr = np.array([float(c.split("@")[1]) for c in ap])
    order = np.argsort(thr)
    return thr[order], [ap[i] for i in order]

def get_algorithm_color(algo_name):
    """Get fixed color for algorithm."""
    return ALGORITHM_COLORS.get(algo_name, "#000000")

def get_algorithm_linestyle(algo_name):
    """Get line style for algorithm."""
    return ALGORITHM_LINESTYLES.get(algo_name, "-")

def get_algorithm_marker(algo_name):
    """Get marker style for algorithm."""
    return ALGORITHM_MARKERS.get(algo_name, "o")

def compute_curve(per_img):
    thr, ap_cols = infer_ap_cols(per_img)
    
    # Ensure we have the full range of thresholds
    shared_thr = np.round(np.arange(0.50, 0.96, 0.05), 2)
    target_cols = [f"AP@{t:.2f}" for t in shared_thr]
    
    curve = per_img.groupby("algorithm")[ap_cols].mean()
    
    # Add missing columns if needed
    for col in target_cols:
        if col not in curve.columns:
            curve[col] = np.nan
    
    curve = curve[target_cols]
    mAP = curve.mean(axis=1)
    
    # Sort by mAP (descending)
    curve = curve.loc[mAP.sort_values(ascending=False).index]
    
    return shared_thr, curve, mAP

# ============================================================================
# MAIN PLOT
# ============================================================================

def main():
    print(f"Loading data from {RESULTS_DIR}...")
    
    per_cell = pd.read_parquet(RESULTS_DIR / "cell_2ch_per_image.parquet")
    per_nuc = pd.read_parquet(RESULTS_DIR / "nuclei_per_image.parquet")
    
    thr_cell, curve_cell, mAP_cell = compute_curve(per_cell)
    thr_nuc, curve_nuc, mAP_nuc = compute_curve(per_nuc)
    
    # Use the same threshold range
    thr = thr_cell
    
    # Create side-by-side plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    axL, axR = axes
    
    # Left panel: Cell segmentation
    for algo, row in curve_cell.iterrows():
        axL.plot(
            thr, row.values,
            color=get_algorithm_color(algo),
            linestyle=get_algorithm_linestyle(algo),
            marker=get_algorithm_marker(algo),
            linewidth=2,
            markersize=5,
            label=f"{get_algorithm_display_name(algo)} (mAP={mAP_cell[algo]:.2f})"
        )
    
    axL.set_xlabel("Precision")
    axL.set_ylabel("Precision")
    axL.set_xlim(thr.min(), thr.max())
    axL.set_ylim(0, 1.0)
    axL.minorticks_on()
    axL.grid(alpha=0.3)
    axL.legend(frameon=False, fontsize=LEGEND_FONT_SIZE)
    axL.text(
        0.02, 0.98, "Cell benchmark",
        transform=axL.transAxes, ha='left', va='top',
        fontsize=11, fontweight='bold'
    )
    
    # Right panel: Nuclei segmentation
    for algo, row in curve_nuc.iterrows():
        axR.plot(
            thr, row.values,
            color=get_algorithm_color(algo),
            linestyle=get_algorithm_linestyle(algo),
            marker=get_algorithm_marker(algo),
            linewidth=2,
            markersize=5,
            label=f"{get_algorithm_display_name(algo)} (mAP={mAP_nuc[algo]:.2f})"
        )
    
    axR.set_xlabel("Precision")
    axR.set_xlim(thr.min(), thr.max())
    axR.set_ylim(0, 1.0)
    axR.minorticks_on()
    axR.grid(alpha=0.3)
    axR.legend(frameon=False, fontsize=LEGEND_FONT_SIZE)
    axR.text(
        0.02, 0.98, "Nuclei benchmark",
        transform=axR.transAxes, ha='left', va='top',
        fontsize=11, fontweight='bold'
    )
    
    plt.tight_layout()
    
    out_pdf = PLOTS_DIR / "cell_vs_nuclei_side_by_side.pdf"
    out_png = PNG_DIR / "cell_vs_nuclei_side_by_side.png"
    # Caption: Precision versus IoU comparison for cell and nuclei benchmarks in parallel.
    save_figure_with_no_legend(
        fig, out_pdf, out_png,
        dpi=FIGURE_DPI,
        transparent=TRANSPARENT_BG
    )
    
    print(f"✓ Saved: {out_pdf}")
    print(f"✓ Saved: {out_png}")
    plt.show()

if __name__ == "__main__":
    main()

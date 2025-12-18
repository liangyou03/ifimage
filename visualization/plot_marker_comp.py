#!/usr/bin/env python3
"""
plot_03_marker_comparison.py

Compare 2-channel vs marker-only for OLIG2 cells.
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

# Dedicated legend font size for these precision-IoU panels
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
        "legend.fontsize": 7,  # Smaller for side-by-side
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

def get_algorithm_linestyle(algo_name, variant="2channel"):
    """
    Get line style for algorithm.
    For marker-only, use dashed line to differentiate.
    """
    if variant == "markeronly":
        return "--"  # Dashed for marker-only
    return ALGORITHM_LINESTYLES.get(algo_name, "-")

def get_algorithm_marker(algo_name):
    """Get marker style for algorithm."""
    return ALGORITHM_MARKERS.get(algo_name, "o")

# ============================================================================
# MAIN PLOT
# ============================================================================

def main():
    print(f"Loading data from {RESULTS_DIR}...")
    
    # Load both variants
    per_2ch = pd.read_parquet(RESULTS_DIR / "cell_2ch_per_image.parquet")
    per_marker = pd.read_parquet(RESULTS_DIR / "cell_marker_per_image.parquet")
    
    # Combine
    per_2ch["variant"] = "2channel"
    per_marker["variant"] = "markeronly"
    per_all = pd.concat([per_2ch, per_marker], ignore_index=True)
    
    # Filter OLIG2
    olig2 = per_all[per_all["base"].str.contains("OLIG2", case=False, na=False)]
    
    thr, ap_cols = infer_ap_cols(olig2)
    
    # Two-panel figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    
    variants = ["2channel", "markeronly"]
    variant_labels = {"2channel": "2-channel", "markeronly": "Marker-only"}
    algs = sorted(olig2["algorithm"].unique())
    
    for ax, variant in zip(axes, variants):
        sub = olig2[olig2["variant"] == variant]
        curve = sub.groupby("algorithm")[ap_cols].mean()
        mAP = curve.mean(axis=1)
        
        sorted_algos = sorted(mAP.index, key=lambda k: mAP[k], reverse=True)
        
        for algo in sorted_algos:
            row = curve.loc[algo]
            ax.plot(
                thr, row.values,
                color=get_algorithm_color(algo),
                linestyle=get_algorithm_linestyle(algo, variant),
                marker=get_algorithm_marker(algo),
                linewidth=1.8,
                markersize=5,
                label=f"{get_algorithm_display_name(algo)} (mAP={mAP[algo]:.3f})"
            )
        
        ax.text(
            0.02, 0.98, f"OLIG2 {variant_labels[variant]}",
            transform=ax.transAxes, ha='left', va='top',
            fontsize=11, fontweight='bold'
        )
        ax.set_xlabel("IOU")
        ax.set_xlim(thr.min(), thr.max())
        ax.set_ylim(0, 1.0)
        ax.minorticks_on()
        ax.grid(alpha=0.3)
    
    axes[0].set_ylabel("Precision")
    axes[1].legend(fontsize=LEGEND_FONT_SIZE, ncols=1, loc="center left", bbox_to_anchor=(1.02, 0.5))
    
    # fig.suptitle("OLIG2: AP curves (2channel vs marker-only)", fontsize=12)
    fig.tight_layout()
    fig.subplots_adjust(top=0.9, right=0.78)
    
    out_pdf = PLOTS_DIR / "marker_variant_comparison.pdf"
    out_png = PNG_DIR / "marker_variant_comparison.png"
    # Caption: Precision versus IoU thresholds for OLIG2 using both input variants.
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

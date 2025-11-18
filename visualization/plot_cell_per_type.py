#!/usr/bin/env python3
"""
plot_02_cell_per_type.py

Plot AP curves for each cell type (OLIG2, NEUN, IBA1, GFAP).
Uses fixed colors for each algorithm defined in config.py.
"""

import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# Import shared configuration
from config import (
    RESULTS_DIR, PLOTS_DIR, FIGURE_DPI, TRANSPARENT_BG,
    CELL_TYPE_GROUPS, ALGORITHM_COLORS, ALGORITHM_LINESTYLES, 
    ALGORITHM_MARKERS, FONT_SIZES
)

# Ensure output directory exists
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

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
        "legend.fontsize": 8,  # Smaller for facet plots
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

# ============================================================================
# MAIN PLOT
# ============================================================================

def main():
    print(f"Loading data from {RESULTS_DIR}...")
    per_img = pd.read_parquet(RESULTS_DIR / "cell_2ch_per_image.parquet")
    
    thr, ap_cols = infer_ap_cols(per_img)
    
    ncols = 2
    nrows = math.ceil(len(CELL_TYPE_GROUPS) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 10), squeeze=False)
    
    for ax, (gname, pattern) in zip(axes.ravel(), CELL_TYPE_GROUPS.items()):
        sub = per_img[per_img["base"].str.contains(pattern, case=False, regex=True, na=False)]
        
        if sub.empty:
            ax.set_title(f"Cell type: {gname} (no images)")
            ax.axis("off")
            continue
        
        curve = sub.groupby("algorithm")[ap_cols].mean()
        mAP = curve.mean(axis=1)
        
        # Sort by mAP (descending)
        sorted_algos = sorted(mAP.index, key=lambda k: mAP[k], reverse=True)
        
        for algo in sorted_algos:
            row = curve.loc[algo]
            ax.plot(
                thr, row.values,
                color=get_algorithm_color(algo),
                linestyle=get_algorithm_linestyle(algo),
                marker=get_algorithm_marker(algo),
                linewidth=2,
                markersize=5,
                label=f"{algo} (mAP={mAP[algo]:.2f})"
            )
        
        ax.set_title(f"Cell type: {gname}")
        ax.set_xlabel("IoU Threshold")
        ax.set_ylabel("Average Precision")
        ax.set_xlim(thr.min(), thr.max())
        ax.set_ylim(0, 1.0)
        ax.minorticks_on()
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=7)
    
    # Hide unused axes
    for j in range(len(CELL_TYPE_GROUPS), nrows * ncols):
        axes.ravel()[j].axis("off")
    
    plt.tight_layout()
    
    out_pdf = PLOTS_DIR / "02_cell_per_type.pdf"
    out_png = PLOTS_DIR / "02_cell_per_type.png"
    fig.savefig(out_pdf, format="pdf", transparent=TRANSPARENT_BG)
    fig.savefig(out_png, format="png", dpi=FIGURE_DPI, transparent=TRANSPARENT_BG)
    
    print(f"✓ Saved: {out_pdf}")
    print(f"✓ Saved: {out_png}")
    plt.show()

if __name__ == "__main__":
    main()
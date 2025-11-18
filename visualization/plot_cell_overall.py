#!/usr/bin/env python3
"""
plot_01_cell_overall.py

Plot overall AP curves for cell segmentation (2-channel).
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# Import shared configuration
from config import RESULTS_DIR, PLOTS_DIR, FIGURE_DPI, TRANSPARENT_BG

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
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
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
    """Find AP columns like 'AP@0.50', 'AP@0.55', etc."""
    ap = [c for c in df.columns if re.match(r"^AP@\d\.\d{2}$", c)]
    if not ap:
        raise ValueError("No AP@xx.xx columns found")
    thr = np.array([float(c.split("@")[1]) for c in ap])
    order = np.argsort(thr)
    return thr[order], [ap[i] for i in order]

# ============================================================================
# MAIN PLOT
# ============================================================================

def main():
    # Load data
    print(f"Loading data from {RESULTS_DIR}...")
    per_img = pd.read_parquet(RESULTS_DIR / "cell_2ch_per_image.parquet")
    
    if per_img.empty:
        print("ERROR: No data found!")
        return
    
    # Compute AP curves
    thr, ap_cols = infer_ap_cols(per_img)
    curve = per_img.groupby("algorithm")[ap_cols].mean()
    mAP = curve.mean(axis=1)
    
    # Sort by mAP (descending)
    curve = curve.loc[mAP.sort_values(ascending=False).index]
    
    # Plot
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    
    for algo, row in curve.iterrows():
        ax.plot(thr, row.values, marker='o', linewidth=2,
                label=f"{algo} (mAP={mAP[algo]:.3f})")
    
    ax.set_xlabel("IoU Threshold")
    ax.set_ylabel("Average Precision")
    ax.set_xlim(thr.min(), thr.max())
    ax.set_ylim(0, 1.0)
    ax.minorticks_on()
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    
    plt.tight_layout()
    
    # Save
    out_pdf = PLOTS_DIR / "01_cell_overall_ap.pdf"
    out_png = PLOTS_DIR / "01_cell_overall_ap.png"
    fig.savefig(out_pdf, format="pdf", transparent=TRANSPARENT_BG)
    fig.savefig(out_png, format="png", dpi=FIGURE_DPI, transparent=TRANSPARENT_BG)
    
    print(f"✓ Saved: {out_pdf}")
    print(f"✓ Saved: {out_png}")
    plt.show()

if __name__ == "__main__":
    main()

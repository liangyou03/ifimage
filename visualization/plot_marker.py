#!/usr/bin/env python3
"""
plot_03_marker_comparison_all.py

Compare 2-channel vs marker-only for all cell types.
Includes: 
  - Pooled (all cell types combined)
  - Per cell type (OLIG2, NEUN, IBA1, GFAP)

No titles, no Chinese text - publication ready.
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
    RESULTS_DIR, PLOTS_DIR, PNG_SUBDIR_NAME, FIGURE_DPI, TRANSPARENT_BG,
    CELL_TYPE_GROUPS, ALGORITHM_COLORS, ALGORITHM_LINESTYLES, 
    ALGORITHM_MARKERS, FONT_SIZES, get_algorithm_display_name,
    save_figure_with_no_legend
)

# Dedicated legend sizes for these precision-IoU figures
LEGEND_FONT_SIZE = 7
LEGEND_FONT_SIZE_SMALL = 6
LEGEND_FONT_SIZE_COMPACT = 9

# Ensure output directory exists
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
PNG_DIR = PLOTS_DIR / PNG_SUBDIR_NAME
PNG_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# STYLE CONFIGURATION
# ============================================================================

try:
    plt.style.use(['science', 'no-latex'])
except Exception:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.size": FONT_SIZES["label"],
        "axes.titlesize": FONT_SIZES["title"],
        "axes.labelsize": FONT_SIZES["label"],
        "legend.fontsize": 7,
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
    """
    Find AP columns like 'AP@0.50', 'AP@0.55', etc.
    """
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
    Use dashed line for marker-only to differentiate.
    """
    if variant == "markeronly":
        return "--"
    return ALGORITHM_LINESTYLES.get(algo_name, "-")


def get_algorithm_marker(algo_name):
    """Get marker style for algorithm."""
    return ALGORITHM_MARKERS.get(algo_name, "o")


def plot_comparison_panel(ax, data, variant, thr, ap_cols, show_legend=False):
    """
    Plot AP curves for one variant (2channel or markeronly) on given axis.
    
    Args:
        ax: matplotlib axis
        data: DataFrame filtered to specific cell type(s) and variant
        variant: "2channel" or "markeronly"
        thr: IoU thresholds
        ap_cols: AP column names
        show_legend: whether to show legend
    """
    if data.empty:
        ax.text(0.5, 0.5, "No data", ha='center', va='center', 
                transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        return
    
    # Compute mean AP curve for each algorithm
    curve = data.groupby("algorithm")[ap_cols].mean()
    mAP = curve.mean(axis=1)
    
    # Sort by mAP (descending)
    sorted_algos = sorted(mAP.index, key=lambda k: mAP[k], reverse=True)
    
    # Plot each algorithm
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
    
    # Axis formatting
    ax.set_xlim(thr.min(), thr.max())
    ax.set_ylim(0, 1.0)
    ax.minorticks_on()
    ax.grid(alpha=0.3)
    
    # Legend
    if show_legend:
        ax.legend(fontsize=LEGEND_FONT_SIZE_SMALL, ncols=1, loc="center left", 
                 bbox_to_anchor=(1.02, 0.5), frameon=False)


# ============================================================================
# MAIN PLOTTING FUNCTIONS
# ============================================================================

def plot_pooled_comparison():
    """
    Plot pooled comparison (all cell types combined).
    Creates a 1x2 figure: 2-channel vs marker-only
    """
    print("Creating pooled comparison plot...")
    
    # Load data
    per_2ch = pd.read_parquet(RESULTS_DIR / "cell_2ch_per_image.parquet")
    per_marker = pd.read_parquet(RESULTS_DIR / "cell_marker_per_image.parquet")
    
    # Tag variants
    per_2ch["variant"] = "2channel"
    per_marker["variant"] = "markeronly"
    
    # Combine all data
    per_all = pd.concat([per_2ch, per_marker], ignore_index=True)
    
    # Get thresholds
    thr, ap_cols = infer_ap_cols(per_all)
    
    # Create figure with 1x2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    
    # Left: 2-channel
    data_2ch = per_all[per_all["variant"] == "2channel"]
    plot_comparison_panel(axes[0], data_2ch, "2channel", thr, ap_cols, 
                         show_legend=False)
    axes[0].set_ylabel("Precision")
    axes[0].set_xlabel("IOU")
    axes[0].text(0.05, 0.95, "2-channel", transform=axes[0].transAxes,
                fontsize=11, va='top', fontweight='bold')
    
    # Right: marker-only
    data_marker = per_all[per_all["variant"] == "markeronly"]
    plot_comparison_panel(axes[1], data_marker, "markeronly", thr, ap_cols, 
                         show_legend=True)
    axes[1].set_xlabel("IOU")
    axes[1].text(0.05, 0.95, "Marker-only", transform=axes[1].transAxes,
                fontsize=11, va='top', fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    fig.subplots_adjust(right=0.78)
    
    # Save
    out_pdf = PLOTS_DIR / "marker_comparison_pooled.pdf"
    out_png = PNG_DIR / "marker_comparison_pooled.png"
    # Caption: Precision versus IoU thresholds comparing pooled 2-channel and marker-only variants.
    save_figure_with_no_legend(
        fig, out_pdf, out_png,
        dpi=FIGURE_DPI,
        transparent=TRANSPARENT_BG
    )
    
    print(f"✓ Saved: {out_pdf}")
    print(f"✓ Saved: {out_png}")
    plt.close(fig)


def plot_per_celltype_comparison():
    """
    Plot per-cell-type comparison.
    Creates a grid: rows = cell types, cols = variants (2ch, marker)
    """
    print("Creating per-cell-type comparison plot...")
    
    # Load data
    per_2ch = pd.read_parquet(RESULTS_DIR / "cell_2ch_per_image.parquet")
    per_marker = pd.read_parquet(RESULTS_DIR / "cell_marker_per_image.parquet")
    
    # Tag variants
    per_2ch["variant"] = "2channel"
    per_marker["variant"] = "markeronly"
    
    # Combine
    per_all = pd.concat([per_2ch, per_marker], ignore_index=True)
    
    # Get thresholds
    thr, ap_cols = infer_ap_cols(per_all)
    
    # Create figure: rows = cell types, cols = 2 (2ch, marker)
    n_types = len(CELL_TYPE_GROUPS)
    fig, axes = plt.subplots(n_types, 2, figsize=(10, 3*n_types), 
                            sharey=True, sharex=True)
    
    # Iterate over cell types
    for i, (cell_type, pattern) in enumerate(CELL_TYPE_GROUPS.items()):
        # Filter data for this cell type
        cell_data = per_all[per_all["base"].str.contains(pattern, 
                                                         case=False, 
                                                         na=False)]
        
        # Left column: 2-channel
        data_2ch = cell_data[cell_data["variant"] == "2channel"]
        plot_comparison_panel(axes[i, 0], data_2ch, "2channel", thr, ap_cols,
                             show_legend=False)
        
        # Right column: marker-only
        data_marker = cell_data[cell_data["variant"] == "markeronly"]
        show_legend = (i == 0)  # Only show legend on first row
        plot_comparison_panel(axes[i, 1], data_marker, "markeronly", thr, ap_cols,
                             show_legend=show_legend)
        
        # Add cell type label on left
        axes[i, 0].text(-0.15, 0.5, cell_type, transform=axes[i, 0].transAxes,
                       fontsize=12, va='center', ha='right', fontweight='bold',
                       rotation=90)
        
        # Y-label only on leftmost column
        if i == n_types // 2:
            axes[i, 0].set_ylabel("Precision")
    
    # Column titles (top row only)
    axes[0, 0].text(0.5, 1.1, "2-channel", transform=axes[0, 0].transAxes,
                   fontsize=12, ha='center', fontweight='bold')
    axes[0, 1].text(0.5, 1.1, "Marker-only", transform=axes[0, 1].transAxes,
                   fontsize=12, ha='center', fontweight='bold')
    
    # X-labels on bottom row only
    axes[-1, 0].set_xlabel("IOU")
    axes[-1, 1].set_xlabel("Precision")
    
    # Adjust layout
    plt.tight_layout()
    fig.subplots_adjust(right=0.85, hspace=0.3, top=0.95, left=0.12)
    
    # Save
    out_pdf = PLOTS_DIR / "marker_comparison_per_type.pdf"
    out_png = PNG_DIR / "marker_comparison_per_type.png"
    # Caption: Precision versus IoU thresholds per cell type for both input variants.
    save_figure_with_no_legend(
        fig, out_pdf, out_png,
        dpi=FIGURE_DPI,
        transparent=TRANSPARENT_BG
    )
    
    print(f"✓ Saved: {out_pdf}")
    print(f"✓ Saved: {out_png}")
    plt.close(fig)


def plot_compact_comparison():
    """
    Plot a compact version: all cell types in one figure.
    Layout: 2 rows x 3 columns
    Row 1: Pooled, OLIG2, NEUN
    Row 2: IBA1, GFAP, (empty)
    Each subplot shows both 2ch (solid) and marker (dashed)
    """
    print("Creating compact comparison plot...")
    
    # Load data
    per_2ch = pd.read_parquet(RESULTS_DIR / "cell_2ch_per_image.parquet")
    per_marker = pd.read_parquet(RESULTS_DIR / "cell_marker_per_image.parquet")
    
    # Tag variants
    per_2ch["variant"] = "2channel"
    per_marker["variant"] = "markeronly"
    
    # Combine
    per_all = pd.concat([per_2ch, per_marker], ignore_index=True)
    
    # Get thresholds
    thr, ap_cols = infer_ap_cols(per_all)
    
    # Create figure: 2x3 grid
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True, sharex=True)
    axes = axes.ravel()
    
    # Plot configurations
    plot_configs = [
        ("Pooled", None),  # All data
        ("OLIG2", "OLIG2"),
        ("NEUN", "NEUN"),
        ("IBA1", "IBA1"),
        ("GFAP", "GFAP"),
    ]
    
    for idx, (title, cell_type) in enumerate(plot_configs):
        ax = axes[idx]
        
        # Filter data
        if cell_type is None:
            # Pooled: all data
            cell_data = per_all
        else:
            # Specific cell type
            cell_data = per_all[per_all["base"].str.contains(cell_type, 
                                                             case=False, 
                                                             na=False)]
        
        # Plot both variants on same axis
        for variant, linestyle_override in [("2channel", "-"), 
                                             ("markeronly", "--")]:
            data = cell_data[cell_data["variant"] == variant]
            
            if data.empty:
                continue
            
            curve = data.groupby("algorithm")[ap_cols].mean()
            mAP = curve.mean(axis=1)
            sorted_algos = sorted(mAP.index, key=lambda k: mAP[k], reverse=True)
            
            # Only show top 5 algorithms to avoid clutter
            for algo in sorted_algos[:5]:
                row = curve.loc[algo]
                label = get_algorithm_display_name(algo) if variant == "2channel" else None
                ax.plot(
                    thr, row.values,
                    color=get_algorithm_color(algo),
                    linestyle=linestyle_override,
                    marker=get_algorithm_marker(algo),
                    linewidth=1.5,
                    markersize=4,
                    label=label
                )
        
        # Formatting
        ax.text(0.05, 0.95, title, transform=ax.transAxes,
               fontsize=11, va='top', fontweight='bold')
        ax.set_xlim(thr.min(), thr.max())
        ax.set_ylim(0, 1.0)
        ax.minorticks_on()
        ax.grid(alpha=0.3)
        
        # Legend only on first plot
        if idx == 0:
            # Custom legend for line styles
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='black', linestyle='-', 
                       label='2-channel', linewidth=1.5),
                Line2D([0], [0], color='black', linestyle='--', 
                       label='Marker-only', linewidth=1.5),
            ]
            ax.legend(handles=legend_elements, loc='lower left', 
                     fontsize=LEGEND_FONT_SIZE_COMPACT, frameon=False)
    
    # Hide last subplot
    axes[5].axis('off')
    
    # Add common labels
    fig.text(0.5, 0.02, 'Precision', ha='center', fontsize=12)
    fig.text(0.02, 0.5, 'Precision', va='center', 
            rotation='vertical', fontsize=12)
    
    # Adjust layout
    plt.tight_layout(rect=[0.03, 0.03, 1, 1])
    
    # Save
    out_pdf = PLOTS_DIR / "marker_comparison_compact.pdf"
    out_png = PNG_DIR / "marker_comparison_compact.png"
    # Caption: Compact precision versus IoU overview for pooled and per-cell-type panels.
    save_figure_with_no_legend(
        fig, out_pdf, out_png,
        dpi=FIGURE_DPI,
        transparent=TRANSPARENT_BG
    )
    
    print(f"✓ Saved: {out_pdf}")
    print(f"✓ Saved: {out_png}")
    plt.close(fig)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """
    Generate all three comparison plots:
    1. Pooled (all cell types)
    2. Per cell type (grid layout)
    3. Compact (all in one figure)
    """
    print("="*70)
    print("2-Channel vs Marker-Only Comparison")
    print("="*70)
    print()
    
    # Generate plots
    plot_pooled_comparison()
    print()
    
    plot_per_celltype_comparison()
    print()
    
    plot_compact_comparison()
    print()
    
    print("="*70)
    print("All plots completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()

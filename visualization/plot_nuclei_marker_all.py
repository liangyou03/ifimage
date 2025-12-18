#!/usr/bin/env python3
"""
plot_nuclei_marker_all.py

Generate all plots for:
  1. Nuclei segmentation
  2. Cell segmentation with marker-only input

Saves results to separate folders:
  - plots/nuclei/
  - plots/marker/

Publication-ready: No Chinese text.
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
    ALGORITHM_MARKERS, DEFAULT_FIGSIZE, FONT_SIZES, get_algorithm_display_name,
    save_figure_with_no_legend
)

# Legend sizes for precision-IoU plots in this module
LEGEND_FONT_SIZE = 9
LEGEND_FONT_SIZE_SMALL = 7

# Create output directories
NUCLEI_DIR = PLOTS_DIR / "nuclei"
MARKER_DIR = PLOTS_DIR / "marker"
NUCLEI_DIR.mkdir(parents=True, exist_ok=True)
MARKER_DIR.mkdir(parents=True, exist_ok=True)
NUCLEI_PNG_DIR = NUCLEI_DIR / PNG_SUBDIR_NAME
MARKER_PNG_DIR = MARKER_DIR / PNG_SUBDIR_NAME
NUCLEI_PNG_DIR.mkdir(parents=True, exist_ok=True)
MARKER_PNG_DIR.mkdir(parents=True, exist_ok=True)
ROOT_PNG_DIR = PLOTS_DIR / PNG_SUBDIR_NAME
ROOT_PNG_DIR.mkdir(parents=True, exist_ok=True)

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
        "legend.fontsize": FONT_SIZES["legend"],
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
    """Find AP columns like 'AP@0.50', 'AP@0.55', etc."""
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


def compute_precision_recall(df):
    """Compute precision and recall at IoU@0.50 from TP, FP, FN."""
    df = df.copy()
    df['Precision@0.50'] = df['TP@0.50'] / (df['TP@0.50'] + df['FP@0.50'])
    df['Precision@0.50'] = df['Precision@0.50'].fillna(0)
    df['Recall@0.50'] = df['TP@0.50'] / (df['TP@0.50'] + df['FN@0.50'])
    df['Recall@0.50'] = df['Recall@0.50'].fillna(0)
    return df


def sanitize_metric_name(metric_name):
    """Create a lowercase, digit-free filename fragment from a metric label."""
    safe = metric_name.lower()
    replacements = {
        ' ': '_',
        '(': '',
        ')': '',
        '=': '',
        '@': 'at'
    }
    for old, new in replacements.items():
        safe = safe.replace(old, new)
    safe = ''.join(ch for ch in safe if ch.isalpha() or ch == '_')
    safe = re.sub(r'_+', '_', safe).strip('_')
    return safe or "metric"


def create_boxplot_with_colors(ax, data, algorithms, metric_col, ylabel, 
                               show_xlabel=True, sort_by_median=True):
    """Create a box plot with algorithm-specific colors."""
    plot_data = data[data['algorithm'].isin(algorithms)].copy()
    
    if plot_data.empty:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                transform=ax.transAxes)
        return
    
    if sort_by_median:
        medians = plot_data.groupby('algorithm')[metric_col].median()
        algorithms = medians.sort_values(ascending=False).index.tolist()
    
    box_data = [plot_data[plot_data['algorithm'] == algo][metric_col].values 
                for algo in algorithms]
    
    bp = ax.boxplot(box_data, patch_artist=True, widths=0.6)
    
    for patch, algo in zip(bp['boxes'], algorithms):
        color = get_algorithm_color(algo)
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        patch.set_edgecolor('black')
        patch.set_linewidth(1)
    
    for element in ['whiskers', 'fliers', 'caps']:
        plt.setp(bp[element], color='black', linewidth=1)
    plt.setp(bp['medians'], color='black', linewidth=2)
    
    ax.set_ylabel(ylabel)
    if show_xlabel:
        ax.set_xlabel('Algorithm')
    ax.set_xticks(range(1, len(algorithms) + 1))
    display_labels = [get_algorithm_display_name(algo) for algo in algorithms]
    ax.set_xticklabels(display_labels, rotation=45, ha='right')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)


def create_pooled_metric_grid(per_img, algorithms, metrics, out_pdf, out_png, caption):
    """Create a stacked grid of pooled box plots for multiple metrics."""
    if per_img.empty:
        print("  WARNING: No data available for pooled grid")
        return
    
    rows = len(metrics)
    fig, axes = plt.subplots(rows, 1, figsize=(8, 4 * rows), squeeze=False)
    axes = axes.ravel()
    
    for ax, (metric_col, metric_name) in zip(axes, metrics):
        create_boxplot_with_colors(
            ax, per_img, algorithms, metric_col, metric_name,
            show_xlabel=True, sort_by_median=True
        )
    
    fig.tight_layout()
    
    save_figure_with_no_legend(
        fig, out_pdf, out_png,
        dpi=FIGURE_DPI,
        transparent=TRANSPARENT_BG
    )
    print(f"  Saved: {out_pdf}")
    plt.close(fig)


# ============================================================================
# NUCLEI SEGMENTATION PLOTS
# ============================================================================

def plot_nuclei_overall_ap():
    """Plot overall AP curves for nuclei segmentation."""
    print("Creating nuclei overall AP plot...")
    
    per_img = pd.read_parquet(RESULTS_DIR / "nuclei_per_image.parquet")
    
    if per_img.empty:
        print("  WARNING: No nuclei data found")
        return
    
    thr, ap_cols = infer_ap_cols(per_img)
    curve = per_img.groupby("algorithm")[ap_cols].mean()
    mAP = curve.mean(axis=1)
    curve = curve.loc[mAP.sort_values(ascending=False).index]
    
    fig = plt.figure(figsize=DEFAULT_FIGSIZE)
    ax = fig.add_subplot(111)
    
    for algo, row in curve.iterrows():
        ax.plot(
            thr, row.values,
            color=get_algorithm_color(algo),
            linestyle=get_algorithm_linestyle(algo),
            marker=get_algorithm_marker(algo),
            linewidth=2, markersize=6,
            label=f"{get_algorithm_display_name(algo)} (mAP={mAP[algo]:.3f})"
        )
    
    ax.set_xlabel("IOU")
    ax.set_ylabel("Precision")
    ax.set_xlim(thr.min(), thr.max())
    ax.set_ylim(0, 1.0)
    ax.minorticks_on()
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=LEGEND_FONT_SIZE)
    plt.tight_layout()
    
    out_pdf = NUCLEI_DIR / "nuclei_overall_ap.pdf"
    out_png = NUCLEI_PNG_DIR / "nuclei_overall_ap.png"
    # Caption: Precision versus IoU thresholds for pooled nuclei segmentation.
    save_figure_with_no_legend(
        fig, out_pdf, out_png,
        dpi=FIGURE_DPI,
        transparent=TRANSPARENT_BG
    )
    print(f"  Saved: {out_pdf}")
    plt.close(fig)


def plot_nuclei_per_type():
    """Plot AP curves per cell type for nuclei."""
    print("Creating nuclei per-cell-type AP plots...")
    
    per_img = pd.read_parquet(RESULTS_DIR / "nuclei_per_image.parquet")
    
    if per_img.empty:
        print("  WARNING: No nuclei data found")
        return
    
    thr, ap_cols = infer_ap_cols(per_img)
    
    ncols = 2
    nrows = math.ceil(len(CELL_TYPE_GROUPS) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 10), squeeze=False)
    
    for ax, (gname, pattern) in zip(axes.ravel(), CELL_TYPE_GROUPS.items()):
        sub = per_img[per_img["base"].str.contains(pattern, case=False, 
                                                   regex=True, na=False)]
        
        if sub.empty:
            ax.text(
                0.5, 0.5, f"{gname}\nNo data",
                transform=ax.transAxes, ha='center', va='center',
                fontsize=11, fontweight='bold'
            )
            ax.axis("off")
            continue
        
        curve = sub.groupby("algorithm")[ap_cols].mean()
        mAP = curve.mean(axis=1)
        sorted_algos = sorted(mAP.index, key=lambda k: mAP[k], reverse=True)
        
        for algo in sorted_algos:
            row = curve.loc[algo]
            ax.plot(
                thr, row.values,
                color=get_algorithm_color(algo),
                linestyle=get_algorithm_linestyle(algo),
                marker=get_algorithm_marker(algo),
                linewidth=2, markersize=5,
                label=f"{get_algorithm_display_name(algo)} (mAP={mAP[algo]:.2f})"
            )
        
        ax.text(0.05, 0.95, gname, transform=ax.transAxes,
               fontsize=11, va='top', fontweight='bold')
        ax.set_xlabel("IOU")
        ax.set_ylabel("Precision")
        ax.set_xlim(thr.min(), thr.max())
        ax.set_ylim(0, 1.0)
        ax.minorticks_on()
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=LEGEND_FONT_SIZE_SMALL)
    
    for j in range(len(CELL_TYPE_GROUPS), nrows * ncols):
        axes.ravel()[j].axis("off")
    
    plt.tight_layout()
    
    out_pdf = NUCLEI_DIR / "nuclei_per_type.pdf"
    out_png = NUCLEI_PNG_DIR / "nuclei_per_type.png"
    # Caption: Precision versus IoU thresholds for nuclei segmentation broken down by cell type.
    save_figure_with_no_legend(
        fig, out_pdf, out_png,
        dpi=FIGURE_DPI,
        transparent=TRANSPARENT_BG
    )
    print(f"  Saved: {out_pdf}")
    plt.close(fig)



def plot_nuclei_metrics_pooled():
    """Plot metrics for nuclei (pooled across all cell types)."""
    print("Creating nuclei metrics per celltype (pooled)...")
    
    per_img = pd.read_parquet(RESULTS_DIR / "nuclei_per_image.parquet")
    
    if per_img.empty:
        print("  WARNING: No nuclei data found")
        return
    
    per_img = compute_precision_recall(per_img)
    algorithms = sorted(per_img['algorithm'].unique())
    
    metrics = [
        ('BF_bestF', 'Boundary F-score'),
        ('Precision@0.50', 'Precision (IoU=0.50)'),
        ('Recall@0.50', 'Recall (IoU=0.50)'),
    ]
    
    for metric_col, metric_name in metrics:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        create_boxplot_with_colors(
            ax, per_img, algorithms, metric_col, metric_name,
            show_xlabel=True, sort_by_median=True
        )
        
        plt.tight_layout()
        
        filename_safe = sanitize_metric_name(metric_name)
        out_pdf = NUCLEI_DIR / f"nuclei_metric_{filename_safe}_pooled.pdf"
        out_png = NUCLEI_PNG_DIR / f"nuclei_metric_{filename_safe}_pooled.png"
        # Caption: Distribution of {metric_name} for nuclei segmentation (pooled).
        save_figure_with_no_legend(
            fig, out_pdf, out_png,
            dpi=FIGURE_DPI,
            transparent=TRANSPARENT_BG
        )
        print(f"  Saved: {out_pdf}")
        plt.close(fig)


def plot_nuclei_metrics_per_celltype():
    """Plot metrics grid for nuclei per cell type (no pooled)."""
    print("Creating nuclei metrics per celltype (separated)...")
    
    per_img = pd.read_parquet(RESULTS_DIR / "nuclei_per_image.parquet")
    
    if per_img.empty:
        print("  WARNING: No nuclei data found")
        return
    
    per_img = compute_precision_recall(per_img)
    algorithms = sorted(per_img['algorithm'].unique())
    
    metrics = [
        ('BF_bestF', 'Boundary F-score'),
        ('Precision@0.50', 'Precision (IoU=0.50)'),
        ('Recall@0.50', 'Recall (IoU=0.50)'),
    ]
    
    # One figure per metric, with subplots for each cell type
    for metric_col, metric_name in metrics:
        n_cols = 2
        n_rows = math.ceil(len(CELL_TYPE_GROUPS) / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows), squeeze=False)
        
        for ax, (cell_type, pattern) in zip(axes.ravel(), CELL_TYPE_GROUPS.items()):
            cell_data = per_img[per_img["base"].str.contains(pattern, 
                                                             case=False, 
                                                             na=False)]
            
            create_boxplot_with_colors(
                ax, cell_data, algorithms, metric_col, metric_name,
                show_xlabel=True, sort_by_median=True
            )
            
            ax.text(0.05, 0.95, cell_type, transform=ax.transAxes,
                   fontsize=12, va='top', fontweight='bold')
        
        # Hide unused subplots
        for j in range(len(CELL_TYPE_GROUPS), n_rows * n_cols):
            axes.ravel()[j].axis("off")
        
        plt.tight_layout()
        
        filename_safe = sanitize_metric_name(metric_name)
        out_pdf = NUCLEI_DIR / f"nuclei_metric_{filename_safe}_per_celltype.pdf"
        out_png = NUCLEI_PNG_DIR / f"nuclei_metric_{filename_safe}_per_celltype.png"
        # Caption: Distribution of {metric_name} for nuclei segmentation per cell type.
        save_figure_with_no_legend(
            fig, out_pdf, out_png,
            dpi=FIGURE_DPI,
            transparent=TRANSPARENT_BG
        )
        print(f"  Saved: {out_pdf}")
        plt.close(fig)


# ============================================================================
# MARKER-ONLY CELL SEGMENTATION PLOTS
# ============================================================================

def plot_marker_overall_ap():
    """Plot overall AP curves for marker-only."""
    print("Creating marker-only overall AP plot...")
    
    per_img = pd.read_parquet(RESULTS_DIR / "cell_marker_per_image.parquet")
    
    if per_img.empty:
        print("  WARNING: No marker-only data found")
        return
    
    thr, ap_cols = infer_ap_cols(per_img)
    curve = per_img.groupby("algorithm")[ap_cols].mean()
    mAP = curve.mean(axis=1)
    curve = curve.loc[mAP.sort_values(ascending=False).index]
    
    fig = plt.figure(figsize=DEFAULT_FIGSIZE)
    ax = fig.add_subplot(111)
    
    for algo, row in curve.iterrows():
        ax.plot(
            thr, row.values,
            color=get_algorithm_color(algo),
            linestyle="--",
            marker=get_algorithm_marker(algo),
            linewidth=2, markersize=6,
            label=f"{get_algorithm_display_name(algo)} (mAP={mAP[algo]:.3f})"
        )
    
    ax.set_xlabel("IOU")
    ax.set_ylabel("Precision")
    ax.set_xlim(thr.min(), thr.max())
    ax.set_ylim(0, 1.0)
    ax.minorticks_on()
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=LEGEND_FONT_SIZE)
    plt.tight_layout()
    
    out_pdf = MARKER_DIR / "marker_overall_ap.pdf"
    out_png = MARKER_PNG_DIR / "marker_overall_ap.png"
    # Caption: Precision versus IoU thresholds for marker-only segmentation (pooled).
    save_figure_with_no_legend(
        fig, out_pdf, out_png,
        dpi=FIGURE_DPI,
        transparent=TRANSPARENT_BG
    )
    print(f"  Saved: {out_pdf}")
    plt.close(fig)


def plot_marker_per_type():
    """Plot AP curves per cell type for marker-only."""
    print("Creating marker-only per-cell-type AP plots...")
    
    per_img = pd.read_parquet(RESULTS_DIR / "cell_marker_per_image.parquet")
    
    if per_img.empty:
        print("  WARNING: No marker-only data found")
        return
    
    thr, ap_cols = infer_ap_cols(per_img)
    
    ncols = 2
    nrows = math.ceil(len(CELL_TYPE_GROUPS) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 10), squeeze=False)
    
    for ax, (gname, pattern) in zip(axes.ravel(), CELL_TYPE_GROUPS.items()):
        sub = per_img[per_img["base"].str.contains(pattern, case=False, 
                                                   regex=True, na=False)]
        
        if sub.empty:
            ax.text(
                0.5, 0.5, f"{gname}\nNo data",
                transform=ax.transAxes, ha='center', va='center',
                fontsize=11, fontweight='bold'
            )
            ax.axis("off")
            continue
        
        curve = sub.groupby("algorithm")[ap_cols].mean()
        mAP = curve.mean(axis=1)
        sorted_algos = sorted(mAP.index, key=lambda k: mAP[k], reverse=True)
        
        for algo in sorted_algos:
            row = curve.loc[algo]
            ax.plot(
                thr, row.values,
                color=get_algorithm_color(algo),
                linestyle="--",
                marker=get_algorithm_marker(algo),
                linewidth=2, markersize=5,
                label=f"{get_algorithm_display_name(algo)} (mAP={mAP[algo]:.2f})"
            )
        
        ax.text(0.05, 0.95, gname, transform=ax.transAxes,
               fontsize=11, va='top', fontweight='bold')
        ax.set_xlabel("IOU")
        ax.set_ylabel("Precision")
        ax.set_xlim(thr.min(), thr.max())
        ax.set_ylim(0, 1.0)
        ax.minorticks_on()
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=LEGEND_FONT_SIZE_SMALL)
    
    for j in range(len(CELL_TYPE_GROUPS), nrows * ncols):
        axes.ravel()[j].axis("off")
    
    plt.tight_layout()
    
    out_pdf = MARKER_DIR / "marker_per_type.pdf"
    out_png = MARKER_PNG_DIR / "marker_per_type.png"
    # Caption: Precision versus IoU thresholds for marker-only segmentation across cell types.
    save_figure_with_no_legend(
        fig, out_pdf, out_png,
        dpi=FIGURE_DPI,
        transparent=TRANSPARENT_BG
    )
    print(f"  Saved: {out_pdf}")
    plt.close(fig)



def plot_marker_metrics_pooled():
    """Plot metrics for marker-only (pooled across all cell types)."""
    print("Creating marker-only metrics per celltype (pooled)...")
    
    per_img = pd.read_parquet(RESULTS_DIR / "cell_marker_per_image.parquet")
    
    if per_img.empty:
        print("  WARNING: No marker-only data found")
        return
    
    per_img = compute_precision_recall(per_img)
    algorithms = sorted(per_img['algorithm'].unique())
    
    metrics = [
        ('BF_bestF', 'Boundary F-score'),
        ('Precision@0.50', 'Precision (IoU=0.50)'),
        ('Recall@0.50', 'Recall (IoU=0.50)'),
    ]
    
    for metric_col, metric_name in metrics:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        create_boxplot_with_colors(
            ax, per_img, algorithms, metric_col, metric_name,
            show_xlabel=True, sort_by_median=True
        )
        
        plt.tight_layout()
        
        filename_safe = sanitize_metric_name(metric_name)
        out_pdf = MARKER_DIR / f"marker_metric_{filename_safe}_pooled.pdf"
        out_png = MARKER_PNG_DIR / f"marker_metric_{filename_safe}_pooled.png"
        # Caption: Distribution of {metric_name} for marker-only segmentation (pooled).
        save_figure_with_no_legend(
            fig, out_pdf, out_png,
            dpi=FIGURE_DPI,
            transparent=TRANSPARENT_BG
        )
        print(f"  Saved: {out_pdf}")
        plt.close(fig)

def create_boxplot_with_colors(ax, data, algorithms, metric_col, ylabel, 
                               show_xlabel=True, sort_by_median=True):
    """
    Create a box plot with algorithm-specific colors.
    """
    # Filter data for this metric
    plot_data = data[data['algorithm'].isin(algorithms)].copy()
    
    if plot_data.empty:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                transform=ax.transAxes)
        return
    
    # 只保留在当前数据中存在的算法
    available_algos = plot_data['algorithm'].unique()
    algorithms = [a for a in algorithms if a in available_algos]
    
    # Sort algorithms by median performance if requested
    if sort_by_median:
        medians = plot_data.groupby('algorithm')[metric_col].median()
        algorithms = medians.loc[algorithms].sort_values(ascending=False).index.tolist()
    
    # Prepare data for box plot
    box_data = [plot_data[plot_data['algorithm'] == algo][metric_col].values 
                for algo in algorithms]
    
    # Create box plot
    bp = ax.boxplot(box_data, patch_artist=True, widths=0.6)
    
    # Color boxes by algorithm - 关键：确保一一对应
    for patch, algo in zip(bp['boxes'], algorithms):
        color = ALGORITHM_COLORS.get(algo, '#808080')
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        patch.set_edgecolor('black')
        patch.set_linewidth(1)
    
    # Style other elements
    for element in ['whiskers', 'fliers', 'caps']:
        plt.setp(bp[element], color='black', linewidth=1)
    
    plt.setp(bp['medians'], color='black', linewidth=2)
    
    # Axis labels
    ax.set_ylabel(ylabel)
    if show_xlabel:
        ax.set_xlabel('Algorithm')
    
    ticks = range(1, len(algorithms) + 1)
    display_labels = [get_algorithm_display_name(algo) for algo in algorithms]
    ax.set_xticks(ticks)
    ax.set_xticklabels(display_labels, rotation=45, ha='right')
    
    # Set y-axis limits
    ax.set_ylim(-0.05, 1.05)
    
    # Grid
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    return algorithms  # 返回实际使用的算法顺序，方便调试
    
def plot_marker_metrics_per_celltype():
    """Plot metrics grid for marker-only per cell type (no pooled)."""
    print("Creating marker-only metrics per celltype (separated)...")
    
    per_img = pd.read_parquet(RESULTS_DIR / "cell_marker_per_image.parquet")
    
    if per_img.empty:
        print("  WARNING: No marker-only data found")
        return
    
    per_img = compute_precision_recall(per_img)
    algorithms = sorted(per_img['algorithm'].unique())
    
    metrics = [
        ('BF_bestF', 'Boundary F-score'),
        ('Precision@0.50', 'Precision (IoU=0.50)'),
        ('Recall@0.50', 'Recall (IoU=0.50)'),
    ]
    
    # One figure per metric, with subplots for each cell type
    for metric_col, metric_name in metrics:
        n_cols = 2
        n_rows = math.ceil(len(CELL_TYPE_GROUPS) / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows), squeeze=False)
        
        for ax, (cell_type, pattern) in zip(axes.ravel(), CELL_TYPE_GROUPS.items()):
            cell_data = per_img[per_img["base"].str.contains(pattern, 
                                                             case=False, 
                                                             na=False)]
            
            create_boxplot_with_colors(
                ax, cell_data, algorithms, metric_col, metric_name,
                show_xlabel=True, sort_by_median=True
            )
            
            ax.text(0.05, 0.95, cell_type, transform=ax.transAxes,
                   fontsize=12, va='top', fontweight='bold')
        
        # Hide unused subplots
        for j in range(len(CELL_TYPE_GROUPS), n_rows * n_cols):
            axes.ravel()[j].axis("off")
        
        plt.tight_layout()
        
        filename_safe = sanitize_metric_name(metric_name)
        out_pdf = MARKER_DIR / f"marker_metric_{filename_safe}_per_celltype.pdf"
        out_png = MARKER_PNG_DIR / f"marker_metric_{filename_safe}_per_celltype.png"
        # Caption: Distribution of {metric_name} for marker-only segmentation per cell type.
        save_figure_with_no_legend(
            fig, out_pdf, out_png,
            dpi=FIGURE_DPI,
            transparent=TRANSPARENT_BG
        )
        print(f"  Saved: {out_pdf}")
        plt.close(fig)


def plot_nuclei_metrics_comprehensive_pooled():
    """Create nuclei-only pooled comprehensive box plots (no per-cell-type panels)."""
    print("Creating nuclei pooled comprehensive box plots (no per-type panels)...")
    
    per_img = pd.read_parquet(RESULTS_DIR / "nuclei_per_image.parquet")
    if per_img.empty:
        print("  WARNING: No nuclei data found")
        return
    
    per_img = compute_precision_recall(per_img)
    algorithms = sorted(per_img['algorithm'].unique())
    metrics = [
        ('BF_bestF', 'Boundary F-score'),
        ('Precision@0.50', 'Precision (IoU=0.50)'),
        ('Recall@0.50', 'Recall (IoU=0.50)'),
    ]
    
    out_pdf = PLOTS_DIR / "metrics_boxplot_comprehensive_nuclei.pdf"
    out_png = ROOT_PNG_DIR / "metrics_boxplot_comprehensive_nuclei.png"
    # Caption: Nuclei-only pooled boundary F-score, precision, and recall distributions.
    create_pooled_metric_grid(per_img, algorithms, metrics, out_pdf, out_png,
                              "Nuclei pooled metrics grid")


def plot_marker_metrics_comprehensive_pooled():
    """Create marker-only pooled comprehensive box plots (no per-cell-type panels)."""
    print("Creating marker-only pooled comprehensive box plots (no per-type panels)...")
    
    per_img = pd.read_parquet(RESULTS_DIR / "cell_marker_per_image.parquet")
    if per_img.empty:
        print("  WARNING: No marker-only data found")
        return
    
    per_img = compute_precision_recall(per_img)
    algorithms = sorted(per_img['algorithm'].unique())
    metrics = [
        ('BF_bestF', 'Boundary F-score'),
        ('Precision@0.50', 'Precision (IoU=0.50)'),
        ('Recall@0.50', 'Recall (IoU=0.50)'),
    ]
    
    out_pdf = PLOTS_DIR / "metrics_boxplot_comprehensive_marker.pdf"
    out_png = ROOT_PNG_DIR / "metrics_boxplot_comprehensive_marker.png"
    # Caption: Marker-only pooled boundary F-score, precision, and recall distributions.
    create_pooled_metric_grid(per_img, algorithms, metrics, out_pdf, out_png,
                              "Marker-only pooled metrics grid")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 70)
    print("GENERATING NUCLEI AND MARKER SEGMENTATION PLOTS")
    print("=" * 70)
    
    # Nuclei plots
    print("\n[NUCLEI SEGMENTATION]")
    plot_nuclei_overall_ap()
    plot_nuclei_per_type()
    plot_nuclei_metrics_pooled()
    plot_nuclei_metrics_per_celltype()
    plot_nuclei_metrics_comprehensive_pooled()
    
    # Marker-only plots
    print("\n[MARKER-ONLY SEGMENTATION]")
    plot_marker_overall_ap()
    plot_marker_per_type()
    plot_marker_metrics_pooled()
    plot_marker_metrics_per_celltype()
    plot_marker_metrics_comprehensive_pooled()
    
    print("\n" + "=" * 70)
    print("ALL PLOTS GENERATED SUCCESSFULLY")
    print("=" * 70)
    print(f"Nuclei plots saved to: {NUCLEI_DIR}")
    print(f"Marker plots saved to: {MARKER_DIR}")


if __name__ == "__main__":
    main()

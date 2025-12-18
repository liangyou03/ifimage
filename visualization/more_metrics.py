#!/usr/bin/env python3
"""
plot_04_advanced_metrics.py

Advanced box plots for multiple metrics:
  - Boundary F-score (at best scale)
  - Precision and Recall (at IoU@0.50)
  
Creates two versions:
  1. Pooled (all cell types combined)
  2. Per cell type (faceted by OLIG2, NEUN, IBA1, GFAP)

No titles, no Chinese text - publication ready.
为多个指标创建高级箱线图，包含pooled和分细胞类型版本。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# Import shared configuration
from config import (
    RESULTS_DIR, PLOTS_DIR, PNG_SUBDIR_NAME, FIGURE_DPI, TRANSPARENT_BG,
    CELL_TYPE_GROUPS, ALGORITHM_COLORS, FONT_SIZES, get_algorithm_display_name,
    save_figure_with_no_legend
)

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

def compute_precision_recall(df):
    """
    Compute precision and recall at IoU@0.50 from TP, FP, FN.
    
    Args:
        df: DataFrame with columns TP@0.50, FP@0.50, FN@0.50
    
    Returns:
        DataFrame with added columns: Precision@0.50, Recall@0.50
    """
    df = df.copy()
    
    # Precision = TP / (TP + FP)
    df['Precision@0.50'] = df['TP@0.50'] / (df['TP@0.50'] + df['FP@0.50'])
    df['Precision@0.50'] = df['Precision@0.50'].fillna(0)
    
    # Recall = TP / (TP + FN)
    df['Recall@0.50'] = df['TP@0.50'] / (df['TP@0.50'] + df['FN@0.50'])
    df['Recall@0.50'] = df['Recall@0.50'].fillna(0)
    
    return df


def create_boxplot_with_colors(ax, data, algorithms, metric_col, ylabel, 
                               show_xlabel=True, sort_by_median=True):
    """
    Create a box plot with algorithm-specific colors.
    
    Args:
        ax: matplotlib axis
        data: DataFrame with 'algorithm' and metric columns
        algorithms: list of algorithm names (determines order)
        metric_col: column name for the metric
        ylabel: label for y-axis
        show_xlabel: whether to show x-axis label
        sort_by_median: sort algorithms by median value
    """
    # Filter data for this metric
    plot_data = data[data['algorithm'].isin(algorithms)].copy()
    
    if plot_data.empty:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                transform=ax.transAxes)
        return
    
    # Sort algorithms by median performance if requested
    if sort_by_median:
        medians = plot_data.groupby('algorithm')[metric_col].median()
        algorithms = medians.sort_values(ascending=False).index.tolist()
    
    # Prepare data for box plot
    box_data = [plot_data[plot_data['algorithm'] == algo][metric_col].values 
                for algo in algorithms]
    
    # Create box plot
    bp = ax.boxplot(box_data, patch_artist=True, widths=0.6)
    
    # Color boxes by algorithm
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
    ax.set_xticks(ticks, display_labels, rotation=45, ha='right')
    
    # Set y-axis limits
    ax.set_ylim(-0.05, 1.05)
    
    # Grid
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)


# ============================================================================
# POOLED PLOTS
# ============================================================================

def plot_pooled_metrics():
    """
    Create pooled box plots for all metrics.
    Layout: 2x2 grid (unused panel hidden when metrics < 4)
    """
    print("Creating pooled metrics box plots...")
    
    # Load data
    per_img = pd.read_parquet(RESULTS_DIR / "cell_2ch_per_image.parquet")
    
    # Compute precision and recall
    per_img = compute_precision_recall(per_img)
    
    # Get list of algorithms
    algorithms = sorted(per_img['algorithm'].unique())
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    # Metrics to plot
    metrics = [
        ('BF_bestF', 'Boundary F-score'),
        ('Precision@0.50', 'Precision @ IoU=0.50'),
        ('Recall@0.50', 'Recall @ IoU=0.50'),
    ]
    
    # Create each subplot
    for idx, (metric_col, ylabel) in enumerate(metrics):
        create_boxplot_with_colors(
            axes[idx], per_img, algorithms, metric_col, ylabel,
            show_xlabel=(idx >= 2),  # Only bottom row shows x-label
            sort_by_median=True
        )
        
        # Add panel label (A, B, C, D)
        panel_label = chr(65 + idx)  # A, B, C, D
        axes[idx].text(-0.1, 1.05, panel_label, transform=axes[idx].transAxes,
                      fontsize=14, fontweight='bold', va='top')
    
    for extra_ax in axes[len(metrics):]:
        extra_ax.axis("off")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save
    out_pdf = PLOTS_DIR / "metrics_boxplot_pooled.pdf"
    out_png = PNG_DIR / "metrics_boxplot_pooled.png"
    # Caption: Pooled box plots summarizing boundary F-score, precision, and recall.
    save_figure_with_no_legend(
        fig, out_pdf, out_png,
        dpi=FIGURE_DPI,
        transparent=TRANSPARENT_BG
    )
    
    print(f"✓ Saved: {out_pdf}")
    print(f"✓ Saved: {out_png}")
    plt.close(fig)


# ============================================================================
# PER-CELL-TYPE PLOTS
# ============================================================================

def plot_per_celltype_single_metric(metric_col, metric_name, filename_suffix):
    """
    Create per-cell-type box plots for a single metric.
    Layout: 1 row x N columns (one per cell type + pooled)
    
    Args:
        metric_col: column name for the metric
        metric_name: display name for the metric
        filename_suffix: suffix for output filename
    """
    print(f"Creating per-cell-type box plots for {metric_name}...")
    
    # Load data
    per_img = pd.read_parquet(RESULTS_DIR / "cell_2ch_per_image.parquet")
    
    # Compute precision and recall if needed
    if 'Precision@0.50' not in per_img.columns:
        per_img = compute_precision_recall(per_img)
    
    # Get list of algorithms
    algorithms = sorted(per_img['algorithm'].unique())
    
    # Create figure: 1 row x (N cell types + 1 pooled)
    n_panels = len(CELL_TYPE_GROUPS) + 1
    fig, axes = plt.subplots(1, n_panels, figsize=(4*n_panels, 5), 
                            sharey=True)
    
    # First panel: Pooled
    create_boxplot_with_colors(
        axes[0], per_img, algorithms, metric_col, metric_name,
        show_xlabel=True, sort_by_median=True
    )
    axes[0].text(0.5, 1.02, 'Pooled', transform=axes[0].transAxes,
                fontsize=11, ha='center', fontweight='bold')
    
    # Remaining panels: Per cell type
    for idx, (cell_type, pattern) in enumerate(CELL_TYPE_GROUPS.items(), start=1):
        # Filter data for this cell type
        cell_data = per_img[per_img["base"].str.contains(pattern, 
                                                         case=False, 
                                                         na=False)]
        
        create_boxplot_with_colors(
            axes[idx], cell_data, algorithms, metric_col, '',
            show_xlabel=True, sort_by_median=True
        )
        axes[idx].text(0.5, 1.02, cell_type, transform=axes[idx].transAxes,
                      fontsize=11, ha='center', fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save
    out_pdf = PLOTS_DIR / f"metrics_{filename_suffix}_per_type.pdf"
    out_png = PNG_DIR / f"metrics_{filename_suffix}_per_type.png"
    # Caption: Per-cell-type box plots for {metric_name}.
    save_figure_with_no_legend(
        fig, out_pdf, out_png,
        dpi=FIGURE_DPI,
        transparent=TRANSPARENT_BG
    )
    
    print(f"✓ Saved: {out_pdf}")
    print(f"✓ Saved: {out_png}")
    plt.close(fig)
    
def plot_per_celltype_all_metrics():
    """
    Create per-cell-type box plots for all metrics.
    Layout: 5 rows (pooled + 4 cell types) x 3 columns (metrics)
    """
    print("Creating comprehensive per-cell-type box plots...")
    
    # Load data
    per_img = pd.read_parquet(RESULTS_DIR / "cell_2ch_per_image.parquet")
    per_img = compute_precision_recall(per_img)
    
    # Get list of algorithms
    all_algorithms = sorted(per_img['algorithm'].unique())
    
    # Metrics to plot
    metrics = [
        ('Precision@0.50', 'Precision @ IoU=0.50'),
        ('Recall@0.50', 'Recall @ IoU=0.50'),
        ('BF_bestF', 'Boundary F-score'),
    ]
    
    # Row labels: Pooled + cell types
    row_labels = ['Pooled'] + list(CELL_TYPE_GROUPS.keys())
    
    # Create figure: 5 rows x 3 columns
    n_rows = len(CELL_TYPE_GROUPS) + 1  # +1 for pooled
    n_cols = len(metrics)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    
    # Iterate over rows (pooled + cell types)
    for row_idx in range(n_rows):
        # Get data for this row
        if row_idx == 0:
            row_data = per_img  # Pooled
            row_label = 'Pooled'
        else:
            cell_type = list(CELL_TYPE_GROUPS.keys())[row_idx - 1]
            pattern = CELL_TYPE_GROUPS[cell_type]
            row_data = per_img[per_img["base"].str.contains(pattern, case=False, na=False)]
            row_label = cell_type
        
        # Iterate over columns (metrics)
        for col_idx, (metric_col, metric_name) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            
            # 只保留在当前数据中存在的算法
            available_algos = row_data['algorithm'].unique()
            algorithms = [a for a in all_algorithms if a in available_algos]
            
            # Sort by median
            medians = row_data.groupby('algorithm')[metric_col].median()
            algorithms = medians.loc[algorithms].sort_values(ascending=False).index.tolist()
            
            # Prepare data for box plot
            box_data = [row_data[row_data['algorithm'] == algo][metric_col].values 
                        for algo in algorithms]
            
            # Create box plot
            bp = ax.boxplot(box_data, patch_artist=True, widths=0.6)
            
            # Color boxes by algorithm - 每个算法固定颜色
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
            
            # Set x-axis labels
            ticks = range(1, len(algorithms) + 1)
            display_labels = [get_algorithm_display_name(algo) for algo in algorithms]
            ax.set_xticks(ticks)
            ax.set_xticklabels(display_labels, rotation=45, ha='right', fontsize=8)
            
            # Set y-axis limits
            ax.set_ylim(-0.05, 1.05)
            
            # Grid
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_axisbelow(True)
            
            # Add metric name on top row
            if row_idx == 0:
                ax.set_title(metric_name, fontsize=12, fontweight='bold')
            
            # Add row label on first column
            if col_idx == 0:
                ax.set_ylabel(row_label, fontsize=11, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.4, wspace=0.08, bottom=0.1)
    
    # Save
    out_pdf = PLOTS_DIR / "metrics_boxplot_comprehensive.pdf"
    out_png = PNG_DIR / "metrics_boxplot_comprehensive.png"
    
    save_figure_with_no_legend(
        fig, out_pdf, out_png,
        dpi=FIGURE_DPI, transparent=TRANSPARENT_BG
    )
    
    print(f"✓ Saved: {out_pdf}")
    print(f"✓ Saved: {out_png}")
    plt.close(fig)


def plot_per_celltype_all_metrics_without_name():
    """
    Create per-cell-type box plots for all metrics.
    Layout: 5 rows (pooled + 4 cell types) x 3 columns (metrics)
    """
    print("Creating comprehensive per-cell-type box plots...")
    
    # Load data
    per_img = pd.read_parquet(RESULTS_DIR / "cell_2ch_per_image.parquet")
    per_img = compute_precision_recall(per_img)
    
    # Get list of algorithms
    algorithms = sorted(per_img['algorithm'].unique())
    
    # Metrics to plot
    metrics = [
        ('BF_bestF', 'Boundary F-score'),
        ('Precision@0.50', 'Precision @ IoU=0.50'),
        ('Recall@0.50', 'Recall @ IoU=0.50'),
    ]
    
    # Row labels: Pooled + cell types
    row_labels = ['Pooled'] + list(CELL_TYPE_GROUPS.keys())
    
    # Create figure: 5 rows x 3 columns
    n_rows = len(CELL_TYPE_GROUPS) + 1  # +1 for pooled
    n_cols = len(metrics)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows),
                            sharey=True)
    
    # Iterate over rows (pooled + cell types)
    for row_idx in range(n_rows):
        # Get data for this row
        if row_idx == 0:
            row_data = per_img  # Pooled
            row_label = 'Pooled'
        else:
            cell_type = list(CELL_TYPE_GROUPS.keys())[row_idx - 1]
            pattern = CELL_TYPE_GROUPS[cell_type]
            row_data = per_img[per_img["base"].str.contains(pattern, 
                                                            case=False, 
                                                            na=False)]
            row_label = cell_type
        
        # Iterate over columns (metrics)
        for col_idx, (metric_col, metric_name) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            
            # Create boxplot without x-labels
            create_boxplot_with_colors(
                ax, row_data, algorithms, metric_col, '',
                show_xlabel=False, sort_by_median=True
            )
            
            # Remove x-tick labels (no algorithm names)
            ax.set_xticklabels([])
            ax.set_xlabel('')
            
            # Add metric name on top row
            if row_idx == 0:
                ax.set_title(metric_name, fontsize=12, fontweight='bold')
            
            # Add row label on first column
            if col_idx == 0:
                ax.set_ylabel(row_label, fontsize=11, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.15, wspace=0.08)
    
    # Save
    out_pdf = PLOTS_DIR / "metrics_boxplot_comprehensive.pdf"
    out_png = PNG_DIR / "metrics_boxplot_comprehensive.png"
    # Caption: Comprehensive grid of metrics across pooled data and each cell type.
    save_figure_with_no_legend(
        fig, out_pdf, out_png,
        dpi=FIGURE_DPI,
        transparent=TRANSPARENT_BG
    )
    
    print(f"✓ Saved: {out_pdf}")
    print(f"✓ Saved: {out_png}")
    plt.close(fig)


# ============================================================================
# VIOLIN PLOT ALTERNATIVE
# ============================================================================

def plot_violin_comparison():
    """
    Create violin plots as an alternative to box plots.
    Shows distribution shape more clearly.
    """
    print("Creating violin plot comparison...")
    
    # Load data
    per_img = pd.read_parquet(RESULTS_DIR / "cell_2ch_per_image.parquet")
    per_img = compute_precision_recall(per_img)
    
    # Get list of algorithms (sorted by median boundary F-score)
    order_metric = 'BF_bestF'
    medians = per_img.groupby('algorithm')[order_metric].median()
    algorithms = medians.sort_values(ascending=False).index.tolist()
    
    metrics = [
        ('BF_bestF', 'Boundary F-score'),
        ('Precision@0.50', 'Precision @ IoU=0.50'),
        ('Recall@0.50', 'Recall @ IoU=0.50'),
    ]
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 6))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    
    # Create each subplot
    for idx, (metric_col, ylabel) in enumerate(metrics):
        # Prepare data
        plot_data = []
        positions = []
        colors = []
        
        for pos, algo in enumerate(algorithms):
            data = per_img[per_img['algorithm'] == algo][metric_col].values
            if len(data) > 0:
                plot_data.append(data)
                positions.append(pos)
                colors.append(ALGORITHM_COLORS.get(algo, '#808080'))
        
        # Create violin plot
        parts = axes[idx].violinplot(plot_data, positions=positions,
                                     widths=0.7, showmeans=True, 
                                     showmedians=True)
        
        # Color violins
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
            pc.set_edgecolor('black')
            pc.set_linewidth(1)
        
        # Style other elements
        for element in ['cmeans', 'cmedians', 'cbars', 'cmaxes', 'cmins']:
            if element in parts:
                parts[element].set_edgecolor('black')
                parts[element].set_linewidth(1.5)
        
        # Axis labels
        axes[idx].set_ylabel(ylabel)
        axes[idx].set_xticks(positions)
        display_labels = [get_algorithm_display_name(algo) for algo in algorithms]
        axes[idx].set_xticklabels(display_labels, rotation=45, ha='right')
        axes[idx].set_ylim(-0.05, 1.05)
        
        # Grid
        axes[idx].grid(True, alpha=0.3, axis='y')
        axes[idx].set_axisbelow(True)
        
        # Panel label
        panel_label = chr(65 + idx)
        axes[idx].text(-0.08, 1.05, panel_label, transform=axes[idx].transAxes,
                      fontsize=14, fontweight='bold', va='top')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save
    out_pdf = PLOTS_DIR / "metrics_violin_pooled.pdf"
    out_png = PNG_DIR / "metrics_violin_pooled.png"
    # Caption: Pooled violin plots highlighting full metric distributions.
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
    Generate all advanced metrics visualizations.
    """
    print("="*70)
    print("Advanced Metrics Box Plots")
    print("="*70)
    print()
    
    # 1. Pooled metrics (2x2 grid)
    plot_pooled_metrics()
    print()
    
    # 2. Per-cell-type for individual metrics
    metrics_to_plot = [
        ('BF_bestF', 'Boundary F-score', 'boundary'),
        ('Precision@0.50', 'Precision @ IoU=0.50', 'precision'),
        ('Recall@0.50', 'Recall @ IoU=0.50', 'recall'),
    ]
    
    for metric_col, metric_name, suffix in metrics_to_plot:
        plot_per_celltype_single_metric(metric_col, metric_name, suffix)
        print()
    
    # 3. Comprehensive grid (all metrics x all cell types)
    plot_per_celltype_all_metrics()
    print()
    
    # 4. Violin plot alternative
    plot_violin_comparison()
    print()
    
    print("="*70)
    print("All advanced metrics plots completed!")
    print("="*70)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
plot_00_color_legend.py

Display the fixed color scheme for all algorithms.
This helps verify that colors are consistent across all plots.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Import shared configuration
from config import (
    PLOTS_DIR, PNG_SUBDIR_NAME, ALGORITHM_COLORS, ALGORITHM_LINESTYLES, 
    ALGORITHM_MARKERS, FIGURE_DPI, get_algorithm_display_name,
    save_figure_with_no_legend, TRANSPARENT_BG
)

# Ensure output directory exists
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
PNG_DIR = PLOTS_DIR / PNG_SUBDIR_NAME
PNG_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# MAIN VISUALIZATION
# ============================================================================

def main():
    """Create a visual legend showing colors, markers, and line styles for each algorithm."""
    
    # Get all algorithms
    algos = sorted(ALGORITHM_COLORS.keys())
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
    
    # ========== LEFT PANEL: Color swatches ==========
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, len(algos))
    ax1.axis('off')
    ax1.text(
        0.5, len(algos) - 0.1, 'Algorithm colors',
        ha='center', va='bottom', fontsize=13, fontweight='bold'
    )
    
    for i, algo in enumerate(algos):
        y = len(algos) - i - 0.5
        color = ALGORITHM_COLORS[algo]
        display_name = get_algorithm_display_name(algo)
        
        # Draw color rectangle
        rect = mpatches.Rectangle((0.05, y - 0.4), 0.15, 0.8, 
                                   facecolor=color, edgecolor='black', linewidth=0.5)
        ax1.add_patch(rect)
        
        # Draw algorithm name
        ax1.text(0.25, y, display_name, va='center', ha='left', fontsize=10)
        
        # Draw hex color code
        ax1.text(0.75, y, color, va='center', ha='right', 
                fontsize=9, family='monospace', color='gray')
    
    # ========== RIGHT PANEL: Line and marker styles ==========
    ax2.set_xlim(0, 3)
    ax2.set_ylim(0, len(algos))
    ax2.axis('off')
    ax2.text(
        1.5, len(algos) - 0.1, 'Line and marker styles',
        ha='center', va='bottom', fontsize=13, fontweight='bold'
    )
    
    for i, algo in enumerate(algos):
        y = len(algos) - i - 0.5
        color = ALGORITHM_COLORS[algo]
        linestyle = ALGORITHM_LINESTYLES.get(algo, '-')
        marker = ALGORITHM_MARKERS.get(algo, 'o')
        
        # Draw sample line with markers
        x = [0.2, 1.2, 2.2]
        y_line = [y, y, y]
        ax2.plot(x, y_line, color=color, linestyle=linestyle, 
                marker=marker, markersize=8, linewidth=2, alpha=0.8)
        
        # Draw style description
        style_desc = f"{linestyle}  {marker}"
        ax2.text(2.5, y, style_desc, va='center', ha='left', 
                fontsize=9, family='monospace')
    
    plt.tight_layout()
    
    # Save
    out_pdf = PLOTS_DIR / "color_legend.pdf"
    out_png = PNG_DIR / "color_legend.png"
    # Caption: Reference colors, line styles, and markers for every algorithm.
    save_figure_with_no_legend(
        fig, out_pdf, out_png,
        dpi=FIGURE_DPI,
        transparent=TRANSPARENT_BG,
        save_kwargs={'bbox_inches': 'tight'}
    )
    
    print(f"✓ Saved: {out_pdf}")
    print(f"✓ Saved: {out_png}")
    print("\nThis reference shows the fixed colors, line styles, and markers")
    print("used for each algorithm across all plots.")
    
    plt.show()

if __name__ == "__main__":
    main()

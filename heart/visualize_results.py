"""
Heart Dataset – Core Visualization (Boxplot Only)
Produces high-quality boxplots for Object Recall, Pixel Recall, Missing Rate.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]


# ============================================================================
# Paths
# ============================================================================

RESULTS_CSV = Path("/ihome/jbwang/liy121/ifimage/heart/evaluation_results.csv")
PLOTS_DIR = Path("/ihome/jbwang/liy121/ifimage/heart/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Algorithm display config
# ============================================================================

ALGORITHM_COLORS = {
    "cellpose": "#1f77b4",
    "cellpose_sam": "#1f77b4",
    "stardist": "#ff7f0e",
    "omnipose": "#8c564b",
    "watershed": "#9467bd",
    "mesmer": "#d62728",
    "lacss": "#e377c2",
    "splinedist": "#7f7f7f",
    "microsam": "#bcbd22",
    "cellsam": "#2ca02c",
}

ALGORITHM_DISPLAY_NAMES = {
    "cellpose": "Cellpose",
    "cellpose_sam": "Cellpose-SAM",
    "stardist": "StarDist",
    "omnipose": "Omnipose",
    "watershed": "Watershed",
    "mesmer": "Mesmer",
    "lacss": "LACSS",
    "splinedist": "SplineDist",
    "microsam": "MicroSAM",
    "cellsam": "CellSAM",
}

# Fixed order for reproducibility
ALGO_ORDER = list(ALGORITHM_COLORS.keys())

# Plot style
plt.style.use("default")
sns.set_style("whitegrid")

FONT_SIZES = {
    "title": 18,
    "label": 18,
    "legend": 14,
    "tick": 16,
}

# ============================================================================
# Helper functions
# ============================================================================

def save_figure(fig, name, bbox_inches='tight'):
    pdf_path = PLOTS_DIR / f"{name}.pdf"
    png_path = PLOTS_DIR / f"{name}.png"

    fig.savefig(pdf_path, format="pdf", bbox_inches=bbox_inches)
    fig.savefig(png_path, format="png", dpi=300, bbox_inches=bbox_inches)
    print(f"Saved: {pdf_path.name}")


def get_display_name(algo):
    return ALGORITHM_DISPLAY_NAMES.get(algo, algo)


# ============================================================================
# Core Plot
# ============================================================================

def plot_boxplot(df):
    print("\n📊 Plotting: Boxplot Distribution (Object/Pixel/Missing)")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    metrics = [
        ("object_recall", "Object Recall (%)"),
        ("pixel_recall", "Pixel Recall (%)"),
        ("missing_rate", "Missing Rate (%)"),
    ]

    # Preprocess
    df_plot = df.copy()
    df_plot["algorithm_display"] = df_plot["algorithm"].apply(get_display_name)

    for ax, (metric, ylabel) in zip(axes, metrics):

        df_plot_metric = df_plot.copy()
        df_plot_metric[metric] = df_plot_metric[metric] * 100

        # Sort by median value
        order = (
            df_plot_metric.groupby("algorithm_display")[metric]
            .median()
            .sort_values(ascending=False)
            .index
        )

        # Collect values for boxplot
        data = [
            df_plot_metric[df_plot_metric["algorithm_display"] == algo][metric].values
            for algo in order
        ]

        # Draw boxplot (no mean → no triangles)
        bp = ax.boxplot(
            data,
            labels=order,
            patch_artist=True,
            showmeans=False,
        )

        # Coloring
        for patch, algo_disp in zip(bp["boxes"], order):
            orig_algo = df_plot[df_plot["algorithm_display"] == algo_disp]["algorithm"].iloc[0]
            patch.set_facecolor(ALGORITHM_COLORS.get(orig_algo, "#333333"))
            patch.set_alpha(0.75)

        # Axis formatting
        ax.set_ylabel(ylabel, fontsize=FONT_SIZES["label"])
        ax.set_title(ylabel.replace(" (%)", ""), fontsize=FONT_SIZES["title"])
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    save_figure(fig, "06_boxplot_distribution")
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    print("\n===============================================")
    print("Heart Evaluation – Boxplot Visualization Only")
    print("===============================================")

    print(f"\nLoading: {RESULTS_CSV}")
    df = pd.read_csv(RESULTS_CSV)

    print(f"Loaded {len(df)} rows")
    print(f"Algorithms: {df['algorithm'].nunique()}")
    print(f"Regions: {df['region'].nunique()}")
    print(f"Cell types: {df['cell_type'].nunique()}")

    plot_boxplot(df)

    print("\nDone. Output saved to:")
    print(PLOTS_DIR)
    print("===============================================\n")


if __name__ == "__main__":
    main()

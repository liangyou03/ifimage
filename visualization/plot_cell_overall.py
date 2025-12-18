#!/usr/bin/env python3
"""
plot_01_cell_overall.py

Plot overall AP curves for cell segmentation (2-channel).
绘制细胞分割（双通道）的整体AP曲线。

Uses fixed colors for each algorithm defined in config.py.
使用config.py中为每个算法定义的固定颜色。
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# Import shared configuration
# 导入共享配置
from config import (
    RESULTS_DIR,
    PLOTS_DIR,
    PNG_SUBDIR_NAME,
    FIGURE_DPI,
    TRANSPARENT_BG,
    ALGORITHM_COLORS,
    ALGORITHM_LINESTYLES,
    ALGORITHM_MARKERS,
    DEFAULT_FIGSIZE,
    FONT_SIZES,
    get_algorithm_display_name,
    save_figure_with_no_legend
)

# Dedicated legend font size for precision vs IoU plots
LEGEND_FONT_SIZE = 9

# Ensure output directory exists
# 确保输出目录存在
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
PNG_DIR = PLOTS_DIR / PNG_SUBDIR_NAME
PNG_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# STYLE CONFIGURATION | 样式配置
# ============================================================================

try:
    # Try to use SciencePlots style if available
    # 如果可用，尝试使用SciencePlots样式
    plt.style.use(['science', 'no-latex'])
except Exception:
    # Fallback to custom Matplotlib configuration
    # 回退到自定义Matplotlib配置
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.size": FONT_SIZES["label"],
        "axes.titlesize": FONT_SIZES["title"],
        "axes.labelsize": FONT_SIZES["label"],
        "legend.fontsize": FONT_SIZES["legend"],
        "xtick.labelsize": FONT_SIZES["tick"],
        "ytick.labelsize": FONT_SIZES["tick"],
        "xtick.direction": "in",          # Tick marks point inward | 刻度线指向内侧
        "ytick.direction": "in",
        "axes.spines.top": False,         # Remove top spine | 移除顶部边框
        "axes.spines.right": False,       # Remove right spine | 移除右侧边框
        "axes.grid": True,                # Enable grid | 启用网格
        "grid.alpha": 0.3,                # Grid transparency | 网格透明度
        "savefig.bbox": "tight",          # Tight bounding box | 紧凑边界框
        "figure.dpi": 200,                # Display resolution | 显示分辨率
    })

# ============================================================================
# HELPER FUNCTIONS | 辅助函数
# ============================================================================

def infer_ap_cols(df):
    """
    Find AP columns like 'AP@0.50', 'AP@0.55', etc.
    查找类似'AP@0.50'、'AP@0.55'等的AP列。
    
    Args | 参数:
        df: DataFrame containing evaluation results
            包含评估结果的DataFrame
    
    Returns | 返回:
        thr: Array of IoU thresholds (sorted)
             IoU阈值数组（已排序）
        ap_cols: List of AP column names (sorted by threshold)
                 AP列名列表（按阈值排序）
    """
    # Find all columns matching the AP@X.XX pattern
    # 查找所有匹配AP@X.XX模式的列
    ap = [c for c in df.columns if re.match(r"^AP@\d\.\d{2}$", c)]
    
    if not ap:
        raise ValueError("No AP@xx.xx columns found | 未找到AP@xx.xx列")
    
    # Extract threshold values from column names
    # 从列名中提取阈值
    thr = np.array([float(c.split("@")[1]) for c in ap])
    
    # Sort by threshold value
    # 按阈值排序
    order = np.argsort(thr)
    return thr[order], [ap[i] for i in order]


def get_algorithm_color(algo_name):
    """
    Get fixed color for algorithm, with fallback to default palette.
    获取算法的固定颜色，如果未定义则使用默认颜色。
    
    Args | 参数:
        algo_name: Name of the algorithm
                   算法名称
    
    Returns | 返回:
        Color hex code | 颜色十六进制代码
    """
    return ALGORITHM_COLORS.get(algo_name, "#000000")  # Black if not defined | 未定义时使用黑色


def get_algorithm_linestyle(algo_name):
    """
    Get line style for algorithm.
    获取算法的线型。
    
    Args | 参数:
        algo_name: Name of the algorithm
                   算法名称
    
    Returns | 返回:
        Line style string (e.g., "-", "--", "-.")
        线型字符串（例如："-"、"--"、"-."）
    """
    return ALGORITHM_LINESTYLES.get(algo_name, "-")


def get_algorithm_marker(algo_name):
    """
    Get marker style for algorithm.
    获取算法的标记样式。
    
    Args | 参数:
        algo_name: Name of the algorithm
                   算法名称
    
    Returns | 返回:
        Marker style string (e.g., "o", "s", "^")
        标记样式字符串（例如："o"、"s"、"^"）
    """
    return ALGORITHM_MARKERS.get(algo_name, "o")


# ============================================================================
# MAIN PLOT GENERATION | 主绘图函数
# ============================================================================

def main():
    """
    Main function to generate the overall AP curve plot for cell segmentation.
    生成细胞分割整体AP曲线图的主函数。
    """
    
    # Load data from saved evaluation results
    # 从保存的评估结果加载数据
    print(f"Loading data from | 从以下位置加载数据: {RESULTS_DIR}...")
    per_img = pd.read_parquet(RESULTS_DIR / "cell_2ch_per_image.parquet")
    
    # Check if data was loaded successfully
    # 检查数据是否成功加载
    if per_img.empty:
        print("ERROR: No data found! | 错误：未找到数据！")
        return
    
    # Compute AP curves for each algorithm
    # 计算每个算法的AP曲线
    print("Computing AP curves | 计算AP曲线...")
    thr, ap_cols = infer_ap_cols(per_img)
    
    # Group by algorithm and compute mean AP across all images
    # 按算法分组并计算所有图像的平均AP
    curve = per_img.groupby("algorithm")[ap_cols].mean()
    
    # Compute mAP (mean AP across all IoU thresholds) for each algorithm
    # 计算每个算法的mAP（所有IoU阈值的平均AP）
    mAP = curve.mean(axis=1)
    
    # Sort algorithms by mAP (descending order) for better legend readability
    # 按mAP降序排列算法，使图例更易读
    curve = curve.loc[mAP.sort_values(ascending=False).index]
    
    # Create figure and axis
    # 创建图形和坐标轴
    print("Creating plot | 创建图表...")
    fig = plt.figure(figsize=DEFAULT_FIGSIZE)
    ax = fig.add_subplot(111)
    
    # Plot AP curve for each algorithm
    # 为每个算法绘制AP曲线
    for algo, row in curve.iterrows():
        ax.plot(
            thr, row.values,
            color=get_algorithm_color(algo),           # Use fixed color | 使用固定颜色
            linestyle=get_algorithm_linestyle(algo),   # Use fixed line style | 使用固定线型
            marker=get_algorithm_marker(algo),         # Use fixed marker | 使用固定标记
            linewidth=2,                               # Line width | 线宽
            markersize=6,                              # Marker size | 标记大小
            label=f"{get_algorithm_display_name(algo)} (mAP={mAP[algo]:.3f})"
        )
    
    # Set axis labels and title
    # 设置坐标轴标签和标题
    ax.set_xlabel("IOU")
    ax.set_ylabel("Precision")
    
    # Set axis limits
    # 设置坐标轴范围
    ax.set_xlim(thr.min(), thr.max())  # X-axis from min to max threshold | X轴从最小到最大阈值
    ax.set_ylim(0, 1.0)                # Y-axis from 0 to 1 (probability) | Y轴从0到1（概率）
    
    # Enable minor ticks for finer grid
    # 启用次刻度以获得更精细的网格
    ax.minorticks_on()
    
    # Show grid
    # 显示网格
    ax.grid(alpha=0.3)
    
    # Add legend (without frame for cleaner look)
    # 添加图例（无边框，更简洁）
    ax.legend(frameon=False, fontsize=LEGEND_FONT_SIZE)
    
    # Adjust layout to prevent label cutoff
    # 调整布局以防止标签被截断
    plt.tight_layout()
    
    # Save figure in both PDF and PNG formats
    # 以PDF和PNG两种格式保存图形
    print("Saving figure | 保存图形...")
    out_pdf = PLOTS_DIR / "cell_overall_ap.pdf"
    out_png = PNG_DIR / "cell_overall_ap.png"
    # Caption: Precision versus IoU thresholds for overall cell segmentation.
    
    save_figure_with_no_legend(
        fig, out_pdf, out_png,
        dpi=FIGURE_DPI,
        transparent=TRANSPARENT_BG
    )
    
    print(f"✓ Saved | 已保存: {out_pdf}")
    print(f"✓ Saved | 已保存: {out_png}")
    
    # Display the plot
    # 显示图表
    plt.show()


# ============================================================================
# SCRIPT ENTRY POINT | 脚本入口
# ============================================================================

if __name__ == "__main__":
    """
    Run the main function when script is executed directly.
    当脚本直接执行时运行主函数。
    """
    main()

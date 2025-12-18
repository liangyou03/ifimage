"""
config.py

Shared configuration for evaluation and visualization scripts.
用于评估和可视化脚本的共享配置文件。

Edit paths here instead of in each individual script.
在这里编辑路径，而不是在每个单独的脚本中修改。
"""

from pathlib import Path

# ============================================================================
# PATHS | 路径配置
# ============================================================================

# Base directory containing GT data
# GT数据所在的基础目录
DATASET_DIR = Path("/ihome/jbwang/liy121/ifimage/00_dataset_withoutpecam")

# Where evaluation results are saved/loaded
# 评估结果的保存/加载位置
RESULTS_DIR = Path("/ihome/jbwang/liy121/ifimage/evaluation_results")

# Where plots are saved
# 图表的保存位置
PLOTS_DIR = Path("/ihome/jbwang/liy121/ifimage/plots_DEC12")

# Name of the subdirectory that stores PNG exports
PNG_SUBDIR_NAME = "png"

# Name of the subdirectory storing versions without legends
NO_LEGEND_SUBDIR_NAME = "no_legend"

# ============================================================================
# ALGORITHM DIRECTORIES | 算法目录配置
# ============================================================================

# Cell segmentation (2-channel: DAPI + marker)
# 细胞分割（双通道：DAPI + 标记物）
CELL_2CH_ALGOS = {
    "CellposeSAM": Path("/ihome/jbwang/liy121/ifimage/01_cellpose_benchmark/refilter_outputs/feat-mean_thr-otsu_area-100_gate-off"),
    "CellposeSAM_Unrefined": Path("/ihome/jbwang/liy121/ifimage/01_cellpose_benchmark/cyto_prediction"),
    "StarDist": Path("/ihome/jbwang/liy121/ifimage/02_stardist_benchmark/refilter_outputs/feat-mean_thr-otsu_area-100_gate-off"),
    "CellSAM": Path("/ihome/jbwang/liy121/ifimage/03_cellsam_benchmark/refilter_outputs/feat-mean_thr-otsu_area-100_gate-off"),
    "MESMER": Path("/ihome/jbwang/liy121/ifimage/04_mesmer_benchmark/refilter_outputs/feat-mean_thr-otsu_area-100_gate-off"),
    "Watershed": Path("/ihome/jbwang/liy121/ifimage/06_watershed_benchmark/refilter_outputs/feat-mean_thr-otsu_area-100_gate-off"),
    "Omnipose": Path("/ihome/jbwang/liy121/ifimage/07_omnipose_benchmark/refilter_outputs/feat-mean_thr-otsu_area-100_gate-off"),
    "LACSS": Path("/ihome/jbwang/liy121/ifimage/011_lacss/refilter_outputs/feat-mean_thr-otsu_area-100_gate-off"),
    "SplineDist": Path("/ihome/jbwang/liy121/ifimage/08_splinedist_benchmark/refilter_outputs/feat-mean_thr-otsu_area-100_gate-off"),
    "MicroSAM": Path("/ihome/jbwang/liy121/ifimage/012_microsam_benchmark/refilter_outputs/feat-mean_thr-otsu_area-100_gate-off"),
}

# Cell segmentation (marker-only)
# 细胞分割（仅标记物通道）
CELL_MARKER_ALGOS = {
    "CellposeSAM": Path("/ihome/jbwang/liy121/ifimage/01_cellpose_benchmark/markeronly"),
    "StarDist": Path("/ihome/jbwang/liy121/ifimage/02_stardist_benchmark/markeronly"),
    "CellSAM": Path("/ihome/jbwang/liy121/ifimage/03_cellsam_benchmark/markeronly"),
    "MESMER": Path("/ihome/jbwang/liy121/ifimage/04_mesmer_benchmark/markeronly"),
    "Watershed": Path("/ihome/jbwang/liy121/ifimage/06_watershed_benchmark/markeronly"),
    "Omnipose": Path("/ihome/jbwang/liy121/ifimage/07_omnipose_benchmark/markeronly"),
    "LACSS": Path("/ihome/jbwang/liy121/ifimage/011_lacss/markeronly"),
    "SplineDist": Path("/ihome/jbwang/liy121/ifimage/08_splinedist_benchmark/markeronly"),
    "MicroSAM": Path("/ihome/jbwang/liy121/ifimage/012_microsam_benchmark/markeronly"),
}

# Nuclei segmentation
# 细胞核分割
NUCLEI_ALGOS = {
    "CellposeSAM": Path("/ihome/jbwang/liy121/ifimage/01_cellpose_benchmark/nuclei_prediction"),
    "StarDist": Path("/ihome/jbwang/liy121/ifimage/02_stardist_benchmark/nuclei_prediction"),
    "CellSAM": Path("/ihome/jbwang/liy121/ifimage/03_cellsam_benchmark/nuclei_prediction"),
    "MESMER": Path("/ihome/jbwang/liy121/ifimage/04_mesmer_benchmark/nuclei_prediction"),
    "Watershed": Path("/ihome/jbwang/liy121/ifimage/06_watershed_benchmark/nuclei_prediction"),
    "Omnipose": Path("/ihome/jbwang/liy121/ifimage/07_omnipose_benchmark/nuclei_prediction"),
    "SplineDist": Path("/ihome/jbwang/liy121/ifimage/08_splinedist_benchmark/nuclei_prediction"),
    "LACSS": Path("/ihome/jbwang/liy121/ifimage/011_lacss/nuclei_prediction"),
    "MicroSAM": Path("/ihome/jbwang/liy121/ifimage/012_microsam_benchmark/nuclei_prediction"),
}

# ============================================================================
# EVALUATION PARAMETERS | 评估参数
# ============================================================================

# IoU thresholds for AP calculation
# 用于AP计算的IoU阈值
AP_THRESHOLDS = tuple([0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])

# Boundary F-score scales
# 边界F-score的尺度因子
BOUNDARY_SCALES = (1.0, 2.0)

# Parallel workers for evaluation
# 评估时使用的并行工作进程数
MAX_WORKERS = 8

# ============================================================================
# PLOT PARAMETERS | 绘图参数
# ============================================================================

# Cell type grouping patterns (regex patterns for filtering images by cell type)
# 细胞类型分组模式（用于按细胞类型筛选图像的正则表达式）
CELL_TYPE_GROUPS = {
    "OLIG2": r"OLIG2",  # Oligodendrocyte marker | 少突胶质细胞标记物
    "NEUN": r"NEUN",    # Neuronal marker | 神经元标记物
    "IBA1": r"IBA1",    # Microglia marker | 小胶质细胞标记物
    "GFAP": r"GFAP",    # Astrocyte marker | 星形胶质细胞标记物
}

# Fixed colors for each algorithm (consistent across all plots)
# 每个算法的固定颜色（在所有图表中保持一致）
# Using professional color palette with good contrast and colorblind-safe colors
# 使用专业配色方案，具有良好对比度且对色盲友好
ALGORITHM_COLORS = {
    # Primary methods (darker, more saturated colors)
    # 主要方法（深色、饱和度较高的颜色）
    "CellposeSAM": "#1f77b4",           # Professional blue | 专业蓝色
    "CellposeSAM_Unrefined": "#aec7e8", # Light blue (for comparison) | 浅蓝色（用于对比）
    "StarDist": "#ff7f0e",              # Orange | 橙色
    "CellSAM": "#2ca02c",               # Green | 绿色
    "MESMER": "#d62728",                # Red | 红色
    
    # Secondary methods (complementary colors)
    # 次要方法（互补色）
    "Watershed": "#9467bd",             # Purple | 紫色
    "Omnipose": "#8c564b",              # Brown | 棕色
    "LACSS": "#e377c2",                 # Pink | 粉色
    "SplineDist": "#7f7f7f",            # Gray | 灰色
    "MicroSAM": "#bcbd22",              # Yellow-green | 黄绿色
    
    # Alternative names (same colors as base algorithms)
    # 替代名称（与基础算法使用相同颜色）
    "Cellpose": "#1f77b4",              # Same as CellposeSAM | 与CellposeSAM相同
    "Cellpose Unrefined": "#aec7e8",    # Same as CellposeSAM_Unrefined | 与CellposeSAM_Unrefined相同
}

# Line styles for different algorithm variants
# 不同算法变体的线型
# Format: "-" solid, "--" dashed, "-." dash-dot, ":" dotted
# 格式："-" 实线, "--" 虚线, "-." 点划线, ":" 点线
ALGORITHM_LINESTYLES = {
    "CellposeSAM": "-",                 # Solid | 实线
    "CellposeSAM_Unrefined": "--",      # Dashed (to distinguish from refined) | 虚线（区分精化版本）
    "StarDist": "-",
    "CellSAM": "-",
    "MESMER": "-",
    "Watershed": "-",
    "Omnipose": "-",
    "LACSS": "-",
    "SplineDist": "-",
    "MicroSAM": "-",
}

# Marker styles for each algorithm
# 每个算法的标记样式
# Common markers: "o" circle, "s" square, "^" triangle, "D" diamond, "*" star
# 常用标记："o" 圆形, "s" 方形, "^" 三角形, "D" 菱形, "*" 星形
ALGORITHM_MARKERS = {
    "CellposeSAM": "o",                 # Circle | 圆形
    "CellposeSAM_Unrefined": "s",       # Square | 方形
    "StarDist": "^",                    # Triangle up | 上三角
    "CellSAM": "v",                     # Triangle down | 下三角
    "MESMER": "D",                      # Diamond | 菱形
    "Watershed": "p",                   # Pentagon | 五边形
    "Omnipose": "h",                    # Hexagon | 六边形
    "LACSS": "*",                       # Star | 星形
    "SplineDist": "X",                  # X | X形
    "MicroSAM": "P",                    # Plus (filled) | 加号（填充）
}

# Short display aliases for algorithms (keeps legends compact)
ALGORITHM_DISPLAY_NAMES = {
    "CellposeSAM_Unrefined": "Cellpose-U",
    "Cellpose Unrefined": "Cellpose-U",
    "cellpose_unrefined": "Cellpose-U",
}


def get_algorithm_display_name(name: str) -> str:
    """Return a compact display label for an algorithm."""
    return ALGORITHM_DISPLAY_NAMES.get(name, name)


def save_figure_with_no_legend(fig, pdf_path, png_path, *, dpi, transparent, save_kwargs=None):
    """
    Save a figure normally plus a legend-free copy under a subfolder.
    
    Args:
        fig: matplotlib figure
        pdf_path: destination PDF path (Path)
        png_path: destination PNG path (Path)
        dpi: DPI for PNG export
        transparent: whether to use transparent background
        save_kwargs: optional dict passed to fig.savefig (e.g., bbox_inches)
    """
    if save_kwargs is None:
        save_kwargs = {}
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(pdf_path, format="pdf", transparent=transparent, **save_kwargs)
    fig.savefig(png_path, format="png", dpi=dpi, transparent=transparent, **save_kwargs)
    
    no_leg_pdf = pdf_path.parent / NO_LEGEND_SUBDIR_NAME / pdf_path.name
    no_leg_png = png_path.parent / NO_LEGEND_SUBDIR_NAME / png_path.name
    no_leg_pdf.parent.mkdir(parents=True, exist_ok=True)
    no_leg_png.parent.mkdir(parents=True, exist_ok=True)
    
    stored_legends = []
    for ax in getattr(fig, "axes", []):
        leg = ax.get_legend()
        if leg is not None:
            stored_legends.append((ax, leg))
            leg.remove()
    
    fig.savefig(no_leg_pdf, format="pdf", transparent=transparent, **save_kwargs)
    fig.savefig(no_leg_png, format="png", dpi=dpi, transparent=transparent, **save_kwargs)
    
    for ax, leg in stored_legends:
        ax.add_artist(leg)

# Figure DPI for saved images
# 保存图像的DPI（分辨率）
FIGURE_DPI = 300

# Whether to use transparent background
# 是否使用透明背景
TRANSPARENT_BG = True

# Default figure size (width, height) in inches

DEFAULT_FIGSIZE = (7, 5)

# Font sizes for different plot elements
# 不同绘图元素的字体大小
FONT_SIZES = {
    "title": 18,    # Figure title | 图表标题
    "label": 12,    # Axis labels | 坐标轴标签
    "legend": 15,    # Legend text | 图例文字
    "tick": 12,      # Tick labels | 刻度标签
}

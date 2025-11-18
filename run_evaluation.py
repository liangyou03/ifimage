#!/usr/bin/env python3
"""
run_evaluation.py

=============================================================================
ONE-TIME EVALUATION SCRIPT | 一次性评估脚本
=============================================================================

English:
This script evaluates all segmentation algorithms and saves results to disk.
You only need to run this ONCE, or when:
  - Adding new algorithms
  - Changing ground truth data
  - Modifying evaluation parameters (IoU thresholds, etc.)

Chinese | 中文：
此脚本评估所有分割算法并将结果保存到磁盘。
您只需要运行一次，或在以下情况下重新运行：
  - 添加新算法
  - 更改ground truth数据
  - 修改评估参数（IoU阈值等）

After running this script, use the visualization scripts in the 
visualization/ folder to generate plots. Those scripts read the saved
results and can be run repeatedly without re-evaluation.

运行此脚本后，使用visualization/文件夹中的可视化脚本生成图表。
这些脚本读取保存的结果，可以反复运行而无需重新评估。

Output | 输出：
  - Per-image metrics (one row per image per algorithm)
    每张图像的指标（每个算法每张图像一行）
  - Summary metrics (one row per algorithm, averaged across images)
    汇总指标（每个算法一行，在所有图像上平均）
  - Both CSV (human-readable) and Parquet (faster loading)
    CSV格式（人类可读）和Parquet格式（加载更快）

Typical runtime: Several hours depending on dataset size and number of workers.
典型运行时间：根据数据集大小和工作进程数，可能需要几个小时。

Author: Research Assistant
Date: 2025
=============================================================================
"""

from pathlib import Path
import pandas as pd
from datetime import datetime

# Import evaluation functions from existing modules
# 从现有模块导入评估函数
from evaluation_tasks import evaluate_cell_benchmark, evaluate_nuclei_benchmark

# Import shared configuration (paths, algorithms, parameters)
# 导入共享配置（路径、算法、参数）
from config import (
    DATASET_DIR,           # Ground truth masks location | GT标注数据位置
    RESULTS_DIR,           # Where to save evaluation results | 评估结果保存位置
    CELL_2CH_ALGOS,        # Cell algorithms (2-channel input) | 细胞算法（双通道输入）
    CELL_MARKER_ALGOS,     # Cell algorithms (marker-only input) | 细胞算法（仅标记物输入）
    NUCLEI_ALGOS,          # Nuclei segmentation algorithms | 细胞核分割算法
    AP_THRESHOLDS,         # IoU thresholds for AP calculation | AP计算的IoU阈值
    BOUNDARY_SCALES,       # Scales for boundary F-score | 边界F-score的尺度
    MAX_WORKERS            # Parallel processing workers | 并行处理工作进程数
)

# Create output directory if it doesn't exist
# 如果输出目录不存在则创建
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# MAIN EVALUATION ROUTINE | 主评估流程
# =============================================================================

def main():
    """
    Main evaluation workflow.
    主评估工作流程。
    
    English:
    This function:
      1. Evaluates cell segmentation with 2-channel input (DAPI + marker)
      2. Evaluates cell segmentation with marker-only input (single channel)
      3. Evaluates nuclei segmentation (DAPI channel)
      4. Saves all results to RESULTS_DIR for later visualization
    
    Chinese | 中文：
    此函数：
      1. 评估使用双通道输入（DAPI + 标记物）的细胞分割
      2. 评估仅使用标记物输入（单通道）的细胞分割
      3. 评估细胞核分割（DAPI通道）
      4. 将所有结果保存到RESULTS_DIR以供后续可视化
    
    Results are saved in both CSV (for manual inspection) and Parquet
    (for faster loading in visualization scripts).
    结果以CSV（便于手动检查）和Parquet（可视化脚本中加载更快）两种格式保存。
    """
    
    # Print header with timestamp
    # 打印带时间戳的标题
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n{'='*70}")
    print(f"SEGMENTATION EVALUATION - Started at {timestamp}")
    print(f"分割评估 - 开始于 {timestamp}")
    print(f"{'='*70}\n")
    
    # =========================================================================
    # STEP 1: EVALUATE CELL SEGMENTATION (2-CHANNEL INPUT)
    # 步骤1：评估细胞分割（双通道输入）
    # =========================================================================
    # Evaluate algorithms that use both DAPI and marker channels as input.
    # This is the standard "2-channel" approach.
    # 评估使用DAPI和标记物两个通道作为输入的算法。
    # 这是标准的"双通道"方法。
    
    print("\n" + "="*70)
    print("[1/3] CELL SEGMENTATION - 2-CHANNEL INPUT")
    print("[1/3] 细胞分割 - 双通道输入")
    print("="*70)
    print(f"Algorithms to evaluate | 待评估算法数: {len(CELL_2CH_ALGOS)}")
    print(f"Dataset directory | 数据集目录: {DATASET_DIR}")
    print(f"IoU thresholds | IoU阈值: {AP_THRESHOLDS}")
    print(f"Parallel workers | 并行工作进程: {MAX_WORKERS}")
    print()
    
    # Run evaluation
    # 运行评估
    # Returns | 返回值：
    #   - per_img_cell: DataFrame with one row per (image, algorithm) pair
    #                   每个（图像，算法）对一行的DataFrame
    #   - sum_cell: DataFrame with one row per algorithm (averaged metrics)
    #               每个算法一行的DataFrame（平均指标）
    per_img_cell, sum_cell = evaluate_cell_benchmark(
        dataset_dir=DATASET_DIR,
        cyto_pred_dirs=CELL_2CH_ALGOS,
        ap_thresholds=AP_THRESHOLDS,
        boundary_scales=BOUNDARY_SCALES,
        max_workers=MAX_WORKERS,
        verbose=True,  # Print progress | 打印进度
    )
    
    # Tag results with variant type for later analysis
    # 为结果标记变体类型以供后续分析
    per_img_cell["variant"] = "2channel"
    
    # Save results to disk
    # 将结果保存到磁盘
    # - CSV format: easy to open in Excel, inspect manually
    #   CSV格式：易于在Excel中打开，手动检查
    # - Parquet format: faster loading, smaller file size
    #   Parquet格式：加载更快，文件更小
    per_img_cell.to_csv(RESULTS_DIR / "cell_2ch_per_image.csv", index=False)
    per_img_cell.to_parquet(RESULTS_DIR / "cell_2ch_per_image.parquet", index=False)
    sum_cell.to_csv(RESULTS_DIR / "cell_2ch_summary.csv", index=False)
    
    print(f"\n✓ Results saved | 结果已保存:")
    print(f"  - {RESULTS_DIR / 'cell_2ch_per_image.csv'}")
    print(f"  - {RESULTS_DIR / 'cell_2ch_per_image.parquet'}")
    print(f"  - {RESULTS_DIR / 'cell_2ch_summary.csv'}")
    
    # =========================================================================
    # STEP 2: EVALUATE CELL SEGMENTATION (MARKER-ONLY INPUT)
    # 步骤2：评估细胞分割（仅标记物输入）
    # =========================================================================
    # Evaluate algorithms using only the marker channel (no DAPI).
    # This tests performance when nuclear information is not available.
    # 评估仅使用标记物通道（无DAPI）的算法。
    # 这测试了在没有细胞核信息时的性能。
    
    print("\n" + "="*70)
    print("[2/3] CELL SEGMENTATION - MARKER-ONLY INPUT")
    print("[2/3] 细胞分割 - 仅标记物输入")
    print("="*70)
    print(f"Algorithms to evaluate | 待评估算法数: {len(CELL_MARKER_ALGOS)}")
    print()
    
    per_img_marker, sum_marker = evaluate_cell_benchmark(
        dataset_dir=DATASET_DIR,
        cyto_pred_dirs=CELL_MARKER_ALGOS,
        ap_thresholds=AP_THRESHOLDS,
        boundary_scales=BOUNDARY_SCALES,
        max_workers=MAX_WORKERS,
        verbose=True,
    )
    
    # Tag as marker-only variant
    # 标记为仅标记物变体
    per_img_marker["variant"] = "markeronly"
    
    # Save marker-only results
    # 保存仅标记物结果
    per_img_marker.to_csv(RESULTS_DIR / "cell_marker_per_image.csv", index=False)
    per_img_marker.to_parquet(RESULTS_DIR / "cell_marker_per_image.parquet", index=False)
    sum_marker.to_csv(RESULTS_DIR / "cell_marker_summary.csv", index=False)
    
    print(f"\n✓ Results saved | 结果已保存:")
    print(f"  - {RESULTS_DIR / 'cell_marker_per_image.csv'}")
    print(f"  - {RESULTS_DIR / 'cell_marker_per_image.parquet'}")
    print(f"  - {RESULTS_DIR / 'cell_marker_summary.csv'}")
    
    # =========================================================================
    # STEP 3: EVALUATE NUCLEI SEGMENTATION
    # 步骤3：评估细胞核分割
    # =========================================================================
    # Evaluate nucleus segmentation algorithms using DAPI channel.
    # Ground truth is the manually annotated nuclei masks.
    # 使用DAPI通道评估细胞核分割算法。
    # Ground truth是手动标注的细胞核掩码。
    
    print("\n" + "="*70)
    print("[3/3] NUCLEI SEGMENTATION")
    print("[3/3] 细胞核分割")
    print("="*70)
    print(f"Algorithms to evaluate | 待评估算法数: {len(NUCLEI_ALGOS)}")
    print()
    
    per_img_nuc, sum_nuc = evaluate_nuclei_benchmark(
        dataset_dir=DATASET_DIR,
        nuclei_pred_dirs=NUCLEI_ALGOS,
        ap_thresholds=AP_THRESHOLDS,
        boundary_scales=BOUNDARY_SCALES,
        max_workers=MAX_WORKERS,
        verbose=True,
    )
    
    # Save nuclei results
    # 保存细胞核结果
    per_img_nuc.to_csv(RESULTS_DIR / "nuclei_per_image.csv", index=False)
    per_img_nuc.to_parquet(RESULTS_DIR / "nuclei_per_image.parquet", index=False)
    sum_nuc.to_csv(RESULTS_DIR / "nuclei_summary.csv", index=False)
    
    print(f"\n✓ Results saved | 结果已保存:")
    print(f"  - {RESULTS_DIR / 'nuclei_per_image.csv'}")
    print(f"  - {RESULTS_DIR / 'nuclei_per_image.parquet'}")
    print(f"  - {RESULTS_DIR / 'nuclei_summary.csv'}")
    
    # =========================================================================
    # FINAL SUMMARY | 最终总结
    # =========================================================================
    
    print(f"\n{'='*70}")
    print("EVALUATION COMPLETED SUCCESSFULLY!")
    print("评估成功完成！")
    print(f"{'='*70}\n")
    
    print(f"All results saved to | 所有结果已保存至: {RESULTS_DIR}\n")
    
    print("Files created | 已创建文件:")
    print("  Cell segmentation (2-channel) | 细胞分割（双通道）:")
    print("    • cell_2ch_per_image.csv/parquet  - Per-image metrics | 每张图像指标")
    print("    • cell_2ch_summary.csv            - Averaged metrics | 平均指标")
    print()
    print("  Cell segmentation (marker-only) | 细胞分割（仅标记物）:")
    print("    • cell_marker_per_image.csv/parquet")
    print("    • cell_marker_summary.csv")
    print()
    print("  Nuclei segmentation | 细胞核分割:")
    print("    • nuclei_per_image.csv/parquet")
    print("    • nuclei_summary.csv")
    print()
    
    print("=" * 70)
    print("NEXT STEPS | 后续步骤:")
    print("=" * 70)
    print("You can now run visualization scripts to generate plots.")
    print("现在可以运行可视化脚本来生成图表。\n")
    print("Examples | 示例:")
    print("  python visualization/plot_01_cell_overall.py")
    print("  python visualization/plot_all.py  # Generate all plots | 生成所有图表")
    print()
    print("The visualization scripts can be run repeatedly without")
    print("re-running this expensive evaluation step.")
    print("可视化脚本可以反复运行，无需重新运行这个耗时的评估步骤。")
    print("=" * 70 + "\n")


# =============================================================================
# ENTRY POINT | 程序入口
# =============================================================================

if __name__ == "__main__":
    """
    Script entry point.
    脚本入口点。
    
    Usage | 用法:
      python run_evaluation.py
      
    This will evaluate all algorithms and save results to disk.
    这将评估所有算法并将结果保存到磁盘。
    
    The process may take several hours depending on:
    根据以下因素，过程可能需要几个小时：
      - Number of images | 图像数量
      - Number of algorithms | 算法数量
      - Number of parallel workers | 并行工作进程数
      - Hardware performance | 硬件性能
    """
    main()
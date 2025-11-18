#!/usr/bin/env python3
"""
diagnose_nuclei_naming.py

Diagnose file naming issues in nuclei segmentation.
诊断核分割中的文件命名问题。

This script checks why algorithms show "no matched pairs".
此脚本检查为什么算法显示"没有匹配的配对"。
"""

from pathlib import Path
from collections import defaultdict

# Import configuration
# 导入配置
from config import DATASET_DIR, NUCLEI_ALGOS

# ============================================================================
# HELPER FUNCTIONS | 辅助函数
# ============================================================================

def extract_base_from_gt(filename: str, strip_tokens: list) -> str:
    """
    Extract base name from GT filename by removing strip tokens.
    通过移除剥离标记从GT文件名提取基础名称。
    
    Args | 参数:
        filename: GT filename (e.g., "olig2_001_dapimultimask.npy")
                  GT文件名（例如："olig2_001_dapimultimask.npy"）
        strip_tokens: List of strings to remove
                      要移除的字符串列表
    
    Returns | 返回:
        Base name (e.g., "olig2_001")
        基础名称（例如："olig2_001"）
    """
    base = filename
    for token in strip_tokens:
        base = base.replace(token, "")
    return base


def extract_base_from_pred(filename: str, strip_tokens: list) -> str:
    """
    Extract base name from prediction filename.
    从预测文件名提取基础名称。
    """
    base = filename
    for token in strip_tokens:
        base = base.replace(token, "")
    return base


# ============================================================================
# DIAGNOSIS | 诊断
# ============================================================================

def diagnose():
    """
    Main diagnosis function.
    主诊断函数。
    """
    
    print("="*70)
    print("NUCLEI SEGMENTATION FILE NAMING DIAGNOSIS")
    print("核分割文件命名诊断")
    print("="*70)
    print()
    
    # ========== STEP 1: Check GT files | 检查GT文件 ==========
    print("[STEP 1] Checking GT files | 检查GT文件")
    print("-"*70)
    
    gt_pattern = "*_dapimultimask.npy"
    gt_files = sorted(DATASET_DIR.glob(gt_pattern))
    
    print(f"GT directory | GT目录: {DATASET_DIR}")
    print(f"GT pattern | GT模式: {gt_pattern}")
    print(f"Found {len(gt_files)} GT files | 找到{len(gt_files)}个GT文件")
    
    if gt_files:
        print("\nFirst 5 GT files | 前5个GT文件:")
        for f in gt_files[:5]:
            print(f"  - {f.name}")
    else:
        print("\n❌ ERROR: No GT files found! | 错误：未找到GT文件！")
        print("Check if GT directory is correct.")
        print("检查GT目录是否正确。")
        return
    
    # Extract base names from GT files
    # 从GT文件提取基础名称
    gt_strip = ["_dapimultimask"]
    gt_bases = {extract_base_from_gt(f.stem, gt_strip): f.name for f in gt_files}
    
    print(f"\nGT base names (after stripping) | GT基础名称（剥离后）:")
    for base in list(gt_bases.keys())[:5]:
        print(f"  - {base}")
    
    print()
    
    # ========== STEP 2: Check each algorithm | 检查每个算法 ==========
    print("[STEP 2] Checking prediction files for each algorithm")
    print("检查每个算法的预测文件")
    print("-"*70)
    print()
    
    results = {}
    
    for algo_name, pred_dir in NUCLEI_ALGOS.items():
        print(f"Algorithm | 算法: {algo_name}")
        print(f"Directory | 目录: {pred_dir}")
        
        # Check if directory exists
        # 检查目录是否存在
        if not pred_dir.exists():
            print(f"  ❌ Directory does not exist! | 目录不存在！")
            print()
            results[algo_name] = {"status": "dir_missing", "matches": 0}
            continue
        
        # Find prediction files
        # 查找预测文件
        pred_files = sorted(pred_dir.glob("*.npy"))
        print(f"  Found {len(pred_files)} prediction files | 找到{len(pred_files)}个预测文件")
        
        if not pred_files:
            print(f"  ❌ No prediction files found! | 未找到预测文件！")
            print()
            results[algo_name] = {"status": "no_files", "matches": 0}
            continue
        
        # Show sample filenames
        # 显示样本文件名
        print(f"  Sample filenames | 样本文件名:")
        for f in pred_files[:3]:
            print(f"    - {f.name}")
        
        # Try different strip patterns to find matches
        # 尝试不同的剥离模式来找到匹配
        strip_patterns = [
            ["_pred_nuc", "_nuc", "_pred"],
            ["_pred_nuclei", "_nuclei", "_pred"],
            ["_prediction", "_pred"],
            ["_nuc_pred", "_pred_nuc"],
            ["_microsam_nuc", "_pred"],
            ["_cellpose_nuc", "_pred"],
            ["_stardist_nuc", "_pred"],
        ]
        
        best_matches = 0
        best_pattern = None
        best_pred_bases = None
        
        for pattern in strip_patterns:
            pred_bases = {extract_base_from_pred(f.stem, pattern): f.name 
                         for f in pred_files}
            matches = set(gt_bases.keys()) & set(pred_bases.keys())
            
            if len(matches) > best_matches:
                best_matches = len(matches)
                best_pattern = pattern
                best_pred_bases = pred_bases
        
        if best_matches > 0:
            print(f"  ✓ Found {best_matches} matches using strip pattern: {best_pattern}")
            print(f"    找到{best_matches}个匹配，使用剥离模式: {best_pattern}")
            
            # Show matched base names
            # 显示匹配的基础名称
            matches = set(gt_bases.keys()) & set(best_pred_bases.keys())
            print(f"  Sample matched bases | 匹配的基础名称样本:")
            for base in list(matches)[:3]:
                print(f"    - {base}")
                print(f"      GT:   {gt_bases[base]}")
                print(f"      Pred: {best_pred_bases[base]}")
            
            results[algo_name] = {
                "status": "success",
                "matches": best_matches,
                "pattern": best_pattern,
                "sample_pred": list(best_pred_bases.values())[:3]
            }
        else:
            print(f"  ❌ No matches found with any strip pattern!")
            print(f"    未找到任何匹配！")
            print(f"  Prediction base names (first 3) | 预测基础名称（前3个）:")
            # Try with minimal stripping
            minimal_bases = {f.stem: f.name for f in pred_files}
            for base in list(minimal_bases.keys())[:3]:
                print(f"    - {base}")
            
            results[algo_name] = {"status": "no_matches", "matches": 0}
        
        print()
    
    # ========== STEP 3: Summary | 总结 ==========
    print("="*70)
    print("[SUMMARY] Results | 结果总结")
    print("="*70)
    print()
    
    for algo_name, info in results.items():
        status = info["status"]
        matches = info["matches"]
        
        if status == "success":
            print(f"✓ {algo_name}: {matches} matches | {matches}个匹配")
            print(f"  Recommended strip pattern | 推荐的剥离模式: {info['pattern']}")
        elif status == "dir_missing":
            print(f"❌ {algo_name}: Directory does not exist | 目录不存在")
        elif status == "no_files":
            print(f"❌ {algo_name}: No prediction files | 无预测文件")
        elif status == "no_matches":
            print(f"❌ {algo_name}: Files exist but no matches | 文件存在但无匹配")
            print(f"   This usually means filename format is different")
            print(f"   这通常意味着文件名格式不同")
        print()
    
    # ========== STEP 4: Recommendations | 建议 ==========
    print("="*70)
    print("[RECOMMENDATIONS] How to fix | 修复建议")
    print("="*70)
    print()
    
    failed_algos = [name for name, info in results.items() 
                   if info["status"] != "success"]
    
    if failed_algos:
        print("For algorithms with no matches, you need to:")
        print("对于没有匹配的算法，您需要：")
        print()
        print("1. Check if prediction files exist in the correct directory")
        print("   检查预测文件是否存在于正确的目录")
        print()
        print("2. Check if filename suffixes match the expected pattern")
        print("   检查文件名后缀是否匹配预期模式")
        print()
        print("3. Update evaluation_tasks.py to add correct strip patterns:")
        print("   更新evaluation_tasks.py以添加正确的剥离模式：")
        print()
        print("   In evaluate_nuclei_benchmark(), modify pred_strip:")
        print("   在evaluate_nuclei_benchmark()中，修改pred_strip：")
        print()
        print("   pred_strip={")
        for algo in failed_algos:
            if algo in results and "pattern" in results[algo]:
                print(f'       "{algo}": {results[algo]["pattern"]},')
            else:
                print(f'       "{algo}": ["_pred_nuc", "_nuc", "_pred"],  # TO BE DETERMINED')
        print("   }")
        print()
    else:
        print("✓ All algorithms have matches! | 所有算法都有匹配！")
        print("The issue may be in the evaluation code configuration.")
        print("问题可能在评估代码配置中。")


# ============================================================================
# MAIN | 主函数
# ============================================================================

if __name__ == "__main__":
    diagnose()

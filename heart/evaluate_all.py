# heart/evaluate_all.py
"""
评估所有算法 - Object-level 和 Pixel-level
使用stardist.matching进行实例匹配
只计算: Recall, Pixel Recall, Missing Rate
"""
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from stardist.matching import matching

# 配置
GT_DIR = Path("/ihome/jbwang/liy121/ifimage/heart/ground_truth_masks")
RESULTS_BASE = Path("/ihome/jbwang/liy121/ifimage/heart/benchmark_results")
OUTPUT_CSV = Path("/ihome/jbwang/liy121/ifimage/heart/evaluation_results.csv")

# 算法列表
ALGORITHMS = [
    'cellpose',
    'cellpose_sam',
    'stardist',
    'omnipose',
    'watershed',
    'mesmer',
    'lacss',
    'microsam',
    'cellsam',
    'splinedist'
]

def calculate_pixel_recall(gt_mask, pred_mask):
    """
    计算Pixel-level Recall
    Pixel Recall = TP pixels / (TP pixels + FN pixels)
    """
    gt_binary = (gt_mask > 0).astype(bool)
    pred_binary = (pred_mask > 0).astype(bool)
    
    tp_pixels = np.logical_and(gt_binary, pred_binary).sum()
    fn_pixels = np.logical_and(gt_binary, ~pred_binary).sum()
    
    if (tp_pixels + fn_pixels) == 0:
        return 0.0
    
    pixel_recall = tp_pixels / (tp_pixels + fn_pixels)
    return pixel_recall

def evaluate_single(gt_mask, pred_mask, iou_threshold=0.5):
    """
    评估单张图像
    
    Returns:
        dict with:
        - n_gt: GT对象数量
        - n_pred: 预测对象数量
        - n_matched: 匹配成功的对象数量
        - n_undetected: 未检测到的对象数量
        - object_recall: Object-level Recall
        - pixel_recall: Pixel-level Recall
        - missing_rate: Missing Rate (未检测率)
    """
    n_gt = len(np.unique(gt_mask)) - 1  # 排除背景0
    n_pred = len(np.unique(pred_mask)) - 1
    
    if n_gt == 0:
        return {
            'n_gt': 0,
            'n_pred': n_pred,
            'n_matched': 0,
            'n_undetected': 0,
            'object_recall': 0.0,
            'pixel_recall': 0.0,
            'missing_rate': 0.0
        }
    
    # 使用stardist.matching进行实例匹配
    matched = matching(gt_mask, pred_mask, thresh=iou_threshold)
    
    # matched包含: tp, fp, fn等信息
    n_matched = matched.tp  # True Positives (成功匹配的GT对象)
    n_undetected = matched.fn  # False Negatives (未检测到的GT对象)
    
    # Object-level Recall
    object_recall = n_matched / n_gt if n_gt > 0 else 0.0
    
    # Missing Rate
    missing_rate = n_undetected / n_gt if n_gt > 0 else 0.0
    
    # Pixel-level Recall
    pixel_recall = calculate_pixel_recall(gt_mask, pred_mask)
    
    return {
        'n_gt': n_gt,
        'n_pred': n_pred,
        'n_matched': n_matched,
        'n_undetected': n_undetected,
        'object_recall': object_recall,
        'pixel_recall': pixel_recall,
        'missing_rate': missing_rate
    }

def find_predictions(algo_name, region, area, channel):
    """
    查找预测文件
    支持不同的文件命名格式
    """
    pred_dir = RESULTS_BASE / f"{algo_name}_predictions" / region
    
    # 尝试不同的文件名格式
    possible_names = [
        f"{area}_{channel}_pred.npy",      # LA1_dapi_pred.npy
        f"{channel}-{area}_pred.npy",      # dapi-LA1_pred.npy
        f"{area}_pred.npy"                 # LA1_pred.npy (只有area)
    ]
    
    for name in possible_names:
        pred_path = pred_dir / name
        if pred_path.exists():
            return pred_path
    
    return None

def main():
    print("=" * 70)
    print("📊 Heart Dataset Evaluation")
    print("=" * 70)
    
    # 加载GT mapping
    gt_mapping = pd.read_csv(GT_DIR / 'file_mapping.csv')
    
    print(f"\n📂 Ground Truth: {len(gt_mapping)} annotations")
    print(f"📂 Algorithms: {len(ALGORITHMS)}")
    print(f"📂 Algorithms: {', '.join(ALGORITHMS)}")
    
    all_results = []
    
    # 遍历每个算法
    for algo_name in ALGORITHMS:
        print(f"\n{'='*70}")
        print(f"🔬 Evaluating: {algo_name}")
        print(f"{'='*70}")
        
        algo_dir = RESULTS_BASE / f"{algo_name}_predictions"
        if not algo_dir.exists():
            print(f"  ⚠️  Prediction directory not found, skipping...")
            continue
        
        n_evaluated = 0
        n_missing = 0
        
        # 遍历每个GT annotation
        for idx, row in tqdm(gt_mapping.iterrows(), 
                            total=len(gt_mapping),
                            desc=f"{algo_name}"):
            
            region = row['region']
            area = row['area']
            cell_type = row['cell_type']
            gt_path = Path(row['mask_absolute_path'])
            
            # 提取channel信息
            # GT文件名格式: Epi-LA1_mask.npy
            channel = f"{cell_type.lower()}"  # 或者用其他mapping
            
            # 查找对应的预测文件
            pred_path = find_predictions(algo_name, region, area, 'dapi')
            
            if pred_path is None:
                n_missing += 1
                continue
            
            try:
                # 加载masks
                gt_mask = np.load(gt_path)
                pred_mask = np.load(pred_path)
                
                # 评估
                metrics = evaluate_single(gt_mask, pred_mask, iou_threshold=0.5)
                
                # 添加元信息
                metrics.update({
                    'algorithm': algo_name,
                    'region': region,
                    'area': area,
                    'cell_type': cell_type,
                    'gt_path': str(gt_path),
                    'pred_path': str(pred_path)
                })
                
                all_results.append(metrics)
                n_evaluated += 1
                
            except Exception as e:
                print(f"\n  ✗ Failed {region}/{area}-{cell_type}: {e}")
                continue
        
        print(f"  ✓ Evaluated: {n_evaluated}/{len(gt_mapping)}")
        if n_missing > 0:
            print(f"  ⚠️  Missing predictions: {n_missing}")
    
    # 保存结果
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(OUTPUT_CSV, index=False)
        
        print("\n" + "=" * 70)
        print("📊 EVALUATION SUMMARY")
        print("=" * 70)
        
        # 按算法汇总
        print("\n🔬 By Algorithm:")
        algo_summary = results_df.groupby('algorithm').agg({
            'object_recall': ['mean', 'std'],
            'pixel_recall': ['mean', 'std'],
            'missing_rate': ['mean', 'std'],
            'n_gt': 'sum',
            'n_matched': 'sum',
            'n_undetected': 'sum'
        }).round(4)
        print(algo_summary)
        
        # 按区域汇总
        print("\n🫀 By Region:")
        region_summary = results_df.groupby('region')[
            ['object_recall', 'pixel_recall', 'missing_rate']
        ].mean().round(4)
        print(region_summary)
        
        # 按细胞类型汇总
        print("\n🧬 By Cell Type:")
        celltype_summary = results_df.groupby('cell_type')[
            ['object_recall', 'pixel_recall', 'missing_rate']
        ].mean().round(4)
        print(celltype_summary)
        
        print(f"\n💾 Results saved to: {OUTPUT_CSV}")
        print("=" * 70)
    else:
        print("\n❌ No results to save!")

if __name__ == "__main__":
    main()
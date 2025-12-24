# heart/sanity_check.py
"""
Sanity Check脚本 - 检查预测结果的质量
检查Omnipose和CellSAM的预测mask是否有问题
"""
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# 配置
GT_DIR = Path("/ihome/jbwang/liy121/ifimage/heart/ground_truth_masks")
PROCESSED_DIR = Path("/ihome/jbwang/liy121/ifimage/heart/processed")
RESULTS_BASE = Path("/ihome/jbwang/liy121/ifimage/heart/benchmark_results")
OUTPUT_DIR = Path("/ihome/jbwang/liy121/ifimage/heart/sanity_check")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 需要检查的算法
ALGORITHMS_TO_CHECK = ['omnipose', 'cellsam', 'cellpose_sam', 'stardist']

def load_image(image_path):
    """加载图像"""
    import tifffile
    img = tifffile.imread(image_path)
    if img.ndim == 3:
        img = img[..., 0]
    return img

def analyze_mask(mask):
    """分析mask的统计信息"""
    n_objects = len(np.unique(mask)) - 1  # 排除背景
    object_sizes = []
    
    if n_objects > 0:
        for obj_id in np.unique(mask)[1:]:
            size = np.sum(mask == obj_id)
            object_sizes.append(size)
    
    return {
        'n_objects': n_objects,
        'total_pixels': np.sum(mask > 0),
        'coverage': np.sum(mask > 0) / mask.size * 100,
        'min_size': min(object_sizes) if object_sizes else 0,
        'max_size': max(object_sizes) if object_sizes else 0,
        'mean_size': np.mean(object_sizes) if object_sizes else 0,
        'median_size': np.median(object_sizes) if object_sizes else 0,
    }

def visualize_sample(sample_info, algorithms, output_path):
    """可视化单个样本的所有算法结果"""
    region = sample_info['region']
    area = sample_info['area']
    cell_type = sample_info['cell_type']
    
    # 加载数据
    gt_mask = np.load(sample_info['gt_path'])
    
    # 找到对应的DAPI图像
    dapi_path = PROCESSED_DIR / region / f"{area}_dapi.tif"
    if not dapi_path.exists():
        print(f"  ⚠️  DAPI image not found: {dapi_path}")
        return
    
    image = load_image(dapi_path)
    
    # 创建子图
    n_algos = len(algorithms) + 1  # +1 for GT
    fig = plt.figure(figsize=(5 * n_algos, 5))
    gs = GridSpec(1, n_algos, figure=fig)
    
    # 显示原图和GT
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(image, cmap='gray')
    ax0.contour(gt_mask > 0, colors='red', linewidths=0.5)
    ax0.set_title(f'Image + GT\n{cell_type} (n={len(np.unique(gt_mask))-1})', 
                  fontsize=10, fontweight='bold')
    ax0.axis('off')
    
    # 显示各算法预测
    for idx, algo in enumerate(algorithms, 1):
        pred_dir = RESULTS_BASE / f"{algo}_predictions" / region
        pred_path = pred_dir / f"{area}_dapi_pred.npy"
        
        ax = fig.add_subplot(gs[0, idx])
        ax.imshow(image, cmap='gray', alpha=0.5)
        
        if pred_path.exists():
            pred_mask = np.load(pred_path)
            stats = analyze_mask(pred_mask)
            
            # 显示预测mask的边界
            ax.contour(pred_mask > 0, colors='cyan', linewidths=0.5)
            
            # 添加统计信息
            title = f'{algo.upper()}\nn={stats["n_objects"]}'
            title += f'\ncov={stats["coverage"]:.1f}%'
            if stats['n_objects'] > 0:
                title += f'\nsize={stats["mean_size"]:.0f}±{np.std([np.sum(pred_mask==i) for i in np.unique(pred_mask)[1:]]):.0f}'
            
            ax.set_title(title, fontsize=10)
            
            # 如果对象数为0，加红框警告
            if stats['n_objects'] == 0:
                for spine in ax.spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(3)
        else:
            ax.set_title(f'{algo.upper()}\nNOT FOUND', fontsize=10, color='red')
            ax.text(0.5, 0.5, 'Prediction\nNot Found', 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=12, color='red', fontweight='bold')
        
        ax.axis('off')
    
    plt.suptitle(f'{region}/{area} - {cell_type}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def check_predictions():
    """检查所有预测文件"""
    print("=" * 70)
    print("🔍 Sanity Check: Prediction Quality Analysis")
    print("=" * 70)
    
    # 加载GT mapping
    gt_mapping = pd.read_csv(GT_DIR / 'file_mapping.csv')
    
    all_stats = []
    
    # 统计每个算法的预测
    for algo in ALGORITHMS_TO_CHECK:
        print(f"\n{'='*70}")
        print(f"🔬 Checking: {algo}")
        print(f"{'='*70}")
        
        pred_dir = RESULTS_BASE / f"{algo}_predictions"
        if not pred_dir.exists():
            print(f"  ⚠️  Directory not found!")
            continue
        
        # 统计
        n_total = 0
        n_found = 0
        n_empty = 0
        n_with_predictions = 0
        
        for idx, row in gt_mapping.iterrows():
            region = row['region']
            area = row['area']
            cell_type = row['cell_type']
            
            pred_path = pred_dir / region / f"{area}_dapi_pred.npy"
            n_total += 1
            
            if pred_path.exists():
                n_found += 1
                mask = np.load(pred_path)
                stats = analyze_mask(mask)
                
                stats.update({
                    'algorithm': algo,
                    'region': region,
                    'area': area,
                    'cell_type': cell_type,
                    'pred_path': str(pred_path)
                })
                all_stats.append(stats)
                
                if stats['n_objects'] == 0:
                    n_empty += 1
                else:
                    n_with_predictions += 1
        
        print(f"\n  📊 Statistics:")
        print(f"    Total GT annotations: {n_total}")
        print(f"    Prediction files found: {n_found} ({n_found/n_total*100:.1f}%)")
        print(f"    Empty predictions (n=0): {n_empty} ({n_empty/n_found*100:.1f}% of found)")
        print(f"    Valid predictions (n>0): {n_with_predictions} ({n_with_predictions/n_found*100:.1f}% of found)")
        
        if n_empty > 0:
            print(f"    ⚠️  WARNING: {n_empty} predictions are empty!")
    
    # 保存统计CSV
    if all_stats:
        stats_df = pd.DataFrame(all_stats)
        stats_csv = OUTPUT_DIR / 'prediction_statistics.csv'
        stats_df.to_csv(stats_csv, index=False)
        print(f"\n💾 Statistics saved to: {stats_csv}")
        
        # 详细统计
        print("\n" + "=" * 70)
        print("📊 DETAILED STATISTICS")
        print("=" * 70)
        
        for algo in ALGORITHMS_TO_CHECK:
            algo_df = stats_df[stats_df['algorithm'] == algo]
            if len(algo_df) == 0:
                continue
            
            print(f"\n{algo.upper()}:")
            print(f"  Total predictions: {len(algo_df)}")
            print(f"  Empty (n=0): {(algo_df['n_objects']==0).sum()}")
            print(f"  Mean objects: {algo_df['n_objects'].mean():.1f} ± {algo_df['n_objects'].std():.1f}")
            print(f"  Mean coverage: {algo_df['coverage'].mean():.1f}% ± {algo_df['coverage'].std():.1f}%")
            print(f"  Mean object size: {algo_df['mean_size'].mean():.1f} pixels")
    
    return stats_df if all_stats else None

def visualize_samples(stats_df, n_samples=5):
    """可视化几个样本进行对比"""
    print("\n" + "=" * 70)
    print("📊 Generating Visual Comparisons")
    print("=" * 70)
    
    # 加载GT mapping
    gt_mapping = pd.read_csv(GT_DIR / 'file_mapping.csv')
    
    # 选择样本：
    # 1. 随机样本
    # 2. Omnipose失败的样本
    # 3. CellSAM失败的样本
    
    samples_to_check = []
    
    # 随机选择一些
    random_samples = gt_mapping.sample(min(3, len(gt_mapping)))
    for _, row in random_samples.iterrows():
        samples_to_check.append({
            'region': row['region'],
            'area': row['area'],
            'cell_type': row['cell_type'],
            'gt_path': row['mask_absolute_path'],
            'type': 'random'
        })
    
    # Omnipose空预测的样本
    if stats_df is not None:
        omni_empty = stats_df[(stats_df['algorithm'] == 'omnipose') & 
                              (stats_df['n_objects'] == 0)]
        for _, row in omni_empty.head(2).iterrows():
            samples_to_check.append({
                'region': row['region'],
                'area': row['area'],
                'cell_type': row['cell_type'],
                'gt_path': gt_mapping[
                    (gt_mapping['region'] == row['region']) &
                    (gt_mapping['area'] == row['area']) &
                    (gt_mapping['cell_type'] == row['cell_type'])
                ]['mask_absolute_path'].values[0],
                'type': 'omnipose_empty'
            })
        
        # CellSAM空预测的样本
        cellsam_empty = stats_df[(stats_df['algorithm'] == 'cellsam') & 
                                 (stats_df['n_objects'] == 0)]
        for _, row in cellsam_empty.head(2).iterrows():
            samples_to_check.append({
                'region': row['region'],
                'area': row['area'],
                'cell_type': row['cell_type'],
                'gt_path': gt_mapping[
                    (gt_mapping['region'] == row['region']) &
                    (gt_mapping['area'] == row['area']) &
                    (gt_mapping['cell_type'] == row['cell_type'])
                ]['mask_absolute_path'].values[0],
                'type': 'cellsam_empty'
            })
    
    # 生成可视化
    for idx, sample in enumerate(samples_to_check[:n_samples]):
        print(f"\n  Visualizing sample {idx+1}/{min(n_samples, len(samples_to_check))}: "
              f"{sample['region']}/{sample['area']} - {sample['cell_type']} ({sample['type']})")
        
        output_path = OUTPUT_DIR / f"visual_check_{idx+1}_{sample['region']}_{sample['area']}_{sample['cell_type']}.png"
        visualize_sample(sample, ALGORITHMS_TO_CHECK, output_path)
        print(f"    Saved: {output_path.name}")

def main():
    # 1. 检查预测文件
    stats_df = check_predictions()
    
    # 2. 可视化样本
    if stats_df is not None:
        visualize_samples(stats_df, n_samples=10)
    
    print("\n" + "=" * 70)
    print("✅ Sanity Check Complete!")
    print(f"📁 Results saved to: {OUTPUT_DIR}")
    print("=" * 70)

if __name__ == "__main__":
    main()
# heart/run_cellsam.py
"""
CellSAM预测脚本 - 分割所有通道
Environment: ifimage_cellsam
"""
import numpy as np
from pathlib import Path
import tifffile
from tqdm import tqdm
from cellSAM import segment_cellular_image

PROCESSED_DIR = Path("/ihome/jbwang/liy121/ifimage/heart/processed")
OUTPUT_DIR = Path("/ihome/jbwang/liy121/ifimage/heart/benchmark_results/cellsam_predictions")

print("=" * 70)
print("🚀 CellSAM Segmentation")
print("=" * 70)

# 检查CUDA
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

tif_files = list(PROCESSED_DIR.glob('*/*.tif'))
print(f"Found {len(tif_files)} TIF images")
print(f"Output: {OUTPUT_DIR}")
print("=" * 70)

failed = []

for tif_path in tqdm(tif_files, desc="CellSAM", unit="img"):
    try:
        image = tifffile.imread(tif_path)
        
        # 确保是2D灰度图
        if image.ndim == 3:
            image = image[..., 0]
        
        # CellSAM分割
        result = segment_cellular_image(image, device=device)
        
        # 检查返回值
        if result is None or result[0] is None:
            # 没有检测到mask，创建空mask
            masks = np.zeros_like(image, dtype=np.int32)
        else:
            masks, _, _ = result
            masks = masks.astype(np.int32, copy=False)
        
        region = tif_path.parent.name
        filename = tif_path.stem
        output_path = OUTPUT_DIR / region / f"{filename}_pred.npy"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, masks)
        
    except Exception as e:
        failed.append(f"{tif_path.name}: {e}")
        # 创建空mask作为fallback
        masks = np.zeros_like(image, dtype=np.int32)
        region = tif_path.parent.name
        filename = tif_path.stem
        output_path = OUTPUT_DIR / region / f"{filename}_pred.npy"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, masks)
        continue

print("\n" + "=" * 70)
print(f"✅ Done! Results saved to: {OUTPUT_DIR}")
if failed:
    print(f"⚠️  {len(failed)} images had issues (saved as empty masks):")
    for f in failed[:5]:
        print(f"  • {f}")
print("=" * 70)
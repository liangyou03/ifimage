# heart/run_cellpose_sam.py
"""
CelloseSAM预测脚本 - 分割所有通道
Environment: ifimage-cpsam-gpu
"""
import numpy as np
from pathlib import Path
import tifffile
from tqdm import tqdm
from cellpose import models

PROCESSED_DIR = Path("/ihome/jbwang/liy121/ifimage/heart/processed")
OUTPUT_DIR = Path("/ihome/jbwang/liy121/ifimage/heart/benchmark_results/cellpose_sam_predictions")

# 检查GPU
def use_gpu():
    try:
        return models.use_gpu()
    except:
        return False

print("=" * 70)
print("🚀 CelloseSAM Segmentation")
print("=" * 70)

gpu = use_gpu()
print(f"GPU available: {gpu}")

# 加载CelloseSAM模型 (默认权重是'cpsam')
print("Loading CelloseSAM model...")
model = models.CellposeModel(gpu=True)

tif_files = list(PROCESSED_DIR.glob('*/*.tif'))
print(f"Found {len(tif_files)} TIF images")
print(f"Output: {OUTPUT_DIR}")
print("=" * 70)

for tif_path in tqdm(tif_files, desc="CelloseSAM", unit="img"):
    image = tifffile.imread(tif_path)
    
    # CelloseSAM分割
    masks, _, _ = model.eval(
        [image],
        diameter=None,
        flow_threshold=0.4,
        cellprob_threshold=0.0,
        do_3D=False,
        batch_size=1,
        resample=True
    )
    
    # 取第一个结果
    mask = masks[0].astype(np.int32)
    
    region = tif_path.parent.name
    filename = tif_path.stem
    output_path = OUTPUT_DIR / region / f"{filename}_pred.npy"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, mask)

print("\n" + "=" * 70)
print(f"✅ Done! Results saved to: {OUTPUT_DIR}")
print("=" * 70)
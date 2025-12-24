# heart/run_mesmer.py
"""
Mesmer预测脚本 - 分割所有通道
Environment: deepcell_retinamask
"""
import numpy as np
from pathlib import Path
import tifffile
from tqdm import tqdm
from deepcell.applications import Mesmer

PROCESSED_DIR = Path("/ihome/jbwang/liy121/ifimage/heart/processed")
OUTPUT_DIR = Path("/ihome/jbwang/liy121/ifimage/heart/benchmark_results/mesmer_predictions")

print("=" * 70)
print("🚀 Mesmer Segmentation")
print("=" * 70)

# 加载Mesmer模型
print("Loading Mesmer model...")
app = Mesmer()

tif_files = list(PROCESSED_DIR.glob('*/*.tif'))
print(f"Found {len(tif_files)} TIF images")
print(f"Output: {OUTPUT_DIR}")
print("=" * 70)

for tif_path in tqdm(tif_files, desc="Mesmer", unit="img"):
    image = tifffile.imread(tif_path)
    
    # Mesmer需要4D输入: [batch, height, width, channels]
    # 而且需要两个通道 [nuclear, cytoplasm]，我们用同一个图像
    if image.ndim == 2:
        # 创建两个通道的图像
        image_2ch = np.stack([image, image], axis=-1)  # (H, W, 2)
    image_4d = np.expand_dims(image_2ch, axis=0)      # (1, H, W, 2)
    
    # Mesmer分割 (返回nuclear和whole-cell masks)
    predictions = app.predict(image_4d, image_mpp=0.5)
    
    # 取nuclear mask (第一个通道)
    mask = predictions[0, ..., 0]
    
    region = tif_path.parent.name
    filename = tif_path.stem
    output_path = OUTPUT_DIR / region / f"{filename}_pred.npy"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, mask)

print("\n" + "=" * 70)
print(f"✅ Done! Results saved to: {OUTPUT_DIR}")
print("=" * 70)
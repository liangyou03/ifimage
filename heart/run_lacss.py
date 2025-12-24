# heart/run_lacss.py
"""
LACSS预测脚本 - 分割所有通道
Environment: lacss
"""
import numpy as np
from pathlib import Path
import tifffile
from tqdm import tqdm
from lacss.deploy import model_urls
from lacss.deploy.predict import Predictor

PROCESSED_DIR = Path("/ihome/jbwang/liy121/ifimage/heart/processed")
OUTPUT_DIR = Path("/ihome/jbwang/liy121/ifimage/heart/benchmark_results/lacss_predictions")

print("=" * 70)
print("🚀 LACSS Segmentation")
print("=" * 70)

# 加载LACSS模型
print("Loading LACSS model...")
predictor = Predictor(model_urls["default"])

tif_files = list(PROCESSED_DIR.glob('*/*.tif'))
print(f"Found {len(tif_files)} TIF images")
print(f"Output: {OUTPUT_DIR}")
print("=" * 70)

for tif_path in tqdm(tif_files, desc="LACSS", unit="img"):
    image = tifffile.imread(tif_path)
    
    # 确保是2D灰度图
    if image.ndim == 3:
        image = image[..., 0]
    
    # LACSS需要 [H, W, 1]
    img_3d = image[..., None]
    
    try:
        # LACSS预测
        out = predictor.predict(img_3d, output_type="label")
        mask = out["pred_label"].astype(np.int32, copy=False)
        
        region = tif_path.parent.name
        filename = tif_path.stem
        output_path = OUTPUT_DIR / region / f"{filename}_pred.npy"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, mask)
        
    except Exception as e:
        print(f"\n  ✗ Failed {tif_path.name}: {e}")
        continue

print("\n" + "=" * 70)
print(f"✅ Done! Results saved to: {OUTPUT_DIR}")
print("=" * 70)
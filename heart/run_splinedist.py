# heart/run_splinedist.py
"""
SplineDist预测脚本 - 分割所有通道
Environment: ifimage_splinedist
"""
import numpy as np
from pathlib import Path
import tifffile
from tqdm import tqdm
from csbdeep.utils import normalize
from splinedist.models import SplineDist2D

PROCESSED_DIR = Path("/ihome/jbwang/liy121/ifimage/heart/processed")
OUTPUT_DIR = Path("/ihome/jbwang/liy121/ifimage/heart/benchmark_results/splinedist_predictions")
PRETRAINED_ROOT = Path("/ihome/jbwang/liy121/ifimage/08_splinedist_benchmark/splinedist_models/bbbc038_8")

# 归一化分位数
P_LOWER, P_UPPER = 1, 99.8

def _pick_sd_model_dir(root):
    """查找SplineDist模型目录"""
    if not root.exists():
        return None
    # 允许根目录本身或其子目录为模型目录
    for p in [root] + list(root.rglob("*")):
        if p.is_dir():
            cfg = p / "config.json"
            has_w = any(p.glob("weights*.h5"))
            if cfg.exists() and has_w:
                return p.parent, p.name
    return None

def _load_model():
    """加载SplineDist模型"""
    picked = _pick_sd_model_dir(PRETRAINED_ROOT)
    if picked is None:
        raise FileNotFoundError(
            f"未在 {PRETRAINED_ROOT.resolve()} 下找到包含 config.json 与 weights_*.h5 的 SplineDist 模型目录"
        )
    basedir, name = picked
    return SplineDist2D(None, name=name, basedir=str(basedir))

print("=" * 70)
print("🚀 SplineDist Segmentation")
print("=" * 70)

# 加载模型
print("Loading SplineDist model...")
model = _load_model()
print(f"Model loaded from: {model.basedir}/{model.name}")

tif_files = list(PROCESSED_DIR.glob('*/*.tif'))
print(f"Found {len(tif_files)} TIF images")
print(f"Output: {OUTPUT_DIR}")
print("=" * 70)

for tif_path in tqdm(tif_files, desc="SplineDist", unit="img"):
    image = tifffile.imread(tif_path)
    
    # 确保是2D灰度图
    if image.ndim == 3:
        image = image[..., 0]
    
    # 归一化
    image_norm = normalize(image, P_LOWER, P_UPPER)
    
    # SplineDist分割
    labels, _ = model.predict_instances(image_norm)
    labels = labels.astype(np.int32, copy=False)
    
    region = tif_path.parent.name
    filename = tif_path.stem
    output_path = OUTPUT_DIR / region / f"{filename}_pred.npy"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, labels)

print("\n" + "=" * 70)
print(f"✅ Done! Results saved to: {OUTPUT_DIR}")
print("=" * 70)
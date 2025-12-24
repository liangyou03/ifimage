# heart/run_microsam.py
"""
MicroSAM预测脚本 - 分割所有通道
Environment: microsam-cuda or micro-sam
"""
import numpy as np
from pathlib import Path
import tifffile
from tqdm import tqdm
import torch
import os
from skimage.transform import resize

from micro_sam import util
from segment_anything import SamAutomaticMaskGenerator

# 全局线程限制
os.environ.setdefault("OMP_NUM_THREADS", "1")
torch.set_num_threads(1)

PROCESSED_DIR = Path("/ihome/jbwang/liy121/ifimage/heart/processed")
OUTPUT_DIR = Path("/ihome/jbwang/liy121/ifimage/heart/benchmark_results/microsam_predictions")
MODEL = "vit_b_lm"
DOWNSCALE = 0.5  # 降采样因子，加速

# AMG参数
def build_amg_kw(downscale):
    dense_grid = 112 if downscale < 1.0 else 64
    return dict(
        pred_iou_thresh=0.5,
        stability_score_thresh=0.5,
        box_nms_thresh=0.95,
        crop_n_layers=1,
        crop_overlap_ratio=0.5,
        crop_n_points_downscale_factor=2,
        crop_nms_thresh=0.95,
        min_mask_region_area=50,
        output_mode="binary_mask",
        points_per_batch=64
    )

def to_u8_robust(img01):
    """[0,1] -> uint8"""
    if img01.size == 0:
        return img01
    g = (np.clip(img01, 0, 1) * 255.0).astype(np.uint8)
    if g.max() == g.min():
        return np.zeros_like(g, np.uint8)
    return g

def normalize01(img_f32):
    """1-99分位标准化到[0,1]"""
    p1, p99 = np.percentile(img_f32, (1, 99))
    if p99 <= p1:
        lo, hi = float(img_f32.min()), float(img_f32.max())
        if hi <= lo:
            return np.zeros_like(img_f32, np.float32)
        return (img_f32 - lo) / (hi - lo)
    return np.clip((img_f32 - p1) / (p99 - p1), 0, 1)

def masks_to_label(masks, shape_hw):
    """将多张二值掩码融合为实例标签图"""
    lab = np.zeros(shape_hw, np.int32)
    for i, m in enumerate(sorted(masks, key=lambda d: d.get("area", 0), reverse=True), 1):
        seg = m["segmentation"]
        put = seg & (lab == 0)
        if put.any():
            lab[put] = i
    return lab

def downscale_rgb(rgb, scale):
    if scale >= 1.0:
        return rgb
    H, W = rgb.shape[:2]
    h2, w2 = max(1, int(H*scale)), max(1, int(W*scale))
    small = resize(rgb, (h2, w2), order=1, preserve_range=True, anti_aliasing=False).astype(rgb.dtype)
    return small

def upsample_label(lab_small, shape_full):
    if lab_small.shape == shape_full:
        return lab_small
    lab_up = resize(lab_small, shape_full, order=0, preserve_range=True, anti_aliasing=False).astype(lab_small.dtype)
    return lab_up

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

print("=" * 70)
print("🚀 MicroSAM Segmentation")
print("=" * 70)

# 加载模型
device = get_device()
print(f"Device: {device.type}")
print(f"Loading MicroSAM model ({MODEL})...")

_, sam = util.get_sam_model(model_type=MODEL, return_sam=True)
sam.to(device)
amg = SamAutomaticMaskGenerator(sam, **build_amg_kw(DOWNSCALE))

tif_files = list(PROCESSED_DIR.glob('*/*.tif'))
print(f"Found {len(tif_files)} TIF images")
print(f"Output: {OUTPUT_DIR}")
print(f"Downscale factor: {DOWNSCALE}")
print("=" * 70)

for tif_path in tqdm(tif_files, desc="MicroSAM", unit="img"):
    image = tifffile.imread(tif_path).astype(np.float32)
    
    # 确保是2D灰度图
    if image.ndim == 3:
        image = image[..., 0]
    
    # 标准化并转为RGB
    img_norm = normalize01(image)
    g = to_u8_robust(img_norm)
    rgb = np.stack([g, g, g], -1)
    
    H, W = rgb.shape[:2]
    rgb_in = downscale_rgb(rgb, DOWNSCALE)
    h2, w2 = rgb_in.shape[:2]
    
    # 生成masks
    masks = amg.generate(rgb_in)
    
    if len(masks) == 0:
        lab_small = np.zeros((h2, w2), np.int32)
    else:
        lab_small = masks_to_label(masks, (h2, w2))
    
    # 上采样到原始尺寸
    lab_full = upsample_label(lab_small, (H, W))
    
    region = tif_path.parent.name
    filename = tif_path.stem
    output_path = OUTPUT_DIR / region / f"{filename}_pred.npy"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, lab_full.astype(np.int32))

if device.type == "cuda":
    torch.cuda.empty_cache()

print("\n" + "=" * 70)
print(f"✅ Done! Results saved to: {OUTPUT_DIR}")
print("=" * 70)
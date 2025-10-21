#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
import os, argparse
import numpy as np
import torch
from tqdm import tqdm
import tifffile as tiff
from skimage.transform import resize

from micro_sam import util
from segment_anything import SamAutomaticMaskGenerator
from utils import SampleDataset, ensure_dir

# ---- 全局线程限制（CPU 上更稳、更快）----
os.environ.setdefault("OMP_NUM_THREADS", "1")
torch.set_num_threads(1)

# ---- 默认路径 ----
DATA_DIR = Path("/ihome/jbwang/liy121/ifimage/00_dataset")
OUT_DIR  = Path("nuclei_prediction")
MODEL    = "vit_b_lm"

# ---- AMG 参数模板（半分辨率时更密；全分辨率时略降密）----
def build_amg_kw(downscale: float):
    dense_grid = 112 if downscale < 1.0 else 64
    return dict(
        pred_iou_thresh=0.5,            # 宽松，召回↑
        stability_score_thresh=0.5,     # 宽松，召回↑
        box_nms_thresh=0.95,             # 少去重
        crop_n_layers=1,                 # 单层裁剪，速度/召回折中
        crop_overlap_ratio=0.5,
        crop_n_points_downscale_factor=2,# 裁剪内采样降密提速
        crop_nms_thresh=0.95,
        min_mask_region_area=50,          # 接受小目标
        output_mode="binary_mask",       # 跳过RLE路径，避坑更快
        points_per_batch=64              # 批量大小，CPU/GPU都稳
    )

# ---- 工具函数 ----
def to_u8_robust(img01: np.ndarray) -> np.ndarray:
    """[0,1] -> uint8，必要时兜底 min-max。"""
    if img01.size == 0:
        return img01
    g = (np.clip(img01, 0, 1) * 255.0).astype(np.uint8)
    if g.max() == g.min():
        return np.zeros_like(g, np.uint8)
    return g

def masks_to_label(masks, shape_hw):
    """将多张二值掩码融合为实例标签图。"""
    lab = np.zeros(shape_hw, np.int32)
    # 面积大优先，且只写空位，避免覆盖
    for i, m in enumerate(sorted(masks, key=lambda d: d.get("area", 0), reverse=True), 1):
        seg = m["segmentation"]
        put = seg & (lab == 0)
        if put.any():
            lab[put] = i
    return lab

def normalize01(img_f32: np.ndarray) -> np.ndarray:
    """1-99分位标准化到[0,1]，失败则min-max。"""
    p1, p99 = np.percentile(img_f32, (1, 99))
    if p99 <= p1:
        lo, hi = float(img_f32.min()), float(img_f32.max())
        if hi <= lo:
            return np.zeros_like(img_f32, np.float32)
        return (img_f32 - lo) / (hi - lo)
    return np.clip((img_f32 - p1) / (p99 - p1), 0, 1)

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def downscale_rgb(rgb: np.ndarray, scale: float) -> np.ndarray:
    if scale >= 1.0:
        return rgb
    H, W = rgb.shape[:2]
    h2, w2 = max(1, int(H*scale)), max(1, int(W*scale))
    small = resize(rgb, (h2, w2), order=1, preserve_range=True, anti_aliasing=False).astype(rgb.dtype)
    return small

def upsample_label(lab_small: np.ndarray, shape_full) -> np.ndarray:
    if lab_small.shape == shape_full:
        return lab_small
    lab_up = resize(lab_small, shape_full, order=0, preserve_range=True, anti_aliasing=False).astype(lab_small.dtype)
    return lab_up

# ---- 主逻辑 ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=Path, default=DATA_DIR)
    ap.add_argument("--out_dir",  type=Path, default=OUT_DIR)
    ap.add_argument("--model",    type=str,  default=MODEL)
    ap.add_argument("--downscale",type=float,default=0.5, help="0<scale<=1.0。0.5 推荐，平衡速度与召回")
    ap.add_argument("--shard_id", type=int,  default=int(os.getenv("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--num_shards",type=int, default=int(os.getenv("SLURM_ARRAY_TASK_COUNT", 1)))
    args = ap.parse_args()

    data_dir, out_dir = args.data_dir, args.out_dir
    ensure_dir(out_dir)

    device = get_device()
    _, sam = util.get_sam_model(model_type=args.model, return_sam=True)
    sam.to(device)
    amg = SamAutomaticMaskGenerator(sam, **build_amg_kw(args.downscale))

    ds = list(SampleDataset(data_dir))  # 需要 len 和索引
    n = len(ds)
    pbar = tqdm(range(n), desc=f"micro-sam:nuc[{args.shard_id}/{args.num_shards}]({device.type})",
                unit="img", dynamic_ncols=True)

    for i in pbar:
        # 分片：只处理属于本 shard 的索引
        if i % max(1, args.num_shards) != args.shard_id:
            continue
        s = ds[i]
        out_path = out_dir / f"{s.base}_microsam_nuc.npy"
        if out_path.exists():
            # 断点恢复：已算过跳过
            continue

        # 读取并标准化
        s.load_images()                 # 得到 s.nuc_chan ∈ [0,1]
        g = to_u8_robust(normalize01(s.nuc_chan))
        rgb = np.stack([g, g, g], -1)

        H, W = rgb.shape[:2]
        rgb_in = downscale_rgb(rgb, args.downscale)
        h2, w2 = rgb_in.shape[:2]

        tqdm.write(f"[{s.base}] HxW={H}x{W} -> {h2}x{w2}, dev={device.type}")

        # 生成候选
        masks = amg.generate(rgb_in)
        tqdm.write(f"  masks={len(masks)}")

        if len(masks) == 0:
            lab_small = np.zeros((h2, w2), np.int32)
        else:
            lab_small = masks_to_label(masks, (h2, w2))

        lab_full = upsample_label(lab_small, (H, W))
        np.save(out_path, lab_full.astype(np.int32))
    if device.type == "cuda":
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()

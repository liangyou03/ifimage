#!/usr/bin/env python3
from pathlib import Path
import numpy as np
from micro_sam import util
from segment_anything import SamAutomaticMaskGenerator
from utils import SampleDataset, ensure_dir  # pure data utils

# ---- paths ----
DATA_DIR = Path("/ihome/jbwang/liy121/ifimage/00_dataset")
MASK_DIR = Path("markeronly_prediction")
MODEL    = "vit_b_lm"

def masks_to_label(masks, shape):
    lab = np.zeros(shape, np.int32)
    for i, m in enumerate(sorted(masks, key=lambda d: d.get("area", 0), reverse=True), 1):
        seg = m["segmentation"]; put = seg & (lab == 0)
        if put.any(): lab[put] = i
    return lab

# ---- run ----
ensure_dir(MASK_DIR)
predictor, sam = util.get_sam_model(model_type=MODEL, return_sam=True)
amg = SamAutomaticMaskGenerator(
    sam,
    points_per_side=32,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    min_mask_region_area=4,
)

for s in SampleDataset(DATA_DIR):
    s.load_images()
    g = (s.cell_chan * 255).astype(np.uint8)
    rgb = np.stack([g, g, g], -1)
    lab = masks_to_label(amg.generate(rgb), g.shape)
    np.save(MASK_DIR / f"{s.base}.npy", lab)

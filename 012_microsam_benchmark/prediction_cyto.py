#!/usr/bin/env python3
from pathlib import Path
import numpy as np
from micro_sam import util
from segment_anything import SamAutomaticMaskGenerator
from utils import SampleDataset, ensure_dir

DATA_DIR = Path("/ihome/jbwang/liy121/ifimage/00_dataset")
MASK_DIR = Path("cyto_prediction")
MODEL    = "vit_b_lm"

def masks_to_label(masks, shape):
    lab = np.zeros(shape, np.int32)
    for i, m in enumerate(sorted(masks, key=lambda d: d.get("area", 0), reverse=True), 1):
        seg = m["segmentation"]; put = seg & (lab == 0)
        if put.any(): lab[put] = i
    return lab

ensure_dir(MASK_DIR)
predictor, sam = util.get_sam_model(model_type=MODEL, return_sam=True)
amg = SamAutomaticMaskGenerator(
    sam,
    points_per_side=32,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    min_mask_region_area=64,
)

for s in SampleDataset(DATA_DIR):
    if s.marker_path is None:
        continue
    s.load_images()                       # s.cell_chan, s.nuc_chan ∈ [0,1]
    if s.cell_chan.shape != s.nuc_chan.shape:
        continue
    m = (s.cell_chan * 255).astype(np.uint8)
    d = (s.nuc_chan  * 255).astype(np.uint8)
    rgb = np.stack([m, d, np.zeros_like(d, np.uint8)], -1)  # R=marker, G=DAPI
    lab = masks_to_label(amg.generate(rgb), d.shape)
    np.save(MASK_DIR / f"{s.base}_microsam_nucmarker.npy", lab)

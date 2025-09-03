#!/usr/bin/env python3
"""
prediction_nuclei_stardist.py — NUCLEI segmentation with StarDist (DAPI only).
Minimal change from CellSAM: swap to StarDist, keep same dataset loop & saving.
"""

from pathlib import Path
import numpy as np
from csbdeep.utils import normalize
from stardist.models import StarDist2D

from utils import SampleDataset, ensure_dir  # NO model logic inside utils

# ---- config (no CLI) ----
DATA_DIR   = Path("/ihome/jbwang/liy121/ifimage/00_dataset")
OUTPUT_DIR = Path("nuclei_prediction")

# StarDist knobs
PRETRAINED   = "2D_versatile_fluo"  # or "2D_versatile_he" for H&E tissue
PROB_THRESH  = None   # None -> use model default; or e.g., 0.5
NMS_THRESH   = None   # None -> use model default; or e.g., 0.3
N_TILES      = None   # e.g., (2,2) for large images / low VRAM

# init once
_MODEL = StarDist2D.from_pretrained(PRETRAINED)

def run_cellsam_single(img2d: np.ndarray) -> np.ndarray:
    """Run StarDist on a single 2D grayscale image → int32 label mask."""
    x = img2d[..., 0] if (img2d.ndim == 3 and img2d.shape[-1] == 1) else img2d
    x = normalize(x)  # recommended normalization for StarDist
    labels, _ = _MODEL.predict_instances(
        x, prob_thresh=PROB_THRESH, nms_thresh=NMS_THRESH, n_tiles=N_TILES
    )
    return labels.astype(np.int32, copy=False)

def main():
    print("== Nuclei prediction (StarDist, DAPI-only) ==")
    print(f"DATA_DIR   : {DATA_DIR.resolve()}")
    print(f"OUTPUT_DIR : {OUTPUT_DIR.resolve()}")
    ensure_dir(OUTPUT_DIR)

    ds = SampleDataset(DATA_DIR)
    print(f"Found {len(ds)} samples with DAPI.")
    n_ok = 0

    for s in ds:
        try:
            s.load_images()                       # fills s.nuc_chan
            mask = run_cellsam_single(s.nuc_chan)
            s.predicted_nuc = mask
            outp = OUTPUT_DIR / f"{s.base}_pred_nuclei.npy"
            np.save(outp, mask)
            n_ok += 1
            print(f"[OK] {s.base} -> {outp.name} (labels: {int(mask.max())})")
        except Exception as e:
            print(f"[FAIL] {s.base}: {e}")

    print(f"Done: nuclei={n_ok}/{len(ds)}")

if __name__ == "__main__":
    main()

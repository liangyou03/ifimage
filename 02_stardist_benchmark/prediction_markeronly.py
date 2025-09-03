#!/usr/bin/env python3
"""
prediction_markeronly_stardist.py — WHOLE-CELL (marker-only) with StarDist.
Note: StarDist预训练模型主要针对“核”，直接用marker做“胞体”可能更难，作为benchmark是OK的。
"""

from pathlib import Path
import numpy as np
from csbdeep.utils import normalize
from stardist.models import StarDist2D

from utils import SampleDataset, ensure_dir

# ---- config ----
DATA_DIR        = Path("/ihome/jbwang/liy121/ifimage/00_dataset")
OUTPUT_DIR      = Path("markeronly_prediction")

# StarDist knobs
PRETRAINED   = "2D_versatile_fluo"  # 可改 "2D_versatile_he" 试试组织型marker
PROB_THRESH  = None
NMS_THRESH   = None
N_TILES      = None

_MODEL = StarDist2D.from_pretrained(PRETRAINED)

def _take_marker_2d(arr: np.ndarray) -> np.ndarray:
    """Robustly get marker 2D."""
    if arr.ndim == 3 and arr.shape[-1] >= 1:
        return arr[..., 0]
    return arr

def run_cellsam_single(marker2d: np.ndarray) -> np.ndarray:
    """Keep the old function name, but call StarDist."""
    x = _take_marker_2d(marker2d)
    x = normalize(x)
    labels, _ = _MODEL.predict_instances(
        x, prob_thresh=PROB_THRESH, nms_thresh=NMS_THRESH, n_tiles=N_TILES
    )
    return labels.astype(np.int32, copy=False)

def main():
    print("== Cytoplasm prediction (StarDist, marker-only) ==")
    print(f"DATA_DIR   : {DATA_DIR.resolve()}")
    print(f"OUTPUT_DIR : {OUTPUT_DIR.resolve()}")
    ensure_dir(OUTPUT_DIR)

    ds = SampleDataset(DATA_DIR)
    print(f"Found {len(ds)} samples (marker optional).")

    n_ok, n_skip = 0, 0
    for s in ds:
        try:
            s.load_images()  # loads/normalizes images
            if s.cell_chan is None:
                n_skip += 1
                print(f"[SKIP] {s.base} (no marker)")
                continue
            marker = s.cell_chan
            mask = run_cellsam_single(marker)
            s.predicted_cell = mask
            outp = OUTPUT_DIR / f"{s.base}_pred_marker_only.npy"
            np.save(outp, mask)
            n_ok += 1
            print(f"[OK] {s.base} -> {outp.name} (labels: {int(mask.max())})")
        except Exception as e:
            print(f"[FAIL] {s.base}: {e}")

    print(f"Done: cyto_ok={n_ok}, cyto_skip(no marker)={n_skip}, total={len(ds)})")

if __name__ == "__main__":
    main()

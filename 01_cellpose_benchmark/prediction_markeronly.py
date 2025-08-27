#!/usr/bin/env python3
"""
prediction_cyto.py — CYTOPLASM segmentation with Cellpose-SAM (DAPI + marker).
Minimal script: import data utils, stack [MARKER, DAPI], run model, save masks.
"""

from pathlib import Path
import numpy as np
from cellpose import models

from utils import SampleDataset, ensure_dir  # pure data utils

# ---- config ----
DATA_DIR   = Path("/ihome/jbwang/liy121/ifimage/00_dataset")
OUTPUT_DIR = Path("outputs/marker_only")

DIAMETER = None
FLOW_THRESHOLD = 0.4
CELLPROB_THRESHOLD = 0.0

def use_gpu() -> bool:
    try:
        return models.use_gpu()
    except Exception:
        return False

def run_cpsam_single(model_obj, img_hw_c: np.ndarray) -> np.ndarray:
    """Run CP-SAM on one HxWxC image (C=1 here) → int32 label mask."""
    masks, _, _ = model_obj.eval(
        [img_hw_c],
        diameter=DIAMETER,
        flow_threshold=FLOW_THRESHOLD,
        cellprob_threshold=CELLPROB_THRESHOLD,
        do_3D=False,
        batch_size=1,
        resample=True,
    )
    return masks[0].astype(np.int32, copy=False)

def main():
    print("== Cytoplasm prediction (Marker only) ==")
    print(f"DATA_DIR   : {DATA_DIR.resolve()}")
    print(f"OUTPUT_DIR : {OUTPUT_DIR.resolve()}")
    ensure_dir(OUTPUT_DIR)

    ds = SampleDataset(DATA_DIR)
    print(f"Found {len(ds)} samples (marker optional).")

    gpu = use_gpu()
    print(f"GPU available: {gpu}")
    model = models.CellposeModel(gpu=gpu)  # default weights = 'cpsam'

    n_ok, n_skip = 0, 0
    for s in ds:
        try:
            s.load_images()  # loads/normalizes images
            if s.cell_chan is None:
                n_skip += 1
                print(f"[SKIP] {s.base} (no marker)")
                continue
            marker = s.cell_chan         # HxWx2, [MARKER, DAPI]
            mask = run_cpsam_single(model, marker)
            s.predicted_cell = mask
            outp = OUTPUT_DIR / f"{s.base}_pred_marker_only.npy"
            np.save(outp, mask)
            n_ok += 1
            print(f"[OK] {s.base} -> {outp.name} (labels: {int(mask.max())})")
        except Exception as e:
            print(f"[FAIL] {s.base}: {e}")

    print(f"Done: cyto_ok={n_ok}, cyto_skip(no marker)={n_skip}, total={len(ds)}")

if __name__ == "__main__":
    main()

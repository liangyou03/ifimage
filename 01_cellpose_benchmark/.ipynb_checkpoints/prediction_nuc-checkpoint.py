#!/usr/bin/env python3
"""
prediction_nuclei.py — NUCLEI segmentation with Cellpose-SAM (DAPI only).
Minimal script: import data utils, run model, save masks.
"""

from pathlib import Path
import numpy as np
from cellpose import models  # pip install cellpose

from utils import SampleDataset, ensure_dir  # NO model logic inside utils

# ---- config (no CLI) ----
DATA_DIR   = Path("/ihome/jbwang/liy121/ifimage/00_dataset")
OUTPUT_DIR = Path("pred_mask")

# CP-SAM knobs
DIAMETER = None
FLOW_THRESHOLD = 0.4
CELLPROB_THRESHOLD = 0.0

def use_gpu() -> bool:
    try:
        return models.use_gpu()
    except Exception:
        return False

def run_cpsam_single(model_obj, img2d: np.ndarray) -> np.ndarray:
    """Run CP-SAM on a single 2D grayscale image → int32 label mask."""
    masks, _, _ = model_obj.eval(
        [img2d],
        diameter=DIAMETER,
        flow_threshold=FLOW_THRESHOLD,
        cellprob_threshold=CELLPROB_THRESHOLD,
        do_3D=False,
        batch_size=1,
        resample=True,
    )
    return masks[0].astype(np.int32, copy=False)

def main():
    print("== Nuclei prediction (DAPI) ==")
    print(f"DATA_DIR   : {DATA_DIR.resolve()}")
    print(f"OUTPUT_DIR : {OUTPUT_DIR.resolve()}")
    ensure_dir(OUTPUT_DIR)

    ds = SampleDataset(DATA_DIR)
    print(f"Found {len(ds)} samples with DAPI.")

    gpu = use_gpu()
    print(f"GPU available: {gpu}")
    model = models.CellposeModel(gpu=gpu)  # default weights = 'cpsam'

    n_ok = 0
    for s in ds:
        try:
            s.load_images()                       # fills s.nuc_chan (and s.cell_chan if exists)
            mask = run_cpsam_single(model, s.nuc_chan)
            s.predicted_nuc = mask               # attach result on the object
            outp = OUTPUT_DIR / f"{s.base}_pred_nuclei.npy"
            np.save(outp, mask)
            n_ok += 1
            print(f"[OK] {s.base} -> {outp.name} (labels: {int(mask.max())})")
        except Exception as e:
            print(f"[FAIL] {s.base}: {e}")

    print(f"Done: nuclei={n_ok}/{len(ds)}")

if __name__ == "__main__":
    main()

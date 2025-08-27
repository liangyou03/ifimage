#!/usr/bin/env python3
"""
prediction_nuclei.py — NUCLEI segmentation with Cellpose-SAM (DAPI only).
Minimal script: import data utils, run model, save masks.
"""

from pathlib import Path
import numpy as np
from cellSAM import segment_cellular_image

from utils import SampleDataset, ensure_dir  # NO model logic inside utils

# ---- config (no CLI) ----
DATA_DIR   = Path("/ihome/jbwang/liy121/ifimage/00_dataset")
OUTPUT_DIR = Path("markeronly")

# CP-SAM knobs
DIAMETER = None
FLOW_THRESHOLD = 0.4
CELLPROB_THRESHOLD = 0.0

def run_cellsam_single(img2d: np.ndarray) -> np.ndarray:
    """Run CP-SAM on a single 2D grayscale image → int32 label mask."""
    masks, _, _ = segment_cellular_image(img2d, device='cuda')
    return masks[0].astype(np.int32, copy=False)

def main():
    print("== Cytoplasm prediction (Marker only) ==")
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
            marker = s.cell_chan         # HxWx2, [MARKER, DAPI]
            mask = run_cellsam_single(marker)
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

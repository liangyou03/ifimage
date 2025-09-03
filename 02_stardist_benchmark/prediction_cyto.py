#!/usr/bin/env python3
"""
prediction_cyto_stardist.py — Two-channel (DAPI+marker) → StarDist.
We fuse 2 channels to single 2D input for pretrained StarDist; minimal changes kept.
"""

from pathlib import Path
import numpy as np
import gc

from csbdeep.utils import normalize
from stardist.models import StarDist2D

from utils import SampleDataset, ensure_dir

# ---- config ----
DATA_DIR        = Path("/ihome/jbwang/liy121/ifimage/00_dataset")
OUT_DIR_CELL    = Path("cyto")

# Keep your original flags; map relevant ones to StarDist
BBOX_THRESHOLD   = 0.4   # repurposed -> PROB_THRESH (object probability threshold)
USE_WSI          = False # if True, use tiling via n_tiles
LOW_CONTRAST_ENH = False # (not used here)
GAUGE_CELL_SIZE  = False # (not used here)

# StarDist knobs
PRETRAINED  = "2D_versatile_fluo"
PROB_THRESH = BBOX_THRESHOLD
NMS_THRESH  = None
N_TILES     = (2, 2) if USE_WSI else None

# How to fuse 2 channels for single-channel StarDist pretrained model
FUSION = "dapi"  # "dapi" | "marker" | "mean" | "max"

_MODEL = StarDist2D.from_pretrained(PRETRAINED)

def _fuse_to_single(dapi: np.ndarray, cyto: np.ndarray) -> np.ndarray:
    """Fuse (H,W) DAPI+marker to one 2D image for StarDist."""
    assert dapi.shape == cyto.shape, "DAPI 与 marker 尺寸必须一致"
    if FUSION == "dapi":
        fused = dapi
    elif FUSION == "marker":
        fused = cyto
    elif FUSION == "mean":
        fused = (dapi.astype(np.float32) + cyto.astype(np.float32)) * 0.5
    elif FUSION == "max":
        fused = np.maximum(dapi, cyto)
    else:
        raise ValueError(f"Unknown FUSION={FUSION}")
    return fused

def _cellsam_cells(dapi: np.ndarray, cyto: np.ndarray) -> np.ndarray:
    """Keep the old function name; internally call StarDist on fused single-channel."""
    x = _fuse_to_single(dapi, cyto)
    x = normalize(x)
    labels, _ = _MODEL.predict_instances(
        x, prob_thresh=PROB_THRESH, nms_thresh=NMS_THRESH, n_tiles=N_TILES
    )
    return labels.astype(np.uint32, copy=False)

def _clear_mem():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()

def main():
    print("== Cell segmentation with StarDist (DAPI + marker, fused) ==")
    print(f"DATA_DIR     : {DATA_DIR.resolve()}")
    ensure_dir(OUT_DIR_CELL); print(f"OUT_DIR_CELL : {OUT_DIR_CELL.resolve()}")

    ds = SampleDataset(DATA_DIR)
    print(f"Found {len(ds)} samples (marker required for cell mask).")

    n_ok, n_skip = 0, 0
    for s in ds:
        try:
            s.load_images()  # need s.nuc_chan & s.cell_chan
            if getattr(s, "cell_chan", None) is None or getattr(s, "nuc_chan", None) is None:
                n_skip += 1
                print(f"[SKIP] {s.base} (missing DAPI or marker)")
                continue

            out_cell = OUT_DIR_CELL / f"{s.base}_pred_cell.npy"
            if out_cell.exists():
                print(f"[SKIP] {s.base} -> exists")
                continue

            cell_mask = _cellsam_cells(s.nuc_chan, s.cell_chan)
            np.save(out_cell, cell_mask)
            n_ok += 1
            print(f"[OK] {s.base} -> {out_cell.name} (cells: {int(cell_mask.max())})")

        except Exception as e:
            print(f"[FAIL] {s.base}: {e}")

        # memory hygiene
        try:
            s.nuc_chan = None; s.cell_chan = None
        except Exception:
            pass
        if "cell_mask" in locals(): del cell_mask
        _clear_mem()

    print(f"Done: cell_ok={n_ok}, cell_skip(no marker)={n_skip}, total={len(ds)})")

if __name__ == "__main__":
    main()

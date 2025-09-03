#!/usr/bin/env python3
"""
prediction_cyto_mesmer_marker_only.py — WHOLE-CELL segmentation with Mesmer (marker-only).
We feed a single-channel marker to Mesmer by constructing a 2-channel tensor without adding new info.
"""

from pathlib import Path
import numpy as np
from deepcell.applications import Mesmer

from utils import SampleDataset, ensure_dir  # NO model logic inside utils

# ---- config (no CLI) ----
DATA_DIR   = Path("/ihome/jbwang/liy121/ifimage/00_dataset")
OUTPUT_DIR = Path("markeronly_prediction")

# Mesmer knobs
IMAGE_MPP   = None           # e.g., 0.5 (µm/px); unknown -> None
COMPARTMENT = "whole-cell"   # we want whole-cell segmentation
NUC_STRATEGY = "copy"        # "copy" | "mem_only"
# "copy"     : nuc = marker, mem = marker  (对 Mesmer更稳定，仍然只有一个信号源)
# "mem_only" : nuc = 0,      mem = marker  (更“严格”的单通道，部分数据会更保守)

# init once
APP = Mesmer()


def _pack_marker_to_two_channels(marker2d: np.ndarray) -> np.ndarray:
    """Return (1, H, W, 2) with channel order [NUC, MEM] for Mesmer, using ONLY the marker signal."""
    x = marker2d
    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[..., 0]
    x = x.astype(np.float32, copy=False)
    vmax = float(x.max())
    if vmax > 0:
        x = x / vmax

    if NUC_STRATEGY == "copy":
        nuc = x
        mem = x
    elif NUC_STRATEGY == "mem_only":
        nuc = np.zeros_like(x, dtype=np.float32)
        mem = x
    else:
        raise ValueError(f"Unknown NUC_STRATEGY={NUC_STRATEGY}")

    X = np.stack([nuc, mem], axis=-1)[None, ...]  # (1, H, W, 2)
    return X


def run_mesmer_marker_only(marker2d: np.ndarray) -> np.ndarray:
    X = _pack_marker_to_two_channels(marker2d)
    if IMAGE_MPP is None:
        y = APP.predict(X, compartment=COMPARTMENT)
    else:
        y = APP.predict(X, image_mpp=IMAGE_MPP, compartment=COMPARTMENT)

    y0 = y[0]
    if y0.ndim == 3:  # sometimes (H,W,1)
        y0 = y0[..., 0]
    return y0.astype(np.int32, copy=False)


def main():
    print("== Cytoplasm prediction (Mesmer, marker-only) ==")
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

            marker = s.cell_chan          # HxW or HxWx1, marker-only
            mask = run_mesmer_marker_only(marker)

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

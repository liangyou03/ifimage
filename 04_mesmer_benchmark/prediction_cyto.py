#!/usr/bin/env python3
"""
prediction_cyto_mesmer.py — Two-channel whole-cell segmentation with Mesmer.
Inputs: DAPI (nuclear) + marker (cyto/membrane).
Output: cell mask (includes nuclei).
"""

from pathlib import Path
import numpy as np
import gc

from deepcell.applications import Mesmer

from utils import SampleDataset, ensure_dir

# ---- config ----
DATA_DIR        = Path("/ihome/jbwang/liy121/ifimage/00_dataset")
OUT_DIR_CELL    = Path("cyto_prediction")

# Mesmer knobs
IMAGE_MPP    = None
COMPARTMENT  = "whole-cell" 

APP = Mesmer()


def _to_float01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    vmax = float(x.max())
    if vmax > 0:
        x /= vmax
    return x


def _make_two_channel_input(dapi: np.ndarray, cyto: np.ndarray) -> np.ndarray:
    """
    Make input for Mesmer: (1, H, W, 2) float32 in [0,1], channels = [DAPI, marker].
    """
    assert dapi.ndim == 2 and cyto.ndim == 2
    assert dapi.shape == cyto.shape
    d = _to_float01(dapi)
    m = _to_float01(cyto)
    X = np.stack([d, m], axis=-1)[None, ...]  # (1, H, W, 2)
    return X


def _mesmer_cells(dapi: np.ndarray, cyto: np.ndarray) -> np.ndarray:
    X = _make_two_channel_input(dapi, cyto)
    # 调用 Mesmer
    if IMAGE_MPP is None:
        y = APP.predict(X, compartment=COMPARTMENT)
    else:
        y = APP.predict(X, image_mpp=IMAGE_MPP, compartment=COMPARTMENT)

    y0 = y[0]
    if y0.ndim == 3:  # (H,W,1) -> (H,W)
        y0 = y0[..., 0]
    return y0.astype(np.uint32, copy=False)


def _clear_mem():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


def main():
    print("== Cell segmentation with Mesmer (DAPI + marker) ==")
    print(f"DATA_DIR     : {DATA_DIR.resolve()}")
    ensure_dir(OUT_DIR_CELL); print(f"OUT_DIR_CELL : {OUT_DIR_CELL.resolve()}")

    ds = SampleDataset(DATA_DIR)
    print(f"Found {len(ds)} samples (marker required for cell mask).")

    n_ok, n_skip = 0, 0
    for s in ds:
        try:
            s.load_images()
            if getattr(s, "cell_chan", None) is None or getattr(s, "nuc_chan", None) is None:
                n_skip += 1
                print(f"[SKIP] {s.base} (missing DAPI or marker)")
                continue

            out_cell = OUT_DIR_CELL / f"{s.base}_pred_cell.npy"
            if out_cell.exists():
                print(f"[SKIP] {s.base} -> exists")
                continue

            cell_mask = _mesmer_cells(s.nuc_chan, s.cell_chan)
            np.save(out_cell, cell_mask)
            n_ok += 1
            print(f"[OK] {s.base} -> {out_cell.name} (cells: {int(cell_mask.max())})")

        except Exception as e:
            print(f"[FAIL] {s.base}: {e}")

        try:
            s.nuc_chan = None; s.cell_chan = None
        except Exception:
            pass
        if "cell_mask" in locals(): del cell_mask
        _clear_mem()

    print(f"Done: cell_ok={n_ok}, cell_skip={n_skip}, total={len(ds)})")


if __name__ == "__main__":
    main()

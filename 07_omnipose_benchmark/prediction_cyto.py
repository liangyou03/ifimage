#!/usr/bin/env python3
"""
prediction_cyto_omni.py — Two-channel whole-cell segmentation with Omnipose.
Inputs: DAPI + marker → cell mask (.npy)
Tries newest models first (cyto3, cyto2_omni), falls back to cyto2. Also supports custom .pth.
"""

from pathlib import Path
import numpy as np
import gc

from cellpose_omni.models import CellposeModel
from utils import SampleDataset, ensure_dir

# ---- paths ----
DATA_DIR     = Path("/ihome/jbwang/liy121/ifimage/00_dataset")
OUT_DIR_CELL = Path("cyto_omnipose")

# ---- model selection ----
MODEL_TYPE         = "auto"
PRETRAINED_MODEL   = None

# ---- knobs (keep minimal) ----
DIAMETER       = None
MASK_THRESHOLD = 0.0
FLOW_THRESHOLD = 0.4
MIN_SIZE       = 30
BATCH_SIZE     = 4
TILE, TILE_OVERLAP = True, 0.1
NORMALIZE, INVERT  = True, False

def _use_gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

def _build_model() -> CellposeModel:
    """Prefer newer weights; fall back gracefully. nclasses=3 avoids branch-specific init bugs."""
    # 1) explicit .pth wins
    if PRETRAINED_MODEL:
        print(f"Using custom weights: {PRETRAINED_MODEL}")
        return CellposeModel(gpu=_use_gpu(), pretrained_model=str(PRETRAINED_MODEL),
                             dim=2, omni=True, nclasses=3, net_avg=True, use_torch=True)
    # 2) auto-try list from newest to stable
    tried = []
    if MODEL_TYPE == "auto":
        candidates = ["cyto3", "cyto2_omni", "cyto2"]
    else:
        candidates = [MODEL_TYPE]
    for name in candidates:
        try:
            print(f"Trying model_type='{name}'...")
            return CellposeModel(gpu=_use_gpu(), model_type=name,
                                 dim=2, omni=True, nclasses=3, net_avg=True, use_torch=True)
        except Exception as e:
            tried.append((name, str(e)))
    # 3) final fallback
    print("Falling back to model_type='cyto2'")
    return CellposeModel(gpu=_use_gpu(), model_type="cyto2",
                         dim=2, omni=True, nclasses=3, net_avg=True, use_torch=True)

APP = _build_model()

def _to_float01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    vmax = float(x.max())
    if vmax > 0: x /= vmax
    return x

def _pack_rgb(dapi: np.ndarray, cyto: np.ndarray) -> np.ndarray:
    """Make pseudo-RGB: G=cyto, B=DAPI (Cellpose channels=[2,3] => cyto=G, nuc=B)."""
    assert dapi.shape == cyto.shape and dapi.ndim == 2
    H, W = dapi.shape
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    rgb[..., 1] = _to_float01(cyto)  # G
    rgb[..., 2] = _to_float01(dapi)  # B
    return rgb

def _eval_omni(img_rgb: np.ndarray):
    """Robust eval(): normalize to (masks, flows, styles, diams)."""
    res = APP.eval(
        [img_rgb], channels=[2, 3], diameter=DIAMETER, omni=True,
        mask_threshold=MASK_THRESHOLD, flow_threshold=FLOW_THRESHOLD,
        min_size=MIN_SIZE, batch_size=BATCH_SIZE,
        normalize=NORMALIZE, invert=INVERT,
        tile=TILE, tile_overlap=TILE_OVERLAP, net_avg=True, show_progress=False
    )
    if isinstance(res, (list, tuple)):  # 3- or 4-tuple
        return (res + (None,))[:4]
    if isinstance(res, dict):
        return res.get("masks"), res.get("flows"), res.get("styles"), res.get("diameters")
    raise RuntimeError(f"Unexpected eval() return type: {type(res)}")

def _omni_cells(dapi: np.ndarray, cyto: np.ndarray) -> np.ndarray:
    img = _pack_rgb(dapi, cyto)
    masks, _, _, _ = _eval_omni(img)
    m = masks[0] if isinstance(masks, (list, tuple)) else masks
    if m is None:
        return np.zeros(img.shape[:2], dtype=np.uint32)
    return m.astype(np.uint32, copy=False)

def _clear_mem():
    try:
        import torch
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()

def main():
    print("== Cell segmentation with Omnipose (DAPI + marker) ==")
    print(f"DATA_DIR     : {DATA_DIR.resolve()}")
    ensure_dir(OUT_DIR_CELL); print(f"OUT_DIR_CELL : {OUT_DIR_CELL.resolve()}")

    ds = SampleDataset(DATA_DIR)
    print(f"Found {len(ds)} samples (marker required).")

    n_ok, n_skip = 0, 0
    for s in ds:
        try:
            s.load_images()
            if getattr(s, "cell_chan", None) is None or getattr(s, "nuc_chan", None) is None:
                n_skip += 1; print(f"[SKIP] {s.base} (missing DAPI or marker)"); continue

            out_cell = OUT_DIR_CELL / f"{s.base}_pred_cell.npy"
            if out_cell.exists():
                print(f"[SKIP] {s.base} -> exists"); continue

            cell_mask = _omni_cells(s.nuc_chan, s.cell_chan)
            np.save(out_cell, cell_mask)
            n_ok += 1
            print(f"[OK] {s.base} -> {out_cell.name} (cells: {int(cell_mask.max())})")
        except KeyboardInterrupt:
            print("Interrupted."); break
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

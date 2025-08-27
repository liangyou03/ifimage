#!/usr/bin/env python3
"""
prediction_cyto_cellsam.py — Multichannel cell segmentation with CellSAM.
Inputs: DAPI (nuclear) + marker (cyto/membrane).
Output: cell mask (includes nuclei).
"""

from pathlib import Path
import numpy as np
import gc

from cellSAM import cellsam_pipeline
from utils import SampleDataset, ensure_dir

# ---- config ----
DATA_DIR        = Path("/ihome/jbwang/liy121/ifimage/00_dataset")
OUT_DIR_CELL    = Path("cyto")

# cellsam_pipeline params
BBOX_THRESHOLD   = 0.4
USE_WSI          = False
LOW_CONTRAST_ENH = False
GAUGE_CELL_SIZE  = False

def _make_multiplex_input(dapi: np.ndarray, cyto: np.ndarray) -> np.ndarray:
    """按 (blank, nuclear, membrane) 拼成 (H,W,3) 给 CellSAM。"""
    assert dapi.shape == cyto.shape, "DAPI 与 cyto 尺寸必须一致"
    H, W = dapi.shape
    seg = np.zeros((H, W, 3), dtype=dapi.dtype)
    seg[..., 1] = dapi       # nuclear -> 第2通道
    seg[..., 2] = cyto       # membrane/cyto -> 第3通道
    return seg

def _cellsam_cells(dapi: np.ndarray, cyto: np.ndarray) -> np.ndarray:
    img = _make_multiplex_input(dapi, cyto)
    mask = cellsam_pipeline(
        img,
        use_wsi=USE_WSI,
        low_contrast_enhancement=LOW_CONTRAST_ENH,
        gauge_cell_size=GAUGE_CELL_SIZE,
        swap_channels=False
    )
    return mask.astype(np.uint32, copy=False)

def _clear_mem():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()

def main():
    print("== Cell segmentation with CellSAM (DAPI + marker) ==")
    print(f"DATA_DIR     : {DATA_DIR.resolve()}")
    ensure_dir(OUT_DIR_CELL); print(f"OUT_DIR_CELL : {OUT_DIR_CELL.resolve()}")

    ds = SampleDataset(DATA_DIR)
    print(f"Found {len(ds)} samples (marker required for cell mask).")

    n_ok, n_skip = 0, 0
    for s in ds:
        try:
            s.load_images()  # 需提供 s.nuc_chan, s.cell_chan
            if getattr(s, "cell_chan", None) is None:
                n_skip += 1
                print(f"[SKIP] {s.base} (no marker)")
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

        # 及时释放，防止累计内存
        try:
            s.nuc_chan = None; s.cell_chan = None
        except Exception:
            pass
        if "cell_mask" in locals(): del cell_mask
        _clear_mem()

    print(f"Done: cell_ok={n_ok}, cell_skip(no marker)={n_skip}, total={len(ds)})")

if __name__ == "__main__":
    main()

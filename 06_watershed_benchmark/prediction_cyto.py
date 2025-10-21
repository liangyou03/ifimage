#!/usr/bin/env python3
"""
prediction_cyto_watershed_from_dapi.py — Two-channel whole-cell segmentation with watershed.
Seeds: DAPI (nuclear). Expansion: marker (cyto/membrane). Output: .npy label mask.
"""

from pathlib import Path
import numpy as np
import gc
from typing import Tuple

from scipy import ndimage as ndi
from skimage import filters, morphology, segmentation, measure, feature, util
from utils import SampleDataset, ensure_dir

# ---- config ----
DATA_DIR     = Path("/ihome/jbwang/liy121/ifimage/00_dataset")
OUT_DIR_CELL = Path("cyto_prediction")

# tuning knobs
NUC_SIGMA        = 1.0   # DAPI smoothing
NUC_MIN_SIZE     = 60    # min nucleus size (px)
NUC_MIN_DISTANCE = 10    # min distance between nuclear peaks (px)

CYTO_SIGMA       = 1.5   # marker smoothing
FG_CLOSE_RADIUS  = 3     # binary closing radius
FG_MIN_HOLE_AREA = 64    # fill small holes (px)
COMPACTNESS      = 0.001 # watershed compactness (0=classic)

def _to_float01(x: np.ndarray) -> np.ndarray:
    x = util.img_as_float32(x, force_copy=False)
    vmax = float(x.max())
    if vmax > 0:
        x /= vmax
    return x

def _prep_nuclear_seeds(dapi: np.ndarray) -> np.ndarray:
    d = _to_float01(dapi)
    if NUC_SIGMA > 0:
        d = filters.gaussian(d, sigma=NUC_SIGMA)
    t = filters.threshold_otsu(d)
    nuc_fg = d > t
    if NUC_MIN_SIZE > 0:
        nuc_fg = morphology.remove_small_objects(nuc_fg, min_size=NUC_MIN_SIZE)
    if nuc_fg.sum() == 0:
        return np.zeros_like(nuc_fg, dtype=np.int32)

    dist = ndi.distance_transform_edt(nuc_fg)
    coords = feature.peak_local_max(dist, labels=nuc_fg, min_distance=NUC_MIN_DISTANCE)
    marker_mask = np.zeros_like(nuc_fg, dtype=bool)
    if coords.size > 0:
        marker_mask[tuple(coords.T)] = True
        seeds = measure.label(marker_mask)
    else:
        seeds = measure.label(nuc_fg)
    return seeds.astype(np.int32, copy=False)

def _prep_cyto_landscape_and_mask(cyto: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (elevation, fg_mask) from marker channel."""
    m = _to_float01(cyto)
    m_smooth = filters.gaussian(m, sigma=CYTO_SIGMA) if CYTO_SIGMA > 0 else m
    elevation = filters.sobel(m_smooth)

    th = filters.threshold_otsu(m_smooth)
    fg = m_smooth > th
    if FG_CLOSE_RADIUS > 0:
        fg = morphology.binary_closing(fg, morphology.disk(FG_CLOSE_RADIUS))
    if FG_MIN_HOLE_AREA > 0:
        fg = morphology.remove_small_holes(fg, area_threshold=FG_MIN_HOLE_AREA)
    return elevation.astype(np.float32, copy=False), fg

def _watershed_cells_from_dapi(dapi: np.ndarray, cyto: np.ndarray) -> np.ndarray:
    assert dapi.ndim == 2 and cyto.ndim == 2 and dapi.shape == cyto.shape
    seeds = _prep_nuclear_seeds(dapi)
    if seeds.max() == 0:
        return np.zeros_like(seeds, dtype=np.uint32)
    elevation, fg_mask = _prep_cyto_landscape_and_mask(cyto)
    if fg_mask.sum() == 0:
        return np.zeros_like(seeds, dtype=np.uint32)

    labels = segmentation.watershed(
        image=elevation,
        markers=seeds,
        mask=fg_mask,
        compactness=COMPACTNESS
    )
    labels = morphology.remove_small_objects(labels, min_size=NUC_MIN_SIZE)
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
    print("== Cell segmentation with watershed (DAPI seeds + marker landscape) ==")
    print(f"DATA_DIR     : {DATA_DIR.resolve()}")
    ensure_dir(OUT_DIR_CELL); print(f"OUT_DIR_CELL : {OUT_DIR_CELL.resolve()}")

    ds = SampleDataset(DATA_DIR)
    print(f"Found {len(ds)} samples (require DAPI + marker).")

    n_ok, n_skip = 0, 0
    for s in ds:
        try:
            s.load_images()
            if getattr(s, "cell_chan", None) is None or getattr(s, "nuc_chan", None) is None:
                n_skip += 1
                print(f"[SKIP] {s.base} (missing DAPI or marker)")
                continue

            out_cell = OUT_DIR_CELL / f"{s.base}.npy"
            if out_cell.exists():
                print(f"[SKIP] {s.base} -> exists")
                continue

            cell_mask = _watershed_cells_from_dapi(s.nuc_chan, s.cell_chan)
            np.save(out_cell, cell_mask)
            print(f"[OK] {s.base} -> {out_cell.name} (cells: {int(cell_mask.max())})")
            n_ok += 1

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

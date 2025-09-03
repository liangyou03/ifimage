#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
refilter_by_marker.py

Refine predicted cell masks by per-cell mean intensity on the marker channel.
- No CSV output
- All parameters are defined in the constants below
"""

from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import tifffile as tiff
from scipy import ndimage as ndi

# -------------------- CONFIG (edit as needed) --------------------
DATA_DIR = Path("../00_dataset")          # Raw data directory (only *_marker.tif(f) is used)
CYTO_DIR = Path("./cyto_prediction")      # Predicted cell masks directory (*.npy)
OUT_DIR  = CYTO_DIR.parent / (CYTO_DIR.name + "_refined")  # Output directory

THRESHOLD_METHOD = "otsu"   # "otsu" or "gmm"
MIN_AREA = 0                # Remove tiny cells by area (in pixels) before intensity filtering; 0 disables
CYTO_GLOB = "*_pred_cyto.npy"  # Pattern to find prediction mask files
MARKER_SUFFIX_CANDIDATES = [   # Candidate suffixes for marker files
    "_marker.tiff", "_marker.tif", "_MARKER.tiff", "_MARKER.tif"
]
# Remove these substrings from the prediction filename stem to recover the shared "base"
# which is then combined with the marker suffix to locate the raw marker image.
REMOVE_FROM_CYTO_STEM = ["_pred_cyto"]

# -------------------- IMPLEMENTATION --------------------
try:
    from skimage.filters import threshold_otsu
except Exception as e:
    threshold_otsu = None
    print(f"[warn] skimage import failed: {e}; Otsu will fall back to a percentile threshold.", file=sys.stderr)

def _marker_path_for_base(base: str) -> Path | None:
    """Find the marker image for a given base name."""
    for suf in MARKER_SUFFIX_CANDIDATES:
        p = DATA_DIR / f"{base}{suf}"
        if p.exists():
            return p
    # Last resort: glob
    hits = list(DATA_DIR.glob(f"{base}_marker.tif*"))
    return hits[0] if hits else None

def _base_from_cyto_path(p: Path) -> str:
    """Derive the shared base name from a prediction file path."""
    stem = p.stem
    for s in REMOVE_FROM_CYTO_STEM:
        stem = stem.replace(s, "")
    return stem

def _load_marker(path: Path) -> np.ndarray:
    """Load a marker image as float32, squeezing to 2D if necessary."""
    img = tiff.imread(str(path))
    img = np.squeeze(img)
    if img.ndim > 2:  # fallback safety
        img = img[..., 0]
    return img.astype(np.float32, copy=False)

def _compute_means(img: np.ndarray, labels_im: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-instance mean intensity over the marker image."""
    labels = np.unique(labels_im)
    labels = labels[labels > 0]
    if labels.size == 0:
        return labels, np.empty((0,), dtype=np.float32)
    means = ndi.mean(img, labels=labels_im, index=labels).astype(np.float32)
    return labels, means

def _thr_otsu(values: np.ndarray) -> float:
    """Otsu threshold on 1D values; fallback to 75th percentile if skimage is missing."""
    if values.size == 0:
        return float("nan")
    if threshold_otsu is not None:
        return float(threshold_otsu(values))
    # simple fallback
    return float(np.percentile(values, 75.0))

def _thr_gmm(values: np.ndarray) -> float:
    """Two-component GMM threshold (midpoint between means); falls back to Otsu if sklearn is unavailable."""
    if values.size == 0:
        return float("nan")
    try:
        from sklearn.mixture import GaussianMixture
    except Exception as e:
        print(f"[warn] sklearn not available ({e}); falling back to Otsu.", file=sys.stderr)
        return _thr_otsu(values)
    x = values.reshape(-1, 1).astype(np.float64)
    gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=0)
    gmm.fit(x)
    mus = np.sort(gmm.means_.ravel())
    return float((mus[0] + mus[1]) * 0.5)

def _decide_threshold(values: np.ndarray) -> float:
    """Dispatch to the selected thresholding method."""
    if THRESHOLD_METHOD.lower() == "otsu":
        return _thr_otsu(values)
    elif THRESHOLD_METHOD.lower() == "gmm":
        return _thr_gmm(values)
    raise ValueError("unsupported threshold method")

def _ensure_out_dir(p: Path) -> None:
    """Create output directory if it does not exist."""
    p.mkdir(parents=True, exist_ok=True)

def _maybe_align_shapes(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Crop both arrays to their overlapping top-left region if shapes differ."""
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    if (h, w) != a.shape:
        a = a[:h, :w]
    if (h, w) != b.shape:
        b = b[:h, :w]
    return a, b

def process_one(marker_path: Path, cyto_path: Path, out_path: Path) -> None:
    """Process a single image: compute per-cell means, threshold, and save filtered mask."""
    marker = _load_marker(marker_path)
    mask = np.load(cyto_path)

    marker, mask = _maybe_align_shapes(marker, mask)

    # Optional area filtering
    if MIN_AREA > 0 and mask.max() > 0:
        areas = ndi.sum(np.ones_like(mask, dtype=np.float32),
                        labels=mask, index=np.arange(1, mask.max() + 1))
        small = set((np.where(areas < float(MIN_AREA))[0] + 1).tolist())
        if small:
            mask = np.where(np.isin(mask, list(small)), 0, mask)

    label_ids, means = _compute_means(marker, mask)
    thr = _decide_threshold(means)
    keep = label_ids[means > thr] if label_ids.size else np.array([], dtype=label_ids.dtype)

    if keep.size == 0:
        filtered = np.zeros_like(mask, dtype=mask.dtype)
    else:
        filtered = np.where(np.isin(mask, keep), mask, 0).astype(mask.dtype, copy=False)

    np.save(out_path, filtered)

def main():
    _ensure_out_dir(OUT_DIR)
    print(f"DATA_DIR : {DATA_DIR.resolve()}")
    print(f"CYTO_DIR : {CYTO_DIR.resolve()}")
    print(f"OUT_DIR  : {OUT_DIR.resolve()}")
    print(f"METHOD   : {THRESHOLD_METHOD}  (min_area={MIN_AREA})")

    files = sorted(CYTO_DIR.glob(CYTO_GLOB))
    if not files:
        print(f"[warn] No prediction files found: pattern `{CYTO_GLOB}`")
        return

    ok = 0
    skipped_marker = 0
    for cyto_path in files:
        base = _base_from_cyto_path(cyto_path)
        marker_path = _marker_path_for_base(base)
        if marker_path is None:
            print(f"[SKIP] {base}: marker image not found")
            skipped_marker += 1
            continue

        out_path = OUT_DIR / cyto_path.name
        try:
            process_one(marker_path, cyto_path, out_path)
            ok += 1
            print(f"[OK]  {base} -> {out_path.name}")
        except Exception as e:
            print(f"[FAIL] {base}: {e}")

    print(f"Done: success={ok}, missing_marker={skipped_marker}, total={len(files)}")

if __name__ == "__main__":
    main()

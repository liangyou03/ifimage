#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_refilter_batch.py  (refine & save only; no evaluation)

- Features: "mean" | "zscore_mean" | "bgcorr_mean" | "ring_mean"
- Thresholds: "gmm" | "otsu"
- Marker gate: OFF / auto(threshold marker) / precomputed(load binary masks)
- Saves refined masks into named subfolders per experiment.

Edit the MAIN SETTINGS in main() to choose:
  • custom experiments or full grid
  • gate mode and directories
  • dataset paths
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple, Optional, List
import sys
import numpy as np
import tifffile as tiff
from scipy import ndimage as ndi

# ---------------- Paths (EDIT THESE) ----------------
DATA_DIR = Path("/ihome/jbwang/liy121/ifimage/00_dataset")                # has raw marker images
CYTO_DIR = Path("/ihome/jbwang/liy121/ifimage/01_cellpose_benchmark/cyto_prediction")
OUT_ROOT = CYTO_DIR.parent / "refilter_outputs"
MARKER_DIR = Path("/ihome/jbwang/liy121/ifimage/00_dataset")

# If using precomputed gate masks (binary marker-foreground):
GATE_DIR = Path("/path/to/marker_gate_masks")  # e.g., your marker-only segmentation dir

# ---------------- File conventions ----------------
CYTO_GLOB = "*_pred_cyto.npy"
MARKER_SUFFIXES = ["_marker.tiff", "_marker.tif", "_MARKER.tiff", "_MARKER.tif"]
REMOVE_FROM_CYTO_STEM = ["_pred_cyto"]

# Expected file names for precomputed gate (try these, then a glob fallback)
GATE_SUFFIXES = ["_markerfg.npy", "_markerfg.tif", "_markerfg.tiff", "_MARKERFG.npy", "_MARKERFG.tif", "_pred_marker_only.npy"]

# ---------------- Fixed internals ----------------
# Rings for bgcorr_mean / ring_mean
RING_INNER, RING_OUTER = 3, 6  # pixels
# Marker gate parameters
MIN_OVERLAP_FRAC = 0.20
GLOBAL_PERC_FALLBACK = 95.0  # used if skimage is missing

# Optional deps
try:
    from skimage.filters import threshold_otsu as _sk_otsu
except Exception as e:
    _sk_otsu = None
    print(f"[warn] skimage.otsu unavailable: {e}", file=sys.stderr)
try:
    from sklearn.mixture import GaussianMixture as _GMM
except Exception:
    _GMM = None

# ---------------- Small utils ----------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def marker_path_for_base(base: str) -> Optional[Path]:
    for suf in MARKER_SUFFIXES:
        p = MARKER_DIR / f"{base}{suf}"
        if p.exists():
            return p
    hits = list(MARKER_DIR.glob(f"{base}_marker.tif*"))
    return hits[0] if hits else None

def gate_path_for_base(base: str) -> Optional[Path]:
    # Try listed suffixes in GATE_DIR; else a generic glob
    for suf in GATE_SUFFIXES:
        p = GATE_DIR / f"{base}{suf}"
        if p.exists():
            return p
    hits = list(GATE_DIR.glob(f"{base}*markerfg*"))
    return hits[0] if hits else None

def base_from_cyto_path(p: Path) -> str:
    stem = p.stem
    for s in REMOVE_FROM_CYTO_STEM:
        stem = stem.replace(s, "")
    return stem

def load_tiff_2d_float(path: Path) -> np.ndarray:
    img = tiff.imread(str(path))
    img = np.squeeze(img)
    if img.ndim > 2:
        img = img[..., 0]
    return img.astype(np.float32, copy=False)

def align_to_common(*arrays: np.ndarray) -> Tuple[np.ndarray, ...]:
    h = min(a.shape[0] for a in arrays)
    w = min(a.shape[1] for a in arrays)
    return tuple(a[:h, :w] for a in arrays)

def valid_labels(lab: np.ndarray) -> np.ndarray:
    ids = np.unique(lab)
    return ids[ids > 0]

# ---------------- Features ----------------
def feat_mean(img: np.ndarray, lab: np.ndarray, ids: np.ndarray) -> np.ndarray:
    return ndi.mean(img, labels=lab, index=ids).astype(np.float32)

def disk_struct(r: int) -> np.ndarray:
    y, x = np.ogrid[-r:r+1, -r:r+1]
    return (x*x + y*y) <= r*r

def ring_mask(lab: np.ndarray, k: int, r_in: int, r_out: int) -> np.ndarray:
    from scipy.ndimage import binary_dilation
    cell = (lab == k)
    if not cell.any():
        return cell
    outer = binary_dilation(cell, structure=disk_struct(r_out))
    inner = binary_dilation(cell, structure=disk_struct(r_in)) if r_in > 0 else cell
    ring = outer & (~inner)
    ring &= (lab == 0)  # exclude other labeled interiors
    return ring

def feat_bgcorr_mean(img: np.ndarray, lab: np.ndarray, ids: np.ndarray) -> np.ndarray:
    out = np.empty(len(ids), dtype=np.float32)
    bg_fallback = float(np.percentile(img, 5.0))
    for i, k in enumerate(ids):
        vals = img[lab == k]
        if vals.size == 0:
            out[i] = np.nan
            continue
        ring = ring_mask(lab, int(k), RING_INNER, RING_OUTER)
        bg = img[ring].mean() if ring.any() else bg_fallback
        out[i] = float(vals.mean() - bg)
    return out

def feat_ring_mean(img: np.ndarray, lab: np.ndarray, ids: np.ndarray) -> np.ndarray:
    out = np.empty(len(ids), dtype=np.float32)
    for i, k in enumerate(ids):
        ring = ring_mask(lab, int(k), RING_INNER, RING_OUTER)
        vals = img[ring]
        out[i] = float(vals.mean()) if vals.size > 0 else np.nan
    return out

def compute_feature(img: np.ndarray, lab: np.ndarray, feature_kind: str) -> Tuple[np.ndarray, np.ndarray]:
    ids = valid_labels(lab)
    if ids.size == 0:
        return ids, np.empty((0,), dtype=np.float32)
    fk = feature_kind.lower()
    if   fk == "mean":
        feats = feat_mean(img, lab, ids)
    elif fk == "zscore_mean":
        vals = feat_mean(img, lab, ids)
        mu = float(np.nanmean(vals)) if np.isfinite(vals).any() else 0.0
        sd = float(np.nanstd(vals))
        if not np.isfinite(sd) or sd == 0.0:
            sd = 1.0
        feats = (vals - mu) / sd
    elif fk == "bgcorr_mean":
        feats = feat_bgcorr_mean(img, lab, ids)
    elif fk == "ring_mean":
        feats = feat_ring_mean(img, lab, ids)
    else:
        raise ValueError(f"Unsupported FEATURE_KIND: {feature_kind}")
    feats = np.asarray(feats, dtype=np.float32)
    if not np.isfinite(feats).any():
        return ids, np.zeros_like(feats)
    feats[~np.isfinite(feats)] = np.nanmin(feats[np.isfinite(feats)])
    return ids, feats

# ---------------- Thresholds ----------------
def thr_otsu(values: np.ndarray) -> float:
    vals = values[np.isfinite(values)]
    if vals.size == 0:
        return float("nan")
    if _sk_otsu is not None:
        return float(_sk_otsu(vals))
    return float(np.quantile(vals, 0.75))  # fallback

def thr_gmm(values: np.ndarray) -> float:
    vals = values[np.isfinite(values)]
    if vals.size == 0:
        return float("nan")
    if _GMM is None:
        print("[warn] sklearn not available; falling back to Otsu.", file=sys.stderr)
        return thr_otsu(vals)
    x = vals.reshape(-1, 1).astype(np.float64)
    gmm = _GMM(n_components=2, covariance_type="full", random_state=0).fit(x)
    mus = np.sort(gmm.means_.ravel())
    return float((mus[0] + mus[1]) * 0.5)

def decide_threshold(values: np.ndarray, method: str) -> float:
    m = method.lower()
    if m == "gmm":  return thr_gmm(values)
    if m == "otsu": return thr_otsu(values)
    raise ValueError("THRESHOLD_METHOD must be 'gmm' or 'otsu'.")

# ---------------- Marker gate builders ----------------
def build_auto_gate(marker: np.ndarray) -> np.ndarray:
    if _sk_otsu is not None:
        thr = float(_sk_otsu(marker))
    else:
        thr = float(np.percentile(marker, GLOBAL_PERC_FALLBACK))
    return marker > thr

def load_precomputed_gate(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        m = np.load(path)
        return (m > 0).astype(bool)
    # tif / tiff
    m = tiff.imread(str(path))
    m = np.squeeze(m)
    if m.ndim > 2:
        m = m[..., 0]
    return (m > 0).astype(bool)

def get_marker_gate(base: str, marker: np.ndarray, gate_mode: str) -> Optional[np.ndarray]:
    """
    gate_mode: "off" | "auto" | "precomputed"
    Returns boolean mask or None.
    """
    if gate_mode == "off":
        return None
    if gate_mode == "auto":
        return build_auto_gate(marker)
    if gate_mode == "precomputed":
        p = gate_path_for_base(base)
        if p is None:
            print(f"[warn] precomputed gate not found for {base}; skipping gate.")
            return None
        g = load_precomputed_gate(p)
        # align to marker shape
        h = min(g.shape[0], marker.shape[0]); w = min(g.shape[1], marker.shape[1])
        return g[:h, :w]
    raise ValueError("gate_mode must be 'off', 'auto', or 'precomputed'.")

# ---------------- Refinement (single image) ----------------
def refine_one(base: str,
               marker: np.ndarray,
               pred_lab: np.ndarray,
               feature_kind: str,
               thr_method: str,
               min_area: int,
               gate_mode: str) -> np.ndarray:
    # Align shapes
    h = min(marker.shape[0], pred_lab.shape[0]); w = min(marker.shape[1], pred_lab.shape[1])
    marker = marker[:h, :w]; lab = pred_lab[:h, :w].copy()

    # Area prefilter
    if min_area > 0 and lab.max() > 0:
        areas = ndi.sum(np.ones_like(lab, dtype=np.float32),
                        labels=lab, index=np.arange(1, lab.max()+1))
        tiny = np.where(areas < float(min_area))[0] + 1
        if tiny.size > 0:
            lab = np.where(np.isin(lab, tiny), 0, lab)

    # Scores
    ids, scores = compute_feature(marker, lab, feature_kind)

    # Threshold
    thr = decide_threshold(scores, thr_method)
    keep_ids = np.array([], dtype=ids.dtype) if (ids.size == 0 or not np.isfinite(thr)) else ids[scores > thr]

    # Marker gate
    gmask = get_marker_gate(base, marker, gate_mode)
    if gmask is not None and ids.size > 0:
        # align to lab shape
        h = min(gmask.shape[0], lab.shape[0]); w = min(gmask.shape[1], lab.shape[1])
        gmask = gmask[:h, :w]
        keep = np.zeros_like(ids, dtype=bool)
        for i, k in enumerate(ids):
            pix = (lab == int(k))
            area = int(np.count_nonzero(pix))
            if area == 0:
                keep[i] = False
                continue
            overlap = int(np.count_nonzero(pix & gmask))
            keep[i] = (overlap / float(area) >= MIN_OVERLAP_FRAC)
        keep_ids = np.intersect1d(keep_ids, ids[keep], assume_unique=False)

    # Build refined mask
    if keep_ids.size == 0:
        return np.zeros_like(lab, dtype=lab.dtype)
    return np.where(np.isin(lab, keep_ids), lab, 0).astype(lab.dtype, copy=False)

# ---------------- Run one config across dataset ----------------
def run_config(feature: str, thr: str, min_area: int, gate_mode: str) -> None:
    slug = f"feat-{feature}_thr-{thr}_area-{min_area}_gate-{gate_mode}"
    out_dir = OUT_ROOT / slug
    ensure_dir(out_dir)
    files = sorted(CYTO_DIR.glob(CYTO_GLOB))
    print(f"[run] {slug}  (n_files={len(files)})")
    saved = 0; skipped_marker = 0

    for f in files:
        base = base_from_cyto_path(f)
        mpath = marker_path_for_base(base)
        if mpath is None:
            if skipped_marker < 5:  # show first few examples
                print(f"[skip] marker missing for {base}  (looked in: {MARKER_DIR})")
            skipped_marker += 1
            continue
        marker = load_tiff_2d_float(mpath)
        pred_lab = np.load(f)
        refined = refine_one(base, marker, pred_lab, feature, thr, min_area, gate_mode)
        np.save(out_dir / f.name, refined)
        saved += 1
        if saved <= 3:  # show a few saves so you see exact paths
            print(f"[save] {out_dir / f.name}")

    print(f"[done] {slug}: saved={saved}, skipped_marker={skipped_marker}, "
          f"out_dir={out_dir.resolve()}")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluation_core.py  (StarDist-backed)

Key changes:
- Average Precision (AP) at multiple IoU thresholds is computed via
  `stardist.matching.matching(...)` when available (fast and standard).
- Fallback to the previous Hungarian implementation if StarDist is not installed.
- Robust handling when no GT/pred pairs are matched (no KeyError).
- Retains AJI and boundary metrics.

Dependencies: numpy, scipy, pandas, tifffile
Optional: stardist
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import warnings

import numpy as np
import pandas as pd
import tifffile as tiff
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment

# Optional StarDist backend
try:
    from stardist.matching import matching as sd_matching
    _SD_AVAILABLE = True
    print("Using StarDist for Average Precision computation.")
except Exception:
    _SD_AVAILABLE = False
    print("StarDist not available; falling back to Hungarian method for Average Precision.")


# --------------------------- Configuration ---------------------------

@dataclass
class EvalConfig:
    gt_dir: Path                               # Directory with GT masks
    pred_dirs: Dict[str, Path]                 # {algorithm_name: directory}
    gt_glob: str = "*.npy"                     # Pattern for GT files
    pred_glob: str = "*.npy"                   # Pattern for prediction files
    gt_strip: List[str] = field(default_factory=lambda: [
        "_cellbodies", "_dapimultimask", "_gt", "_GT"
    ])
    pred_strip: Dict[str, List[str]] = field(default_factory=lambda: {
        # Example per algorithm: {"cyto": ["_pred_cyto"], "nuc": ["_pred_nuc"]}
    })
    ap_thresholds: Iterable[float] = (0.5, 0.75, 0.9)
    boundary_scales: Iterable[float] = (0.5, 1.0, 2.0)  # fraction of median diameter for tolerance radius


# --------------------------- I/O helpers ---------------------------

def _read_mask_any(p: Path) -> np.ndarray:
    """Read a mask from .npy or .tif/.tiff. If binary, label connected components.
    Returns int32 label image where 0 is background."""
    if p.suffix.lower() == ".npy":
        arr = np.load(p)
    else:
        arr = tiff.imread(str(p))
        arr = np.squeeze(arr)
        if arr.ndim > 2:
            arr = arr[..., 0]
    if arr.dtype == bool:
        arr = arr.astype(np.uint8)
    if arr.ndim != 2:
        raise ValueError(f"mask must be 2D: {p}")
    uniq = np.unique(arr)
    if uniq.size <= 3 and uniq.min() == 0:
        labeled, _ = ndi.label(arr > 0)
        return labeled.astype(np.int32, copy=False)
    return arr.astype(np.int32, copy=False)


def _to_base(stem: str, strips: List[str]) -> str:
    """Strip known suffixes from a filename stem to obtain the shared basename."""
    base = stem
    for s in strips:
        base = base.replace(s, "")
    return base


def _pair_by_base(
    gt_dir: Path,
    pred_dir: Path,
    gt_glob: str,
    pred_glob: str,
    gt_strip: List[str],
    pred_strip: List[str],
) -> List[Tuple[str, Path, Path]]:
    """Pair GT and prediction files using a shared basename after stripping suffixes."""
    gt_files = sorted(gt_dir.glob(gt_glob))
    pr_files = sorted(pred_dir.glob(pred_glob))
    gt_map = {_to_base(p.stem, gt_strip): p for p in gt_files}
    pr_map = {_to_base(p.stem, pred_strip): p for p in pr_files}
    keys = sorted(set(gt_map) & set(pr_map))
    pairs = [(k, gt_map[k], pr_map[k]) for k in keys]

    missing_pred = sorted(set(gt_map) - set(pr_map))
    missing_gt = sorted(set(pr_map) - set(gt_map))
    if missing_pred:
        warnings.warn(f"{pred_dir.name}: {len(missing_pred)} GT files had no prediction (e.g., {missing_pred[:3]}).")
    if missing_gt:
        warnings.warn(f"{pred_dir.name}: {len(missing_gt)} predictions had no GT (e.g., {missing_gt[:3]}).")
    return pairs


def _align_shapes(gt: np.ndarray, pr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """If shapes differ, crop both to the overlapping top-left region."""
    if gt.shape == pr.shape:
        return gt, pr
    h = min(gt.shape[0], pr.shape[0])
    w = min(gt.shape[1], pr.shape[1])
    if (h, w) != gt.shape:
        gt = gt[:h, :w]
    if (h, w) != pr.shape:
        pr = pr[:h, :w]
    return gt, pr


# --------------------------- Core helpers ---------------------------

def _label_overlap(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Fast contingency table via vectorized bincount (no Python loops)."""
    if true.size != pred.size:
        raise ValueError("true and pred must have the same number of pixels")
    n_true = int(true.max()) + 1
    n_pred = int(pred.max()) + 1
    t = true.astype(np.int64, copy=False).ravel()
    p = pred.astype(np.int64, copy=False).ravel()
    pair = t * n_pred + p
    counts = np.bincount(pair, minlength=n_true * n_pred)
    return counts.reshape(n_true, n_pred)


def iou_matrix(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute the IoU matrix including background row/column."""
    overlap = _label_overlap(true, pred)
    t_sum = overlap.sum(axis=1, keepdims=True).astype(np.float64)
    p_sum = overlap.sum(axis=0, keepdims=True).astype(np.float64)
    denom = t_sum + p_sum - overlap
    denom[denom == 0] = 1.0
    return (overlap / denom).astype(np.float32)


# --------------------------- Metrics ---------------------------

def aji(true: np.ndarray, pred: np.ndarray) -> float:
    """Aggregated Jaccard Index (AJI)."""
    if true.max() == 0 and pred.max() == 0:
        return 1.0
    overlap = _label_overlap(true, pred)[1:, 1:]
    iou = iou_matrix(true, pred)[1:, 1:]
    n_t, n_p = iou.shape
    area_t = overlap.sum(axis=1)
    area_p = overlap.sum(axis=0)
    if n_t == 0 and n_p == 0:
        return 1.0
    if n_t == 0 or n_p == 0:
        return 0.0
    # greedy by IoU
    pairs, used_t, used_p = [], set(), set()
    order = np.argsort(iou.ravel())[::-1]
    for idx in order:
        ti = int(idx // n_p); pi = int(idx % n_p)
        if ti in used_t or pi in used_p:
            continue
        if iou[ti, pi] <= 0.0:
            break
        pairs.append((ti, pi)); used_t.add(ti); used_p.add(pi)
    inter = sum(overlap[ti, pi] for ti, pi in pairs)
    unions = sum(area_t[ti] + area_p[pi] - overlap[ti, pi] for ti, pi in pairs)
    unmatched_t = [i for i in range(n_t) if i not in used_t]
    unmatched_p = [j for j in range(n_p) if j not in used_p]
    denom = unions + area_t[unmatched_t].sum() + area_p[unmatched_p].sum()
    return float(inter / denom) if denom > 0 else 0.0


def _sd_to_dict(m):
    """Normalize StarDist Matching object across versions."""
    # newish versions: namedtuple-like with _asdict()
    if hasattr(m, "_asdict"):
        return m._asdict()
    # some versions expose to_dict()
    if hasattr(m, "to_dict"):
        return m.to_dict()
    # last resort: pull common attributes directly
    keys = ["tp", "fp", "fn", "precision", "recall", "f1",
            "n_true", "n_pred", "thresh", "criterion",
            "mean_true_score", "mean_matched_score", "panoptic_quality"]
    d = {}
    for k in keys:
        if hasattr(m, k):
            d[k] = getattr(m, k)
    return d

def _average_precision_stardist(true: np.ndarray, pred: np.ndarray, thresholds):
    """AP via StarDist matching backend (robust across versions)."""
    T = len(thresholds)
    ap = np.zeros((T,), np.float32)
    tp = np.zeros((T,), np.int32)
    fp = np.zeros((T,), np.int32)
    fn = np.zeros((T,), np.int32)
    for k, th in enumerate(thresholds):
        m = sd_matching(true, pred, thresh=float(th), criterion="iou")
        s = _sd_to_dict(m)
        tp[k] = int(s.get("tp", 0))
        fp[k] = int(s.get("fp", 0))
        fn[k] = int(s.get("fn", 0))
        denom = tp[k] + fp[k] + fn[k]
        ap[k] = float(tp[k] / denom) if denom > 0 else 0.0
    return ap, tp, fp, fn



def _average_precision_hungarian(true: np.ndarray, pred: np.ndarray, thresholds) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Previous implementation using Hungarian assignment."""
    iou = iou_matrix(true, pred)[1:, 1:]
    n_true, n_pred = iou.shape
    T = len(thresholds)
    ap = np.zeros((T,), np.float32)
    tp = np.zeros((T,), np.int32)
    fp = np.zeros((T,), np.int32)
    fn = np.zeros((T,), np.int32)
    for k, th in enumerate(thresholds):
        if n_true == 0 and n_pred == 0:
            ap[k] = 1.0; tp[k] = fp[k] = fn[k] = 0; continue
        if n_true == 0:
            ap[k] = 0.0; tp[k] = 0; fp[k] = n_pred; fn[k] = 0; continue
        if n_pred == 0:
            ap[k] = 0.0; tp[k] = 0; fp[k] = 0; fn[k] = n_true; continue
        n_min = min(n_true, n_pred)
        cost = -(iou >= th).astype(float) - iou / (2.0 * n_min)
        ti, pi = linear_sum_assignment(cost)
        matched = iou[ti, pi] >= th
        tp[k] = int(matched.sum())
        fp[k] = int(n_pred - tp[k])
        fn[k] = int(n_true - tp[k])
        denom = tp[k] + fp[k] + fn[k]
        ap[k] = float(tp[k] / denom) if denom > 0 else 0.0
    return ap, tp, fp, fn


def average_precision(true: np.ndarray, pred: np.ndarray, thresholds=(0.5, 0.75, 0.9)) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Dispatch AP to StarDist backend if available; otherwise fallback to Hungarian."""
    if _SD_AVAILABLE:
        return _average_precision_stardist(true, pred, thresholds)
    return _average_precision_hungarian(true, pred, thresholds)


# --------------------------- Boundary metrics ---------------------------

def _binary_outline(lbl: np.ndarray) -> np.ndarray:
    """Return a binary outline map for a labeled mask (outer boundary of instances)."""
    b = lbl > 0
    er = ndi.binary_erosion(b, structure=np.ones((3, 3), bool), border_value=0)
    return b ^ er


def _median_eq_diameter(lbl: np.ndarray) -> float:
    """Median equivalent diameter of instances; used to scale the tolerance radius."""
    if lbl.max() == 0:
        return 1.0
    idx = np.arange(1, lbl.max() + 1)
    areas = ndi.sum(np.ones_like(lbl, dtype=np.float32), labels=lbl, index=idx)
    areas = np.asarray(areas, dtype=np.float32)
    areas = areas[areas > 0]
    if areas.size == 0:
        return 1.0
    eq_d = np.sqrt(4.0 * areas / np.pi)
    return float(np.median(eq_d))


def _disk(radius: float) -> np.ndarray:
    """Binary disk structuring element with given radius in pixels."""
    r = max(1, int(np.ceil(radius)))
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= (r * r)


def boundary_scores(true: np.ndarray, pred: np.ndarray, scales=(0.5, 1.0, 2.0)) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Boundary precision/recall/F1 with a tolerance band.
    The tolerance radius = 0.5 * scale * median_equivalent_diameter."""
    diam = _median_eq_diameter(true)
    otrue = _binary_outline(true)
    opred = _binary_outline(pred)
    precisions, recalls, fscores = [], [], []
    for s in scales:
        se = _disk(radius=0.5 * s * diam)
        tol_true = ndi.binary_dilation(otrue, structure=se)
        tol_pred = ndi.binary_dilation(opred, structure=se)
        tp = np.logical_and(opred, tol_true).sum()
        fp = np.logical_and(opred, ~tol_true).sum()
        fn = np.logical_and(otrue, ~tol_pred).sum()
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        precisions.append(p); recalls.append(r); fscores.append(f)
    return np.array(precisions), np.array(recalls), np.array(fscores)


# --------------------------- Per-image evaluation ---------------------------

def eval_one_pair(
    gt_path: Path,
    pred_path: Path,
    ap_thresholds=(0.5, 0.75, 0.9),
    boundary_scales=(0.5, 1.0, 2.0),
) -> dict:
    """Evaluate a single GT/prediction pair and return a flat dict of metrics."""
    gt = _read_mask_any(gt_path)
    pr = _read_mask_any(pred_path)
    gt, pr = _align_shapes(gt, pr)

    aji_val = aji(gt, pr)
    ap, tp, fp, fn = average_precision(gt, pr, thresholds=ap_thresholds)
    bp, br, bf = boundary_scores(gt, pr, scales=boundary_scales)
    best_idx = int(np.argmax(bf))

    out = {
        "n_true": int(gt.max()),
        "n_pred": int(pr.max()),
        "AJI": float(aji_val),
        **{f"AP@{t:.2f}": float(v) for t, v in zip(ap_thresholds, ap)},
        **{f"TP@{t:.2f}": int(v) for t, v in zip(ap_thresholds, tp)},
        **{f"FP@{t:.2f}": int(v) for t, v in zip(ap_thresholds, fp)},
        **{f"FN@{t:.2f}": int(v) for t, v in zip(ap_thresholds, fn)},
        **{f"BF_P@{s:g}": float(v) for s, v in zip(boundary_scales, bp)},
        **{f"BF_R@{s:g}": float(v) for s, v in zip(boundary_scales, br)},
        **{f"BF_F@{s:g}": float(v) for s, v in zip(boundary_scales, bf)},
        "BF_bestF": float(bf[best_idx]),
        "BF_best_scale": float(list(boundary_scales)[best_idx]),
    }
    return out


# --------------------------- Directory-level evaluation ---------------------------

def evaluate_dir(
    gt_dir: Path,
    pred_dir: Path,
    algo_name: str,
    gt_glob: str = "*.npy",
    pred_glob: str = "*.npy",
    gt_strip: List[str] | None = None,
    pred_strip: List[str] | None = None,
    ap_thresholds=(0.5, 0.75, 0.9),
    boundary_scales=(0.5, 1.0, 2.0),
):
    """Evaluate all paired images for one algorithm directory.
    Returns (per_image_df, summary_df[one row for this algorithm])."""
    gt_strip = gt_strip or ["_cellbodies", "_dapimultimask", "_gt", "_GT"]
    pred_strip = pred_strip or ["_pred_cyto", "_pred_nuc", "_pred", "_refined"]

    pairs = _pair_by_base(gt_dir, pred_dir, gt_glob, pred_glob, gt_strip, pred_strip)

    rows = []
    for base, g, p in pairs:
        res = eval_one_pair(g, p, ap_thresholds=ap_thresholds, boundary_scales=boundary_scales)
        res.update({"base": base, "algorithm": algo_name})
        rows.append(res)

    df = pd.DataFrame(rows)
    if df.empty:
        # Return empty per-image table and a minimal summary row
        return df, pd.DataFrame([{"algorithm": algo_name, "note": "no matched pairs"}])

    per_img = df.sort_values(["algorithm", "base"]).reset_index(drop=True)
    metrics_cols = [c for c in per_img.columns if c not in ("base", "algorithm")]
    summary = per_img.groupby("algorithm")[metrics_cols].mean().reset_index()
    return per_img, summary


def evaluate_all(cfg: EvalConfig):
    """Evaluate multiple algorithms as specified in EvalConfig.
    Returns (per_image_df, summary_df)."""
    all_per = []
    all_sum = []
    for algo, pdir in cfg.pred_dirs.items():
        strips = cfg.pred_strip.get(algo, ["_pred_cyto", "_pred_nuc", "_pred", "_refined"])
        per_img, summ = evaluate_dir(
            gt_dir=cfg.gt_dir,
            pred_dir=pdir,
            algo_name=algo,
            gt_glob=cfg.gt_glob,
            pred_glob=cfg.pred_glob,
            gt_strip=cfg.gt_strip,
            pred_strip=strips,
            ap_thresholds=tuple(cfg.ap_thresholds),
            boundary_scales=tuple(cfg.boundary_scales),
        )
        all_per.append(per_img)
        all_sum.append(summ)

    per_df = pd.concat(all_per, ignore_index=True) if any(len(df) for df in all_per) else pd.DataFrame()
    sum_df = pd.concat(all_sum, ignore_index=True) if all_sum else pd.DataFrame()
    return per_df, sum_df

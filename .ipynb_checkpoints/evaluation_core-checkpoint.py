#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
evaluation_core.py  (StarDist-only, parallel + autosave)

- StarDist matching for AP only (no fallback).
- ProcessPool parallelism at directory level.
- Saves per-image and summary CSVs per algorithm and global ALL tables.
- Correct cell counting via unique labels > 0.

Dependencies: numpy, scipy, pandas, tifffile, stardist
"""

import os
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import tifffile as tiff
from scipy import ndimage as ndi

try:
    from tqdm import tqdm  # optional
except Exception:
    tqdm = None

# Require StarDist
try:
    from stardist.matching import matching as sd_matching
except Exception as e:
    raise ImportError(
        f"[FATAL] StarDist 'matching' unavailable: {e}\n"
        "Install it first: pip install stardist csbdeep"
    )


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
    pred_strip: Dict[str, List[str]] = field(default_factory=dict)
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
    print(f"Now Evaluating {pred_dir}")
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
    if hasattr(m, "_asdict"):
        return m._asdict()
    if hasattr(m, "to_dict"):
        return m.to_dict()
    keys = ["tp", "fp", "fn", "precision", "recall", "f1",
            "n_true", "n_pred", "thresh", "criterion",
            "mean_true_score", "mean_matched_score", "panoptic_quality"]
    d = {}
    for k in keys:
        if hasattr(m, k):
            d[k] = getattr(m, k)
    return d


def average_precision(true: np.ndarray, pred: np.ndarray, thresholds=(0.5, 0.75, 0.9)):
    """AP from StarDist matching at given IoU thresholds."""
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

    # Correct counts: number of unique labels > 0
    n_true = int(np.count_nonzero(np.unique(gt) > 0))
    n_pred = int(np.count_nonzero(np.unique(pr) > 0))

    aji_val = aji(gt, pr)
    ap, tp, fp, fn = average_precision(gt, pr, thresholds=ap_thresholds)
    bp, br, bf = boundary_scores(gt, pr, scales=boundary_scales)
    best_idx = int(np.argmax(bf))
    map_val = float(np.mean(ap))  # mAP across all IoU thresholds

    out = {
        "n_true": n_true,
        "n_pred": n_pred,
        "AJI": float(aji_val),
        "mAP": map_val,
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


# --------------------------- Parallel task helper ---------------------------

def _eval_task(args) -> dict:
    """Top-level picklable task for ProcessPoolExecutor."""
    base, g, p, ap_thresholds, boundary_scales, algo_name = args
    res = eval_one_pair(g, p, ap_thresholds=ap_thresholds, boundary_scales=boundary_scales)
    res.update({"base": base, "algorithm": algo_name})
    return res


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
    max_workers: int | None = None,
    save_dir: Path | None = None,
    run_id: str | None = None,
    verbose: bool = True,
):
    """Evaluate all paired images for one algorithm directory."""
    gt_strip = gt_strip or ["_cellbodies", "_dapimultimask", "_gt", "_GT"]
    pred_strip = pred_strip or ["_pred_cyto", "_pred_nuc", "_pred_marker_only","_pred", "_refined"]

    pairs = _pair_by_base(gt_dir, pred_dir, gt_glob, pred_glob, gt_strip, pred_strip)

    if not pairs:
        per_img = pd.DataFrame()
        summ = pd.DataFrame([{"algorithm": algo_name, "note": "no matched pairs"}])
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            rid = run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
            summ.to_csv(save_dir / f"{algo_name}__{rid}__summary.csv", index=False)
        return per_img, summ

    job_args = [(base, g, p, tuple(ap_thresholds), tuple(boundary_scales), algo_name) for base, g, p in pairs]

    rows: List[dict] = []
    workers = max_workers or max(1, (os.cpu_count() or 1))

    if verbose:
        print(f"[eval] {algo_name}: {len(job_args)} image pairs | workers={workers}")
    t0 = time.perf_counter()

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_eval_task, a) for a in job_args]
        if verbose and tqdm is not None:
            bar = tqdm(total=len(job_args), desc=f"{algo_name}", unit="img", dynamic_ncols=True)
            for f in as_completed(futs):
                rows.append(f.result())
                bar.update(1)
            bar.close()
        else:
            n_done = 0
            for f in as_completed(futs):
                rows.append(f.result())
                n_done += 1
                if verbose and (n_done % 20 == 0 or n_done == len(job_args)):
                    print(f"[eval] {algo_name}: {n_done}/{len(job_args)} done")

    if verbose:
        dt = time.perf_counter() - t0
        print(f"[eval] {algo_name}: finished in {dt:.2f}s")

    df = pd.DataFrame(rows)
    per_img = df.sort_values(["algorithm", "base"]).reset_index(drop=True)

    if per_img.empty:
        summ = pd.DataFrame([{"algorithm": algo_name, "note": "no matched pairs"}])
    else:
        metrics_cols = [c for c in per_img.columns if c not in ("base", "algorithm")]
        summ = per_img.groupby("algorithm")[metrics_cols].mean().reset_index()

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        rid = run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
        per_img.to_csv(save_dir / f"{algo_name}__{rid}__per_image.csv", index=False)
        summ.to_csv(save_dir / f"{algo_name}__{rid}__summary.csv", index=False)

    return per_img, summ


def evaluate_all(
    cfg: EvalConfig,
    max_workers: int | None = None,
    out_dir: Path | None = None,
    run_id: str | None = None,
    verbose: bool = True,
):
    """Evaluate multiple algorithms as specified in EvalConfig."""
    rid = run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    all_per: List[pd.DataFrame] = []
    all_sum: List[pd.DataFrame] = []

    if verbose:
        print(f"[eval] starting ALL: {len(cfg.pred_dirs)} algorithms")

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
            max_workers=max_workers,
            save_dir=out_dir,
            run_id=rid,
            verbose=verbose,
        )
        all_per.append(per_img)
        all_sum.append(summ)

    per_df = pd.concat([df for df in all_per if not df.empty], ignore_index=True) if any(len(df) for df in all_per) else pd.DataFrame()
    sum_df = pd.concat(all_sum, ignore_index=True) if all_sum else pd.DataFrame()

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        per_df.to_csv(out_dir / f"ALL__{rid}__per_image.csv", index=False)
        sum_df.to_csv(out_dir / f"ALL__{rid}__summary.csv", index=False)

    if verbose:
        print("[eval] ALL done")

    return per_df, sum_df

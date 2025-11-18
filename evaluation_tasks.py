#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
evaluation_tasks.py

Thin wrappers around evaluation_core for two benchmarks:
1) Nuclei segmentation: GT (DAPI multi-mask) vs nuclei predictions
2) Cell segmentation:   GT (marker cellbodies) vs cyto-refined or raw predictions

Adds IoU@0.50 per-image, keeps AP (incl. AP@0.50), AJI, boundary metrics,
and cell counts (n_true, n_pred). Optionally writes merged CSVs.
"""

from pathlib import Path
from typing import Dict, Mapping, Tuple, List, Optional

import pandas as pd

from evaluation_core import (
    EvalConfig,
    evaluate_all,
    _read_mask_any,
    _align_shapes,
    _pair_by_base,
)

# Require StarDist
try:
    from stardist.matching import matching as sd_matching
except Exception as e:
    raise ImportError(
        f"[FATAL] StarDist 'matching' unavailable: {e}\n"
        "Install it first: pip install stardist csbdeep"
    )


# ---------------- IoU@0.50 helpers ----------------

def _mean_iou_at_050_from_arrays(gt, pr) -> float:
    """Mean IoU of matched pairs at threshold 0.50. 0.0 if no matches."""
    gt, pr = _align_shapes(gt, pr)
    m = sd_matching(gt, pr, thresh=0.5, criterion="iou")
    if hasattr(m, "mean_matched_score"):
        return float(m.mean_matched_score)
    if hasattr(m, "to_dict"):
        return float(m.to_dict().get("mean_matched_score", 0.0))
    if hasattr(m, "_asdict"):
        return float(m._asdict().get("mean_matched_score", 0.0))
    return 0.0



def _compute_iou50_table(
    gt_dir: Path,
    pred_dir: Path,
    algo_name: str,
    gt_glob: str,
    pred_glob: str,
    gt_strip: List[str],
    pred_strip: List[str],
    *,
    verbose: bool = False,
    debug_limit: Optional[int] = None,
) -> pd.DataFrame:
    """Compute IoU@0.50 per image for one algorithm directory, with debug prints."""
    pairs = _pair_by_base(gt_dir, pred_dir, gt_glob, pred_glob, gt_strip, pred_strip)
    if verbose:
        print(f"[IoU50] algo={algo_name}  pairs={len(pairs)}  gt_glob={gt_glob}  pred_glob={pred_glob}")
        if pairs:
            print(f"[IoU50] sample pair[0]: base={pairs[0][0]}\n"
                  f"        gt={pairs[0][1]}\n        pred={pairs[0][2]}")

    rows = []
    for idx, (base, g, p) in enumerate(pairs):
        if debug_limit is not None and idx >= debug_limit:
            if verbose: print(f"[IoU50] stopping early at idx={idx} due to debug_limit={debug_limit}")
            break
        gt = _read_mask_any(g)
        pr = _read_mask_any(p)
        if verbose:
            print(f"[IoU50] {algo_name} base={base}  gt.shape={getattr(gt,'shape',None)}  pr.shape={getattr(pr,'shape',None)}")
        val = _mean_iou_at_050_from_arrays(gt, pr)
        if verbose:
            print(f"[IoU50] {algo_name} base={base}  IoU@0.50={val:.4f}")
        rows.append({"base": base, "algorithm": algo_name, "IoU@0.50": float(val)})
    return pd.DataFrame(rows)


def _to_paths(d: Mapping[str, str | Path]) -> Dict[str, Path]:
    """Normalize a mapping of {name: path-like} into {name: Path(path)}."""
    return {k: (v if isinstance(v, Path) else Path(v)) for k, v in d.items()}


# ---------------- Public APIs ----------------

def evaluate_nuclei_benchmark(
    dataset_dir: str | Path,
    nuclei_pred_dirs: Mapping[str, str | Path],
    *,
    ap_thresholds: Tuple[float, ...] = (0.5, 0.75, 0.9),
    boundary_scales: Tuple[float, ...] = (0.5, 1.0, 2.0),
    pred_glob: str = "*.npy",
    out_dir: str | Path | None = None,
    run_id: str | None = None,
    max_workers: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate nuclei task and append IoU@0.50 to per-image results.
    评估核分割任务并将IoU@0.50添加到每张图像的结果中。

    GT:   <dataset_dir>/*_dapimultimask.npy
    PRED: folders with *.npy label masks
    """
    cfg = EvalConfig(
        gt_dir=Path(dataset_dir),
        pred_dirs=_to_paths(nuclei_pred_dirs),
        gt_glob="*_dapimultimask.npy",
        pred_glob=pred_glob,
        gt_strip=["_dapimultimask"],
        
        # ========== 修复部分 | FIXED SECTION ==========
        # 旧代码 | OLD CODE:
        # pred_strip={
        #     algo: ["_pred_nuc", "_nuc","_pred_cell", "_pred_marker_only","_pred", ...]
        #     for algo in nuclei_pred_dirs
        # },
        
        # 新代码 | NEW CODE:
        pred_strip={
            # 大多数算法使用 _pred_nuclei | Most use _pred_nuclei
            "CellposeSAM": ["_pred_nuclei", "_nuclei", "_pred", "_nuc"],
            "StarDist": ["_pred_nuclei", "_nuclei", "_pred", "_nuc"],
            "MESMER": ["_pred_nuclei", "_nuclei", "_pred", "_nuc"],
            "Watershed": ["_pred_nuclei", "_nuclei", "_pred", "_nuc"],
            "Omnipose": ["_pred_nuclei", "_nuclei", "_pred", "_nuc"],
            "LACSS": ["_pred_nuclei", "_nuclei", "_pred", "_nuc"],
            
            # 少数算法使用简单命名 | Some use simpler naming
            "CellSAM": ["_pred_nuc", "_nuc", "_pred", "_nuclei"],
            "SplineDist": ["_pred_nuc", "_nuc", "_pred", "_nuclei"],
            "MicroSAM": ["_pred_nuc", "_nuc", "_pred", "_nuclei"],
        },
        # ========== 修复结束 | END FIX ==========
        
        ap_thresholds=ap_thresholds,
        boundary_scales=boundary_scales,
    )

    per_img, summ = evaluate_all(
        cfg, max_workers=max_workers, out_dir=Path(out_dir) if out_dir else None,
        run_id=run_id, verbose=verbose
    )
    if per_img.empty:
        return per_img, summ

    # IoU@0.50 per-image
    add_tables = []
    for algo, pdir in cfg.pred_dirs.items():
        strips = cfg.pred_strip.get(algo, ["_pred_nuclei", "_nuclei", "_pred", "_nuc"])
        add_tables.append(
            _compute_iou50_table(cfg.gt_dir, pdir, algo, cfg.gt_glob, cfg.pred_glob, 
                                cfg.gt_strip, strips)
        )
    iou_df = pd.concat(add_tables, ignore_index=True) if add_tables else pd.DataFrame()
    if not iou_df.empty:
        per_img = per_img.merge(iou_df, on=["base", "algorithm"], how="left")
        # per-algorithm means
        if "AP@0.50" in per_img.columns:
            ap50_mean = per_img.groupby("algorithm")["AP@0.50"].mean().reset_index()
            summ = summ.merge(ap50_mean, on="algorithm", how="left", suffixes=("", "_mean"))
        iou50_mean = per_img.groupby("algorithm")["IoU@0.50"].mean().reset_index()
        summ = summ.merge(iou50_mean, on="algorithm", how="left", suffixes=("", "_mean"))

    # Save merged tables if requested
    if out_dir:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        rid = run_id if run_id else "latest"
        per_img.to_csv(out_dir / f"ALL__{rid}__per_image.csv", index=False)
        summ.to_csv(out_dir / f"ALL__{rid}__summary.csv", index=False)

    return per_img, summ


def evaluate_cell_benchmark(
    dataset_dir: str | Path,
    cyto_pred_dirs: Mapping[str, str | Path],
    *,
    ap_thresholds: Tuple[float, ...] = (0.5, 0.75, 0.9),
    boundary_scales: Tuple[float, ...] = (0.5, 1.0, 2.0),
    pred_glob: str = "*.npy",
    out_dir: str | Path | None = None,
    run_id: str | None = None,
    max_workers: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate cytoplasm/cell task and append IoU@0.50 to per-image results.

    GT:   <dataset_dir>/*_cellbodies.npy
    PRED: cyto-refined or raw folders with *.npy label masks
    """
    cfg = EvalConfig(
        gt_dir=Path(dataset_dir),
        pred_dirs=_to_paths(cyto_pred_dirs),
        gt_glob="*_cellbodies.npy",
        pred_glob=pred_glob,
        gt_strip=["_cellbodies"],
        pred_strip={
            algo: ["_pred_cyto", "_cyto", "_pred_marker_only","_pred_cell","_pred", "_prediction", "_refined", "_filter", "_filtered", "_cyto_filter"]
            for algo in cyto_pred_dirs
        },
        ap_thresholds=ap_thresholds,
        boundary_scales=boundary_scales,
    )

    per_img, summ = evaluate_all(
        cfg, max_workers=max_workers, out_dir=Path(out_dir) if out_dir else None,
        run_id=run_id, verbose=verbose
    )
    if per_img.empty:
        return per_img, summ

    # IoU@0.50 per-image
    add_tables = []
    for algo, pdir in cfg.pred_dirs.items():
        strips = cfg.pred_strip.get(algo, ["_pred_cyto","_pred_marker_only", "_pred_cell","_pred","_cyto", "_cell", "_prediction", "_refined", "_filter", "_filtered", "_cyto_filter"])
        add_tables.append(
            _compute_iou50_table(cfg.gt_dir, pdir, algo, cfg.gt_glob, cfg.pred_glob, cfg.gt_strip, strips)
        )
    iou_df = pd.concat(add_tables, ignore_index=True) if add_tables else pd.DataFrame()
    if not iou_df.empty:
        per_img = per_img.merge(iou_df, on=["base", "algorithm"], how="left")
        # per-algorithm means
        if "AP@0.50" in per_img.columns:
            ap50_mean = per_img.groupby("algorithm")["AP@0.50"].mean().reset_index()
            summ = summ.merge(ap50_mean, on="algorithm", how="left", suffixes=("", "_mean"))
        iou50_mean = per_img.groupby("algorithm")["IoU@0.50"].mean().reset_index()
        summ = summ.merge(iou50_mean, on="algorithm", how="left", suffixes=("", "_mean"))

    # Save merged tables if requested
    if out_dir:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        rid = run_id if run_id else "latest"
        per_img.to_csv(out_dir / f"ALL__{rid}__per_image.csv", index=False)
        summ.to_csv(out_dir / f"ALL__{rid}__summary.csv", index=False)

    return per_img, summ

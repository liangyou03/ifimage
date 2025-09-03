#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluation_tasks.py

Thin convenience wrappers around evaluation_core for two benchmarks:
1) Nuclei segmentation: GT (DAPI multi-mask scribble) vs algorithm predictions
2) Cell segmentation:   GT (marker cellbodies scribble) vs cyto-filter predictions

You pass multiple algorithm folders; the function pairs files by basename and
returns (per_image_df, summary_df) ready for visualization in a notebook.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Mapping, Tuple

import pandas as pd

from evaluation_core import EvalConfig, evaluate_all


def _to_paths(d: Mapping[str, str | Path]) -> Dict[str, Path]:
    """Normalize a mapping of {name: path-like} into {name: Path(path)}."""
    return {k: (v if isinstance(v, Path) else Path(v)) for k, v in d.items()}


def evaluate_nuclei_benchmark(
    dataset_dir: str | Path,
    nuclei_pred_dirs: Mapping[str, str | Path],
    *,
    ap_thresholds=(0.5, 0.75, 0.9),
    boundary_scales=(0.5, 1.0, 2.0),
    pred_glob="*.npy",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate the nuclei task.

    GT:
        00_dataset / *_dapimultimask.npy   (DAPI mask manual scribble)
    Predictions:
        multiple algorithm folders (each contains *.npy label masks)

    Returns:
        per_image_df, summary_df
    """
    cfg = EvalConfig(
        gt_dir=Path(dataset_dir),
        pred_dirs=_to_paths(nuclei_pred_dirs),
        gt_glob="*_dapimultimask.npy",
        pred_glob=pred_glob,
        gt_strip=["_dapimultimask"],
        # Default suffixes to strip from prediction stems to recover the shared base.
        # Override per algorithm in your notebook if your naming differs.
        pred_strip={
            algo: ["_pred_nuc", "_nuc", "_nuclei", "_prediction", "_refined", "_filter", "_filtered"]
            for algo in nuclei_pred_dirs
        },
        ap_thresholds=ap_thresholds,
        boundary_scales=boundary_scales,
    )
    return evaluate_all(cfg)


def evaluate_cell_benchmark(
    dataset_dir: str | Path,
    cyto_pred_dirs: Mapping[str, str | Path],
    *,
    ap_thresholds=(0.5, 0.75, 0.9),
    boundary_scales=(0.5, 1.0, 2.0),
    pred_glob="*.npy",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate the cell (cytoplasm) task.

    GT:
        00_dataset / *_cellbodies.npy      (marker mask manual scribble)
    Predictions:
        multiple *cyto-filtered* folders (refined masks), one per algorithm

    Returns:
        per_image_df, summary_df
    """
    cfg = EvalConfig(
        gt_dir=Path(dataset_dir),
        pred_dirs=_to_paths(cyto_pred_dirs),
        gt_glob="*_cellbodies.npy",
        pred_glob=pred_glob,
        gt_strip=["_cellbodies"],
        pred_strip={
            # Common suffixes for cyto predictions (raw or refined).
            # Add/remove items in your notebook if your filenames differ.
            algo: ["_pred_cyto", "_cyto", "_cell", "_prediction",
                   "_refined", "_filter", "_filtered", "_cyto_filter"]
            for algo in cyto_pred_dirs
        },
        ap_thresholds=ap_thresholds,
        boundary_scales=boundary_scales,
    )
    return evaluate_all(cfg)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inspection_panels_v9.py
- 2x2 panels:
  [1] Cell overlap (prediction vs GT)
  [2] DAPI + nuclei prediction
  [3] Nuclei overlap (prediction vs GT)
  [4] DAPI + nuclei GT
- Overlap colors: yellow=overlap, magenta=prediction only, cyan=GT only
- Metrics shown: mAP, AP@0.50, IoU@0.50
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import tifffile as tiff
from evaluation_core import eval_one_pair, iou_matrix

# ---------------- CONFIG ----------------
DATASET_DIR   = Path("00_dataset")
CYTO_PRED_DIR = Path("01_cellpose_benchmark/cyto_prediction_refined")
NUC_PRED_DIR  = Path("01_cellpose_benchmark/nuclei_prediction")
OUT_DIR       = Path("inspection_panels_cellpose")

CELL_GT_SUFFIX = "_cellbodies.npy"
NUC_GT_SUFFIX  = "_dapimultimask.npy"
DAPI_SUFFIXES  = (".tiff", ".tif")

CYTO_PRED_SUFFIXES = ("_pred_cyto.npy", ".npy")
NUC_PRED_SUFFIXES  = ("_pred_nuc.npy",  ".npy")

THR = np.round(np.arange(0.50, 0.96, 0.05), 2)
BOUNDARY_SCALES = (1.0, 2.0)

FIGSIZE = (14, 12)
DPI = 300
SUPTITLE_FZ = 20
PANEL_TITLE_FZ = 16
CAPTION_FZ = 14
LEGEND_FZ = 12
# ----------------------------------------

OUT_DIR.mkdir(parents=True, exist_ok=True)

def _overlap_rgb(pred_mask: np.ndarray, gt_mask: np.ndarray) -> np.ndarray:
    """RGB: yellow=overlap, magenta=pred only, cyan=GT only."""
    pred = pred_mask > 0
    gt   = gt_mask > 0
    H, W = pred.shape
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    rgb[pred & gt]  = (1,1,0)
    rgb[pred & ~gt] = (1,0,1)
    rgb[~pred & gt] = (0,1,1)
    return rgb

def _mean_iou_at(thr: float, true: np.ndarray, pred: np.ndarray) -> float:
    from scipy.optimize import linear_sum_assignment
    M = iou_matrix(true, pred)[1:,1:]
    if M.size == 0: return 0.0
    n_true, n_pred = M.shape
    cost = -(M>=thr).astype(float) - M/(2.0*max(min(n_true,n_pred),1))
    ti, pi = linear_sum_assignment(cost)
    vals = M[ti,pi][M[ti,pi]>=thr]
    return float(vals.mean()) if vals.size else 0.0

def _safe_minmax(img: np.ndarray):
    vmin,vmax = np.percentile(img,2), np.percentile(img,98)
    return (vmin,vmax) if vmax>vmin else (img.min(),img.max())

def _load_tif(base: str) -> np.ndarray|None:
    for suf in DAPI_SUFFIXES:
        p = DATASET_DIR/f"{base}{suf}"
        if p.exists():
            arr = tiff.imread(str(p))
            arr = np.squeeze(arr)
            return arr[...,0] if arr.ndim>2 else arr
    return None

def _find_pred(base: str, pred_dir: Path, suffixes: tuple[str, ...]) -> Path|None:
    for suf in suffixes:
        p = pred_dir/f"{base}{suf}"
        if p.exists(): return p
    hits = sorted(pred_dir.glob(f"{base}*.npy"))
    return hits[0] if hits else None

# iterate over nuclei GT files
for gt_nuc_path in sorted(DATASET_DIR.glob(f"*{NUC_GT_SUFFIX}")):
    base = gt_nuc_path.stem.replace(NUC_GT_SUFFIX.replace(".npy",""), "")
    gt_cell = np.load(DATASET_DIR/f"{base}{CELL_GT_SUFFIX}") if (DATASET_DIR/f"{base}{CELL_GT_SUFFIX}").exists() else None
    gt_nuc  = np.load(gt_nuc_path)

    pred_cyto_path = _find_pred(base, CYTO_PRED_DIR, CYTO_PRED_SUFFIXES)
    pred_nuc_path  = _find_pred(base, NUC_PRED_DIR, NUC_PRED_SUFFIXES)
    pred_cyto = np.load(pred_cyto_path) if pred_cyto_path else None
    pred_nuc  = np.load(pred_nuc_path)  if pred_nuc_path else None
    dapi = _load_tif(base)

    # metrics
    cell_metrics, nuc_metrics = {},{}
    if pred_cyto is not None and gt_cell is not None:
        res = eval_one_pair(DATASET_DIR/f"{base}{CELL_GT_SUFFIX}", pred_cyto_path,
                            ap_thresholds=tuple(THR), boundary_scales=BOUNDARY_SCALES)
        ap_vals = [res[f"AP@{t:.2f}"] for t in THR]
        cell_metrics = {"mAP":np.mean(ap_vals),"AP50":res["AP@0.50"],"IoU50":_mean_iou_at(0.5,gt_cell,pred_cyto)}
    if pred_nuc is not None:
        res = eval_one_pair(gt_nuc_path, pred_nuc_path,
                            ap_thresholds=tuple(THR), boundary_scales=BOUNDARY_SCALES)
        ap_vals = [res[f"AP@{t:.2f}"] for t in THR]
        nuc_metrics = {"mAP":np.mean(ap_vals),"AP50":res["AP@0.50"],"IoU50":_mean_iou_at(0.5,gt_nuc,pred_nuc)}

    # plot
    fig,axes = plt.subplots(2,2,figsize=FIGSIZE,dpi=DPI)
    (ax1,ax2),(ax3,ax4)=axes

    # 1) Cell overlap
    ax1.set_title("Cell (prediction vs GT) — overlap",fontsize=PANEL_TITLE_FZ)
    if gt_cell is not None and pred_cyto is not None:
        ax1.imshow(_overlap_rgb(pred_cyto,gt_cell),interpolation="nearest")
    elif gt_cell is not None:
        ax1.imshow(_overlap_rgb(np.zeros_like(gt_cell),gt_cell),interpolation="nearest")
    ax1.axis("off")
    ax1.legend([Patch(color="yellow"),Patch(color="magenta"),Patch(color="cyan")],
               ["Overlap","Prediction only","GT only"],loc="lower right",fontsize=LEGEND_FZ)

    # 2) DAPI + nuclei prediction
    ax2.set_title("DAPI + nuclei prediction",fontsize=PANEL_TITLE_FZ)
    if dapi is not None:
        vmin,vmax=_safe_minmax(dapi)
        ax2.imshow(dapi,cmap="gray",vmin=vmin,vmax=vmax,interpolation="nearest")
    if pred_nuc is not None:
        pred_bin = pred_nuc > 0
        H, W = pred_bin.shape
        rgb = np.zeros((H, W, 3), dtype=np.float32)
        rgb[pred_bin] = (1.0, 0.84, 0.0)   # gold
        ax2.imshow(rgb, alpha=0.6, interpolation="nearest")

    ax2.axis("off")

    # 3) Nuclei overlap
    ax3.set_title("Nuclei (prediction vs GT) — overlap",fontsize=PANEL_TITLE_FZ)
    if pred_nuc is not None:
        ax3.imshow(_overlap_rgb(pred_nuc,gt_nuc),interpolation="nearest")
    else:
        ax3.imshow(_overlap_rgb(np.zeros_like(gt_nuc),gt_nuc),interpolation="nearest")
    ax3.axis("off")
    ax3.legend([Patch(color="yellow"),Patch(color="magenta"),Patch(color="cyan")],
               ["Overlap","Prediction only","GT only"],loc="lower right",fontsize=LEGEND_FZ)

    # 4) DAPI + nuclei GT
    ax4.set_title("DAPI + nuclei GT",fontsize=PANEL_TITLE_FZ)
    if dapi is not None:
        vmin,vmax=_safe_minmax(dapi)
        ax4.imshow(dapi,cmap="gray",vmin=vmin,vmax=vmax,interpolation="nearest")
    ax4.imshow(_overlap_rgb(np.zeros_like(gt_nuc),gt_nuc),alpha=0.6,interpolation="nearest")
    ax4.axis("off")

    # caption
    lines=[]
    if cell_metrics: lines.append(f"[CELL] mAP={cell_metrics['mAP']:.3f} | AP@0.50={cell_metrics['AP50']:.3f} | IoU@0.50={cell_metrics['IoU50']:.3f}")
    else: lines.append("[CELL] missing")
    if nuc_metrics: lines.append(f"[NUCLEI] mAP={nuc_metrics['mAP']:.3f} | AP@0.50={nuc_metrics['AP50']:.3f} | IoU@0.50={nuc_metrics['IoU50']:.3f}")
    else: lines.append("[NUCLEI] missing")
    fig.suptitle(f"{base}",fontsize=SUPTITLE_FZ)
    fig.text(0.01,0.02,"\n".join(lines),fontsize=CAPTION_FZ,va="bottom",ha="left")

    plt.subplots_adjust(bottom=0.12,top=0.94,wspace=0.05,hspace=0.12)
    fig.savefig(OUT_DIR/f"{base}.png",bbox_inches="tight")
    plt.close(fig)

print(f"Done. Panels saved to: {OUT_DIR.resolve()}")

# --- Build a folder of high-res inspection panels for every sample ---
# Panels: (1) Cellpose (cyto) vs cellbodies scribble, (2) DAPI+prediction, (3) DAPI+GT
# Metrics shown (per image): AP@{0.50..0.95}, mAP, AJI, Boundary-F(best), mean IoU@0.50

from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff

# Import your evaluation core (StarDist-backed AP if available)
from evaluation_core import eval_one_pair, iou_matrix

# ------------------ CONFIG (edit as needed) ------------------
DATASET_DIR = Path("00_dataset")
CYTO_PRED_DIR = Path("01_cellpose_benchmark/cyto_prediction_refined")  # cellpose (cyto) refined masks
NUC_PRED_DIR  = Path("01_cellpose_benchmark/nuclei_prediction")        # cellpose nuclei masks
OUT_DIR       = Path("inspection_panels_cellpose")                     # output folder for PNGs

# Filenaming assumptions
CELL_GT_SUFFIX = "_cellbodies.npy"         # manual scribble (marker-based)
NUC_GT_SUFFIX  = "_dapimultimask.npy"      # manual scribble (DAPI-based)
MARKER_SUFFIXES = ("_marker.tiff", "_marker.tif")
DAPI_SUFFIXES   = (".tiff", ".tif")        # e.g., gfap_XXXX.tiff as raw DAPI
CYTO_PRED_SUFFIXES = ("_pred_cyto.npy", ".npy")   # search order inside CYTO_PRED_DIR
NUC_PRED_SUFFIXES  = ("_pred_nuc.npy", ".npy")    # search order inside NUC_PRED_DIR

# IoU thresholds for COCO-style mAP
THR = np.round(np.arange(0.50, 0.96, 0.05), 2)
AP_COLS = [f"AP@{t:.2f}" for t in THR]
BOUNDARY_SCALES = (1.0, 2.0)

# Known cell-type tokens to detect in filename (customize if needed)
KNOWN_TYPES = ["OLIG2", "NEUN", "IBA1", "GFAP", "PECAM"]

# Figure size & dpi
FIGSIZE = (13, 5.5)
DPI = 300
# ------------------------------------------------------------

OUT_DIR.mkdir(parents=True, exist_ok=True)

def _binary_outline(lbl: np.ndarray) -> np.ndarray:
    """Binary outline from a labeled mask (outer boundary)."""
    from scipy import ndimage as ndi
    b = lbl > 0
    er = ndi.binary_erosion(b, structure=np.ones((3,3), bool), border_value=0)
    return b ^ er

def _load_tif_any(base: str, suffixes: tuple[str, ...], root: Path) -> np.ndarray | None:
    for suf in suffixes:
        p = root / f"{base}{suf}"
        if p.exists():
            arr = tiff.imread(str(p))
            arr = np.squeeze(arr)
            if arr.ndim > 2:
                arr = arr[..., 0]
            return arr
    return None

def _find_pred(base: str, pred_dir: Path, suffixes: tuple[str, ...]) -> Path | None:
    # Try the strict suffixes in order, then fall back to any .npy that startswith base
    for suf in suffixes:
        p = pred_dir / f"{base}{suf}"
        if p.exists():
            return p
    hits = sorted(pred_dir.glob(f"{base}*.npy"))
    return hits[0] if hits else None

def _mean_iou_at(threshold: float, true: np.ndarray, pred: np.ndarray) -> float:
    """Mean IoU of matched pairs at a given threshold (Hungarian matching)."""
    from scipy.optimize import linear_sum_assignment
    M = iou_matrix(true, pred)[1:, 1:]
    if M.size == 0:
        return 0.0
    n_true, n_pred = M.shape
    n_min = min(n_true, n_pred)
    cost = -(M >= threshold).astype(float) - M / (2.0 * max(n_min, 1))
    ti, pi = linear_sum_assignment(cost)
    miou = M[ti, pi]
    sel = miou >= threshold
    return float(miou[sel].mean()) if np.any(sel) else 0.0

def _cell_type_from_base(base: str) -> str:
    for t in KNOWN_TYPES:
        if t.lower() in base.lower():
            return t.upper()
    return "UNKNOWN"

def _safe_minmax(img: np.ndarray) -> tuple[float, float]:
    vmin, vmax = float(np.percentile(img, 2)), float(np.percentile(img, 98))
    if vmax <= vmin:
        vmax = float(img.max())
        vmin = float(img.min())
    return vmin, vmax

# Identify all bases from the GT nuclei scribbles (this guarantees 40 images)
gt_nuc_files = sorted(DATASET_DIR.glob(f"*{NUC_GT_SUFFIX}"))
bases = [p.stem.replace(NUC_GT_SUFFIX.replace(".npy",""), "") for p in gt_nuc_files]

for gt_nuc_path in gt_nuc_files:
    base = gt_nuc_path.stem.replace(NUC_GT_SUFFIX.replace(".npy",""), "")
    # Paths for this sample
    gt_cell_path = DATASET_DIR / f"{base}{CELL_GT_SUFFIX}"
    gt_nuc_path  = DATASET_DIR / f"{base}{NUC_GT_SUFFIX}"
    dapi = _load_tif_any(base, DAPI_SUFFIXES, DATASET_DIR)
    marker = _load_tif_any(base, MARKER_SUFFIXES, DATASET_DIR)
    pred_cyto_path = _find_pred(base, CYTO_PRED_DIR, CYTO_PRED_SUFFIXES)
    pred_nuc_path  = _find_pred(base, NUC_PRED_DIR,  NUC_PRED_SUFFIXES)

    # Load arrays (skip gracefully if something essential is missing)
    try:
        gt_cell = np.load(gt_cell_path) if gt_cell_path.exists() else None
        gt_nuc  = np.load(gt_nuc_path)
        pred_cyto = np.load(pred_cyto_path) if pred_cyto_path else None
        pred_nuc  = np.load(pred_nuc_path)  if pred_nuc_path  else None
    except Exception as e:
        print(f"[SKIP] {base}: failed to load arrays ({e})")
        continue

    # ----- Compute metrics -----
    # Cell (cyto) metrics are computed only if both pred_cyto and gt_cell exist
    cell_metrics = {}
    if pred_cyto is not None and gt_cell is not None:
        res_cell = eval_one_pair(gt_cell_path, pred_cyto_path,
                                 ap_thresholds=tuple(THR), boundary_scales=BOUNDARY_SCALES)
        # mAP over 0.50..0.95
        ap_values = [res_cell[f"AP@{t:.2f}"] for t in THR]
        mAP_cell = float(np.mean(ap_values))
        miou50_cell = _mean_iou_at(0.50, gt_cell, pred_cyto)
        cell_metrics = {
            "AJI": res_cell["AJI"],
            "mAP": mAP_cell,
            "AP@0.50": res_cell["AP@0.50"],
            "AP@0.75": res_cell.get("AP@0.75", np.nan),
            "BF_bestF": res_cell["BF_bestF"],
            "meanIoU@0.50": miou50_cell,
        }

    # Nuclei metrics (pred_nuc vs gt_nuc)
    nuc_metrics = {}
    if pred_nuc is not None:
        res_nuc = eval_one_pair(gt_nuc_path, pred_nuc_path,
                                ap_thresholds=tuple(THR), boundary_scales=BOUNDARY_SCALES)
        ap_values = [res_nuc[f"AP@{t:.2f}"] for t in THR]
        mAP_nuc = float(np.mean(ap_values))
        miou50_nuc = _mean_iou_at(0.50, gt_nuc, pred_nuc)
        nuc_metrics = {
            "AJI": res_nuc["AJI"],
            "mAP": mAP_nuc,
            "AP@0.50": res_nuc["AP@0.50"],
            "AP@0.75": res_nuc.get("AP@0.75", np.nan),
            "BF_bestF": res_nuc["BF_bestF"],
            "meanIoU@0.50": miou50_nuc,
        }

    # ----- Build the figure -----
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE, dpi=DPI)
    ax1, ax2, ax3 = axes

    # (1) Cellpose (cyto) vs manual scribble
    ax1.set_title("Cell (cellpose vs manual scribble)")
    if marker is not None:
        vmin, vmax = _safe_minmax(marker)
        ax1.imshow(marker, cmap="gray", vmin=vmin, vmax=vmax)
    else:
        ax1.imshow(np.zeros_like(gt_nuc), cmap="gray")

    if gt_cell is not None:
        ax1.contour(_binary_outline(gt_cell), levels=[0.5], colors=["lime"], linewidths=1.2, alpha=0.9)
    if pred_cyto is not None:
        ax1.contour(_binary_outline(pred_cyto), levels=[0.5], colors=["orangered"], linewidths=1.0, alpha=0.9)
    ax1.axis("off")

    # (2) DAPI + prediction (nuclei)
    ax2.set_title("DAPI + nuclei prediction")
    if dapi is not None:
        vmin, vmax = _safe_minmax(dapi)
        ax2.imshow(dapi, cmap="gray", vmin=vmin, vmax=vmax)
    else:
        ax2.imshow(np.zeros_like(gt_nuc), cmap="gray")

    if pred_nuc is not None:
        ax2.contour(_binary_outline(pred_nuc), levels=[0.5], colors=["gold"], linewidths=1.0, alpha=0.9)
    ax2.axis("off")

    # (3) DAPI + ground truth (nuclei)
    ax3.set_title("DAPI + nuclei GT")
    if dapi is not None:
        vmin, vmax = _safe_minmax(dapi)
        ax3.imshow(dapi, cmap="gray", vmin=vmin, vmax=vmax)
    else:
        ax3.imshow(np.zeros_like(gt_nuc), cmap="gray")

    ax3.contour(_binary_outline(gt_nuc), levels=[0.5], colors=["cyan"], linewidths=1.0, alpha=0.9)
    ax3.axis("off")

    # Title with cell type and code
    cell_type = _cell_type_from_base(base)
    fig.suptitle(f"{cell_type} — {base}", y=0.995, fontsize=12)

    # Metrics block (English)
    lines = []
    if cell_metrics:
        lines.append("[CELL (cyto vs scribble)]")
        lines.append(f"AJI={cell_metrics['AJI']:.3f} | mAP(0.50–0.95)={cell_metrics['mAP']:.3f} | "
                     f"AP@0.50={cell_metrics['AP@0.50']:.3f} | AP@0.75={cell_metrics['AP@0.75']:.3f} | "
                     f"BF_bestF={cell_metrics['BF_bestF']:.3f} | meanIoU@0.50={cell_metrics['meanIoU@0.50']:.3f}")
    else:
        lines.append("[CELL] prediction or GT missing")

    if nuc_metrics:
        lines.append("[NUCLEI (pred vs GT)]")
        lines.append(f"AJI={nuc_metrics['AJI']:.3f} | mAP(0.50–0.95)={nuc_metrics['mAP']:.3f} | "
                     f"AP@0.50={nuc_metrics['AP@0.50']:.3f} | AP@0.75={nuc_metrics['AP@0.75']:.3f} | "
                     f"BF_bestF={nuc_metrics['BF_bestF']:.3f} | meanIoU@0.50={nuc_metrics['meanIoU@0.50']:.3f}")
    else:
        lines.append("[NUCLEI] prediction or GT missing")

    fig.text(0.01, 0.01, "\n".join(lines), fontsize=8, va="bottom", ha="left")

    # Save
    out_path = OUT_DIR / f"{base}.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=DPI)
    plt.close(fig)

print(f"Done. Panels saved to: {OUT_DIR.resolve()}")

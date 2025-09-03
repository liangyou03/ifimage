
#!/usr/bin/env python3
"""
evaluate_instances.py
=====================
Metrics-only evaluator for instance segmentation (cells/nuclei).

What it does
------------
- Reads a *ground-truth* labels folder and one or more *prediction* folders.
- Matches files by base name (with configurable suffix stripping).
- Computes per-image metrics and per-method aggregates.
- Writes results to CSV/JSON; prints a compact summary.

It does *not* run any model and never changes your images (no downsampling, etc.).

Supported label formats
-----------------------
- .npy (numpy arrays with integer instance labels)
- .tif / .tiff / .png (integer instance labels)

Matching files
--------------
We match by *base name* after stripping optional suffixes.
Example: GT file "sample_001_gt.npy" with --strip-gt-suffixes "_gt"
matches prediction "sample_001_pred_cell.npy" with --strip-pred-suffixes "_pred_cell".

Usage examples
--------------
# One method
python evaluate_instances.py \
  --gt /path/to/gt_cell \
  --pred cellsam:/path/to/preds/cellsam_cell \
  --strip-gt-suffixes _gt \
  --strip-pred-suffixes _pred_cell \
  --out outputs/metrics_cell

# Multiple methods at once
python evaluate_instances.py \
  --gt /path/to/gt_cell \
  --pred cellsam:/path/to/preds/cellsam_cell \
  --pred cellpose:/path/to/preds/cellpose_cell \
  --strip-gt-suffixes _gt \
  --strip-pred-suffixes _pred_cell _pred \
  --out outputs/metrics_cell

Command-line arguments
----------------------
--gt DIR                        Ground-truth directory (required)
--pred NAME:DIR [NAME:DIR ...]  One or more prediction folders with method names
--exts .npy .tif .tiff .png     Extensions to look for (default covers the common ones)
--strip-gt-suffixes ...         Suffixes to remove from GT base names when matching
--strip-pred-suffixes ...       Suffixes to remove from prediction base names when matching
--iou-thr 0.5                   IoU threshold for instance matching (default 0.5)
--boundary-tol 2                Pixel tolerance for boundary F1 (default 2)
--out DIR                       Output directory for metrics (default: outputs/metrics_eval)

Outputs
-------
<out>/per_image_metrics.csv        (one row per image per method)
<out>/per_image_metrics.jsonl      (same as CSV, JSON lines)
<out>/aggregate_metrics.csv        (one row per method with macro/micro stats)
<out>/missing_pairs.txt            (files skipped due to no match)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

try:
    import tifffile as tiff
except Exception:
    tiff = None

try:
    from skimage import segmentation, morphology
except Exception:
    segmentation = None
    morphology = None


# --------------------------- IO helpers ---------------------------

SUPPORTED_EXTS = [".npy", ".tif", ".tiff", ".png"]

def load_label_any(p: Path) -> np.ndarray:
    if not p.exists():
        raise FileNotFoundError(p)
    if p.suffix.lower() == ".npy":
        return np.load(p)
    if p.suffix.lower() in (".tif", ".tiff", ".png"):
        if tiff is None:
            raise RuntimeError("Reading TIFF/PNG requires 'tifffile'. Install via conda/pip.")
        arr = tiff.imread(str(p))
        return arr
    raise ValueError(f"Unsupported extension: {p.suffix}")


def list_label_files(root: Path, allowed_exts: Iterable[str]) -> List[Path]:
    files: List[Path] = []
    for ext in allowed_exts:
        files.extend(sorted(root.glob(f"*{ext}")))
    return files


def strip_suffix(name: str, suffixes: Iterable[str]) -> str:
    for s in sorted(suffixes, key=len, reverse=True):
        if s and name.endswith(s):
            return name[: -len(s)]
    return name


def base_from_path(p: Path) -> str:
    return p.stem  # filename without extension


# --------------------------- Metrics ---------------------------

def to_binary(mask: np.ndarray) -> np.ndarray:
    return (mask.astype(np.int64) > 0).astype(np.uint8)


def iou_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    inter = np.logical_and(y_true, y_pred).sum(dtype=np.int64)
    uni = np.logical_or(y_true, y_pred).sum(dtype=np.int64)
    return float(inter / uni) if uni > 0 else (1.0 if inter == 0 else 0.0)


def dice_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    inter = np.logical_and(y_true, y_pred).sum(dtype=np.int64)
    s = y_true.sum(dtype=np.int64) + y_pred.sum(dtype=np.int64)
    return float(2.0 * inter / s) if s > 0 else (1.0 if inter == 0 else 0.0)


def compute_iou_matrix(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """IoU for instance labels (>0). Returns [n_gt, n_pred] float32."""
    gt_ids = np.unique(gt)
    gt_ids = gt_ids[gt_ids > 0]
    pr_ids = np.unique(pred)
    pr_ids = pr_ids[pr_ids > 0]

    n_gt, n_pr = len(gt_ids), len(pr_ids)
    if n_gt == 0 and n_pr == 0:
        return np.zeros((0, 0), dtype=np.float32)
    if n_gt == 0 or n_pr == 0:
        return np.zeros((n_gt, n_pr), dtype=np.float32)

    gt_map = {int(lbl): i for i, lbl in enumerate(gt_ids)}
    pr_map = {int(lbl): j for j, lbl in enumerate(pr_ids)}

    inter = np.zeros((n_gt, n_pr), dtype=np.int64)

    gt_f = gt.ravel()
    pr_f = pred.ravel()
    both = np.where((gt_f > 0) & (pr_f > 0))[0]
    gt_b = gt_f[both]
    pr_b = pr_f[both]
    for g, p in zip(gt_b, pr_b):
        i = gt_map.get(int(g))
        j = pr_map.get(int(p))
        if i is not None and j is not None:
            inter[i, j] += 1

    gt_areas = np.array([(gt == gid).sum(dtype=np.int64) for gid in gt_ids], dtype=np.int64)
    pr_areas = np.array([(pred == pid).sum(dtype=np.int64) for pid in pr_ids], dtype=np.int64)

    union = gt_areas[:, None] + pr_areas[None, :] - inter
    with np.errstate(divide='ignore', invalid='ignore'):
        iou = np.where(union > 0, inter / union, 0.0).astype(np.float32)
    return iou


def match_instances(iou_mat: np.ndarray, iou_thr: float) -> Tuple[int, int, int, List[Tuple[int,int,float]]]:
    """Greedy matching by IoU to compute TP/FP/FN and matched pairs."""
    if iou_mat.size == 0:
        n_gt, n_pr = (iou_mat.shape if iou_mat.ndim == 2 else (0, 0))
        return 0, n_pr, n_gt, []

    n_gt, n_pr = iou_mat.shape
    used_gt = np.zeros(n_gt, dtype=bool)
    used_pr = np.zeros(n_pr, dtype=bool)
    matches: List[Tuple[int,int,float]] = []

    idxs = np.dstack(np.unravel_index(np.argsort(-iou_mat, axis=None), (n_gt, n_pr)))[0]
    for gi, pj in idxs:
        if used_gt[gi] or used_pr[pj]:
            continue
        iou = float(iou_mat[gi, pj])
        if iou < iou_thr:
            break
        used_gt[gi] = True
        used_pr[pj] = True
        matches.append((int(gi), int(pj), iou))

    tp = int(len(matches))
    fp = int(n_pr - tp)
    fn = int(n_gt - tp)
    return tp, fp, fn, matches


def boundary_f1(gt_bin: np.ndarray, pr_bin: np.ndarray, tol: int = 2) -> float:
    """Boundary F1 with pixel tolerance; requires scikit-image."""
    if segmentation is None or morphology is None:
        return float('nan')
    gt_b = segmentation.find_boundaries(gt_bin, mode='inner')
    pr_b = segmentation.find_boundaries(pr_bin, mode='inner')
    se = morphology.disk(tol) if hasattr(morphology, 'disk') else morphology.square(2*tol+1)
    gt_d = morphology.binary_dilation(gt_b, selem=se)
    pr_d = morphology.binary_dilation(pr_b, selem=se)
    # Precision: pred boundary pixels matched by dilated GT boundary
    tp_p = np.logical_and(pr_b, gt_d).sum(dtype=np.int64)
    pp = pr_b.sum(dtype=np.int64)
    prec = tp_p / pp if pp > 0 else 1.0
    # Recall: GT boundary pixels matched by dilated pred boundary
    tp_r = np.logical_and(gt_b, pr_d).sum(dtype=np.int64)
    gp = gt_b.sum(dtype=np.int64)
    rec = tp_r / gp if gp > 0 else 1.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def per_image_metrics(gt: np.ndarray, pr: np.ndarray, iou_thr: float, boundary_tol: int) -> Dict[str, float]:
    gt = gt.astype(np.int64, copy=False)
    pr = pr.astype(np.int64, copy=False)
    gt_bin = (gt > 0).astype(np.uint8)
    pr_bin = (pr > 0).astype(np.uint8)

    iou_bin = iou_binary(gt_bin, pr_bin)
    dice_bin = dice_binary(gt_bin, pr_bin)

    iou_mat = compute_iou_matrix(gt, pr)
    tp, fp, fn, matches = match_instances(iou_mat, iou_thr=iou_thr)
    prec = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if tp == 0 and fp == 0 else 0.0)
    rec  = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if tp == 0 and fn == 0 else 0.0)
    f1_inst = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    mean_iou_matched = float(np.mean([m[2] for m in matches])) if matches else 0.0

    bf1 = boundary_f1(gt_bin, pr_bin, tol=boundary_tol)

    return {
        "n_gt": int(np.max(gt)) if gt.size else 0,
        "n_pred": int(np.max(pr)) if pr.size else 0,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision_inst@{:.2f}".format(iou_thr): float(prec),
        "recall_inst@{:.2f}".format(iou_thr): float(rec),
        "f1_inst@{:.2f}".format(iou_thr): float(f1_inst),
        "mean_iou_matched": float(mean_iou_matched),
        "iou_binary": float(iou_bin),
        "dice_binary": float(dice_bin),
        "boundary_f1_tol{}".format(boundary_tol): float(bf1),
    }


def aggregate_metrics(per_img: List[Dict[str, float]], iou_thr: float, boundary_tol: int) -> Dict[str, float]:
    if not per_img:
        return {}
    # Extract keys except counts
    skip = {"n_gt", "n_pred", "tp", "fp", "fn"}
    keys = [k for k in per_img[0].keys() if k not in skip]

    # Macro average
    macro = {f"macro_{k}": float(np.mean([d[k] for d in per_img])) for k in keys}

    # Micro instance pooling
    tp = sum(int(d.get("tp", 0)) for d in per_img)
    fp = sum(int(d.get("fp", 0)) for d in per_img)
    fn = sum(int(d.get("fn", 0)) for d in per_img)
    prec = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if tp == 0 and fp == 0 else 0.0)
    rec  = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if tp == 0 and fn == 0 else 0.0)
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    micro = {
        "micro_precision_inst@{:.2f}".format(iou_thr): float(prec),
        "micro_recall_inst@{:.2f}".format(iou_thr): float(rec),
        "micro_f1_inst@{:.2f}".format(f1),
        "micro_tp": int(tp),
        "micro_fp": int(fp),
        "micro_fn": int(fn),
    }
    return {**macro, **micro}


# --------------------------- Matching & evaluation ---------------------------

from dataclasses import dataclass

@dataclass
class MethodSpec:
    name: str
    dir: Path


def parse_pred_specs(specs: List[str]) -> List[MethodSpec]:
    out: List[MethodSpec] = []
    for s in specs:
        if ":" not in s:
            raise ValueError(f"--pred expects NAME:DIR, got {s}")
        name, path = s.split(":", 1)
        out.append(MethodSpec(name=name, dir=Path(path)))
    return out


def build_index(root: Path, exts: List[str], strip_suffixes: List[str]) -> Dict[str, Path]:
    idx: Dict[str, Path] = {}
    files = list_label_files(root, exts)
    for p in files:
        base = base_from_path(p)
        base = strip_suffix(base, strip_suffixes)
        if base in idx:
            # Prefer .npy over image formats if duplicates
            if p.suffix.lower() == ".npy" or idx[base].suffix.lower() != ".npy":
                idx[base] = p
        else:
            idx[base] = p
    return idx


def evaluate_method(gt_idx: Dict[str, Path],
                    pr_idx: Dict[str, Path],
                    method_name: str,
                    iou_thr: float,
                    boundary_tol: int,
                    out_dir: Path,
                    csv_writer,
                    jsonl_f) -> Tuple[List[Dict[str, float]], List[Tuple[str, str]]]:
    per_list: List[Dict[str, float]] = []
    missing: List[Tuple[str, str]] = []  # (base, reason)

    for base, gt_path in gt_idx.items():
        pr_path = pr_idx.get(base)
        if pr_path is None:
            missing.append((base, "pred_missing"))
            continue
        try:
            gt = load_label_any(gt_path)
            pr = load_label_any(pr_path)
            if gt.shape != pr.shape:
                missing.append((base, f"shape_mismatch gt={gt.shape} pred={pr.shape}"))
                continue

            m = per_image_metrics(gt, pr, iou_thr=iou_thr, boundary_tol=boundary_tol)
            row = {"method": method_name, "base": base, **m}
            csv_writer.writerow(row)
            jsonl_f.write(json.dumps(row) + "\n")
            per_list.append(row)
        except Exception as e:
            missing.append((base, f"error:{e}"))
            continue

    # Write aggregate
    agg = aggregate_metrics(per_list, iou_thr=iou_thr, boundary_tol=boundary_tol)
    agg_path = out_dir / f"aggregate_{method_name}.json"
    with agg_path.open("w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)
    return per_list, missing


def main():
    ap = argparse.ArgumentParser(description="Instance segmentation metrics evaluator (cells/nuclei).")
    ap.add_argument("--gt", type=Path, required=True, help="Ground-truth directory")
    ap.add_argument("--pred", type=str, required=True, nargs="+",
                    help="One or more NAME:DIR specs for predictions, e.g., cellsam:/path/to/preds")
    ap.add_argument("--exts", type=str, nargs="*", default=SUPPORTED_EXTS,
                    help="Extensions to look for (default: .npy .tif .tiff .png)")
    ap.add_argument("--strip-gt-suffixes", type=str, nargs="*", default=[],
                    help="Suffixes to strip from GT base names (e.g., _gt)")
    ap.add_argument("--strip-pred-suffixes", type=str, nargs="*", default=[],
                    help="Suffixes to strip from prediction base names (e.g., _pred_cell)")
    ap.add_argument("--iou-thr", type=float, default=0.5, help="IoU threshold for instance matching")
    ap.add_argument("--boundary-tol", type=int, default=2, help="Pixel tolerance for boundary F1")
    ap.add_argument("--out", type=Path, default=Path("outputs/metrics_eval"), help="Output directory")
    args = ap.parse_args()

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parse methods
    methods = parse_pred_specs(args.pred)

    # Build indices
    gt_idx = build_index(args.gt, args.exts, args.strip_gt_suffixes)

    # Open global per-image outputs
    per_csv_path  = out_dir / "per_image_metrics.csv"
    per_jsonl_path= out_dir / "per_image_metrics.jsonl"
    agg_csv_path  = out_dir / "aggregate_metrics.csv"
    miss_path     = out_dir / "missing_pairs.txt"

    per_fields = ["method", "base",
                  "n_gt","n_pred","tp","fp","fn",
                  f"precision_inst@{args.iou_thr:.2f}",
                  f"recall_inst@{args.iou_thr:.2f}",
                  f"f1_inst@{args.iou_thr:.2f}",
                  "mean_iou_matched",
                  "iou_binary","dice_binary",
                  f"boundary_f1_tol{args.boundary_tol}"]

    with per_csv_path.open("w", newline="", encoding="utf-8") as per_csv_f, \
         per_jsonl_path.open("w", encoding="utf-8") as per_jsonl_f:

        csv_writer = csv.DictWriter(per_csv_f, fieldnames=per_fields)
        csv_writer.writeheader()

        all_missing: List[Tuple[str, str, str]] = []  # (method, base, reason)
        agg_rows: List[Dict[str, object]] = []

        for m in methods:
            pr_idx = build_index(m.dir, args.exts, args.strip_pred_suffixes)
            # Evaluate method m
            per_list, missing = evaluate_method(
                gt_idx, pr_idx, m.name, args.iou_thr, args.boundary_tol, out_dir,
                csv_writer, per_jsonl_f
            )
            # Aggregate table row
            agg = aggregate_metrics(per_list, iou_thr=args.iou_thr, boundary_tol=args.boundary_tol)
            agg_row = {"method": m.name, **agg}
            agg_rows.append(agg_row)
            # Missing pairs
            for base, reason in missing:
                all_missing.append((m.name, base, reason))

    # Write aggregate CSV
    # Determine all keys present
    all_keys = set()
    for r in agg_rows:
        all_keys |= set(r.keys())
    all_keys = ["method"] + sorted([k for k in all_keys if k != "method"])
    with agg_csv_path.open("w", newline="", encoding="utf-8") as f:
        cw = csv.DictWriter(f, fieldnames=all_keys)
        cw.writeheader()
        for r in agg_rows:
            cw.writerow(r)

    # Write missing pairs
    if all_missing:
        with miss_path.open("w", encoding="utf-8") as mf:
            for method, base, reason in all_missing:
                mf.write(f"{method}\t{base}\t{reason}\n")

    # Print a compact summary
    print(f"[DONE] Per-image metrics: {per_csv_path}")
    print(f"[DONE] Per-method aggregate: {agg_csv_path}")
    if all_missing:
        print(f"[WARN] Some pairs were skipped due to missing/mismatch. See: {miss_path}")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
# (content identical to previous cell; shortened header)
from __future__ import annotations
import argparse, csv, json, warnings
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import numpy as np, pandas as pd
from stardist.matching import matching
from cellpose import metrics as cpmetrics
from utils import SampleDataset

def load_npy(p: Path): return np.load(p)
def try_load(p: Path): 
    try: return load_npy(p)
    except Exception: return None

def iter_samples_with_gt(ds: SampleDataset, task: str):
    out = []
    for s in ds:
        gt = s.nuc_scribble_path if task=="nuclei" else s.marker_scribble_path
        if gt is not None and gt.exists(): out.append((s.base, gt))
    return out

def pred_path_for(base: str, method_dir: Path, task: str, pred_suffix: str|None):
    if pred_suffix is not None:
        return method_dir / f"{base}{pred_suffix}"
    return method_dir / (f"{base}_pred_nuclei.npy" if task=="nuclei" else f"{base}_pred_cell.npy")

def aji(gt, pr):
    try: return float(cpmetrics.aggregated_jaccard_index(gt, pr))
    except TypeError: return float(cpmetrics.aggregated_jaccard_index(pr, gt))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["cell","nuclei"], required=True)
    ap.add_argument("--gt-root", type=Path, required=True)
    ap.add_argument("--pred", type=str, nargs="+", required=True)  # NAME:DIR
    ap.add_argument("--pred-suffix", type=str, default=None)
    ap.add_argument("--iou-start", type=float, default=0.5)
    ap.add_argument("--iou-stop", type=float, default=0.95)
    ap.add_argument("--iou-step", type=float, default=0.05)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    out_dir: Path = args.out; out_dir.mkdir(parents=True, exist_ok=True)
    ths = np.arange(args.iou_start, args.iou_stop+1e-9, args.iou_step)

    ds = SampleDataset(args.gt_root)
    pairs = iter_samples_with_gt(ds, args.task)
    if not pairs: raise SystemExit(f"No GT found for task={args.task} under {args.gt_root}")

    methods = []
    for spec in args.pred:
        if ":" not in spec: raise SystemExit(f"--pred expects NAME:DIR, got {spec}")
        name, p = spec.split(":",1); methods.append((name, Path(p)))

    per_rows, agg_rows = [], []
    for method, mdir in methods:
        curves, n_used, n_missing, n_shape = [], 0, 0, 0
        for base, gt_path in pairs:
            pr_path = pred_path_for(base, mdir, args.task, args.pred_suffix)
            pr = try_load(pr_path)
            if pr is None: n_missing += 1; continue
            gt = load_npy(gt_path)
            if gt.shape != pr.shape: n_shape += 1; continue

            precisions = []
            for t in ths:
                try:
                    m = matching(gt, pr, thresh=float(t))
                    precisions.append(float(m.precision))
                except Exception as exc:
                    warnings.warn(f"matching() failed on {base} ({method}, thr={t}): {exc}")
                    precisions.append(0.0)
            precisions = np.asarray(precisions, dtype=np.float32)
            curves.append(precisions); n_used += 1

            try: a = aji(gt, pr)
            except Exception: a = float("nan")

            per_rows.append({
                "task": args.task, "method": method, "base": base,
                f"mAP@[{args.iou_start:.2f}:{args.iou_stop:.2f}:{args.iou_step:.2f}]": float(precisions.mean()),
                "AJI": float(a),
                **{f"precision@IoU={t:.2f}": float(p) for t,p in zip(ths, precisions)},
            })

        if curves:
            M = np.stack(curves, axis=0); mean_curve = M.mean(axis=0)
            agg_rows.append({
                "task": args.task, "method": method,
                "n_images": int(n_used), "n_missing_pred": int(n_missing), "n_shape_mismatch": int(n_shape),
                f"mAP@[{args.iou_start:.2f}:{args.iou_stop:.2f}:{args.iou_step:.2f}]": float(mean_curve.mean()),
                **{f"mean_precision@IoU={t:.2f}": float(v) for t,v in zip(ths, mean_curve)},
            })
        else:
            agg_rows.append({
                "task": args.task, "method": method,
                "n_images": 0, "n_missing_pred": int(n_missing), "n_shape_mismatch": int(n_shape),
                f"mAP@[{args.iou_start:.2f}:{args.iou_stop:.2f}:{args.iou_step:.2f}]": float("nan"),
            })

    import pandas as pd
    per_df = pd.DataFrame(per_rows); agg_df = pd.DataFrame(agg_rows)
    per_csv = out_dir / f"per_image_{args.task}.csv"
    agg_csv = out_dir / f"aggregate_{args.task}.csv"
    per_jsonl = out_dir / f"per_image_{args.task}.jsonl"
    per_df.to_csv(per_csv, index=False); agg_df.to_csv(agg_csv, index=False)
    with open(per_jsonl,"w",encoding="utf-8") as f:
        for _,row in per_df.iterrows(): f.write(row.to_json() + "\n")
    print(f"[DONE] per-image: {per_csv}"); print(f"[DONE] aggregate: {agg_csv}"); print(f"[INFO] jsonl: {per_jsonl}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按“细胞类型 + 半径”混合列统计半径分布。
假设半径列中，第一个词是细胞类型，其后包含一个数字为半径值。
示例值： "Tcell 12.3", "B-cell: 9.8", "macro 7", "NK 10um"

输出：
- stats_by_type.csv         （按细胞类型聚合）
- stats_by_type_image.csv   （按细胞类型、image_key 聚合；若无该列则跳过）

用法：
python stat_cell_radius_by_type.py \
  --csv cell_features.csv \
  --radius_col cell_radius \
  --image_key_col image_key \
  --out_dir ./stats_out
"""
import argparse, os, re, math
import numpy as np
import pandas as pd

TYPE_RE   = re.compile(r'^\s*([A-Za-z0-9_\-]+)')  # 第一个“词”作为类型
FLOAT_RE  = re.compile(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?')  # 第一个浮点数作为半径

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def parse_type_and_radius(val):
    """从字符串中解析 (cell_type, radius_float)。失败返回 (None, NaN)。"""
    if pd.isna(val):
        return (None, math.nan)
    s = str(val)
    mtype = TYPE_RE.search(s)
    mnum  = FLOAT_RE.search(s)
    ctype = mtype.group(1) if mtype else None
    rad   = float(mnum.group(0)) if mnum else math.nan
    return (ctype, rad)

def agg_stats(s: pd.Series) -> pd.Series:
    s = s.dropna().astype(float)
    if s.empty:
        return pd.Series({
            "count": 0, "mean": np.nan, "median": np.nan, "std": np.nan,
            "min": np.nan, "p10": np.nan, "p90": np.nan, "max": np.nan
        })
    return pd.Series({
        "count": s.size,
        "mean": s.mean(),
        "median": s.median(),
        "std": s.std(ddof=1) if s.size > 1 else 0.0,
        "min": s.min(),
        "p10": np.percentile(s, 10),
        "p90": np.percentile(s, 90),
        "max": s.max()
    })

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="cell_features.csv")
    ap.add_argument("--radius_col", default="equivalent_radius_px")
    ap.add_argument("--image_key_col", default="image_key")
    ap.add_argument("--out_dir", default="./stats_out")
    args = ap.parse_args()
    ensure_dir(args.out_dir)

    if not os.path.isfile(args.csv):
        raise FileNotFoundError(args.csv)

    df = pd.read_csv(args.csv)

    if args.radius_col not in df.columns:
        raise KeyError(f"未找到列: {args.radius_col}")

    parsed = df[args.radius_col].apply(parse_type_and_radius)
    df["_cell_type"]  = parsed.apply(lambda x: x[0])
    df["_cell_radius"] = parsed.apply(lambda x: x[1])

    # 仅保留解析出类型与半径的行
    use = df[~df["_cell_type"].isna() & ~df["_cell_radius"].isna()].copy()
    if use.empty:
        raise ValueError("无法从半径列解析出任何类型或数值。请检查源数据。")

    # 1) 按类型聚合
    by_type = use.groupby("_cell_type")["_cell_radius"].apply(agg_stats).unstack()
    by_type = by_type.sort_values(["count","mean"], ascending=[False, False])
    out1 = os.path.join(args.out_dir, "stats_by_type.csv")
    by_type.to_csv(out1, float_format="%.6g")

    # 2) 按类型 + image_key 聚合（若存在该列）
    if args.image_key_col in use.columns:
        by_img = (
            use.groupby([args.image_key_col, "_cell_type"])["_cell_radius"]
               .apply(agg_stats).unstack()
               .reset_index()
        )
        out2 = os.path.join(args.out_dir, "stats_by_type_image.csv")
        by_img.to_csv(out2, index=False, float_format="%.6g")

    # 可选：打印简要摘要（不要求输出结果，可保留）
    top = by_type.head(min(10, len(by_type)))
    print("Top types by count:")
    print(top[["count","mean","median","p10","p90"]])

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch feature extraction for nuclei (dapimultimask) and cell bodies masks.
- Scans a folder for *dapimultimask.npy and *cellbodies.npy
- Optionally reads matching raw TIFFs: *_marker.tiff for cell intensity, *.tiff for DAPI intensity
- Computes per-object features and descriptive stats
- Saves CSVs in output directory


python 10_eda/extract_features.py\
    --data_dir /ihome/jbwang/liy121/ifimage/00_dataset_sep27\
        --out_dir /ihome/jbwang/liy121/ifimage/10_eda


"""

import os
import re
import math
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from skimage.measure import label, regionprops_table
from skimage.segmentation import relabel_sequential
from tifffile import imread

def load_mask(path: Path):
    arr = np.load(path) if path.suffix.lower() == ".npy" else imread(path)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Mask at {path} is not 2D. Shape={arr.shape}")
    if arr.dtype == bool or set(np.unique(arr).tolist()) <= {0,1}:
        lab = label(arr.astype(bool))
    else:
        lab = arr.astype(np.int64)
        lab, _, _ = relabel_sequential(lab)
    return lab

def load_optional(path: Path):
    try:
        x = imread(path)
        return np.squeeze(x)
    except Exception:
        return None

def base_key(p: Path):
    s = p.stem
    s = re.sub(r"_(dapimultimask|cellbodies|marker)$", "", s, flags=re.IGNORECASE)
    return s

def find_pairs(data_dir: Path):
    nuc_files = list(data_dir.glob("*dapimultimask.npy"))
    cell_files = list(data_dir.glob("*cellbodies.npy"))
    tiffs = list(data_dir.glob("*.tif")) + list(data_dir.glob("*.tiff"))
    nuc_map = {base_key(p): p for p in nuc_files}
    cell_map = {base_key(p): p for p in cell_files}
    marker_map, dapi_map = {}, {}
    for p in tiffs:
        st = p.stem.lower()
        if st.endswith("_marker"):
            marker_map[base_key(p)] = p
        else:
            dapi_map[base_key(p)] = p
    keys = sorted(set(nuc_map) | set(cell_map) | set(marker_map) | set(dapi_map))
    out = []
    for k in keys:
        out.append({
            "key": k,
            "nucleus_mask": nuc_map.get(k),
            "cell_mask": cell_map.get(k),
            "marker_tiff": marker_map.get(k),
            "dapi_tiff": dapi_map.get(k),
        })
    return out

def per_object_features(label_img, raw=None, role="nucleus", image_key=""):
    props = [
        "label","area","bbox_area","eccentricity","equivalent_diameter","extent",
        "feret_diameter_max","major_axis_length","minor_axis_length","orientation",
        "perimeter","solidity","centroid"
    ]
    table = regionprops_table(
        label_img,
        intensity_image=raw if raw is not None else None,
        properties=props + (["intensity_mean"] if raw is not None else []),
    )
    df = pd.DataFrame(table)
    if df.empty:
        return df
    df["equivalent_radius_px"] = np.sqrt(df["area"] / np.pi)
    df["circularity"] = (4*np.pi*df["area"]) / (np.maximum(df["perimeter"], 1e-6)**2)
    df["aspect_ratio"] = np.divide(df["major_axis_length"], np.maximum(df["minor_axis_length"], 1e-6))
    df["role"] = role
    df["image_key"] = image_key
    return df

def overlap_mapping(nuc_labels, cell_labels):
    if nuc_labels.max()==0 or cell_labels.max()==0:
        return {}
    pairs = np.stack([nuc_labels.ravel(), cell_labels.ravel()], axis=1)
    mask = (pairs[:,0]>0) & (pairs[:,1]>0)
    if not np.any(mask):
        return {}
    pairs = pairs[mask]
    # count overlaps
    # numpy bincount on encoded ids for speed
    max_n = nuc_labels.max()+1
    encoded = pairs[:,0].astype(np.int64)* (cell_labels.max()+1) + pairs[:,1].astype(np.int64)
    unique, counts = np.unique(encoded, return_counts=True)
    # decode
    best = {}
    temp = {}
    for code, cnt in zip(unique, counts):
        n = int(code // (cell_labels.max()+1))
        c = int(code %  (cell_labels.max()+1))
        if n not in best or cnt > best[n][1]:
            best[n] = (c, cnt)
    return {n:c for n,(c,_) in best.items()}

def summarize(df, role_col="role"):
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    group_cols = ["image_key", role_col]
    agg = {
        "area": ["count","mean","median","std"],
        "equivalent_radius_px": ["mean","median","std"],
        "perimeter": ["mean","median"],
        "circularity": ["mean","median"],
        "aspect_ratio": ["mean","median"],
    }
    per_image = df.groupby(group_cols).agg(agg)
    per_image.columns = ["_".join(col).strip() for col in per_image.columns.values]
    per_image = per_image.reset_index()
    global_stats = df.groupby(role_col).agg(agg)
    global_stats.columns = ["_".join(col).strip() for col in global_stats.columns.values]
    global_stats = global_stats.reset_index()
    return per_image, global_stats

def main():
    parser = argparse.ArgumentParser(description="Extract features from DAPI nuclei and cell body masks.")
    parser.add_argument("--data_dir", type=Path, required=True, help="Folder with masks and raw tiffs")
    parser.add_argument("--out_dir", type=Path, default=Path("."), help="Output folder for CSVs")
    parser.add_argument("--px_size_um", type=float, default=None, help="Pixel size in micrometers. If given, adds µm metrics.")
    args = parser.parse_args()

    pairs = find_pairs(args.data_dir)
    all_nuc, all_cell = [], []

    for rec in pairs:
        key = rec["key"]
        nuc_path = rec["nucleus_mask"]
        cell_path = rec["cell_mask"]
        if nuc_path is None and cell_path is None:
            continue

        nuc_lab = load_mask(nuc_path) if nuc_path else None
        cell_lab = load_mask(cell_path) if cell_path else None

        marker_raw = load_optional(rec["marker_tiff"]) if rec["marker_tiff"] else None
        dapi_raw   = load_optional(rec["dapi_tiff"]) if rec["dapi_tiff"] else None

        if nuc_lab is None:
            nuc_lab = np.zeros_like(cell_lab)
        if cell_lab is None:
            cell_lab = np.zeros_like(nuc_lab)

        H = min(nuc_lab.shape[0], cell_lab.shape[0])
        W = min(nuc_lab.shape[1], cell_lab.shape[1])
        nuc_lab = nuc_lab[:H,:W]
        cell_lab = cell_lab[:H,:W]
        if marker_raw is not None:
            marker_raw = np.squeeze(marker_raw)[:H,:W]
        if dapi_raw is not None:
            dapi_raw = np.squeeze(dapi_raw)[:H,:W]

        df_n = per_object_features(nuc_lab, raw=dapi_raw, role="nucleus", image_key=key)
        df_c = per_object_features(cell_lab, raw=marker_raw, role="cell", image_key=key)

        mapping = overlap_mapping(nuc_lab, cell_lab)
        if not df_n.empty:
            df_n["cell_label"] = df_n["label"].map(mapping).fillna(0).astype(int)

        all_nuc.append(df_n)
        all_cell.append(df_c)

    nuc_df = pd.concat(all_nuc, ignore_index=True) if all_nuc else pd.DataFrame()
    cell_df = pd.concat(all_cell, ignore_index=True) if all_cell else pd.DataFrame()

    # Unit conversion if pixel size provided
    if args.px_size_um is not None and not nuc_df.empty:
        for df in [nuc_df, cell_df]:
            if df.empty: 
                continue
            df["equivalent_radius_um"] = df["equivalent_radius_px"] * args.px_size_um
            df["area_um2"] = df["area"] * (args.px_size_um ** 2)
            df["perimeter_um"] = df["perimeter"] * args.px_size_um

    per_image_nuc, global_nuc = summarize(nuc_df, role_col="role")
    per_image_cell, global_cell = summarize(cell_df, role_col="role")
    summary_per_image = pd.concat([per_image_nuc, per_image_cell], ignore_index=True) if not per_image_nuc.empty or not per_image_cell.empty else pd.DataFrame()
    summary_global = pd.concat([global_nuc, global_cell], ignore_index=True) if not global_nuc.empty or not global_cell.empty else pd.DataFrame()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    nuc_csv = args.out_dir / "nuclei_features.csv"
    cell_csv = args.out_dir / "cell_features.csv"
    per_image_csv = args.out_dir / "summary_per_image.csv"
    global_csv = args.out_dir / "summary_global.csv"

    nuc_df.to_csv(nuc_csv, index=False)
    cell_df.to_csv(cell_csv, index=False)
    summary_per_image.to_csv(per_image_csv, index=False)
    summary_global.to_csv(global_csv, index=False)

    print("Saved:")
    print(nuc_csv)
    print(cell_csv)
    print(per_image_csv)
    print(global_csv)

if __name__ == "__main__":
    main()

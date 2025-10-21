#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

import os, sys, json, subprocess
from pathlib import Path

import numpy as np
from skimage.draw import polygon

# 可选：用 openslide 读 WSI 尺寸；读失败再用 tifffile
try:
    import openslide
except Exception:
    openslide = None
try:
    from tifffile import TiffFile
except Exception:
    TiffFile = None

DATA_DIR = Path("/ihome/jbwang/liy121/ifimage/00_dataset_sep27")
OUT_DIR  = Path("pred_cellvit_nuc")

MODEL    = "SAM"        # or "HIPT"
TAX      = "binary"     # or "pannuke"
GPU      = "0"
BATCH    = "8"

INSTANCE_FIELD = "instance_id"    # 若无该字段则按顺序赋值 1..N
CLASS_FIELD    = "class_id"       # 若存在则同时写类别图

def to_pyramid(src: Path) -> Path:
    ext = src.suffix.lower()
    if ext in [".svs", ".ndpi"] or ".pyr.tiff" in src.name.lower():
        return src
    tgt = src.with_suffix(".pyr.tiff")
    if not tgt.exists():
        tgt.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            "vips","tiffsave",str(src),str(tgt),
            "--tile","--tile-width","256","--tile-height","256",
            "--pyramid","--compression","jpeg","--Q","90"
        ], check=True)
    return tgt


def run_cellvit(wsi: Path, outdir: Path):
    """调用 cellvit-inference 对单张 WSI 推理并导出 GeoJSON。"""
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "cellvit-inference",
        "--model", MODEL,
        "--nuclei_taxonomy", TAX,
        "--gpu", GPU,
        "--batch_size", BATCH,
        "--geojson",
        "--outdir", str(outdir),
        "process_wsi",
        "--wsi_path", str(wsi)
    ]
    subprocess.run(cmd, check=True)


def get_image_size(wsi_path: Path):
    """读取 level-0 尺寸，优先 openslide；失败则用 tifffile。"""
    # OpenSlide
    if openslide is not None:
        try:
            slide = openslide.OpenSlide(str(wsi_path))
            w, h = slide.dimensions
            slide.close()
            return (h, w)
        except Exception:
            pass
    # tifffile
    if TiffFile is not None:
        try:
            with TiffFile(str(wsi_path)) as tf:
                page = tf.pages[0]
                h, w = int(page.imagelength), int(page.imagewidth)
                return (h, w)
        except Exception:
            pass
    raise RuntimeError(f"无法读取图像尺寸: {wsi_path}")


def rasterize_geojson(geojson_path: Path, out_shape, instance_field=INSTANCE_FIELD, class_field=CLASS_FIELD):
    """将 GeoJSON 多边形栅格化为 label 图。返回 (inst_mask, class_mask or None)。"""
    H, W = out_shape
    with open(geojson_path, "r") as f:
        gj = json.load(f)

    inst = np.zeros((H, W), np.uint32)
    cls  = np.zeros((H, W), np.uint16) if class_field else None

    next_id = 1
    feats = gj["features"] if "features" in gj else []
    for feat in feats:
        geom = feat.get("geometry", {})
        if geom.get("type") != "Polygon":
            # 只处理 Polygon；MultiPolygon 可按需扩展
            continue
        coords = geom.get("coordinates", [])
        if not coords:
            continue
        ring = np.asarray(coords[0], dtype=float)  # 外环
        if ring.ndim != 2 or ring.shape[1] != 2:
            continue

        # skimage.draw.polygon 需要行(y)列(x)
        rr, cc = polygon(ring[:, 1], ring[:, 0], shape=(H, W))

        # 实例ID
        val = feat.get("properties", {}).get(instance_field, None)
        if val is None:
            val = next_id
            next_id += 1
        val = int(val)

        inst[rr, cc] = val

        # 类别图（可选）
        if cls is not None and class_field:
            cval = feat.get("properties", {}).get(class_field, 0)
            try:
                cval = int(cval)
            except Exception:
                cval = 0
            cls[rr, cc] = cval

    return inst, cls


def find_geojson(dir_path: Path):
    """在 CellViT 输出目录中寻找一个 geojson 文件路径。"""
    cands = list(dir_path.rglob("*.geojson")) + list(dir_path.rglob("*.json"))
    # 优先 *.geojson
    cands.sort(key=lambda p: (p.suffix != ".geojson", len(str(p))))
    return cands[0] if cands else None


def process_one_case(src_img: Path):
    """单样本：转金字塔→推理→栅格化→保存 .npy"""
    wsi = to_pyramid(src_img)
    case_out = OUT_DIR / src_img.stem
    run_cellvit(wsi, case_out)

    gj = find_geojson(case_out)
    if gj is None:
        print(f"[WARN] 未找到 GeoJSON: {case_out}")
        return

    H, W = get_image_size(wsi)
    inst, cls = rasterize_geojson(gj, (H, W))

    np.save(case_out / "mask_instance.npy", inst)
    if cls is not None:
        np.save(case_out / "mask_class.npy", cls)

    print(f"[OK] {src_img.name} -> {case_out}")


def main():
    OUT_DIR.mkdir(exist_ok=True)
    imgs = sorted([p for p in Path(DATA_DIR).rglob("*")
                   if p.suffix.lower() in [".svs", ".tiff", ".tif", ".ndpi"]])
    if not imgs:
        print(f"[EMPTY] 未在 {DATA_DIR} 找到可处理的图像")
        sys.exit(0)

    ok, fail = 0, 0
    for img in imgs:
        try:
            process_one_case(img)
            ok += 1
        except Exception as e:
            print(f"[FAIL] {img}: {e}")
            fail += 1
    print(f"Done. ok={ok}, fail={fail}, total={len(imgs)}")


if __name__ == "__main__":
    main()

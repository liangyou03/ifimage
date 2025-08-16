#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert NPY instance-id masks + TIFF images into COCO Instance JSON.
Now supports a SINGLE directory with mixed .tiff and *_cellbodies.npy / *_dapimultimask.npy.
If no original image is found, fallback to mask.shape and print warnings.

Usage:
  python npy_to_coco_single.py \
    --data-dir /path/to/processed_dataset/images \
    --task cellbodies \
    --out /path/to/annotations_cell.json
    
python npy_to_coco_single.py \
    --data-dir /ihome/jbwang/liy121/ifimage/processed_dataset/images \
    --task cellbodies \
    --out /ihome/jbwang/liy121/ifimage/processed_dataset/annotations_cell.json
    
python npy_to_coco_single.py \
  --data-dir /ihome/jbwang/liy121/ifimage/processed_dataset/images \
  --task dapimultimask \
  --out /ihome/jbwang/liy121/ifimage/processed_dataset/annotations_nucleus.json


"""

import argparse
import json
import re
from pathlib import Path
import numpy as np

try:
    import tifffile as tiff
except Exception:
    tiff = None

try:
    from pycocotools import mask as maskUtils
except Exception:
    maskUtils = None


def natural_key(s: str):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def rle_encode_from_binary(bin_mask: np.ndarray):
    if maskUtils is None:
        raise RuntimeError("pycocotools not installed. Please pip install pycocotools")
    if bin_mask.dtype != np.uint8:
        bin_mask = bin_mask.astype(np.uint8)
    rle = maskUtils.encode(np.asfortranarray(bin_mask))
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle


def to_annotations_from_label(mask: np.ndarray, image_id: int, category_id: int, start_ann_id: int = 1):
    anns = []
    inst_ids = np.unique(mask)
    inst_ids = inst_ids[inst_ids > 0]
    ann_id = start_ann_id

    for iid in inst_ids:
        bin_mask = (mask == iid)
        if not bin_mask.any():
            continue
        rle = rle_encode_from_binary(bin_mask)
        area = int(float(maskUtils.area(rle)))
        bbox = [float(x) for x in maskUtils.toBbox(rle).tolist()]

        anns.append({
            "id": int(ann_id),
            "image_id": int(image_id),
            "category_id": int(category_id),
            "iscrowd": 0,
            "segmentation": rle,
            "bbox": bbox,
            "area": area,
        })
        ann_id += 1
    return anns, ann_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True, help='Directory with .tiff and .npy masks together.')
    parser.add_argument('--task', type=str, required=True, choices=['cellbodies', 'dapimultimask'],
                        help='Which mask suffix to convert.')
    parser.add_argument('--out', type=str, required=True, help='Output COCO JSON path.')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    suffix = args.task

    if suffix == 'cellbodies':
        categories = [{"id": 1, "name": "cell"}]
        category_id = 1
    else:
        categories = [{"id": 2, "name": "nucleus"}]
        category_id = 2

    mask_files = sorted(data_dir.glob(f'*_{suffix}.npy'), key=lambda p: natural_key(p.name))
    if not mask_files:
        raise FileNotFoundError(f"No '*_{suffix}.npy' found under {data_dir}")

    images, annotations = [], []
    ann_id, image_id = 1, 1
    image_id_map = {}
    missing_images = []

    for npy_path in mask_files:
        stem = npy_path.stem[:-(len(suffix)+1)]
        img_path = None
        for ext in ('.tiff', '.tif', '.png', '.jpg'):
            cand = data_dir / f"{stem}{ext}"
            if cand.exists():
                img_path = cand
                break

        mask = np.load(npy_path)
        mask = np.squeeze(mask).astype(np.int32)

        if stem not in image_id_map:
            if img_path and tiff is not None:
                try:
                    with tiff.TiffFile(str(img_path)) as tf:
                        page = tf.pages[0]
                        h, w = int(page.shape[-2]), int(page.shape[-1])
                    file_name = img_path.name
                except Exception:
                    h, w = mask.shape[0], mask.shape[1]
                    file_name = f"{stem}.tiff"
                    missing_images.append(stem)
                    print(f"[WARN] Could not read {img_path}, using mask shape instead.")
            else:
                h, w = mask.shape[0], mask.shape[1]
                file_name = f"{stem}.tiff"
                missing_images.append(stem)
                print(f"[WARN] No image found for {stem}, using mask shape.")

            images.append({
                "id": int(image_id),
                "file_name": file_name,
                "height": int(h),
                "width": int(w),
            })
            image_id_map[stem] = image_id
            cur_image_id = image_id
            image_id += 1
        else:
            cur_image_id = image_id_map[stem]

        anns, ann_id = to_annotations_from_label(mask, cur_image_id, category_id, ann_id)
        annotations.extend(anns)

    coco = {
        "info": {"description": f"{suffix} dataset", "version": "1.0"},
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    with open(args.out, 'w') as f:
        json.dump(coco, f)
    print(f"[OK] Wrote {args.out} with {len(images)} images and {len(annotations)} annotations.")

    if missing_images:
        print("\n[SUMMARY] Missing original images for the following cases (used mask shape instead):")
        for s in missing_images:
            print(f"  - {s}")


if __name__ == '__main__':
    main()

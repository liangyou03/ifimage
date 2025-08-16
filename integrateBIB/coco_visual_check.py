#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import os
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from PIL import Image

# ===== CONFIG =====
ANNOT_PATH = "/ihome/jbwang/liy121/ifimage/processed_dataset/annotations_cell.json"
IMG_ROOT   = "/ihome/jbwang/liy121/ifimage/processed_dataset/images"
OUT_DIR    = "sanity_check_outputs"
NUM_SAMPLES = 5   # how many images to visualize
# ==================

os.makedirs(OUT_DIR, exist_ok=True)

coco = COCO(ANNOT_PATH)
img_ids = coco.getImgIds()
chosen_ids = random.sample(img_ids, min(NUM_SAMPLES, len(img_ids)))

for img_id in chosen_ids:
    img_info = coco.loadImgs([img_id])[0]
    file_name = img_info['file_name']
    img_path = os.path.join(IMG_ROOT, file_name)

    if not os.path.exists(img_path):
        print(f"[WARN] Image file not found: {img_path}, skipping.")
        continue

    # load image
    img = np.array(Image.open(img_path))

    # get annotations
    ann_ids = coco.getAnnIds(imgIds=[img_id])
    anns = coco.loadAnns(ann_ids)

    plt.figure(figsize=(10, 10))
    plt.imshow(img, cmap="gray")
    plt.axis("off")

    # overlay masks
    for ann in anns:
        if 'segmentation' in ann:
            rle = ann['segmentation']
            m = maskUtils.decode(rle)
            # random color
            color = np.random.rand(3)
            img_mask = np.zeros((*m.shape, 4))
            img_mask[m == 1] = [*color, 0.5]  # RGBA
            plt.imshow(img_mask)

            # draw bbox
            x, y, w, h = ann['bbox']
            plt.gca().add_patch(
                plt.Rectangle((x, y), w, h, fill=False, edgecolor=color, linewidth=2)
            )

    out_path = os.path.join(OUT_DIR, f"check_{img_id}.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved sanity check -> {out_path}")


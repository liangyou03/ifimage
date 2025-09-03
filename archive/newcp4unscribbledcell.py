# %%
import importlib
import ifimage_tools
importlib.reload(ifimage_tools)
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from preprocessing import SamplePreprocessor,batch_process_zip_rois,batch_rename,mask_to_rois_zip

# %%
import os
import numpy as np
from stardist.matching import matching
image_dir = "Reorgnized Ground Truth"
masks_dir = "Reorgnized Ground Truth/mask"
dataset = ifimage_tools.IfImageDataset(image_dir, masks_dir, {})
dataset.load_data()

old_version_sample_ids = ["6390", "8408", "8406", "8405v2", "8405", "8407"]
for sample_id in old_version_sample_ids:
    if sample_id in dataset.samples:
        del dataset.samples[sample_id]

# %%
to_process = []
for sample_id, sample in dataset.samples.items():
    has_cellbodies = (sample.cellbodies_mask is not None) or (sample.cellbodies_multimask is not None)
    has_dapi_multi = sample.dapi_multi_mask is not None
    if has_cellbodies and not has_dapi_multi:
        to_process.append(sample_id)

print(f"—— 一共发现 {len(to_process)} 个样本，需要先跑 apply_nuc_pipeline，然后生成 ROI：{to_process}")

# %%
# 输出目录
output_zip_dir = "roi_zips"
os.makedirs(output_zip_dir, exist_ok=True)

for sid in to_process:
    sample = dataset.samples[sid]
    print(f"\n[INFO] 开始处理样本：{sid}")

    # 1. 调用 apply_nuc_pipeline 生成 dapi_multi_mask
    print(f"  • 运行 apply_nuc_pipeline...")
    sample.apply_nuc_pipeline()

    if sample.masks is None:
        print(f"  [WARN] 样本 {sid} 在运行 apply_nuc_pipeline 后仍然没有 dapi_multi_mask，跳过。")
        continue

    dapi_mask = sample.masks.copy()["cellposeSAM"]
    # 如果是 0/1 二值图，就先做 connected-components 标号
    from skimage.measure import label as sklabel
    if dapi_mask.max() <= 1:
        dapi_mask = sklabel(dapi_mask)

    # 3. 调用 mask_to_rois_zip，把 dapi_mask 转成 ImageJ ROI 并打包
    celltype=sample.celltype
    zip_path = os.path.join(output_zip_dir, f"{celltype}_{sid}_nuclei_roi.zip")
    print(f"  • 生成 ROI ZIP: {zip_path} ...")
    mask_to_rois_zip(dapi_mask, zip_path)

    print(f"  [OK] 样本 {sid} 的 nuclei ROI ZIP 已保存。")

print("\n全部样本处理完毕，请到 “roi_zips” 目录查看各个 *_nuclei_roi.zip 文件。")

# %%




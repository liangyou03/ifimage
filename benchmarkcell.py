# %%
import importlib, os, numpy as np
import ifimage_tools                           # 你自己的 utils 包
importlib.reload(ifimage_tools)                # 热重载
from tqdm.auto import tqdm                     # 可选：进度条
import os
import numpy as np
from stardist.matching import matching

# %%
image_dir  = "Reorgnized Ground Truth"
masks_dir  = "merged_mask"
dataset = ifimage_tools.IfImageDataset(image_dir, masks_dir, {})
dataset.load_data()

for sid in ["6390", "8408", "8406", "8405v2", "8405", "8407"]:
    dataset.samples.pop(sid, None)

# --------------------------- 2. 分割 & 保存 -----------------------------------
output_root = "cell_masks"            # 新目录，不要跟 nuclei 混
os.makedirs(output_root, exist_ok=True)

# %%
for sid, sample in dataset.samples.items():
    if sample.marker is None: continue
    m = sample.marker
    print(f"{sid}: marker ndim={m.ndim}, shape={m.shape}, min={m.min():.3f}, max={m.max():.3f}")
    break

# %%
dataset.samples.items()

# %%
methods = ["cellpose","cellpose2chan"]

for sample_id, sample in tqdm(dataset.samples.items(), desc="cell seg"):
    # 跑细胞质阳性分割
    sample.get_positive_cyto_pipline(methods=methods)

    # mkdir cell_masks/<sample_id>/
    sample_dir = os.path.join(output_root, sample_id)
    os.makedirs(sample_dir, exist_ok=True)

    # 保存每种算法的 mask
    for m in methods:
        mask = sample.cyto_positive_masks.get(m)
        if mask is None:
            print("失败！")
            continue                      # 方法失败/未跑
        out_path = os.path.join(sample_dir, f"{m}.npy")
        np.save(out_path, mask.astype(np.uint32))
        print(f"[{sample_id}] saved {m} → {out_path}")




#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import tifffile as tiff
from skimage.filters import threshold_otsu
from scipy import ndimage as ndi

DATASET_DIR = Path("/ihome/jbwang/liy121/ifimage/00_dataset")  # 原始强度图所在处
# 每个算法的“原始胞质预测”目录
CYTO_PRED_DIRS = {
    "cellpose":   Path("/ihome/jbwang/liy121/ifimage/01_cellpose_benchmark/cyto_prediction"),
    "cellsam":    Path("/ihome/jbwang/liy121/ifimage/03_cellsam_benchmark/cyto_prediction"),
    "mesmer":     Path("/ihome/jbwang/liy121/ifimage/04_mesmer_benchmark/cyto_prediction"),
    "watershed":  Path("/ihome/jbwang/liy121/ifimage/06_watershed_benchmark/cyto_prediction"),
    "omnipose":   Path("/ihome/jbwang/liy121/ifimage/07_omnipose_benchmark/cyto_prediction"),
}

# 预测文件名里需要剥离的后缀以得到 base name
STRIPS = ["_pred_cyto","_cyto","_cell","_prediction","_refined","_filter","_filtered","_cyto_filter"]

def _read_any_mask(p: Path) -> np.ndarray:
    if p.suffix.lower()==".npy": arr=np.load(p)
    else: arr=tiff.imread(str(p))
    arr=np.squeeze(arr)
    if arr.dtype==bool: arr=arr.astype(np.uint8)
    if arr.ndim!=2: raise ValueError(f"mask must be 2D: {p}")
    # 若是二值，连通域标记
    if np.unique(arr).size<=3 and arr.min()==0:
        arr, _ = ndi.label(arr>0)
    return arr.astype(np.int32, copy=False)

def _read_any_image(p: Path) -> np.ndarray:
    if p.suffix.lower()==".npy": img=np.load(p)
    else: img=tiff.imread(str(p))
    return np.squeeze(img).astype(np.float32)

def _base(stem: str) -> str:
    b=stem
    for s in STRIPS: b=b.replace(s,"")
    return b

def _intensity_path(base: str) -> Path:
    # 原始强度图：<base>_cellbodies.(tif|tiff|npy)
    for ext in [".tif",".tiff",".npy"]:
        p = DATASET_DIR / f"{base}_cellbodies{ext}"
        if p.exists(): return p
    raise FileNotFoundError(f"no intensity for {base}")

def refine_one(pred_path: Path, min_area: int = 100) -> np.ndarray:
    lbl = _read_any_mask(pred_path)
    base = _base(pred_path.stem)
    img  = _read_any_image(_intensity_path(base))

    ids = np.unique(lbl); ids = ids[ids>0]
    if ids.size==0: return lbl*0

    # Otsu 阈值在“预测区域的像素”上估计
    union = img[lbl>0]
    thr = threshold_otsu(union) if union.size>0 else np.inf

    keep = []
    for k in ids:
        mask = (lbl==k)
        area = int(mask.sum())
        if area < min_area: 
            continue
        meanv = float(img[mask].mean())
        if meanv >= thr:   # gate-off：不使用额外gate，仅按均值与Otsu
            keep.append(k)

    out = np.zeros_like(lbl, np.int32)
    if keep:
        # 重新顺序化标签
        out_mask = np.isin(lbl, keep)
        out[out_mask] = ndi.label(out_mask)[0][out_mask]
    return out

def run():
    for algo, in_dir in CYTO_PRED_DIRS.items():
        if not in_dir.exists(): 
            print(f"[skip] {algo}: {in_dir} not found"); 
            continue
        out_dir = in_dir.parent / "cyto_prediction_refined_mean_otsu_area100"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[refine] {algo}: {in_dir.name} -> {out_dir.name}")
        files = sorted(list(in_dir.glob("*.npy")) + list(in_dir.glob("*.tif")) + list(in_dir.glob("*.tiff")))
        for p in files:
            try:
                ref = refine_one(p, min_area=100)
                out_path = out_dir / (p.stem + ".npy")
                np.save(out_path, ref.astype(np.int32, copy=False))
            except Exception as e:
                print(f"[error] {p.name}: {e}")

if __name__=="__main__":
    run()

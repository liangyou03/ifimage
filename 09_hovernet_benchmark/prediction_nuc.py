#!/usr/bin/env python3
from pathlib import Path
import subprocess, sys, json, csv, os
import numpy as np
import cv2
import tifffile as tiff
from scipy.io import loadmat

# -------- paths --------
ROOT       = Path(__file__).resolve().parent
HOVER      = ROOT / "hover_net"                         # .../09_hovernet_benchmark/hover_net
RUN_INFER  = HOVER / "run_infer.py"

DATA_DIR   = Path("/ihome/jbwang/liy121/ifimage/00_dataset_sep27")  # 原始图像目录
READY_DIR  = ROOT / "hovernet_ready"                                # 预处理后可读的PNG
RAW_OUTDIR = ROOT / "pred_hovernet"                                 # hovernet原始输出
MASK_DIR   = ROOT / "pred_mask"                                     # 收集的npy

MODEL      = Path("/ihome/jbwang/liy121/ifimage/09_hovernet_benchmark/hovernet_original_consep_notype_tf2pytorch.tar")

# -------- hovernet config --------
GPU        = "0"
MODEL_MODE = "original"   # PanNuke=fast
NR_TYPES   = 0        # PanNuke类型数

ALLOWED = {".png",".jpg",".jpeg",".tif",".tiff"}

def to_uint8_rgb(img):
    # img: np.ndarray, any dtype/ndim
    if img.ndim == 2:  # gray -> 3ch
        img = np.stack([img]*3, axis=-1)
    if img.ndim == 3 and img.shape[2] == 4:  # RGBA -> RGB
        img = img[..., :3]
    # normalize to uint8
    if img.dtype != np.uint8:
        vmin, vmax = np.percentile(img, (0.5, 99.5))
        if vmax <= vmin:
            vmax = img.max() if img.max() > 0 else 1.0
            vmin = 0.0
        img = np.clip((img - vmin) / (vmax - vmin) * 255.0, 0, 255).astype(np.uint8)
    return img

def prep_images():
    READY_DIR.mkdir(parents=True, exist_ok=True)
    imgs = [p for p in sorted(DATA_DIR.iterdir()) if p.suffix.lower() in ALLOWED]
    if not imgs:
        print(f"no images found in {DATA_DIR}")
        return 0
    n_ok, n_conv = 0, 0
    for p in imgs:
        out = READY_DIR / (p.stem + ".png")
        # quick path: try OpenCV direct read
        arr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if arr is None or arr.size == 0:
            # fallback robust read via tifffile
            try:
                arr = tiff.imread(str(p))
                arr = to_uint8_rgb(arr)
                # OpenCV写PNG期望BGR
                cv2.imwrite(str(out), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
                n_conv += 1
            except Exception as e:
                print(f"[SKIP] cannot read {p.name}: {e}")
                continue
        else:
            # 确保3通道uint8
            if arr.ndim == 2:
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            if arr.dtype != np.uint8:
                arr = (np.clip(arr, 0, 255)).astype(np.uint8)
            cv2.imwrite(str(out), arr)
        n_ok += 1
    print(f"prepared {n_ok} images, converted {n_conv}, saved to {READY_DIR}")
    return n_ok

def run_hovernet():
    RAW_OUTDIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(RUN_INFER),
        f"--gpu={GPU}",
        f"--nr_types={NR_TYPES}",
        f"--model_path={str(MODEL)}",
        f"--model_mode={MODEL_MODE}",
    ]
    type_info = HOVER / "type_info.json"
    if type_info.exists():
        cmd += ["--type_info_path", str(type_info)]
    cmd += [
        "tile",
        "--input_dir", str(READY_DIR),
        "--output_dir", str(RAW_OUTDIR),
        "--mem_usage", "0.1",
        "--draw_dot",
        "--save_qupath",
    ]
    print("RUN:", " ".join(cmd))
    # 建议：若只想先验证流程，可强制CPU跑一次：
    # env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    # subprocess.run(cmd, check=True, cwd=str(HOVER), env=env)
    subprocess.run(cmd, check=True, cwd=str(HOVER))

def collect_outputs():
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    mat_dir = RAW_OUTDIR / "mat"
    mats = sorted(mat_dir.glob("*.mat"))
    rows = [("file", "n_instances", "saved_npy")]
    summary = {}
    for m in mats:
        d = loadmat(m)
        inst = d.get("inst_map")
        if inst is None:
            print(f"[WARN] inst_map missing: {m.name}")
            continue
        inst = np.asarray(inst).astype(np.int32)
        stem = m.stem
        out_npy = MASK_DIR / f"{stem}_pred_nuclei.npy"
        np.save(out_npy, inst)
        n_inst = int(inst.max())
        rows.append((stem, n_inst, out_npy.name))
        summary[stem] = {"n_instances": n_inst, "npy": out_npy.name}
        print(f"[OK] {stem}: {n_inst} -> {out_npy.name}")
    with open(MASK_DIR / "summary.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)
    with open(MASK_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("done. raw:", RAW_OUTDIR, "collected:", MASK_DIR)

def main():
    assert RUN_INFER.exists(), f"run_infer.py 未找到: {RUN_INFER}"
    assert DATA_DIR.exists(), f"DATA_DIR 不存在: {DATA_DIR}"
    n = prep_images()
    if n == 0:
        return
    run_hovernet()
    collect_outputs()

if __name__ == "__main__":
    main()

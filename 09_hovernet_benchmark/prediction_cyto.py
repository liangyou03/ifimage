#!/usr/bin/env python3
# prediction_cyto_hovernet.py —— DAPI+marker 合成伪RGB→HoVer-Net(seg-only)
from pathlib import Path
import subprocess, sys
import numpy as np, cv2
from scipy.io import loadmat
from utils import SampleDataset, ensure_dir

ROOT      = Path(__file__).resolve().parent
HOVER     = ROOT / "hover_net"
RUN_INFER = HOVER / "run_infer.py"

DATA_DIR   = Path("/ihome/jbwang/liy121/ifimage/00_dataset")
READY_DIR  = ROOT / "hovernet_ready_cyto"
RAW_OUTDIR = ROOT / "pred_hovernet_cyto"
OUT_DIR    = ROOT / "cyto_prediction"

MODEL      = Path("/ihome/jbwang/liy121/ifimage/09_hovernet_benchmark/hovernet_original_consep_notype_tf2pytorch.tar")
GPU, MODEL_MODE, NR_TYPES = "0", "original", 0

def to_uint8(x):
    if x.dtype == np.uint8: return x
    x = x.astype(np.float32); lo, hi = np.percentile(x, (0.5, 99.5)); hi = max(hi, lo + 1e-6)
    return np.clip((x - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)

def fuse_to_rgb(marker: np.ndarray, dapi: np.ndarray) -> np.ndarray:
    m = to_uint8(marker); d = to_uint8(dapi)
    # 2通道→假彩RGB：R=marker, G=dapi, B=(0.5m+0.5d)
    b = ((m.astype(np.uint16) + d.astype(np.uint16)) // 2).astype(np.uint8)
    return np.stack([m, d, b], -1)

def prep_ready_pngs():
    ensure_dir(READY_DIR)
    ds = SampleDataset(DATA_DIR)
    n = 0
    for s in ds:
        s.load_images()
        if s.nuc_chan is None or s.cell_chan is None:  # 需要成对
            continue
        rgb = fuse_to_rgb(s.cell_chan, s.nuc_chan)
        cv2.imwrite(str(READY_DIR / f"{s.base}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        n += 1
    return n

def run_hovernet():
    ensure_dir(RAW_OUTDIR)
    cmd = [sys.executable, str(RUN_INFER),
           f"--gpu={GPU}", f"--nr_types={NR_TYPES}",
           f"--model_path={str(MODEL)}", f"--model_mode={MODEL_MODE}",
           "tile", "--input_dir", str(READY_DIR),
           "--output_dir", str(RAW_OUTDIR),
           "--mem_usage", "0.1", "--draw_dot", "--save_qupath"]
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(HOVER))

def collect_inst_maps():
    ensure_dir(OUT_DIR)
    for m in sorted((RAW_OUTDIR/"mat").glob("*.mat")):
        inst = loadmat(m).get("inst_map")
        if inst is None: continue
        np.save(OUT_DIR / f"{m.stem}_pred_cyto.npy", np.asarray(inst, dtype=np.int32))
        print(f"[OK] {m.stem} -> {m.stem}_pred_cyto.npy")

def main():
    assert RUN_INFER.exists()
    if prep_ready_pngs() == 0:
        print("no paired DAPI+marker."); return
    run_hovernet(); collect_inst_maps()

if __name__ == "__main__":
    main()

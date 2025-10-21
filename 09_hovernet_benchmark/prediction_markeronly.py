#!/usr/bin/env python3
# prediction_markeronly_hovernet.py —— marker-only→HoVer-Net(seg-only)
from pathlib import Path
import subprocess, sys
import numpy as np, cv2
from scipy.io import loadmat
from utils import SampleDataset, ensure_dir

ROOT      = Path(__file__).resolve().parent
HOVER     = ROOT / "hover_net"
RUN_INFER = HOVER / "run_infer.py"

DATA_DIR   = Path("/ihome/jbwang/liy121/ifimage/00_dataset")
READY_DIR  = ROOT / "hovernet_ready_marker"
RAW_OUTDIR = ROOT / "pred_hovernet_marker"
OUT_DIR    = ROOT / "outputs/marker_only"

MODEL      = Path("/ihome/jbwang/liy121/ifimage/09_hovernet_benchmark/hovernet_original_consep_notype_tf2pytorch.tar")
GPU, MODEL_MODE, NR_TYPES = "0", "original", 0

def to_uint8_rgb(gray: np.ndarray) -> np.ndarray:
    if gray.dtype != np.uint8:
        g = gray.astype(np.float32)
        lo, hi = np.percentile(g, (0.5, 99.5)); hi = max(hi, lo + 1e-6)
        g = np.clip((g - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    else:
        g = gray
    return np.stack([g, g, g], -1)

def prep_ready_pngs():
    ensure_dir(READY_DIR)
    ds = SampleDataset(DATA_DIR)
    n = 0
    for s in ds:
        s.load_images()
        if s.cell_chan is None:           # 仅 marker-only
            continue
        rgb = to_uint8_rgb(s.cell_chan)
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
        np.save(OUT_DIR / f"{m.stem}_pred_marker_only.npy", np.asarray(inst, dtype=np.int32))
        print(f"[OK] {m.stem} -> {m.stem}_pred_marker_only.npy")

def main():
    assert RUN_INFER.exists()
    if prep_ready_pngs() == 0:
        print("no marker-only samples."); return
    run_hovernet(); collect_inst_maps()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
prediction_nuclei_mesmer.py — NUCLEI segmentation with Mesmer (DAPI only).
Minimal changes from CellSAM: swap model, stack a zero membrane channel.
"""

from pathlib import Path
import numpy as np
from deepcell.applications import Mesmer
from utils import SampleDataset, ensure_dir
DATA_DIR   = Path("/ihome/jbwang/liy121/ifimage/00_dataset")
OUTPUT_DIR = Path("markeronly_prediction")

# Mesmer knobs
IMAGE_MPP = None
COMPARTMENT = "nuclear"
APP = Mesmer()

def run_mesmer_single(img2d: np.ndarray) -> np.ndarray:
    """Run Mesmer on a single 2D nuclear image → int32 label mask."""
    # 归一化到 [0,1]（保险起见）
    x = img2d.astype(np.float32, copy=False)
    vmax = float(x.max())
    if vmax > 0:
        x /= vmax

    # Mesmer 期望 [batch, H, W, 2]（核在前，膜在后）；DAPI-only -> 膜通道全 0。:contentReference[oaicite:3]{index=3}
    mem = np.zeros_like(x, dtype=np.float32)
    x2 = np.stack([x, mem], axis=-1)[None, ...]  # (1, H, W, 2)

    y = APP.predict(x2, image_mpp=IMAGE_MPP, compartment=COMPARTMENT)  # 可能返回 (1,H,W) 或 (1,H,W,1)
    y0 = y[0]
    if y0.ndim == 3:
        y0 = y0[..., 0]
    return y0.astype(np.int32, copy=False)


def main():
    print("== Nuclei prediction (Mesmer, DAPI-only) ==")
    print(f"DATA_DIR   : {DATA_DIR.resolve()}")
    print(f"OUTPUT_DIR : {OUTPUT_DIR.resolve()}")
    print(f"Mesmer training mpp (for reference): {APP.model_mpp} µm/px")
    ensure_dir(OUTPUT_DIR)

    ds = SampleDataset(DATA_DIR)
    print(f"Found {len(ds)} samples with DAPI.")
    n_ok = 0

    for s in ds:
        try:
            s.load_images()                       # fills s.nuc_chan
            mask = run_mesmer_single(s.cell_chan)
            s.predicted_nuc = mask
            outp = OUTPUT_DIR / f"{s.base}.npy"
            np.save(outp, mask)
            n_ok += 1
            print(f"[OK] {s.base} -> {outp.name} (labels: {int(mask.max())})")
        except Exception as e:
            print(f"[FAIL] {s.base}: {e}")

    print(f"Done: nuclei={n_ok}/{len(ds)}")

if __name__ == "__main__":
    main()

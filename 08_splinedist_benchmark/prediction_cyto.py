#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prediction_cyto.py — CYTOPLASM segmentation with SplineDist (DAPI + marker fused to 1ch).
Minimal script: import data utils, stack [MARKER, DAPI] -> fuse -> run pretrained SplineDist, save masks.
"""

# ---- Force CPU & quiet TF logs (avoid CUDA/TensorRT noise) ----
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from pathlib import Path
import numpy as np

from csbdeep.utils import normalize
from splinedist.models import SplineDist2D

from utils import SampleDataset, ensure_dir  # pure data utils

# ---- config ----
DATA_DIR    = Path("/ihome/jbwang/liy121/ifimage/00_dataset")
OUTPUT_DIR  = Path("cyto_prediction")

# pretrained model:
# A) use a known tag if available in your install, e.g. "paper_dsb2018"
PRETRAINED_NAME = None
# B) or load from local weights: PRETRAINED_ROOT/<model_name>/{config.json,weights_*.h5}
PRETRAINED_ROOT = Path("/ihome/jbwang/liy121/ifimage/08_splinedist_benchmark/splinedist_models/bbbc038_8")

# per-image robust normalization
P_LOWER, P_UPPER = 1, 99.8

# tiling for large images (None = off)
N_TILES = None

# fuse knobs
# mode: "max" | "wmean"
FUSE_MODE   = "wmean"
W_MARKER    = 0.7
W_DAPI      = 0.3


# ---- model init ----
def _pick_sd_model_dir(root: Path):
    if not root.exists():
        return None
    for p in [root] + list(root.rglob("*")):
        if p.is_dir() and (p / "config.json").exists() and any(p.glob("weights*.h5")):
            return p.parent, p.name
    return None

def _load_model() -> SplineDist2D:
    if PRETRAINED_NAME:
        try:
            return SplineDist2D.from_pretrained(PRETRAINED_NAME)
        except Exception:
            pass
    picked = _pick_sd_model_dir(PRETRAINED_ROOT)
    if picked is None:
        raise FileNotFoundError(
            f"No SplineDist model found. Set PRETRAINED_NAME or put weights under {PRETRAINED_ROOT.resolve()}"
        )
    basedir, name = picked
    return SplineDist2D(None, name=name, basedir=str(basedir))

MODEL = _load_model()


# ---- core ----
def _to_float01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    vmax = float(x.max())
    if vmax > 0:
        x /= vmax
    return x

def _fuse_two_channels(img2c: np.ndarray) -> np.ndarray:
    """
    Input: HxWx2, channel order = [MARKER, DAPI]
    Output: HxW (float32), fused
    """
    assert img2c.ndim == 3 and img2c.shape[-1] >= 2, "expect HxWx2"
    m = _to_float01(img2c[..., 0])
    d = _to_float01(img2c[..., 1])
    if FUSE_MODE == "max":
        f = np.maximum(m, d)
    else:  # weighted mean
        f = W_MARKER * m + W_DAPI * d
    return f.astype(np.float32, copy=False)

def run_splinedist_fused(model_obj: SplineDist2D, img2c: np.ndarray) -> np.ndarray:
    """
    Fuse two channels -> single channel, robust normalize, predict instances.
    """
    x = _fuse_two_channels(img2c)
    x = normalize(x, P_LOWER, P_UPPER).astype(np.float32, copy=False)  # (H,W)
    if N_TILES is None:
        labels, _ = model_obj.predict_instances(x)
    else:
        labels, _ = model_obj.predict_instances(x, n_tiles=N_TILES)
    return labels.astype(np.int32, copy=False)


# ---- main ----
def main():
    print("== Cytoplasm prediction (DAPI + marker fused, SplineDist pretrained) ==")
    print(f"DATA_DIR   : {DATA_DIR.resolve()}")
    print(f"OUTPUT_DIR : {OUTPUT_DIR.resolve()}")
    try:
        print(f"MODEL_DIR  : {MODEL.basedir}/{MODEL.name}")
    except Exception:
        pass
    ensure_dir(OUTPUT_DIR)

    ds = SampleDataset(DATA_DIR)
    print(f"Found {len(ds)} samples (marker required).")
    n_ok, n_skip = 0, 0

    for s in ds:
        try:
            s.load_images()  # fills s.cell_chan (marker) and s.nuc_chan (DAPI)
            if s.cell_chan is None or s.nuc_chan is None:
                n_skip += 1
                print(f"[SKIP] {s.base} (missing marker or DAPI)")
                continue

            img2c = s.two_channel_input()  # HxWx2, [MARKER, DAPI]
            outp  = OUTPUT_DIR / f"{s.base}_pred_cyto.npy"
            if outp.exists():
                print(f"[SKIP] {s.base} -> exists")
                continue

            mask = run_splinedist_fused(MODEL, img2c)
            s.predicted_cell = mask
            np.save(outp, mask)
            n_ok += 1
            print(f"[OK] {s.base} -> {outp.name} (labels: {int(mask.max())})")

        except Exception as e:
            print(f"[FAIL] {s.base}: {e!r}")

    print(f"Done: cyto_ok={n_ok}, cyto_skip={n_skip}, total={len(ds)})")


if __name__ == "__main__":
    main()

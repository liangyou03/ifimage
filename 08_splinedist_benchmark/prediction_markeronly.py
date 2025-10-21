#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prediction_markeronly.py — 
"""

from pathlib import Path
import numpy as np
from typing import Optional, Tuple

from csbdeep.utils import normalize
from splinedist.models import SplineDist2D

from utils import SampleDataset, ensure_dir 

DATA_DIR   = Path("/ihome/jbwang/liy121/ifimage/00_dataset")
OUTPUT_DIR = Path("nuclei_prediction_splinedist_pretrained")
PRETRAINED_ROOT = Path("/ihome/jbwang/liy121/ifimage/08_splinedist_benchmark/splinedist_models/bbbc038_8")

P_LOWER, P_UPPER = 1, 99.8


def _pick_sd_model_dir(root: Path) -> Optional[Tuple[Path, str]]:
    if not root.exists():
        return None
    # 允许根目录本身或其一级/多级子目录为模型目录
    for p in [root] + list(root.rglob("*")):
        if p.is_dir():
            cfg = p / "config.json"
            has_w = any(p.glob("weights*.h5"))
            if cfg.exists() and has_w:
                return p.parent, p.name
    return None


def _load_model() -> SplineDist2D:
    picked = _pick_sd_model_dir(PRETRAINED_ROOT)
    if picked is None:
        raise FileNotFoundError(
            f"No {PRETRAINED_ROOT.resolve()} "
        )
    basedir, name = picked
    return SplineDist2D(None, name=name, basedir=str(basedir))


MODEL = _load_model()


def run_splinedist_single(img2d: np.ndarray) -> np.ndarray:
    """Run SplineDist (pretrained) on a single 2D grayscale image → int32 label mask."""
    x = normalize(img2d, P_LOWER, P_UPPER)
    labels, _ = MODEL.predict_instances(x)
    return labels.astype(np.int32, copy=False)


def main():
    print("== Nuclei prediction (DAPI, SplineDist pretrained) ==")
    print(f"DATA_DIR   : {DATA_DIR.resolve()}")
    print(f"OUTPUT_DIR : {OUTPUT_DIR.resolve()}")
    print(f"MODEL_DIR  : {MODEL.basedir}/{MODEL.name}")
    ensure_dir(OUTPUT_DIR)

    ds = SampleDataset(DATA_DIR)  # must provide .load_images(), .nuc_chan, .base
    print(f"Found {len(ds)} samples with DAPI.")
    n_ok = 0

    for s in ds:
        try:
            s.load_images()                 # fills s.nuc_chan
            mask = run_splinedist_single(s.cell_chan)
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

#!/usr/bin/env python3
"""
prediction_nuclei_omnipose.py — Nuclei segmentation with Omnipose (DAPI only).
- Robust to different eval() return signatures (3-tuple or 4-tuple).
- Uses CellposeModel with nclasses=3 to avoid init error on some branches.
- Saves int32 label masks (.npy) per sample.
"""

from pathlib import Path
import numpy as np

from cellpose_omni.models import CellposeModel
from utils import SampleDataset, ensure_dir

DATA_DIR   = Path("/ihome/jbwang/liy121/ifimage/00_dataset")
OUTPUT_DIR = Path("markeronly_prediction")

USE_GPU        = True
MODEL_TYPE     = "cyto"
DIAMETER       = None
MASK_THRESHOLD = 0.0
FLOW_THRESHOLD = 0.4
MIN_SIZE       = 15
BATCH_SIZE     = 4
TILE           = True
TILE_OVERLAP   = 0.1
NORMALIZE      = True
INVERT         = False        # DAPI usually bright-on-dark

APP = CellposeModel(
    gpu=USE_GPU,
    model_type=MODEL_TYPE,
    net_avg=True,
    use_torch=True,
    nclasses=3,
    dim=2,
    omni=True,
)


def _eval_omni(x: np.ndarray):
    """Call APP.eval robustly across versions; always return (masks, flows, styles, diams)."""
    res = APP.eval(
        [x],
        channels=[0, 0],                # grayscale DAPI
        diameter=DIAMETER,
        omni=True,
        mask_threshold=MASK_THRESHOLD,
        flow_threshold=FLOW_THRESHOLD,
        min_size=MIN_SIZE,
        batch_size=BATCH_SIZE,
        normalize=NORMALIZE,
        invert=INVERT,
        tile=TILE,
        tile_overlap=TILE_OVERLAP,
        net_avg=True,
        show_progress=False,
    )
    if isinstance(res, (list, tuple)):
        if len(res) == 4:
            masks, flows, styles, diams = res
        elif len(res) == 3:
            masks, flows, styles = res
            diams = None
        else:
            raise RuntimeError(f"Unexpected eval() return length {len(res)}")
    elif isinstance(res, dict):
        masks = res.get("masks")
        flows = res.get("flows")
        styles = res.get("styles")
        diams = res.get("diameters", None)
    else:
        raise RuntimeError(f"Unexpected eval() return type: {type(res)}")
    return masks, flows, styles, diams


def run_omni_single(img2d: np.ndarray) -> np.ndarray:
    """Run Omnipose on one 2D DAPI image -> int32 label mask."""
    x = img2d
    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[..., 0]
    x = x.astype(np.float32, copy=False)

    masks, flows, styles, diams = _eval_omni(x)
    m = masks[0] if isinstance(masks, (list, tuple)) else masks
    if m is None:
        return np.zeros_like(x, dtype=np.int32)
    return m.astype(np.int32, copy=False)


def main():
    print("== Nuclei prediction (Omnipose, DAPI-only) ==")
    print(f"DATA_DIR   : {DATA_DIR.resolve()}")
    print(f"OUTPUT_DIR : {OUTPUT_DIR.resolve()}")
    ensure_dir(OUTPUT_DIR)

    ds = SampleDataset(DATA_DIR)
    print(f"Found {len(ds)} samples with DAPI.")
    n_ok, n_fail = 0, 0

    for s in ds:
        try:
            s.load_images()
            outp = OUTPUT_DIR / f"{s.base}.npy"
            if outp.exists():
                print(f"[SKIP] {s.base} -> exists")
                continue

            mask = run_omni_single(s.cell_chan)
            np.save(outp, mask)
            print(f"[OK] {s.base} -> {outp.name} (labels: {int(mask.max())})")
            n_ok += 1

        except KeyboardInterrupt:
            print("Interrupted by user.")
            break
        except Exception as e:
            n_fail += 1
            print(f"[FAIL] {s.base}: {e}")

    print(f"Done: nuclei_ok={n_ok}, nuclei_fail={n_fail}, total={len(ds)})")


if __name__ == "__main__":
    main()

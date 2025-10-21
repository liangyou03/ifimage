#!/usr/bin/env python3
# prediction_cyto_lacss.py — cytoplasm via LACSS (marker + DAPI, robust 2C with 1C fallback)

from pathlib import Path
import numpy as np
from lacss.deploy import model_urls
from lacss.deploy.predict import Predictor
from utils import SampleDataset, ensure_dir

DATA_DIR   = Path("/ihome/jbwang/liy121/ifimage/00_dataset")
OUTPUT_DIR = Path("cyto_prediction_lacss")
MODEL_KEY  = "default"  # change to another key if needed

def pick_weights():
    try:
        return model_urls[MODEL_KEY]
    except KeyError:
        # fallback to the first available weight
        return next(iter(model_urls.values()))

def to_2d(arr):
    """Make any array 2D by max-projection over non-spatial dims."""
    a = np.asarray(arr)
    a = np.squeeze(a)
    while a.ndim > 2:
        a = a.max(axis=-1 if a.shape[-1] <= 6 else 0)
    return np.nan_to_num(a.astype(np.float32), copy=False)

def align_stack(a, b):
    """Crop two 2D arrays to common size and stack as HxWx2 float32."""
    h = min(a.shape[0], b.shape[0]); w = min(a.shape[1], b.shape[1])
    a = a[:h, :w]; b = b[:h, :w]
    return np.stack([a, b], axis=-1).astype(np.float32, copy=False)

def pad_to_multiple(img, k=128):
    """Pad HxWxC to multiples of k. Return padded image and original (H,W)."""
    H, W, C = img.shape
    H2 = ((H + k - 1) // k) * k
    W2 = ((W + k - 1) // k) * k
    pad = ((0, H2 - H), (0, W2 - W), (0, 0))
    return np.pad(img, pad, mode="constant"), (H, W)

def norm01(x):
    """Percentile-based 0-1 normalize."""
    lo, hi = np.percentile(x, 1), np.percentile(x, 99)
    if hi <= lo:
        return np.clip(x, 0, None)
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0, 1).astype(np.float32, copy=False)

def merge_to_1c(cell2d, nuc2d, mode="sum"):
    """Merge two 2D channels into 1C image."""
    c = norm01(cell2d); n = norm01(nuc2d)
    if mode == "sum":
        m = c + n
    elif mode == "max":
        m = np.maximum(c, n)
    else:
        m = 0.6 * c + 0.4 * n
    return m[..., None].astype(np.float32, copy=False)

def main():
    print("== Cytoplasm prediction (MARKER + DAPI) with LACSS ==")
    ensure_dir(OUTPUT_DIR)

    predictor = Predictor(pick_weights())
    ds = SampleDataset(DATA_DIR)

    n_ok, n_fallback, n_skip = 0, 0, 0
    for s in ds:
        try:
            s.load_images()
            if s.cell_chan is None or s.nuc_chan is None:
                n_skip += 1
                print(f"[SKIP] {s.base} (missing channel)")
                continue

            cell2 = to_2d(s.cell_chan)
            nuc2  = to_2d(s.nuc_chan)
            img2c = align_stack(cell2, nuc2)
            img2c_p, (H, W) = pad_to_multiple(img2c, 128)

            try:
                out = predictor.predict(img2c_p, output_type="label")
                mask = out["pred_label"].astype(np.int32, copy=False)[:H, :W]
            except Exception as e:
                # fallback: merge to 1C and retry
                n_fallback += 1
                img1c = merge_to_1c(cell2, nuc2, mode="sum")
                img1c_p, (H1, W1) = pad_to_multiple(img1c, 128)
                out = predictor.predict(img1c_p, output_type="label")
                mask = out["pred_label"].astype(np.int32, copy=False)[:H1, :W1]
                print(f"[FALLBACK→1C] {s.base}: {e}")

            np.save(OUTPUT_DIR / f"{s.base}_pred_cyto.npy", mask)
            n_ok += 1
            print(f"[OK] {s.base} labels={int(mask.max())}")

        except Exception as e:
            print(f"[FAIL] {s.base}: {e}")

    print(f"Done: ok={n_ok}, fallback={n_fallback}, skip={n_skip}, total={len(ds)})")

if __name__ == "__main__":
    main()

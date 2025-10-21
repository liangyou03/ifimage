#!/usr/bin/env python3
# prediction_markeronly_lacss.py — cytoplasm via LACSS (marker only)

from pathlib import Path
import numpy as np
from lacss.deploy import model_urls
from lacss.deploy.predict import Predictor
from utils import SampleDataset, ensure_dir

DATA_DIR   = Path("/ihome/jbwang/liy121/ifimage/00_dataset")
OUTPUT_DIR = Path("outputs_lacss/marker_only")

def main():
    print("== Cytoplasm prediction (Marker only) with LACSS ==")
    ensure_dir(OUTPUT_DIR)

    ds = SampleDataset(DATA_DIR)
    predictor = Predictor(model_urls["default"])

    n_ok, n_skip = 0, 0
    for s in ds:
        try:
            s.load_images()
            if s.cell_chan is None:
                n_skip += 1
                print(f"[SKIP] {s.base} (no marker)")
                continue
            img1c = s.cell_chan[..., None]        # HxWx1
            out = predictor.predict(img1c, output_type="label")
            mask = out["pred_label"].astype(np.int32, copy=False)
            np.save(OUTPUT_DIR / f"{s.base}_pred_marker_only.npy", mask)
            n_ok += 1
            print(f"[OK] {s.base} (labels: {int(mask.max())})")
        except Exception as e:
            print(f"[FAIL] {s.base}: {e}")

    print(f"Done: ok={n_ok}, skip={n_skip}, total={len(ds)}")

if __name__ == "__main__":
    main()

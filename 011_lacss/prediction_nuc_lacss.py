#!/usr/bin/env python3
# prediction_nuc_lacss.py — nuclei segmentation via LACSS (DAPI only)

from pathlib import Path
import numpy as np
from lacss.deploy import model_urls
from lacss.deploy.predict import Predictor
import numpy as np
from utils import SampleDataset, ensure_dir  # reuse your data utils

DATA_DIR   = Path("/ihome/jbwang/liy121/ifimage/00_dataset")
OUTPUT_DIR = Path("pred_mask_lacss")

def main():
    print("== Nuclei prediction (DAPI) with LACSS ==")
    ensure_dir(OUTPUT_DIR)

    ds = SampleDataset(DATA_DIR)
    predictor = Predictor(model_urls["default"])

    n_ok = 0
    for s in ds:
        try:
            s.load_images()               # expects s.nuc_chan as 2D array
            img = s.nuc_chan[..., None]   # make it HxWx1
            out = predictor.predict(img, output_type="label")
            mask = out["pred_label"].astype(np.int32, copy=False)
            np.save(OUTPUT_DIR / f"{s.base}_pred_nuclei.npy", mask)
            n_ok += 1
            print(f"[OK] {s.base} (labels: {int(mask.max())})")
        except Exception as e:
            print(f"[FAIL] {s.base}: {e}")

    print(f"Done: nuclei={n_ok}/{len(ds)}")

if __name__ == "__main__":
    main()

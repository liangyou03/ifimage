#!/usr/bin/env python3
"""
prediction_nuclei_watershed.py — Nuclei segmentation baseline with watershed.
Input: DAPI-only images
Output: label mask (.npy)
"""

from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from skimage import filters, segmentation, measure, feature, morphology
from utils import SampleDataset, ensure_dir

# ---- config ----
DATA_DIR   = Path("/ihome/jbwang/liy121/ifimage/00_dataset")
OUTPUT_DIR = Path("markeronly_prediction")

# watershed tuning knobs
MIN_DISTANCE = 20   # distance between seeds
SIGMA        = 1.5  # smoothing for distance map
MIN_SIZE     = 100   # remove small objects


def run_watershed_single(img2d: np.ndarray) -> np.ndarray:
    """Run watershed on a single 2D nuclear image → int32 label mask."""
    x = img2d.astype(np.float32, copy=False)
    vmax = float(x.max())
    if vmax > 0:
        x /= vmax

    # Step 1: threshold to get foreground
    thresh = filters.threshold_otsu(x)
    mask = x > thresh
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=np.int32)

    # Step 2: distance transform + smoothing
    dist = ndi.distance_transform_edt(mask)
    dist = filters.gaussian(dist, sigma=SIGMA)

    # Step 3: local maxima as seeds
    coords = feature.peak_local_max(dist, labels=mask, min_distance=MIN_DISTANCE)
    marker_mask = np.zeros_like(mask, dtype=bool)
    marker_mask[tuple(coords.T)] = True
    markers = measure.label(marker_mask)
    if markers.max() == 0:
        return mask.astype(np.int32)

    # Step 4: watershed
    labels = segmentation.watershed(-dist, markers, mask=mask)

    # Step 5: remove tiny fragments
    labels = morphology.remove_small_objects(labels, min_size=MIN_SIZE)

    return labels.astype(np.int32, copy=False)


def main():
    print("== Nuclei prediction (Watershed, DAPI-only) ==")
    print(f"DATA_DIR   : {DATA_DIR.resolve()}")
    print(f"OUTPUT_DIR : {OUTPUT_DIR.resolve()}")
    ensure_dir(OUTPUT_DIR)

    ds = SampleDataset(DATA_DIR)
    print(f"Found {len(ds)} samples with DAPI.")
    n_ok, n_fail = 0, 0

    for s in ds:
        try:
            s.load_images()
            mask = run_watershed_single(s.cell_chan)
            outp = OUTPUT_DIR / f"{s.base}.npy"
            np.save(outp, mask)
            n_ok += 1
            print(f"[OK] {s.base} -> {outp.name} (labels: {int(mask.max())})")
        except Exception as e:
            n_fail += 1
            print(f"[FAIL] {s.base}: {e}")

    print(f"Done: nuclei_ok={n_ok}, nuclei_fail={n_fail}, total={len(ds)})")                           


if __name__ == "__main__":
    main()
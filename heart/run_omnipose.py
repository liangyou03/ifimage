#!/usr/bin/env python3
"""
Generic Omnipose/Cellpose segmentation script for arbitrary data.

This script is a convenience wrapper around the Cellpose‑Omnipose API that
automatically handles images with varying numbers of channels.  It inspects
each input image, ensures that a two‑channel array is available (creating a
"pseudo" second channel when necessary), normalizes the data, runs
segmentation with a pretrained model, and saves the resulting masks.  A cell
count for each image is printed to stdout for quick inspection.

Key features:

* Accepts single‑channel or multi‑channel input.  If an image has only
  one channel, that channel is duplicated to form a two‑channel stack.  If
  the image has more than two channels, the first two are used by default.
* Normalizes each image using the 99th‑percentile method recommended by
  Omnipose (via ``transforms.normalize99``).  This improves segmentation
  robustness.
* Uses a pretrained model specified by ``MODEL_TYPE``.  By default this is
  ``"cyto2_omni"``, which is the Omnipose variant trained on the cyto2
  dataset, but this can be changed to any of the models listed in the
  Omnipose documentation (e.g. ``"cyto2"``, ``"cyto2_omni"``, ``"bact_phase_omni"``, etc.).
* Handles GPU/CPU selection automatically via the ``use_gpu`` function.
* Saves each segmentation mask as a NumPy ``.npy`` file into the output
  directory, preserving the directory structure of the input (useful if
  processing nested folders).

To use this script, update the ``PROCESSED_DIR`` and ``OUTPUT_DIR``
constants below to point to your data and desired output location.  Then run

    python run_omnipose_generic.py

The script will process all images with the extensions ``.tif``, ``.tiff``,
``.png``, or ``.jpg`` found recursively under ``PROCESSED_DIR``.

Note:  While duplicating a single channel does satisfy the model's
requirements for two channels, the biological relevance of the segmentation
may be limited.  For brightfield or phase contrast images, consider using
an affinity model or training a custom network.  See the Omnipose
documentation for details.
"""

import os
from pathlib import Path
from typing import Iterable

import numpy as np
import tifffile

from cellpose_omni import models, core, transforms


# -----------------------------------------------------------------------------
# Configuration section
# -----------------------------------------------------------------------------

# Path to the directory containing images to be segmented.  Update this to
# match the location of your data.  The script will search recursively
# through subdirectories.
PROCESSED_DIR = Path("/ihome/jbwang/liy121/ifimage/heart/processed")
OUTPUT_DIR = Path("/ihome/jbwang/liy121/ifimage/heart/benchmark_results/omnipose_predictions")


# Choose which pretrained model to use.  See the Omnipose documentation
# (https://omnipose.readthedocs.io/models.html) for available options.  The
# default ``cyto2_omni`` model is trained on cytoplasmic fluorescence images
# and is generally applicable to many microscopy modalities.  You can also
# specify ``cyto2`` for the Cellpose version or other names such as
# ``bact_phase_omni``, ``worm_omni``, etc.
MODEL_TYPE = "cyto2_omni"

# Approximate cell diameter in pixels.  Adjust this depending on your
# magnification.  A value of ``None`` lets Cellpose/Omnipose estimate it
# automatically, but specifying a diameter can improve speed and accuracy.
DIAMETER = 30

# Threshold for mask probability.  Lower values are more permissive and
# produce more masks.  Values around 0 to −1 are useful when segmenting
# low‑contrast images; positive values are stricter.  This list applies a
# constant threshold to all images.  You can make this a list if you wish to
# experiment with different thresholds.
MASK_THRESHOLD = -1.0

# Threshold for flow error; set to 0.0 for speed but use 0.4 to clean up
# spurious masks if necessary.
FLOW_THRESHOLD = 0.0

# Minimum size of objects to keep (in pixels).  Objects smaller than this
# value will be removed from the output mask.  Increase to remove noise.
MIN_SIZE = 15

# Whether to run dynamics on a rescaled grid.  Setting this to ``False``
# processes the image at its original resolution; ``True`` may improve
# accuracy at the cost of speed.
RESAMPLE = False

# Whether to run the model on the GPU if available.  ``core.use_gpu()``
# internally checks for a CUDA device and returns a boolean.  You can
# override this by setting ``USE_GPU = False`` if you want to force CPU.
USE_GPU = core.use_gpu()


def find_images(base_dir: Path) -> Iterable[Path]:
    """Recursively yield image files from ``base_dir``.

    The function looks for files with extensions commonly used in
    microscopy: ``.tif``, ``.tiff``, ``.png``, and ``.jpg``.  Feel free to
    extend this list if your data uses other formats.
    """
    exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
    for path in base_dir.rglob("*"):
        if path.suffix.lower() in exts and path.is_file():
            yield path


def to_two_channels(img: np.ndarray) -> np.ndarray:
    """Ensure that ``img`` has shape (2, H, W).

    If ``img`` is 2‑D (H, W), its single channel is duplicated to form
    (2, H, W).  If ``img`` is 3‑D with a channel dimension of size C, the
    first two channels are selected and reordered to shape (2, H, W).  If
    fewer than two channels are available, the last available channel is
    duplicated.  If ``img`` has a different arrangement (e.g. channel
    dimension first), this function assumes channels are last; modify as
    necessary for your data.
    """
    if img.ndim == 2:
        # grayscale image: duplicate the single channel
        return np.stack((img, img))
    elif img.ndim == 3:
        h, w, c = img.shape
        if c >= 2:
            ch1 = img[..., 0]
            ch2 = img[..., 1]
        else:
            ch1 = img[..., 0]
            ch2 = img[..., 0]
        return np.stack((ch1, ch2))
    elif img.ndim == 4:
        # if channel axis comes first (C, H, W) or last (H, W, C)?  For 4D we
        # interpret it as (Z, H, W, C) and take the first slice.  This is
        # unlikely for typical segmentation tasks but handled for completeness.
        if img.shape[-1] <= 4:
            # assume channels last: pick first z slice
            slice0 = img[0]
            return to_two_channels(slice0)
        else:
            # assume channels first: pick first two channels
            c, h, w, _ = img.shape
            ch1 = img[0]
            ch2 = img[1] if c >= 2 else img[0]
            return np.stack((ch1, ch2))
    else:
        raise ValueError(f"Unsupported image shape {img.shape}")


def main() -> None:
    """Segment all images under ``PROCESSED_DIR`` using Omnipose/Cellpose.

    Creates the output directory if it does not exist and writes one
    ``.npy`` file per input image with the same filename stem appended
    with ``_mask.npy``.  Prints a running summary of successes and empty
    segmentations to stdout.
    """
    if not PROCESSED_DIR.exists():
        raise FileNotFoundError(f"Input directory does not exist: {PROCESSED_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize model
    model = models.CellposeModel(
        gpu=USE_GPU,
        model_type=MODEL_TYPE,
    )
    print(f"Using model '{MODEL_TYPE}', GPU: {USE_GPU}")

    n_ok = 0
    n_empty = 0
    n_fail = 0

    for img_path in find_images(PROCESSED_DIR):
        try:
            img = tifffile.imread(img_path)
            # Convert to 2‑channel (2, H, W)
            img2 = to_two_channels(img)
            # Convert to float32
            img2 = img2.astype(np.float32, copy=False)
            # Normalize using Omnipose's recommended method
            img2 = transforms.normalize99(img2, omni=True)

            # Channels specification: [1,2] means cytoplasm channel=1, nuclear=2
            # on the provided two‑channel array; this matches the order of
            # ``to_two_channels``.  Cellpose uses 1‑based indexing.
            channels = [1, 2]
            # Evaluate on a list of one image
            masks_list, flows_list, styles_list = model.eval(
                [img2],
                channels=channels,
                diameter=DIAMETER,
                mask_threshold=MASK_THRESHOLD,
                flow_threshold=FLOW_THRESHOLD,
                min_size=MIN_SIZE,
                resample=RESAMPLE,
                verbose=0,
            )
            mask = masks_list[0]
            # Count the number of objects (labels > 0)
            cell_count = int(mask.max())

            # Derive output file path preserving relative structure
            # 保持原有子目录结构
            rel_path = img_path.relative_to(PROCESSED_DIR)

            # 输出目录（例如 OUTPUT_DIR / RA）
            out_dir = OUTPUT_DIR / rel_path.parent
            out_dir.mkdir(parents=True, exist_ok=True)

            # 输出文件名：与其他算法统一使用 _pred.npy
            out_name = rel_path.stem + "_pred.npy"
            out_path = out_dir / out_name

            # 保存 mask
            np.save(out_path, mask.astype(np.int32))

            print(f"{rel_path}: {cell_count} objects")
            if cell_count == 0:
                n_empty += 1
            n_ok += 1
        except Exception as e:
            print(f"FAILED {img_path}: {e}")
            n_fail += 1

    # Summary
    total = n_ok + n_fail
    print("\nSummary:")
    print(f"Processed: {total}")
    print(f"Successful: {n_ok}")
    print(f"Empty masks: {n_empty}")
    print(f"Failed: {n_fail}")
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

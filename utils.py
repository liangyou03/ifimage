#!/usr/bin/env python3
"""
utils.py — model-agnostic data utilities for IF image projects.

Provides:
  • Sample: per-case container with attributes:
      cell_type, nuc_chan, cell_chan, nuc_scribble, marker_scribble,
      predicted_nuc, predicted_cell
  • SampleDataset: index a folder into Sample objects
  • safe_read(): TIFF/TIF → SINGLE-CHANNEL 2D float32 (first channel/plane only)
  • robust_norm(), robust_norm_per_channel()
  • ensure_dir()
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional
import re
import numpy as np
import tifffile as tiff

__all__ = [
    "Sample",
    "SampleDataset",
    "ensure_dir",
    "safe_read",
    "robust_norm",
    "robust_norm_per_channel",
]

# ---- filename conventions ----
TIFF_RE = re.compile(r".+\.(tif|tiff)$", re.IGNORECASE)

# ---- IO helpers (no model imports here) ----
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def safe_read(path: Path) -> np.ndarray:
    """
    Read a TIFF/TIF and return a SINGLE-CHANNEL 2D float32 image.
    If the image has channels/planes (C/Z/T), keep ONLY the first along extra dims
    until shape is (H, W).
    """
    arr = tiff.imread(path)
    if arr.ndim == 2:
        return arr.astype(np.float32, copy=False)
    a = arr
    while a.ndim > 2:
        # If last axis looks like channel (<=4), drop it; otherwise drop first axis (Z/T).
        if a.shape[-1] <= 4:
            a = a[..., 0]
        else:
            a = a[0]
    return a.astype(np.float32, copy=False)

def robust_norm(img: np.ndarray) -> np.ndarray:
    """
    Normalize a 2D image to [0,1] using 1st–99th percentiles.
    Falls back to min–max; returns zeros if flat.
    """
    img = img.astype(np.float32, copy=False)
    lo, hi = np.percentile(img, (1, 99))
    if hi <= lo:
        lo, hi = float(img.min()), float(img.max())
        if hi <= lo:
            return np.zeros_like(img, dtype=np.float32)
    img = (img - lo) / (hi - lo)
    return np.clip(img, 0.0, 1.0).astype(np.float32, copy=False)

def robust_norm_per_channel(img_hw_c: np.ndarray) -> np.ndarray:
    """
    Normalize each channel of an HxWxC image independently to [0,1].
    """
    out = np.empty_like(img_hw_c, dtype=np.float32)
    for c in range(img_hw_c.shape[-1]):
        out[..., c] = robust_norm(img_hw_c[..., c])
    return out

# ---- data objects ----
@dataclass
class Sample:
    """
    Per-sample data container identified by <base>.

    Attributes (filled by loader/predictor later):
      cell_type, nuc_chan, cell_chan, nuc_scribble, marker_scribble,
      predicted_nuc, predicted_cell
    """
    base: str
    dapi_path: Path
    marker_path: Optional[Path] = None
    nuc_scribble_path: Optional[Path] = None        # <base>_dapimultimask.npy
    marker_scribble_path: Optional[Path] = None     # <base>_cellbodies.npy

    # Derived + loaded fields
    cell_type: str = field(init=False)
    nuc_chan: Optional[np.ndarray] = field(default=None, init=False)     # 2D float32 [0,1]
    cell_chan: Optional[np.ndarray] = field(default=None, init=False)    # 2D float32 [0,1]
    nuc_scribble: Optional[np.ndarray] = field(default=None, init=False) # 2D any dtype
    marker_scribble: Optional[np.ndarray] = field(default=None, init=False)
    predicted_nuc: Optional[np.ndarray] = field(default=None, init=False)  # int labels
    predicted_cell: Optional[np.ndarray] = field(default=None, init=False) # int labels

    def __post_init__(self):
        # e.g., "gfap_6390" → "gfap"
        self.cell_type = self.base.split("_", 1)[0].lower()

    # --- pure data loading (no model here) ---
    def load_images(self) -> None:
        """Load and normalize DAPI (required) and MARKER (optional) to [0,1]."""
        if self.nuc_chan is None:
            self.nuc_chan = robust_norm(safe_read(self.dapi_path))
        if (self.marker_path is not None) and (self.cell_chan is None):
            self.cell_chan = robust_norm(safe_read(self.marker_path))

    def load_scribbles(self) -> None:
        """Load optional scribbles if present."""
        if self.nuc_scribble is None and self.nuc_scribble_path and self.nuc_scribble_path.exists():
            self.nuc_scribble = np.load(self.nuc_scribble_path)
        if self.marker_scribble is None and self.marker_scribble_path and self.marker_scribble_path.exists():
            self.marker_scribble = np.load(self.marker_scribble_path)

    def two_channel_input(self) -> np.ndarray:
        """
        Return HxWx2 array for cytoplasm segmentation: [MARKER, DAPI].
        Requires both channels loaded and shapes to match.
        """
        if self.cell_chan is None or self.nuc_chan is None:
            raise RuntimeError(f"[{self.base}] call load_images() first (need both channels)")
        if self.cell_chan.shape != self.nuc_chan.shape:
            raise ValueError(f"[{self.base}] shape mismatch: marker {self.cell_chan.shape} vs dapi {self.nuc_chan.shape}")
        return np.stack([self.cell_chan, self.nuc_chan], axis=-1).astype(np.float32, copy=False)

class SampleDataset:
    """
    Index a folder into Sample objects.

    Conventions:
      DAPI   : <base>.tif[f]
      MARKER : <base>_marker.tif[f]
      Scribs : <base>_dapimultimask.npy (nuc), <base>_cellbodies.npy (marker)
    Only bases with a DAPI image are kept.
    """

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.samples: List[Sample] = self._index()

    def _index(self) -> List[Sample]:
        table: Dict[str, Dict[str, Path]] = {}
        for p in sorted(self.data_dir.iterdir()):
            if p.suffix.lower() == ".npy":
                nm = p.name
                if nm.endswith("_dapimultimask.npy"):
                    base = nm[:-len("_dapimultimask.npy")]
                    table.setdefault(base, {})["nuc_scrib"] = p
                elif nm.endswith("_cellbodies.npy"):
                    base = nm[:-len("_cellbodies.npy")]
                    table.setdefault(base, {})["marker_scrib"] = p
                continue

            if p.is_file() and TIFF_RE.match(p.name):
                low = p.name.lower()
                if low.endswith("_marker.tif") or low.endswith("_marker.tiff"):
                    base = p.stem[:-7]  # strip "_marker"
                    table.setdefault(base, {})["marker"] = p
                else:
                    base = p.stem
                    table.setdefault(base, {})["dapi"] = p

        out: List[Sample] = []
        for base, d in sorted(table.items()):
            if "dapi" not in d:
                continue
            out.append(Sample(
                base=base,
                dapi_path=d["dapi"],
                marker_path=d.get("marker"),
                nuc_scribble_path=d.get("nuc_scrib"),
                marker_scribble_path=d.get("marker_scrib"),
            ))
        return out

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[Sample]:
        return iter(self.samples)
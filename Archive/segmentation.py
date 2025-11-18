# segmentaion.py  – self-contained segmentation utilities
#
# * SegResult         – uniform result object
# * FindMarker        – marker-positive detection algorithms
# * NUC_SEG_METHODS   – nucleus segmentation wrappers
# * CYTO_SEG_METHODS  – cytoplasm segmentation wrappers
#
# Usage:
#     from segmentaion import NUC_SEG_METHODS, CYTO_SEG_METHODS
#     res = NUC_SEG_METHODS['cellposeSAM'](dapi_img)
#     mask = res.mask

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

from scipy.ndimage import distance_transform_edt
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.measure import label, regionprops
from skimage.feature import peak_local_max
from cellpose import models
from stardist.matching import matching


# ─────────────────────────── result object ───────────────────────────────────
@dataclass
class SegResult:
    mask:   np.ndarray
    method: str
    stats:  Dict[str, Any] = field(default_factory=dict)
    ok:     bool = True
    error:  Optional[str] = None


__all__ = ['SegResult', 'FindMarker', 'NUC_SEG_METHODS', 'CYTO_SEG_METHODS']


# ───────────────────────── heavy-model cache ─────────────────────────────────
_CACHE: Dict[str, Any] = {}

def _lazy_cp_sam(gpu: bool = True):
    if 'cp_sam' not in _CACHE:
        print("[_lazy_cp_sam] Initializing Cellpose SAM model…")        # ← 新增
        _CACHE['cp_sam'] = models.CellposeModel(gpu=gpu)
    else:
        print("[_lazy_cp_sam] Reusing cached Cellpose SAM model")       # ← 新增
    return _CACHE['cp_sam']



# ───────────────────────────── FindMarker ────────────────────────────────────
class Thresholder:
    @staticmethod
    def otsu_threshold(intensities):
        from skimage.filters import threshold_otsu
        return threshold_otsu(intensities)

    @staticmethod
    def gmm_threshold(intensities, n_components=2):
        from sklearn.mixture import GaussianMixture
        gmm = GaussianMixture(n_components=n_components, random_state=0)
        gmm.fit(intensities.reshape(-1, 1))
        return np.sort(gmm.means_.ravel())[0]


class FindMarker:
    def __init__(self, max_expansion=10, threshold_method='otsu'):
        self.max_expansion = max_expansion
        self.threshold_method = threshold_method
        self.thresholder = Thresholder()

    # ───────── helper ────────
    def _apply_threshold(self, values):
        if self.threshold_method == 'otsu':
            return self.thresholder.otsu_threshold(values)
        elif self.threshold_method == 'gmm':
            return self.thresholder.gmm_threshold(values)
        else:
            raise ValueError('unsupported threshold method')

    def _avg_intensity(self, img, mask):
        labels = np.unique(mask)[1:]
        tot = ndi.sum(img, labels=mask, index=labels)
        area = ndi.sum(np.ones_like(img), labels=mask, index=labels)
        return tot / area

    # ───────── algorithms ────────
    def cell_expansion(self, marker, dapi_multi):
        props = regionprops(dapi_multi)
        centroids = np.array([p.centroid for p in props])
        markers = np.zeros_like(dapi_multi, int)
        for i, (y, x) in enumerate(centroids):
            markers[int(round(y)), int(round(x))] = i + 1

        dist, (iy, ix) = distance_transform_edt(markers == 0, return_indices=True)
        limited = np.zeros_like(dapi_multi, int)
        within = dist <= self.max_expansion
        limited[within] = markers[iy[within], ix[within]]

        meanI = self._avg_intensity(marker, limited)
        thr = self._apply_threshold(meanI)
        labels = np.unique(limited)[1:]
        posLab = labels[meanI > thr]
        posMask = np.where(np.isin(limited, posLab), limited, 0)
        return len(labels), len(posLab) / len(labels), len(posLab), limited, posMask

    def watershed_cellpose_with_dapi(self, marker, dapi_multi):
        cyto = watershed(marker.astype(float), markers=dapi_multi)
        meanI = self._avg_intensity(marker, cyto)
        thr = self._apply_threshold(meanI)
        labels = np.unique(cyto)[1:]
        posLab = labels[meanI > thr]
        posMask = np.where(np.isin(cyto, posLab), cyto, 0)
        return len(labels), len(posLab) / len(labels), len(posLab), cyto, posMask

    def watershed_only_cyto(self, marker, _dapi=None):
        coords = peak_local_max(marker, min_distance=30, threshold_abs=0.3)
        seeds = np.zeros_like(marker, bool)
        seeds[tuple(coords.T)] = True
        cyto = watershed(marker.astype(float), markers=label(seeds))
        meanI = self._avg_intensity(marker, cyto)
        thr = self._apply_threshold(meanI)
        labels = np.unique(cyto)[1:]
        posLab = labels[meanI > thr]
        posMask = np.where(np.isin(cyto, posLab), cyto, 0)
        return len(labels), len(posLab) / len(labels), len(posLab), cyto, posMask

    def cellpose_only_cyto(self, marker, _dapi=None):
        mdl = _lazy_cp_sam(True)
        marker_f = marker.astype(np.float32)
        if marker_f.max() > 1.5:
            marker_f /= 255.0
        masks, *_ = mdl.eval(marker_f, channels=[0, 0])
        return masks

    def cellpose2chan(self, marker, dapi_multi):
        dapi_f = marker.astype(np.float32)
        if dapi_f.max() > 1.5:
            dapi_f /= 255.0
        stacked = np.stack([dapi_f, dapi_multi.astype(float), np.zeros_like(dapi_f)], 2)
        mdl = _lazy_cp_sam(True)
        cyto, *_ = mdl.eval(stacked, channels=[2, 1])
        meanI = self._avg_intensity(marker, cyto)
        thr = self._apply_threshold(meanI)
        labels = np.unique(cyto)[1:]
        posLab = labels[meanI > thr]
        return np.where(np.isin(cyto, posLab), cyto, 0)

    def compute_confusion_matrix_stardist(self, gt, pred, iou_threshold=0.01):
        return matching(gt.astype(np.uint32), pred.astype(np.uint32), thresh=iou_threshold)


# ─────────────────── nucleus segmentation wrappers ───────────────────────────
def nuc_cellpose_sam(dapi, *, gpu=True) -> SegResult:
    try:
        mask, *_ = _lazy_cp_sam(gpu).eval(dapi, diameter=None, channels=[0, 0])
        return SegResult(mask=mask.astype(np.uint32), method='cellposeSAM',
                         stats={'n_nuclei': int(mask.max())})
    except Exception as e:
        return SegResult(mask=np.zeros_like(dapi, np.uint32),
                         method='cellposeSAM', ok=False, error=str(e))


def nuc_watershed(dapi) -> SegResult:
    try:
        dist = distance_transform_edt(dapi > 0)
        seeds = (dapi > 0).astype(int)
        mask = watershed(-dist, seeds, mask=dapi > 0)
        return SegResult(mask=mask.astype(np.uint32), method='watershed',
                         stats={'n_nuclei': int(mask.max())})
    except Exception as e:
        return SegResult(mask=np.zeros_like(dapi, np.uint32),
                         method='watershed', ok=False, error=str(e))


NUC_SEG_METHODS = {
    'cellposeSAM': nuc_cellpose_sam,
    'watershed': nuc_watershed,
}


# ────────────────── cytoplasm segmentation wrappers ──────────────────────────
def _mk_wrapper(fname):
    def _wrap(marker, dapi_multi, **kw):
        fm = FindMarker(**kw)
        try:
            res = getattr(fm, fname)(marker, dapi_multi)
            if fname.startswith('cellpose'):
                mask, stats = (res, {})
            else:
                tot, ratio, num, _cyto, mask = res
                stats = {'total_cell_num': int(tot),
                         'marker_cell_ratio': float(ratio),
                         'marker_cell_num': int(num)}
            return SegResult(mask=mask.astype(np.uint32), method=fname, stats=stats)
        except Exception as e:
            h, w = marker.shape[:2]
            return SegResult(mask=np.zeros((h, w), np.uint32),
                             method=fname, ok=False, error=str(e))
    return _wrap


CYTO_SEG_METHODS = {
    'cell_expansion': _mk_wrapper('cell_expansion'),
    'watershed': _mk_wrapper('watershed_cellpose_with_dapi'),
    'watershed_only_cyto': _mk_wrapper('watershed_only_cyto'),
    'cellpose': _mk_wrapper('cellpose_only_cyto'),
    'cellpose2chan': _mk_wrapper('cellpose2chan'),
}

import os, glob, warnings
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import skimage.io as skio
from stardist.matching import matching
# NOTE: fix the typo: was `segmentaion`
from segmentation import NUC_SEG_METHODS, CYTO_SEG_METHODS

# ---------- warnings ----------
class MissingComponentWarning(UserWarning):
    """Raised when an expected image/mask/component is missing or empty."""

warnings.simplefilter("always", MissingComponentWarning)  # make sure they show

# ---------- io helper ----------
def safe_read(path):
    """Read .npy or image; warn on failure or empty content."""
    try:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".npy":
            arr = np.load(path, allow_pickle=False)
        else:
            arr = skio.imread(path)
    except Exception as e:
        warnings.warn(f"Read failed: {path} -> {e}", category=MissingComponentWarning)
        return None

    if arr is None or (isinstance(arr, np.ndarray) and arr.size == 0):
        warnings.warn(f"Empty read: {path}", category=MissingComponentWarning)
        return None
    return arr


class ImageSample:
    def __init__(self, sample_id, manual_cell_count=None):
        self.sample_id = sample_id
        self.celltype = None
        self.manual_cell_count = manual_cell_count

        # raw channels
        self.dapi   = None
        self.marker = None

        # legacy/GT
        self.dapi_mask = None                # legacy
        self.dapi_multi_mask = None          # GT nuclei (instance labels)
        self.cellbodies_mask = None          # legacy
        self.cellbodies_multimask = None     # GT cell bodies (instance labels)

        # predicted / working
        self.masks = {                       # nuclei-related predictions
            "cyto": None, "cyto2": None, "cyto3": None,
            "cellposeSAM": None, "watershed": None, "StarDist2D": None
        }
        self.cyto_positive_masks = {         # cell-body predictions
            "cellpose": None,
            "watershed": None,
            "cell_expansion": None,
            "watershed_only_cyto": None,
            "cellpose2chan": None,
        }

        # misc
        self.nuc_metadata  = {}  # {method: SegResult-like}
        self.cyto_metadata = {}
        self.pos_nuclei_mask = None
        self.limited_mask = None

    def __str__(self):
        return f"ImageSample:{self.celltype}_{self.sample_id}"

    # --------- adders ----------
    def add_image_file(self, file):
        filename = os.path.basename(file).lower()
        arr = safe_read(file)
        if arr is None:
            return
        if "dapimask" in filename:
            self.dapi_mask = arr
        elif "marker." in filename:
            self.marker = arr
        elif "dapimultimask" in filename:
            self.dapi_multi_mask = arr
            if isinstance(arr, np.ndarray) and arr.max() == 0:
                warnings.warn(f"[{self.sample_id}] dapi_multi_mask is all zeros.",
                              MissingComponentWarning)
        elif "cellbodiesmultimask" in filename or "cellbodies.npy" in filename:
            self.cellbodies_multimask = arr
            if isinstance(arr, np.ndarray) and arr.max() == 0:
                warnings.warn(f"[{self.sample_id}] cellbodies_multimask is all zeros.",
                              MissingComponentWarning)
        elif "cellbodies.tif" in filename:
            self.cellbodies_mask = arr
        else:
            # heuristic: files named like "{celltype}_{sampleid}.*" are the DAPI
            parts = os.path.splitext(os.path.basename(file))[0].split("_")
            if len(parts) == 2:
                self.dapi = arr

    def add_mask_file(self, file):
        filename = os.path.basename(file)
        arr = safe_read(file)
        if arr is None:
            return
        name = os.path.splitext(filename)[0]
        # decide which dict to drop into by common names
        if "cyto2" in name:
            self.masks["cyto2"] = arr
        elif "cyto3" in name:
            self.masks["cyto3"] = arr
        elif "watershed" in name and "cyto" not in name:
            self.masks["watershed"] = arr
        elif "stardist2d" in name.lower():
            self.masks["StarDist2D"] = arr
        elif "cyto" in name:
            self.masks["cyto"] = arr
        # empty array check
        if isinstance(arr, np.ndarray) and arr.max() == 0:
            warnings.warn(f"[{self.sample_id}] mask '{name}' is all zeros.",
                          MissingComponentWarning)

    # --------- pipelines (legacy compatibility) ----------
    def apply_nuc_pipeline(self, methods=None):
        if self.dapi is None:
            warnings.warn(f"[{self.sample_id}] no DAPI, skip nuclei.",
                          MissingComponentWarning)
            return
        methods = methods or ["cellposeSAM", "watershed"]
        for m in methods:
            if m not in NUC_SEG_METHODS:
                warnings.warn(f"[{self.sample_id}] nuclei method '{m}' not in NUC_SEG_METHODS",
                              MissingComponentWarning)
                continue
            res = NUC_SEG_METHODS[m](self.dapi)
            self.nuc_metadata[m] = res
            self.masks[m] = getattr(res, "mask", None)

    def get_positive_cyto_pipline(self, methods=("cellpose","cellpose2chan","watershed","cell_expansion","watershed_only_cyto")):
        if self.marker is None:
            warnings.warn(f"[{self.sample_id}] no marker, skip cyto pipeline.",
                          MissingComponentWarning)
            return
        marker_ch = self.marker[..., 0] if self.marker.ndim == 3 else self.marker

        dapi_multi = self.dapi_multi_mask
        if dapi_multi is None:
            if self.masks.get("cellposeSAM") is None:
                self.apply_nuc_pipeline(methods=["cellposeSAM"])
            dapi_multi = self.masks.get("cellposeSAM")
            if dapi_multi is None or (isinstance(dapi_multi, np.ndarray) and dapi_multi.max() == 0):
                warnings.warn(f"[{self.sample_id}] fallback nucleus mask empty → skip.",
                              MissingComponentWarning)
                return

        for m in methods:
            if m not in CYTO_SEG_METHODS:
                warnings.warn(f"[{self.sample_id}] cyto method '{m}' not in CYTO_SEG_METHODS",
                              MissingComponentWarning)
                continue
            res = CYTO_SEG_METHODS[m](marker_ch, dapi_multi)
            self.cyto_metadata[m] = res
            self.cyto_positive_masks[m] = getattr(res, "mask", None)


class IfImageDataset:
    def __init__(self, image_dir, nuclei_masks_dir, cell_masks_dir=None, manual_cell_counts=None):
        self.image_dir         = image_dir
        self.nuclei_masks_dir  = nuclei_masks_dir
        self.cell_masks_dir    = cell_masks_dir
        self.manual_cell_counts= manual_cell_counts or {}
        self.samples           = {}

    # ---------- public api ----------
    def load_data(self):
        # 1) raw images
        image_files = (
            glob.glob(os.path.join(self.image_dir, "*.tif")) +
            glob.glob(os.path.join(self.image_dir, "*.tiff")) +
            glob.glob(os.path.join(self.image_dir, "*.npy"))
        )
        for f in image_files:
            self._process_image_file(f)

        # 2) nuclei masks
        nuc_glob = os.path.join(self.nuclei_masks_dir, "*", "*.npy")
        nuc_files = glob.glob(nuc_glob)
        if not nuc_files:
            warnings.warn(f"No nuclei mask files found under {self.nuclei_masks_dir}",
                          MissingComponentWarning)
        for f in nuc_files:
            self._process_mask_file(f, target="nuclei")

        # 3) cell-body masks
        if self.cell_masks_dir:
            cell_glob = os.path.join(self.cell_masks_dir, "*", "*.npy")
            cell_files = glob.glob(cell_glob)
            if not cell_files:
                warnings.warn(f"No cell-body mask files found under {self.cell_masks_dir}",
                              MissingComponentWarning)
            for f in cell_files:
                self._process_mask_file(f, target="cyto")

        # 4) post-load validation warnings
        self._warn_missing_components()

    def summary(self, table=False):
        print("Total samples:", len(self.samples))
        # counts by cell type
        celltype_counts = {}
        for sample in self.samples.values():
            celltype_counts[sample.celltype] = celltype_counts.get(sample.celltype, 0) + 1
        print("\nSamples by cell type:")
        for ct, cnt in celltype_counts.items():
            print(f"  {ct}: {cnt}")

        has_dapi = sum(1 for s in self.samples.values() if s.dapi is not None)
        has_marker = sum(1 for s in self.samples.values() if s.marker is not None)
        has_manual_count = sum(1 for s in self.samples.values() if s.manual_cell_count is not None)

        print("\nAttribute statistics:")
        print(f"  Samples with DAPI: {has_dapi}")
        print(f"  Samples with marker: {has_marker}")
        print(f"  Samples with manual count: {has_manual_count}")

        if table:
            print("\nSamples missing key components:")
            print(f"{'Sample ID':<15} {'Cell Type':<15} {'DAPI':<10} {'Marker':<10} {'DAPI Mask':<15} {'Cell Bodies':<15}")
            print("-"*80)
            for sid, s in self.samples.items():
                d = "✓" if s.dapi is not None else "✗"
                m = "✓" if s.marker is not None else "✗"
                dm = "✓" if s.dapi_multi_mask is not None else "✗"
                cb = "✓" if s.cellbodies_multimask is not None else "✗"
                if "✗" in [d, m, dm, cb]:
                    print(f"{sid:<15} {s.celltype or 'Unknown':<15} {d:<10} {m:<10} {dm:<15} {cb:<15}")

    def __getitem__(self, sample_id):
        if sample_id not in self.samples:
            raise KeyError(f"Sample ID {sample_id} not found in the dataset.")
        return self.samples[sample_id]

    def get_random_sample(self):
        if not self.samples:
            print("No samples available in the dataset.")
            return None
        sid = np.random.choice(list(self.samples.keys()))
        s = self.samples[sid]
        print(f"Sample ID: {s.sample_id}, Cell Type: {s.celltype}")
        return s

    def run_all_nuclei_segmentation(self, output_root):
        for sample_id, sample in self.samples.items():
            sample.apply_nuc_pipeline()
            sample_dir = os.path.join(output_root, sample_id)
            os.makedirs(sample_dir, exist_ok=True)
            for mask_name, mask_array in sample.masks.items():
                if mask_array is None:
                    continue
                out_path = os.path.join(sample_dir, f"{mask_name}.npy")
                np.save(out_path, mask_array)
                print(f"Saved {mask_name} → {out_path}")

    def evaluate_nuclei(self, methods=None, iou_thresholds=np.arange(0.5, 1.0, 0.05)):
        rows = []
        for sample_id, sample in tqdm(self.samples.items(), desc="nuclei eval"):
            gt_mask = sample.dapi_multi_mask
            if gt_mask is None:
                warnings.warn(f"[{sample_id}] lacks GT nuclei; skipping…",
                              MissingComponentWarning)
                continue
            celltype = sample.celltype or ""
            for method in methods or list(sample.masks.keys()):
                pred_mask = sample.masks.get(method)
                if pred_mask is None:
                    warnings.warn(f"[{sample_id}] no nuclei mask for '{method}'; skip",
                                  MissingComponentWarning)
                    continue
                for thr in iou_thresholds:
                    try:
                        m = matching(gt_mask, pred_mask, thresh=thr)
                        precision = float(m.precision)
                    except Exception as exc:
                        warnings.warn(f"matching() error [{sample_id}/{method}/{thr}]: {exc}",
                                      MissingComponentWarning)
                        precision = 0.0
                    rows.append({"sample_id": sample_id, "celltype": celltype,
                                 "method": method, "iou": thr, "precision": precision})
        return pd.DataFrame(rows)

    def evaluate_cell(self, methods=None, iou_thresholds=np.arange(0.5, 1.0, 0.05)):
        rows = []
        for sample_id, sample in tqdm(self.samples.items(), desc="cell eval"):
            gt_mask = sample.cellbodies_multimask
            if gt_mask is None:
                warnings.warn(f"[{sample_id}] lacks GT cell mask; skipping…",
                              MissingComponentWarning)
                continue
            celltype = sample.celltype or ""
            for method in methods or list(sample.cyto_positive_masks.keys()):
                src = sample.cyto_positive_masks.get(method)
                if src is None:
                    warnings.warn(f"[{sample_id}] no cell mask for '{method}'; skip",
                                  MissingComponentWarning)
                    continue
                try:
                    pred_mask = src if isinstance(src, np.ndarray) else np.load(src)
                except Exception as exc:
                    warnings.warn(f"[{sample_id}/{method}] load error: {exc}",
                                  MissingComponentWarning)
                    continue
                for thr in iou_thresholds:
                    try:
                        m = matching(gt_mask, pred_mask, thresh=thr)
                        precision = float(m.precision)
                    except Exception as exc:
                        warnings.warn(f"matching() error [{sample_id}/{method}/{thr}]: {exc}",
                                      MissingComponentWarning)
                        precision = 0.0
                    rows.append({"sample_id": sample_id, "celltype": celltype,
                                 "method": method, "iou": thr, "precision": precision})
        return pd.DataFrame(rows)

    # ---------- internals ----------
    def _process_image_file(self, file):
        filename = os.path.basename(file)
        name, _ = os.path.splitext(filename)
        parts = name.split("_")
        if len(parts) < 2:
            warnings.warn(f"Unexpected image filename (skip): {filename}",
                          MissingComponentWarning)
            return
        celltype, sample_id = parts[0], parts[1]
        if sample_id not in self.samples:
            manual = self.manual_cell_counts.get(sample_id)
            self.samples[sample_id] = ImageSample(sample_id, manual)
        sample = self.samples[sample_id]
        sample.celltype = celltype
        sample.add_image_file(file)

    def _process_mask_file(self, file, target="nuclei"):
        sample_id = os.path.basename(os.path.dirname(file))
        mask_name = os.path.splitext(os.path.basename(file))[0]
        arr = safe_read(file)

        if sample_id not in self.samples:
            manual = self.manual_cell_counts.get(sample_id)
            self.samples[sample_id] = ImageSample(sample_id, manual)
        sample = self.samples[sample_id]

        if arr is None:
            warnings.warn(f"[{sample_id}] failed to read mask '{mask_name}'.",
                          MissingComponentWarning)
            return

        if target == "nuclei":
            sample.masks[mask_name] = arr
        elif target == "cyto":
            sample.cyto_positive_masks[mask_name] = arr
        else:
            raise ValueError(f"Unknown mask target: {target}")

        if isinstance(arr, np.ndarray) and arr.max() == 0:
            warnings.warn(f"[{sample_id}] mask '{mask_name}' is all zeros.",
                          MissingComponentWarning)

    def _warn_missing_components(self):
        for sid, s in self.samples.items():
            if s.dapi is None:
                warnings.warn(f"[{sid}] missing DAPI image.", MissingComponentWarning)
            if s.marker is None:
                warnings.warn(f"[{sid}] missing marker image.", MissingComponentWarning)
            if s.dapi_multi_mask is None:
                warnings.warn(f"[{sid}] missing GT nuclei (dapi_multi_mask).",
                              MissingComponentWarning)
            if self.cell_masks_dir and s.cellbodies_multimask is None:
                warnings.warn(f"[{sid}] missing GT cell bodies (cellbodies_multimask).",
                              MissingComponentWarning)

            # no predicted masks present
            has_nuc_pred = any(v is not None for v in s.masks.values())
            if not has_nuc_pred:
                warnings.warn(f"[{sid}] no nuclei predictions loaded.",
                              MissingComponentWarning)
            has_cyto_pred = any(v is not None for v in s.cyto_positive_masks.values())
            if self.cell_masks_dir and not has_cyto_pred:
                warnings.warn(f"[{sid}] no cell-body predictions loaded.",
                              MissingComponentWarning)



from sklearn.mixture import GaussianMixture

class Thresholder:
    @staticmethod
    def otsu_threshold(intensities):
        """Apply Otsu's thresholding method."""
        from skimage.filters import threshold_otsu
        return threshold_otsu(intensities)

    @staticmethod
    def gmm_threshold(intensities, n_components=2):
        """Apply Gaussian Mixture Model (GMM) for thresholding."""
        gmm = GaussianMixture(n_components=n_components, random_state=0)
        gmm.fit(intensities.reshape(-1, 1))
        thresholds = np.sort(gmm.means_.ravel())
        return thresholds[0]  # Use the first threshold for binary classification


import matplotlib.patches as mpatches
def draw_overlap(predicted, ground_truth, ax=None, title=None, display_legend=False):
    binary_mask1 = (predicted > 0).astype(np.uint8)
    binary_mask2 = (ground_truth > 0).astype(np.uint8)
    overlap = binary_mask1 & binary_mask2
    overlay = np.ones((*predicted.shape, 3), dtype=np.uint8) * 255
    overlay[binary_mask1 == 1] = [255, 0, 0]
    overlay[binary_mask2 == 1] = [0, 255, 0]
    overlay[overlap == 1] = [255, 255, 0]
    
    # Create legend patches
    red_patch = mpatches.Patch(color=(1, 0, 0), label='Only Ground Truth')
    green_patch = mpatches.Patch(color=(0, 1, 0), label='Only Predicted')
    yellow_patch = mpatches.Patch(color=(1, 1, 0), label='Overlap')
    
    if ax is None:
        #plt.axis(False)
        if title is not None:
            plt.title(title)
        if display_legend:
            plt.legend(handles=[red_patch, green_patch, yellow_patch])
        plt.imshow(overlay)
        return 0
    
    ax.imshow(overlay)
    ax.set_xticks([])
    ax.set_yticks([])
    if display_legend:
        ax.legend(handles=[red_patch, green_patch, yellow_patch])


import os
def display_sample_stats(sample, dapi_result):
    if dapi_result is None:
        print("No DAPI result available for this sample.")
        return

    if isinstance(sample.cellbodies_multimask, np.ndarray):
        Posinum = sample.cellbodies_multimask.max()
        print(f" Ground truth cellbodies number: {Posinum}")
    comparing_stats = dapi_result["comparing_stats"]
    total_cell_num = dapi_result['total_cell_num']
    marker_cell_ratio = dapi_result['marker_cell_ratio']
    marker_cell_num = dapi_result['marker_cell_num']
    comparing_stats_watershed = dapi_result['comparing_stats_watershed']

    print(f"  Celltype: {sample.celltype}")
    print(f"  Total Cell numbers: {total_cell_num}")
    print(f"  Marker Positive Number_ce: {marker_cell_num}")
    print(f"  Marker Positive Ratio_ce: {marker_cell_ratio:.2f}")
    print(f"  Marker Positive Number_ce: {marker_cell_num}")
    print(f"  Marker Positive Ratio_ce: {marker_cell_ratio:.2f}")
    print(f"  Cell Expansion Summary: {comparing_stats}")
    print(f"  Watershed Summary: {comparing_stats_watershed}")

def safe_read(path):
    """Safely read image or numpy files and handle multi-channel images.

    Args:
        path (str): Path to the image or numpy file to read

    Returns:
        numpy.ndarray: The loaded image/mask data, converted to 2D if needed.
                      Returns None if file doesn't exist or on error.
    """
    if not os.path.exists(path):
        print(f"File does not exist: {path}")
        return None

    try:
        if path.endswith('.npy'):
            mask = np.load(path)
        else:
            mask = skio.imread(path)
        if mask.ndim == 3:
            if mask.shape[-1] == 1:
                mask = np.squeeze(mask, axis=-1)  # Remove singleton dimension
            else:
                # print(f"Warning: Multi-channel mask detected in {path}, converting to 2D.")
                mask = np.max(mask, axis=-1)  # Use max projection for now
        return mask
    except Exception as e:
        print(f"Error reading file {path}: {e}")
        return None


def calculate_overall_iou(mask1, mask2):
    """
    Calculate Intersection over Union (IoU) between two binary masks.
    
    Args:
        mask1 (numpy.ndarray): First binary mask
        mask2 (numpy.ndarray): Second binary mask
        
    Returns:
        float: IoU score between 0 and 1
    """
    if mask1 is None or mask2 is None:
        print("One or both masks are None")
        return 0.0
        
    if mask1.shape != mask2.shape:
        print("Masks have different shapes")
        return 0.0
        
    # Convert to boolean masks if not already
    mask1 = mask1 > 0
    mask2 = mask2 > 0
    
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score



import numpy as np
from skimage.measure import label, regionprops
from scipy.spatial.distance import cdist

class CentroidAnalyzer:
    @staticmethod
    def find_centers(mask):
        labeled_mask = label(mask)
        props = regionprops(labeled_mask)
        centers = np.array([prop.centroid for prop in props])
        labels = np.array([prop.label for prop in props])
        return centers, labels

    @staticmethod
    def match_cells(cell_centers, cell_labels, nuclei_centers, nuclei_labels):
        distances = cdist(cell_centers, nuclei_centers)
        closest_indices = distances.argmin(axis=1)
        mapping = {cell_labels[i]: nuclei_labels[closest_indices[i]] for i in range(len(cell_labels))}
        return mapping

    @staticmethod
    def plot_centroids(gt_cyto, pred_cyto, linkage=False):
        gt_centers, gt_labels = CentroidAnalyzer.find_centers(gt_cyto)
        pred_centers, pred_labels = CentroidAnalyzer.find_centers(pred_cyto)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(gt_centers[:, 1], gt_centers[:, 0], c='green', label='Ground Truth', alpha=0.7, s=50)
        ax.scatter(pred_centers[:, 1], pred_centers[:, 0], c='red', label='Predicted', alpha=0.7, s=50)

        if linkage:
            distances = cdist(gt_centers, pred_centers)
            # Create a list to keep track of which predicted centroids have been assigned
            assigned = [False] * len(pred_centers)
            for i in range(len(gt_centers)):
                # Find the closest unassigned predicted centroid
                min_dist = np.inf
                closest_idx = -1
                for j in range(len(pred_centers)):
                    if not assigned[j] and distances[i, j] < min_dist:
                        min_dist = distances[i, j]
                        closest_idx = j

                if closest_idx != -1:
                    # Assign this predicted centroid to the ground truth centroid
                    assigned[closest_idx] = True
                    ax.plot([gt_centers[i, 1], pred_centers[closest_idx, 1]],
                            [gt_centers[i, 0], pred_centers[closest_idx, 0]], 'k--', alpha=0.5)
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title('Centroids of Ground Truth and Predicted Cells')
        ax.legend()
        plt.show()

    @staticmethod
    def plot_centroids_with_cyto(gt_cyto, pred_cyto, linkage=False, title="Centroids of Ground Truth and Predicted Cells"):
        gt_centers, gt_labels = CentroidAnalyzer.find_centers(gt_cyto)
        pred_centers, pred_labels = CentroidAnalyzer.find_centers(pred_cyto)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(gt_cyto, cmap='Greens', alpha=0.3)
        ax.imshow(pred_cyto, cmap='Reds', alpha=0.3)

        # Plot the centroids
        ax.scatter(gt_centers[:, 1], gt_centers[:, 0], c='green', label='Ground Truth', alpha=0.7, s=5)
        ax.scatter(pred_centers[:, 1], pred_centers[:, 0], c='red', label='Predicted', alpha=0.7, s=5)

        if linkage:
            distances = cdist(gt_centers, pred_centers)
            closest_indices = distances.argmin(axis=1)

            for i, idx in enumerate(closest_indices):
                    ax.plot([gt_centers[i, 1], pred_centers[idx, 1]], [gt_centers[i, 0], pred_centers[idx, 0]], 'k--', alpha=0.5)
        ax.set_title(f'{title}')
        ax.legend()
        plt.show()

    def analyze_image_sample(image_sample,
                             gt_mask_attr='cellbodies_multimask',
                             pred_mask_attr='cellpose',
                             linkage=False):
        gt_mask = getattr(image_sample, gt_mask_attr, None)
        if gt_mask is None:
            print(f"ImageSample lack {gt_mask_attr} attribute")
            return
        if pred_mask_attr in image_sample.__dict__:
            pred_mask = getattr(image_sample, pred_mask_attr)
        elif isinstance(image_sample.cyto_positive_masks, dict) and pred_mask_attr in image_sample.cyto_positive_masks:
            pred_mask = image_sample.cyto_positive_masks[pred_mask_attr]
        else:
            print(f"ImageSample lack {pred_mask_attr} attribute")
            return

        gt_centers, gt_labels = CentroidAnalyzer.find_centers(gt_mask)
        pred_centers, pred_labels = CentroidAnalyzer.find_centers(pred_mask)

        if gt_centers.size == 0 or pred_centers.size == 0:
            print("No centers, please check")
            return

        mapping = CentroidAnalyzer.match_cells(gt_centers, gt_labels, pred_centers, pred_labels)
        print("Mapping:", mapping)
        CentroidAnalyzer.plot_centroids(gt_centers, pred_centers, linkage=linkage)

    def evaluation(gt_cyto, pred_cyto):
        # still considering how to evaluate two masks based on centeroids
        # ???
        pass

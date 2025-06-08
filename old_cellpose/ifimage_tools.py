import glob
import scipy.ndimage as ndimage
import skimage.io as skio
from cellpose import models
from imagecodecs.imagecodecs import none_check
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import watershed
from stardist.matching import matching
from stardist.models import StarDist2D
from csbdeep.utils import normalize


# --- Cellpose monkey-patch: skip size-model download -----------------------
from cellpose import models as _cp_models

if not getattr(_cp_models, "_size_patch_done", False):
    # short-circuit the downloader for every run in this process
    _cp_models.size_model_path = lambda *a, **k: None
    _cp_models._size_patch_done = True
# --------------------------------------------------------------------------


class ImageSample:
    def __init__(self, sample_id, manual_cell_count=None):
        self.sample_id = sample_id
        self.celltype = None
        self.manual_cell_count = manual_cell_count
        self.dapi = None
        self.marker = None
        self.dapi_mask = None
        self.dapi_multi_mask = None
        self.cellbodies_mask = None
        self.cellbodies_multimask = None
        self.masks = {
            "cyto": None,
            "cyto2": None,
            "cyto3": None,
            "watershed": None,
            "StarDist2D": None
        }
        self.cyto_positive_masks = {
            "cellpose": None,
            "watershed": None,
            "cell_expansion": None,
            "watershed_only_cyto":None,
            "cellpose2chan": None
        }
        self.pos_nuclei_mask = None
        self.limited_mask = None
    
    def __str__(self):
        return "ImageSample:{}_{}".format(self.celltype,self.sample_id)

    def add_image_file(self, file):
        filename = os.path.basename(file)
        if "dapimask" in filename:
            self.dapi_mask = safe_read(file)
        elif "marker." in filename:
            self.marker = safe_read(file)
        elif "dapimultimask" in filename:
            self.dapi_multi_mask = safe_read(file)
        elif "cellbodiesmultimask" in filename:
            self.cellbodies_multimask = safe_read(file)
        elif "cellbodies.npy" in filename:
            self.cellbodies_multimask = safe_read(file)
        elif "cellbodies.tif" in filename:
            self.cellbodies_mask = safe_read(file)
        elif len(filename.split("_")) == 2:
            self.dapi = safe_read(file)

    def add_mask_file(self, file):
        filename = os.path.basename(file)
        if "cyto2" in filename:
            self.masks["cyto2"] = safe_read(file)
        elif "cyto3" in filename:
            self.masks["cyto3"] = safe_read(file)
        elif "watershed" in filename:
            self.masks["watershed"] = safe_read(file)
        elif "StarDist2D" in filename:
            self.masks["StarDist2D"] = safe_read(file)
        elif "cyto" in filename:
            self.masks["cyto"] = safe_read(file)

    def apply_nuc_pipeline(self):
        """Apply nucleus segmentation pipeline using various methods."""
        if self.dapi is None:
            print("Missing DAPI image; cannot run nucleus segmentation pipeline.")
            return

        #—— Cellpose 'cyto3' only (you can add back cyto, cyto2 similarly) ——
        if self.masks.get("cyto3") is None:
            print("Running Cellpose 'cyto3'…")
            model_cp = models.Cellpose(gpu=True, model_type='cyto3')
            masks_cp, *_ = model_cp.eval(self.dapi, diameter=15, channels=[0,0])
            self.masks["cyto3"] = masks_cp
        else:
            print("Skip 'cyto3' (already present)")

        # —— StarDist2D ——  
        # 只在第一次尝试
        # print("🏃 Running StarDist2D ...")
        # try:
        #     sd_model = StarDist2D.from_pretrained("2D_versatile_fluo")
        #     img_norm = normalize(self.dapi)
        #     labels_sd, _ = sd_model.predict_instances(img_norm, n_tiles=None)
        #     self.masks["StarDist2D"] = labels_sd
        # except Exception as e:
        #     print(f"⚠️  StarDist2D failed: {e}")
        #     self.masks["StarDist2D"] = "FAILED"


        # —— Watershed segmentation ——  
        # if self.masks.get("watershed") is None:
        #     print("Running watershed…")
        #     distance = distance_transform_edt(self.dapi > 0)
        #     markers = ndimage.label(self.dapi > 0)[0]
        #     ws = watershed(-distance, markers, mask=self.dapi > 0)
        #     self.masks["watershed"] = ws
        # else:
        #     print("Skip 'watershed' (already present)")

    def get_positive_cyto_pipline(self, methods=["cellpose", "cellpose2chan" , "watershed", "cell_expansion","watershed_only_cyto"]):
        if self.marker is None or self.dapi_multi_mask is None:
            print("Missing marker or dapi_multi_mask; cannot run cytoplasm positive pipeline.")
            return

        marker_channel = self.marker[:, :, 0] if self.marker.ndim == 3 else self.marker
        multi_mask = self.dapi_multi_mask
        finder = FindMarker(max_expansion=10)

        if 'watershed' in methods:
            _, _, _, cyto_mask, predicted_positive_mask_ws = finder.watershed_cellpose_with_dapi(marker_channel, multi_mask)
            self.cyto_positive_masks["watershed"] = predicted_positive_mask_ws

        if 'cell_expansion' in methods or 'cell_expansion' in methods:
            _, _, _, limited_mask, predicted_positive_mask_ce = finder.cell_expansion(marker_channel, multi_mask)
            self.cyto_positive_masks["cell_expansion"] = predicted_positive_mask_ce

        if "watershed_only_cyto" in methods:
            _, _, _, _, predicted_positive_mask_ws_only_cyto = finder.watershed_only_cyto(marker_channel, multi_mask)
            self.cyto_positive_masks["watershed_only_cyto"]=predicted_positive_mask_ws_only_cyto

        if 'cellpose' in methods:
            predicted_positive_mask_cp = finder.cellpose_only_cyto(marker_channel, multi_mask)
            self.cyto_positive_masks["cellpose"] = predicted_positive_mask_cp

        if 'cellpose2chan' in methods:
            predicted_positive_mask_cp = finder.cellpose2chan(marker_channel, multi_mask)
            self.cyto_positive_masks["cellpose2chan"] = predicted_positive_mask_cp


class IfImageDataset:
    def __init__(self, image_dir, masks_dir, manual_cell_counts):
        self.image_dir = image_dir
        self.masks_dir = masks_dir
        self.manual_cell_counts = manual_cell_counts
        self.samples = {}

    def load_data(self):
        image_files = glob.glob(os.path.join(self.image_dir, "*.tif")) + \
                      glob.glob(os.path.join(self.image_dir, "*.tiff")) + \
                      glob.glob(os.path.join(self.image_dir, "*.npy"))
                      
        mask_files = glob.glob(os.path.join(self.masks_dir, "*.npy"))

        for file in image_files:
            self._process_image_file(file)
        for file in mask_files:
            self._process_mask_file(file)

    def summary(self,table=False):
        print("Total samples:", len(self.samples))
        
        # Count samples by cell type
        celltype_counts = {}
        for sample in self.samples.values():
            if sample.celltype not in celltype_counts:
                celltype_counts[sample.celltype] = 0
            celltype_counts[sample.celltype] += 1
            
        print("\nSamples by cell type:")
        for celltype, count in celltype_counts.items():
            print(f"  {celltype}: {count}")
            
        # Count samples with various attributes
        has_dapi = sum(1 for s in self.samples.values() if s.dapi is not None)
        has_marker = sum(1 for s in self.samples.values() if s.marker is not None)
        has_manual_count = sum(1 for s in self.samples.values() if s.manual_cell_count is not None)
        
        print("\nAttribute statistics:")
        print(f"  Samples with DAPI: {has_dapi}")
        print(f"  Samples with marker: {has_marker}")
        print(f"  Samples with manual count: {has_manual_count}")
        
        if table:
            # Create table showing samples lacking key components
            print("\nSamples missing key components:")
            print(f"{'Sample ID':<15} {'Cell Type':<15} {'DAPI':<10} {'Marker':<10} {'DAPI Mask':<15} {'Cell Bodies':<15}")
            print("-" * 80)
            
            for sample_id, sample in self.samples.items():
                has_dapi = "✓" if sample.dapi is not None else "✗"
                has_marker = "✓" if sample.marker is not None else "✗"
                has_dapi_mask = "✓" if sample.dapi_multi_mask is not None else "✗"
                has_cellbodies = "✓" if sample.cellbodies_multimask is not None else "✗"
                
                # Only show in table if missing at least one component
                if "✗" in [has_dapi, has_marker, has_dapi_mask, has_cellbodies]:
                    print(f"{sample_id:<15} {sample.celltype or 'Unknown':<15} {has_dapi:<10} {has_marker:<10} {has_dapi_mask:<15} {has_cellbodies:<15}")
            
        

    def _process_image_file(self, file):
        filename = os.path.basename(file)
        name, ext = os.path.splitext(filename)
        parts = name.split("_")
        if len(parts) < 2:
            return
        sample_id = parts[1]
        celltype = parts[0]
        if sample_id not in self.samples:
            manual_count = self.manual_cell_counts.get(sample_id)
            self.samples[sample_id] = ImageSample(sample_id, manual_count)
        self.samples[sample_id].celltype = celltype
        self.samples[sample_id].add_image_file(file)

    def _process_mask_file(self, file):
        filename = os.path.basename(file)
        name, ext = os.path.splitext(filename)
        parts = name.split("_")
        if len(parts) < 2:
            return
        sample_id = parts[0]
        if sample_id not in self.samples:
            manual_count = self.manual_cell_counts.get(sample_id)
            self.samples[sample_id] = ImageSample(sample_id, manual_count)
        self.samples[sample_id].add_mask_file(file)

    def __getitem__(self, sample_id):
        """Retrieve a sample by its ID."""
        if sample_id not in self.samples:
            raise KeyError(f"Sample ID {sample_id} not found in the dataset.")
        return self.samples[sample_id]

    def get_random_sample(self):
        """Get a random sample and print its cell type and ID."""
        if not self.samples:
            print("No samples available in the dataset.")
            return None

        sample_id = np.random.choice(list(self.samples.keys()))
        sample = self.samples[sample_id]
        print(f"Sample ID: {sample.sample_id}, Cell Type: {sample.celltype}")
        return sample




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


class FindMarker:
    def __init__(self, max_expansion=10, threshold_method="otsu"):
        self.max_expansion = max_expansion
        self.threshold_method = threshold_method
        self.thresholder = Thresholder()

    def get_average_intensity(self, marker_channel, mask, normalize=False):
        labels_unique = np.unique(mask)
        labels_unique = labels_unique[labels_unique != 0]
        total_marker_intensity = ndimage.sum(marker_channel, labels=mask, index=labels_unique)
        region_area = ndimage.sum(np.ones_like(marker_channel), labels=mask, index=labels_unique)

        average_marker_intensity = total_marker_intensity / region_area

        if normalize:
            average_marker_intensity = (average_marker_intensity - average_marker_intensity.min()) / \
                                      (average_marker_intensity.max() - average_marker_intensity.min())
        return average_marker_intensity

    def _apply_threshold(self, intensities):
        """Apply the selected thresholding method."""
        if self.threshold_method == "otsu":
            return self.thresholder.otsu_threshold(intensities)
        elif self.threshold_method == "gmm":
            return self.thresholder.gmm_threshold(intensities)
        else:
            raise ValueError(f"Unsupported thresholding method: {self.threshold_method}")

    def cell_expansion(self, marker_channel, dapi_multi_mask):
        properties = regionprops(dapi_multi_mask)
        centroids = np.array([prop.centroid for prop in properties])
        markers = np.zeros_like(dapi_multi_mask, dtype=int)

        for idx, (y, x) in enumerate(centroids):
            markers[int(round(y)), int(round(x))] = idx + 1

        distance, (inds_y, inds_x) = distance_transform_edt(markers == 0, return_indices=True)
        limited_mask = np.zeros_like(dapi_multi_mask, dtype=int)
        within_limit = distance <= self.max_expansion
        limited_mask[within_limit] = markers[inds_y[within_limit], inds_x[within_limit]]

        average_marker_intensity = self.get_average_intensity(marker_channel, limited_mask)
        threshold = self._apply_threshold(average_marker_intensity)
        labels_unique = np.unique(limited_mask)
        labels_unique = labels_unique[labels_unique != 0]
        regions_positive = labels_unique[average_marker_intensity > threshold]
        predicted_positive_nuclei_mask = np.where(np.isin(limited_mask, regions_positive), limited_mask, 0)
        total_cell_num = len(labels_unique)
        marker_cell_num = len(regions_positive)
        marker_cell_ratio = marker_cell_num / total_cell_num if total_cell_num > 0 else 0
        return total_cell_num, marker_cell_ratio, marker_cell_num, limited_mask, predicted_positive_nuclei_mask

    def watershed_cellpose_with_dapi(self, marker_channel, dapi_multi_mask):
        gradient = marker_channel
        cytoplasm_mask = watershed(gradient, markers=dapi_multi_mask)

        average_marker_intensity = self.get_average_intensity(marker_channel, cytoplasm_mask)
        threshold = self._apply_threshold(average_marker_intensity)
        labels_unique = np.unique(cytoplasm_mask)
        labels_unique = labels_unique[labels_unique != 0]
        regions_positive = labels_unique[average_marker_intensity > threshold]

        predicted_positive_mask = np.where(np.isin(cytoplasm_mask, regions_positive), cytoplasm_mask, 0)
        total_cell_num = len(labels_unique)
        marker_cell_num = len(regions_positive)
        marker_cell_ratio = marker_cell_num / total_cell_num if total_cell_num > 0 else 0

        return total_cell_num, marker_cell_ratio, marker_cell_num, cytoplasm_mask, predicted_positive_mask

    def cellpose_only_cyto(self, marker_channel, dapi_multi_mask):
        model = models.Cellpose(model_type='cyto3', gpu=True)
        masks, flows, styles, diams = model.eval(
            marker_channel,
            channels=[0, 0],
            diameter=None,
            flow_threshold=0.4,
            cellprob_threshold=0.0
        )
        
        return masks

    def cellpose2chan(self, marker_channel, dapi_multi_mask):
        from cellpose import models
        dapi = marker_channel
        marker = dapi_multi_mask
        stacked = np.stack([dapi, marker, np.zeros_like(dapi)], axis=2)

        model = models.Cellpose(model_type='cyto3',gpu=True)
        cytoplasm_mask, flows, styles, diams = model.eval(
            stacked,  # must be a list of images
            channels=[2, 1],
            diameter=None
        )
        average_marker_intensity = self.get_average_intensity(marker_channel, cytoplasm_mask)
        threshold = self._apply_threshold(average_marker_intensity)
        labels_unique = np.unique(cytoplasm_mask)
        labels_unique = labels_unique[labels_unique != 0]
        regions_positive = labels_unique[average_marker_intensity > threshold]

        predicted_positive_mask = np.where(np.isin(cytoplasm_mask, regions_positive), cytoplasm_mask, 0)
        total_cell_num = len(labels_unique)
        marker_cell_num = len(regions_positive)
        marker_cell_ratio = marker_cell_num / total_cell_num if total_cell_num > 0 else 0

        return predicted_positive_mask

    def watershed_only_cyto(self, marker_channel, dapi_multi_mask=None):
        from skimage.feature import peak_local_max
        coordinates = peak_local_max(marker_channel, min_distance=30, threshold_abs=0.3)
        seeds = np.zeros_like(marker_channel, dtype=bool)
        seeds[tuple(coordinates.T)] = True
        seeds_labeled = label(seeds)
        gradient = marker_channel
        cytoplasm_mask = watershed(gradient, markers=seeds_labeled)
        # print(cytoplasm_mask.max())
        average_marker_intensity = self.get_average_intensity(marker_channel, cytoplasm_mask)
        threshold = self._apply_threshold(average_marker_intensity)
        labels_unique = np.unique(cytoplasm_mask)
        labels_unique = labels_unique[labels_unique != 0]
        regions_positive = labels_unique[average_marker_intensity > threshold]

        predicted_positive_mask = np.where(np.isin(cytoplasm_mask, regions_positive), cytoplasm_mask, 0)
        total_cell_num = len(labels_unique)
        marker_cell_num = len(regions_positive)
        print(marker_cell_num)
        marker_cell_ratio = marker_cell_num / total_cell_num if total_cell_num > 0 else 0

        return total_cell_num, marker_cell_ratio, marker_cell_num, cytoplasm_mask, predicted_positive_mask

    def compute_confusion_matrix_stardist(self, cellbodies_mask, predicted_positive_nuclei_mask, iou_threshold=0.01):
        predicted_positive_nuclei_mask = predicted_positive_nuclei_mask.astype(np.uint32)
        cellbodies_mask = cellbodies_mask.astype(np.uint32)
        stats = matching(cellbodies_mask, predicted_positive_nuclei_mask, thresh=iou_threshold)
        return stats




import matplotlib.pyplot as plt
def visualize_multimask(multi_mask, ax):
    unique_masks = np.unique(multi_mask)
    unique_masks = unique_masks[unique_masks != 0]
    num_masks = len(unique_masks)
    cmap = plt.get_cmap('gist_ncar', num_masks)
    colored_mask = np.ones((*multi_mask.shape, 3))
    for i, mask_id in enumerate(unique_masks):
        color = cmap(i)[:3]
        colored_mask[multi_mask == mask_id] = color
    ax.imshow(colored_mask)
    ax.set_xticks([])
    ax.set_yticks([])


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
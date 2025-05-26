import os
import glob
import argparse
import numpy as np
import cv2
from roifile import ImagejRoi
from skimage.draw import polygon
import matplotlib.pyplot as plt


import os
import zipfile
import re
import shutil
import numpy as np
from skimage.io import imread
from roifile import ImagejRoi
from skimage.draw import polygon



# batch_process_zip_rois->batch_rename

class SamplePreprocessor:
    def __init__(self, raw_data_dir, output_dir, width=1388, height=1024):
        self.raw_data_dir = raw_data_dir
        self.output_dir = output_dir
        self.width = width
        self.height = height
        self.manual_cell_counts = {}

        os.makedirs(self.output_dir, exist_ok=True)

    def preprocess_all(self):
        for filename in os.listdir(self.raw_data_dir):
            path = os.path.join(self.raw_data_dir, filename)
            if filename.endswith('.zip') and "RoiSet" in filename:
                self._process_roi_zip(path)
            elif filename.endswith('.tif') or filename.endswith('.tiff'):
                self._process_tiff(path)

    def _extract_info(self, filename):
        celltypes = ['olig2', 'gfap', 'neun', 'iba1', 'pecam']
        filename_lower = filename.lower()

        # Try to find a sample_id (number between 1 and 20000)
        id_matches = re.findall(r'\d{3,5}', filename)
        sample_id = None
        for match in id_matches:
            num = int(match)
            if 1 <= num <= 20000:
                sample_id = str(num)
                break

        # Try to find a known celltype
        celltype = None
        for ct in celltypes:
            if ct in filename_lower:
                celltype = ct
                break

        return sample_id, celltype
    
    def _process_roi_zip(self, zip_path):
        filename = os.path.basename(zip_path)
        sample_id, celltype = self._extract_info(filename)
        if not sample_id:
            print(f"[WARN] Could not parse sample_id from {filename}")
            return

        # Convert .roi -> mask
        mask = self._rois_to_mask(zip_path, self.width, self.height)

        # Figure out output filename
        if "ALL_DAPI" in filename:
            out_name = f"{celltype}_{sample_id}_dapimultimask.npy"
            self.manual_cell_counts[sample_id] = int(np.max(mask))
        elif "without_Olig2" in filename:
            out_name = f"{celltype}_{sample_id}_dapimask.npy"
        elif "CellBodies" in filename:
            out_name = f"{celltype}_{sample_id}_cellbodiesmultimask.npy"
        elif "Olig2_DAPI" in filename:
            out_name = f"{celltype}_{sample_id}_marker.npy"
        else:
            out_name = f"{celltype}_{sample_id}_unknown.npy"

        np.save(os.path.join(self.output_dir, out_name), mask)
        print(f"[OK] Saved mask: {out_name}")

    def _process_tiff(self, path):
        filename = os.path.basename(path)
        match = re.match(r"Snap-(\d+)_b0c(\d)x\d+-\d+y\d+-\d+.tiff", filename)
        if not match:
            return
        sample_id = match.group(1)
        channel = int(match.group(2))
        celltype = "Unknown"
        image_type = "dapi" if channel == 0 else "marker"
        out_name = f"{celltype}_{sample_id}_{image_type}.tiff"
        shutil.copy(path, os.path.join(self.output_dir, out_name))
        print(f"[OK] Copied TIFF: {out_name}")

    def _rois_to_mask(self, roi_path, width, height):
        mask = np.zeros((height, width), dtype=np.uint16)
        roi_entries = []

        with zipfile.ZipFile(roi_path, 'r') as zf:
            for name in zf.namelist():
                if name.endswith('.roi'):
                    roi_bytes = zf.read(name)
                    roi_entries.append((name, roi_bytes))

        label = 1
        for name, roi_bytes in roi_entries:
            try:
                roi = ImagejRoi.frombytes(roi_bytes)
                coords = roi.coordinates()
                if coords is None or len(coords) == 0:
                    if roi.roitype == 10:  # Point ROI
                        cx, cy = int(roi.left), int(roi.top)
                        coords = np.array([
                            [cx - 1, cy - 1],
                            [cx + 1, cy - 1],
                            [cx + 1, cy + 1],
                            [cx - 1, cy + 1]
                        ])
                    else:
                        coords = np.array([
                            [roi.left, roi.top],
                            [roi.right, roi.top],
                            [roi.right, roi.bottom],
                            [roi.left, roi.bottom]
                        ])
                rr, cc = polygon(coords[:, 1], coords[:, 0], shape=mask.shape)
                mask[rr, cc] = label
                label += 1
            except Exception as e:
                print(f"[ERROR] Failed to parse ROI {name}: {e}")
                continue
        return mask



def rois_to_mask(roi_path, width, height):
    """
    Convert all ImageJ .roi files in a directory or ZIP archive to a multi-valued mask.
    
    Each ROI is read (assuming it is of type polygon, freehand, rectangle, oval, or point).
    The ROI is drawn into a blank mask (with the given width and height) using a unique label.
    If the ROI's polygon coordinates are empty, the function falls back on using its
    bounds or point location.
    
    Parameters:
        roi_path (str): Path to a directory containing .roi files or a ZIP file containing them.
        width (int): Width of the target mask image.
        height (int): Height of the target mask image.
        
    Returns:
        numpy.ndarray: A 2D mask (dtype=uint16) of shape (height, width) where the background is 0
                       and each ROI is filled with its unique label.
    """
    import os, glob, zipfile
    import numpy as np
    from roifile import ImagejRoi
    from skimage.draw import polygon

    # Create an empty mask
    mask = np.zeros((height, width), dtype=np.uint16)
    roi_entries = []
    
    # Determine if roi_path is a ZIP file or a directory
    if roi_path.lower().endswith(".zip"):
        with zipfile.ZipFile(roi_path, 'r') as zf:
            for name in zf.namelist():
                if name.lower().endswith(".roi"):
                    # Read the ROI file bytes from the ZIP archive
                    roi_bytes = zf.read(name)
                    roi_entries.append((name, roi_bytes))
    else:
        # Assume roi_path is a directory; read each .roi file as bytes.
        for fp in glob.glob(os.path.join(roi_path, "*.roi")):
            try:
                with open(fp, 'rb') as f:
                    roi_bytes = f.read()
                roi_entries.append((os.path.basename(fp), roi_bytes))
            except Exception as e:
                print(f"Error reading file '{fp}': {e}")
    
    label = 1  # Start labeling from 1
    for name, roi_bytes in roi_entries:
        try:
            # Read ROI from bytes
            roi = ImagejRoi.frombytes(roi_bytes)
        except Exception as e:
            print(f"Error reading ROI '{name}': {e}")
            continue

        # Attempt to get polygon coordinates
        coords = roi.coordinates()
        # If coordinates are empty, fall back on bounds or point ROI handling.
        if coords is None or len(coords) == 0:
            if hasattr(roi, 'roitype'):
                # If ROI type is 10 (point ROI), make a small square (3x3) centered at the point.
                if roi.roitype == 10:
                    cx = int(round(roi.left))
                    cy = int(round(roi.top))
                    coords = np.array([
                        [cx - 1, cy - 1],
                        [cx + 1, cy - 1],
                        [cx + 1, cy + 1],
                        [cx - 1, cy + 1]
                    ], dtype=np.float32)
                # Otherwise, if ROI has bounds, create a rectangle polygon.
                elif (hasattr(roi, 'left') and hasattr(roi, 'top') and 
                      hasattr(roi, 'right') and hasattr(roi, 'bottom')):
                    coords = np.array([
                        [roi.left, roi.top],
                        [roi.right, roi.top],
                        [roi.right, roi.bottom],
                        [roi.left, roi.bottom]
                    ], dtype=np.float32)
                else:
                    print(f"ROI '{name}' has no usable coordinates; skipping.")
                    continue
            else:
                print(f"ROI '{name}' has no coordinates; skipping.")
                continue
        
        # Draw the ROI polygon into the mask.
        # Note: skimage.draw.polygon expects row (y) and col (x) coordinates.
        rr, cc = polygon(coords[:, 1], coords[:, 0], shape=mask.shape)
        mask[rr, cc] = label
        print(f"ROI '{name}' drawn with label {label}.")
        label += 1

    return mask

def mask_to_rois_zip(mask, roi_zip_path):
    """
    Convert a mask to a set of ImageJ ROI files, capturing irregular shapes.
    
    For each unique label (excluding background 0), this function extracts the contour
    of the ROI using skimage.measure.find_contours and then creates a polygon ROI.
    
    Parameters:
        mask (numpy.ndarray): A 2D mask where each unique value represents a different ROI.
        roi_zip_path (str): The path where the zipped ROI files will be saved.
    
    Returns:
        None
    """
    import numpy as np
    from roifile import ImagejRoi, ROI_TYPE  # import ROI_TYPE constant
    from skimage import measure
    import zipfile

    unique_labels = np.unique(mask)
    if unique_labels.size == 0:
        print("No ROIs found in the mask.")
        return

    with zipfile.ZipFile(roi_zip_path, 'w') as zf:
        for label in unique_labels:
            if label == 0:
                continue  # Skip background.
            
            roi_mask = (mask == label).astype(np.uint8)
            contours = measure.find_contours(roi_mask, level=0.5)
            if len(contours) == 0:
                print(f"No contour found for label {label}.")
                continue
            contour = max(contours, key=len)
            coords = np.fliplr(contour)
            
            try:
                # Create a polygon ROI using the provided coordinates.
                roi = ImagejRoi.frompoints(coords.astype(np.float32))
                # Set the ROI type using the proper constant.
                roi.roitype = ROI_TYPE.POLYGON
                roi_bytes = roi.tobytes()
            except Exception as e:
                print(f"Error creating ROI for label {label}: {e}")
                continue
            
            roi_filename = f"roi_{label}.roi"
            zf.writestr(roi_filename, roi_bytes)
            print(f"ROI for label {label} saved as '{roi_filename}'.")



def batch_process_zip_rois(folders, width=1388, height=1024):
    import os, zipfile, glob
    import numpy as np
    import cv2
    from roifile import ImagejRoi
    from skimage.draw import polygon

    def rois_to_mask(roi_source, width, height):
        """_summary_

        Args:
            roi_source (_type_): _description_
            width (_type_): _description_
            height (_type_): _description_

        Returns:
            _type_: _description_
        """
        
        mask = np.zeros((height, width), dtype=np.uint16)
        roi_entries = []
        # 如果传入的是ZIP文件路径
        if isinstance(roi_source, str) and roi_source.lower().endswith(".zip"):
            with zipfile.ZipFile(roi_source, 'r') as zf:
                for name in zf.namelist():
                    if name.lower().endswith(".roi"):
                        try:
                            roi_bytes = zf.read(name)
                            roi_entries.append((name, roi_bytes))
                        except Exception as e:
                            print(f"读取ZIP中ROI '{name}' 出错: {e}")
        else:
            # 否则假设 roi_source 为包含 .roi 文件的目录
            for fp in glob.glob(os.path.join(roi_source, "*.roi")):
                try:
                    with open(fp, 'rb') as f:
                        roi_bytes = f.read()
                    roi_entries.append((os.path.basename(fp), roi_bytes))
                except Exception as e:
                    print(f"读取文件 '{fp}' 出错: {e}")
        
        label = 1
        for name, roi_bytes in roi_entries:
            try:
                roi = ImagejRoi.frombytes(roi_bytes)
            except Exception as e:
                print(f"解析ROI '{name}' 出错: {e}")
                continue

            # 尝试获取多边形顶点
            coords = roi.coordinates()
            if coords is None or len(coords) == 0:
                # 若没有坐标，尝试使用 ROI 的矩形边界构造多边形
                if hasattr(roi, 'left') and hasattr(roi, 'top') and hasattr(roi, 'right') and hasattr(roi, 'bottom'):
                    coords = np.array([
                        [roi.left, roi.top],
                        [roi.right, roi.top],
                        [roi.right, roi.bottom],
                        [roi.left, roi.bottom]
                    ], dtype=np.float32)
                # 如果是点ROI (roitype==10)，构造一个小方框（3x3）
                elif hasattr(roi, 'roitype') and roi.roitype == 10:
                    cx = int(round(roi.left))
                    cy = int(round(roi.top))
                    coords = np.array([
                        [cx-1, cy-1],
                        [cx+1, cy-1],
                        [cx+1, cy+1],
                        [cx-1, cy+1]
                    ], dtype=np.float32)
                else:
                    print(f"ROI '{name}' 没有可用坐标，跳过。")
                    continue

            # skimage.draw.polygon() 接受 row (y) 和 column (x)
            rr, cc = polygon(coords[:, 1], coords[:, 0], shape=mask.shape)
            mask[rr, cc] = label
            print(f"ROI '{name}' 以 label {label} 绘制。")
            label += 1

        return mask

    for folder in folders:
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.lower().endswith(".zip"):
                    zip_path = os.path.join(root, file)
                    print(f"正在处理ZIP文件: {zip_path}")
                    # 生成多值mask
                    multi_mask = rois_to_mask(zip_path, width, height)
                    
                    # 构造多值mask输出文件名（保存为 .npy）
                    base = os.path.splitext(file)[0]
                    multi_mask_filename = os.path.join(root, base + ".npy")
                    
                    try:
                        np.save(multi_mask_filename, multi_mask)
                        print(f"已保存多值mask: {multi_mask_filename}")
                    except Exception as e:
                        print(f"保存多值mask失败 {zip_path}: {e}")
                    
                    binary_mask = ((multi_mask > 0).astype(np.uint8)) * 255
                    binary_mask_filename = os.path.join(root, base + ".tif")
                    if cv2.imwrite(binary_mask_filename, binary_mask):
                        print(f"已保存二值mask: {binary_mask_filename}")
                    else:
                        print(f"保存二值mask失败 {zip_path}")
                        
import os
import re
import shutil

import os
import re
import shutil

def batch_rename(top_root, dest_root):
    """
    三层遍历：top_root 下 → 细胞类型组 → 样本文件夹 → 文件
    样本 dir 名格式必须像 TYPE_ID（不论大小写，TYPE 里不能有下划线）
    重命名规则：
      1. Snap-..._b0c1... → {type}_{id}_marker.tiff
      2. Snap-..._b0c0... → {type}_{id}.tiff
      3. RoiSet_{id}_DAPI.npy → {type}_{id}_dapimultimask.npy
      4. RoiSet_{id}_PECAM.npy → {type}_{id}_cellbodiesmultimask.npy
      5. RoiSet_CellBodies_Final.npy → {type}_{id}_cellbodies.npy
    """
    sample_dir_re = re.compile(r"^([A-Za-z0-9]+)_(\d+)$")
    snap_base      = re.compile(r"Snap-(\d+)_b0c0.*\.tiff?$")
    snap_marker    = re.compile(r"Snap-(\d+)_b0c1.*\.tiff?$")
    roi_dapi       = re.compile(r"RoiSet_(\d+)_DAPI\.npy$")
    roi_pecam      = re.compile(r"RoiSet_(\d+)_PECAM\.npy$")
    roi_cellbodies = re.compile(r"RoiSet_CellBodies_Final\.npy$")

    os.makedirs(dest_root, exist_ok=True)

    for group in os.listdir(top_root):
        group_path = os.path.join(top_root, group)
        if not os.path.isdir(group_path):
            continue
        for sample in os.listdir(group_path):
            sample_path = os.path.join(group_path, sample)
            m = sample_dir_re.match(sample)
            if not (os.path.isdir(sample_path) and m):
                continue
            cell_type, sample_id = m.group(1).lower(), m.group(2)
            prefix = f"{cell_type}_{sample_id}"

            # 准备目标子文件夹
            dst_folder = os.path.join(dest_root, prefix)
            os.makedirs(dst_folder, exist_ok=True)

            # 第三层：处理样本目录里的文件
            for fn in os.listdir(sample_path):
                src = os.path.join(sample_path, fn)
                new_fn = None

                if snap_marker.match(fn):
                    new_fn = f"{prefix}_marker.tiff"
                elif snap_base.match(fn):
                    new_fn = f"{prefix}.tiff"
                elif roi_dapi.match(fn):
                    new_fn = f"{prefix}_dapimultimask.npy"
                elif roi_pecam.match(fn):
                    new_fn = f"{prefix}_cellbodiesmultimask.npy"
                elif roi_cellbodies.match(fn):
                    new_fn = f"{prefix}_cellbodies.npy"

                if new_fn:
                    dst = os.path.join(dst_folder, new_fn)
                    shutil.copy2(src, dst)
                    print(f"✅ {sample}/{fn} → {prefix}/{new_fn}")
                else:
                    print(f"Skip {sample}/{fn}")

    print("All files renamed and copied to:", dest_root)

# 调用示例：
# batch_rename(
#     '/Users/macbookair/Downloads/new524',
#     '/Users/macbookair/Downloads/new524/renamed'
# )

import os

def find_sample_dir(path, sample_id):
    """Recursively scan using scandir; return path when you spot sample_id."""
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_dir(follow_symlinks=False):
                if entry.name == sample_id:
                    return entry.path
                # recurse into subdir
                found = find_sample_dir(entry.path, sample_id)
                if found:
                    return found
    return None

def get_raw(sample_id: str, input_dir: str):
    sample_path = find_sample_dir(input_dir, sample_id)
    if not sample_path:
        raise FileNotFoundError(f"No folder named '{sample_id}' in '{input_dir}'")
    dapi = marker = None
    with os.scandir(sample_path) as it:
        for entry in it:
            if not entry.is_file() or not entry.name.lower().endswith('.tiff'):
                continue
            if 'b0c0' in entry.name:
                dapi = entry.path
            elif 'b0c1' in entry.name:
                marker = entry.path
            if dapi and marker:
                break
    return dapi, marker


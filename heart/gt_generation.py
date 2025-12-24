import os
import numpy as np
from pathlib import Path
from PIL import Image
import zipfile
import struct
from tqdm import tqdm
import json

def read_roi_zip(zip_path):
    """从ImageJ ROI zip文件读取ROI坐标"""
    rois = []
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for name in zf.namelist():
            if not name.endswith('.roi'):
                continue
            with zf.open(name) as f:
                roi = parse_roi(f.read())
                if roi is not None:
                    rois.append(roi)
    return rois

def parse_roi(roi_bytes):
    """解析ImageJ ROI格式 - 支持多种类型"""
    if len(roi_bytes) < 64:
        return None
    
    # ROI header parsing
    magic = roi_bytes[:4]
    if magic != b'Iout':
        return None
    
    # Get ROI type and coordinates
    version = struct.unpack('>h', roi_bytes[4:6])[0]
    roi_type = struct.unpack('>h', roi_bytes[6:8])[0]
    top = struct.unpack('>h', roi_bytes[8:10])[0]
    left = struct.unpack('>h', roi_bytes[10:12])[0]
    bottom = struct.unpack('>h', roi_bytes[12:14])[0]
    right = struct.unpack('>h', roi_bytes[14:16])[0]
    n_coordinates = struct.unpack('>H', roi_bytes[16:18])[0]  # unsigned short
    
    # ROI Types:
    # 0 = polygon
    # 1 = rect
    # 2 = oval
    # 3 = line
    # 4 = freeline
    # 5 = polyline
    # 6 = noROI
    # 7 = freehand (traced)
    # 8 = traced
    # 9 = angle
    # 10 = point
    # 1792 = traced/freehand (0x700)
    
    if roi_type in [0, 7, 1792]:  # Polygon or Freehand or Traced
        # For version 228+, coordinates are at offset 64
        coords_offset = 64
        x_coords = []
        y_coords = []
        
        for i in range(n_coordinates):
            if coords_offset + (i+1)*2 > len(roi_bytes):
                break
            x = struct.unpack('>h', roi_bytes[coords_offset + i*2:coords_offset + i*2 + 2])[0]
            x_coords.append(x + left)
        
        y_offset = coords_offset + n_coordinates * 2
        for i in range(n_coordinates):
            if y_offset + (i+1)*2 > len(roi_bytes):
                break
            y = struct.unpack('>h', roi_bytes[y_offset + i*2:y_offset + i*2 + 2])[0]
            y_coords.append(y + top)
        
        if len(x_coords) == len(y_coords) and len(x_coords) > 0:
            return {'x': x_coords, 'y': y_coords, 'type': roi_type}
        
    elif roi_type == 1:  # Rectangle
        return {
            'x': [left, right, right, left],
            'y': [top, top, bottom, bottom],
            'type': roi_type
        }
    
    elif roi_type == 10:  # Point
        # For point ROIs, use the bounding box center
        return {
            'x': [(left + right) // 2],
            'y': [(top + bottom) // 2],
            'type': roi_type
        }
    
    return None

def create_mask_from_rois(rois, img_shape):
    """从ROI列表创建mask，每个细胞核用唯一ID标记"""
    from skimage.draw import polygon
    mask = np.zeros(img_shape, dtype=np.uint16)
    
    for idx, roi in enumerate(rois, start=1):
        if roi is None:
            continue
        
        x_coords = np.array(roi['x'])
        y_coords = np.array(roi['y'])
        
        # Ensure coordinates are within image bounds
        x_coords = np.clip(x_coords, 0, img_shape[1] - 1)
        y_coords = np.clip(y_coords, 0, img_shape[0] - 1)
        
        if len(x_coords) >= 3:  # Need at least 3 points for polygon
            rr, cc = polygon(y_coords, x_coords, shape=img_shape)
            mask[rr, cc] = idx
        elif len(x_coords) > 0:  # Single point
            mask[int(y_coords[0]), int(x_coords[0])] = idx
    
    return mask

def process_heart_annotations(base_dir, output_dir):
    """处理所有心脏区域的标注，保持目录结构"""
    base_path = Path(base_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    regions = ['LA', 'RA', 'LV', 'RV', 'SEP']
    cell_types = ['Epi', 'Immune', 'Mural']
    
    stats = []
    mapping = []  # 保存对应关系
    
    for region in regions:
        region_dir = base_path / region
        if not region_dir.exists():
            print(f"⚠️ Region {region} not found")
            continue
        
        # 创建输出子目录
        output_region_dir = output_path / region
        output_region_dir.mkdir(parents=True, exist_ok=True)
        
        # Get TIF files to determine image dimensions
        tif_files = list(region_dir.glob('*.tif'))
        
        for tif_file in tif_files:
            area_name = tif_file.stem  # e.g., 'LA1'
            img = Image.open(tif_file)
            img_shape = (img.height, img.width)
            
            print(f"\n📍 Processing {region}/{area_name} (shape: {img_shape})")
            
            for cell_type in cell_types:
                zip_file = region_dir / f"{cell_type}-{area_name}.zip"
                
                if not zip_file.exists():
                    print(f"  ⚠️ {cell_type}: file not found")
                    continue
                
                # Read ROIs
                rois = read_roi_zip(zip_file)
                print(f"  ✓ {cell_type}: {len(rois)} nuclei")
                
                if len(rois) == 0:
                    print(f"    ⚠️ Warning: No valid ROIs parsed!")
                    continue
                
                # Create mask
                mask = create_mask_from_rois(rois, img_shape)
                
                # Save mask with same directory structure
                mask_filename = f"{cell_type}-{area_name}_mask.npy"
                mask_path = output_region_dir / mask_filename
                np.save(mask_path, mask)
                
                # 记录对应关系
                mapping.append({
                    'region': region,
                    'area': area_name,
                    'cell_type': cell_type,
                    'image_path': str(tif_file.relative_to(base_path)),
                    'roi_zip_path': str(zip_file.relative_to(base_path)),
                    'mask_path': str(mask_path.relative_to(output_path)),
                    'image_absolute_path': str(tif_file),
                    'roi_zip_absolute_path': str(zip_file),
                    'mask_absolute_path': str(mask_path),
                    'n_nuclei': len(rois),
                    'image_shape': img_shape
                })
                
                stats.append({
                    'region': region,
                    'area': area_name,
                    'cell_type': cell_type,
                    'n_nuclei': len(rois),
                    'image_shape': img_shape,
                })
    
    # Save statistics
    import pandas as pd
    df_stats = pd.DataFrame(stats)
    df_stats.to_csv(output_path / 'annotation_stats.csv', index=False)
    
    # Save mapping (对应关系)
    df_mapping = pd.DataFrame(mapping)
    df_mapping.to_csv(output_path / 'file_mapping.csv', index=False)
    
    # Save mapping as JSON for easier programmatic access
    with open(output_path / 'file_mapping.json', 'w') as f:
        json.dump(mapping, f, indent=2)
    
    print(f"\n✅ Processing complete!")
    print(f"📊 Stats saved to: {output_path / 'annotation_stats.csv'}")
    print(f"🔗 Mapping saved to: {output_path / 'file_mapping.csv'} and {output_path / 'file_mapping.json'}")
    print(f"\nTotal masks created: {len(stats)}")
    print(f"\n📈 Summary by cell type:")
    print(df_stats.groupby('cell_type')['n_nuclei'].agg(['sum', 'mean', 'std']))
    print(f"\n📈 Summary by region:")
    print(df_stats.groupby('region')['n_nuclei'].sum())

if __name__ == "__main__":
    base_dir = "/ihome/jbwang/liy121/ifimage/heart/raw"
    output_dir = "./ground_truth_masks"
    
    process_heart_annotations(base_dir, output_dir)
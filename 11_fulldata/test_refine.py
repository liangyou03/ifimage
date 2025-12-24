"""
TEST VERSION: Refine segmentation using marker channel intensity
Only processes 5 images to verify logic
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tifffile import imread
from skimage.filters import threshold_otsu
from scipy import ndimage as ndi

# Paths
base_path = Path('/ihome/jbwang/liy121/ifimage/11_fulldata/raw')
seg_base = Path('/ihome/jbwang/liy121/ifimage/11_fulldata/segmentation')

markers = ['GFAP', 'iba1', 'NeuN', 'Olig2', 'PECAM']

# Load file mapping
df_map = pd.read_csv(seg_base / 'file_mapping_with_stats.csv')
df_map = df_map[df_map['mask_exists'] == True]
print(f"Loaded {len(df_map)} entries with masks")


def find_marker_channel(c0_path, marker):
    """Find c1 (marker) file based on c0 path and marker type"""
    c0_str = str(c0_path)
    if marker == 'NeuN':
        c1 = Path(c0_str.replace('dapi-b0c0', 'NeuN-b0c1'))
    else:
        c1 = Path(c0_str.replace('c0', 'c1'))
    return c1


def refine_mask(mask, marker_img, min_area=100):
    """
    Refine mask using marker intensity
    """
    ids = np.unique(mask)
    ids = ids[ids > 0]
    
    if ids.size == 0:
        return 0, 0, mask * 0, None
    
    # Calculate Otsu threshold
    union = marker_img[mask > 0]
    if union.size == 0:
        return 0, 0, mask * 0, None
    
    try:
        thr = threshold_otsu(union)
    except:
        thr = np.median(union)
    
    n_total = 0
    positive_ids = []
    cell_stats = []
    
    for k in ids:
        cell_mask = (mask == k)
        area = int(cell_mask.sum())
        
        if area < min_area:
            continue
        
        n_total += 1
        mean_intensity = float(marker_img[cell_mask].mean())
        is_positive = mean_intensity >= thr
        
        cell_stats.append({
            'cell_id': k,
            'area': area,
            'mean_intensity': mean_intensity,
            'threshold': thr,
            'is_positive': is_positive
        })
        
        if is_positive:
            positive_ids.append(k)
    
    n_positive = len(positive_ids)
    
    # Create refined mask
    refined = np.zeros_like(mask, dtype=np.int32)
    if positive_ids:
        out_mask = np.isin(mask, positive_ids)
        refined[out_mask] = ndi.label(out_mask)[0][out_mask]
    
    return n_total, n_positive, refined, cell_stats


# Test on 5 images (1 per marker)
print("\n" + "="*60)
print("TESTING REFINEMENT ON 5 IMAGES")
print("="*60)

for marker in markers:
    df_marker = df_map[df_map['marker'] == marker].head(1)
    
    if len(df_marker) == 0:
        print(f"\n[{marker}] No data found")
        continue
    
    row = df_marker.iloc[0]
    print(f"\n[{marker}] Testing: {row['sample_name']}")
    print(f"  Participant: {row['participant_id']}")
    
    # Load mask
    out_dir = Path(row['out_dir'])
    mask_file = out_dir / 'mask_merged.npy'
    
    if not mask_file.exists():
        print(f"  ✗ Mask not found: {mask_file}")
        continue
    
    mask = np.load(mask_file)
    print(f"  Mask shape: {mask.shape}, unique labels: {len(np.unique(mask))}")
    
    # Find marker channel
    c0_path = Path(row['c0_path'])
    c1_path = find_marker_channel(c0_path, marker)
    
    if not c1_path.exists():
        print(f"  ✗ Marker channel not found: {c1_path}")
        continue
    
    print(f"  c1 path: {c1_path.name}")
    
    # Load marker image
    marker_img = imread(c1_path)
    print(f"  Marker image shape: {marker_img.shape}, dtype: {marker_img.dtype}")
    
    # Convert to grayscale if needed
    if marker_img.ndim == 3:
        marker_img = marker_img[:, :, 0].astype(np.float32)
    else:
        marker_img = marker_img.astype(np.float32)
    
    print(f"  Marker intensity range: [{marker_img.min():.1f}, {marker_img.max():.1f}]")
    
    # Refine
    n_total, n_positive, refined, cell_stats = refine_mask(mask, marker_img, min_area=100)
    
    ratio = n_positive / n_total if n_total > 0 else 0
    
    print(f"\n  RESULTS:")
    print(f"    Total cells (area>=100): {n_total}")
    print(f"    Marker positive: {n_positive}")
    print(f"    Ratio: {ratio:.3f}")
    
    if cell_stats:
        print(f"\n  Sample cell stats (first 5):")
        for cs in cell_stats[:5]:
            status = "+" if cs['is_positive'] else "-"
            print(f"    Cell {cs['cell_id']}: area={cs['area']}, "
                  f"mean={cs['mean_intensity']:.1f}, thr={cs['threshold']:.1f} [{status}]")

print("\n" + "="*60)
print("TEST COMPLETE")
print("="*60)
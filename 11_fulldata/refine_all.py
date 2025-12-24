"""
Refine segmentation using marker channel intensity (Otsu threshold)
Then recalculate marker positive ratio

Logic:
1. Load merged mask (segmentation result)
2. Load marker channel (c1) intensity image
3. For each cell in mask, check if mean marker intensity >= Otsu threshold
4. Keep only marker-positive cells
5. Calculate ratio = marker_positive_cells / total_cells
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tifffile import imread
from skimage.filters import threshold_otsu
from scipy import ndimage as ndi
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Paths
base_path = Path('/ihome/jbwang/liy121/ifimage/11_fulldata/raw')
seg_base = Path('/ihome/jbwang/liy121/ifimage/11_fulldata/segmentation')
out_base = Path('/ihome/jbwang/liy121/ifimage/11_fulldata/segmentation_refined')
out_base.mkdir(exist_ok=True)

markers = ['GFAP', 'iba1', 'NeuN', 'Olig2', 'PECAM']

# Load file mapping
df_map = pd.read_csv(seg_base / 'file_mapping_with_stats.csv')
print(f"Loaded {len(df_map)} entries")


def find_marker_channel(c0_path, marker):
    """Find c1 (marker) file based on c0 path and marker type"""
    c0_str = str(c0_path)
    if marker == 'NeuN':
        c1 = Path(c0_str.replace('dapi-b0c0', 'NeuN-b0c1'))
    else:
        c1 = Path(c0_str.replace('c0', 'c1'))
    return c1


def refine_mask(mask, marker_img, min_area=10):
    """
    Refine mask using marker intensity
    
    Args:
        mask: segmentation mask (labeled)
        marker_img: marker channel intensity image
        min_area: minimum cell area to keep
    
    Returns:
        n_total: total cells in mask
        n_positive: marker-positive cells
        refined_mask: mask with only marker-positive cells
    """
    # Get unique cell IDs
    ids = np.unique(mask)
    ids = ids[ids > 0]
    
    if ids.size == 0:
        return 0, 0, mask * 0
    
    # Calculate Otsu threshold on all masked pixels
    union = marker_img[mask > 0]
    if union.size == 0:
        return 0, 0, mask * 0
    
    try:
        thr = threshold_otsu(union)
    except:
        thr = np.median(union)
    
    n_total = 0
    positive_ids = []
    
    for k in ids:
        cell_mask = (mask == k)
        area = int(cell_mask.sum())
        
        if area < min_area:
            continue
        
        n_total += 1
        mean_intensity = float(marker_img[cell_mask].mean())
        
        if mean_intensity >= thr:
            positive_ids.append(k)
    
    n_positive = len(positive_ids)
    
    # Create refined mask
    refined = np.zeros_like(mask, dtype=np.int32)
    if positive_ids:
        out_mask = np.isin(mask, positive_ids)
        refined[out_mask] = ndi.label(out_mask)[0][out_mask]
    
    return n_total, n_positive, refined


# Process all files
results = []
errors = []

for idx, row in tqdm(df_map.iterrows(), total=len(df_map), desc="Refining"):
    try:
        marker = row['marker']
        out_dir = Path(row['out_dir'])
        c0_path = Path(row['c0_path'])
        
        # Skip if mask doesn't exist
        mask_file = out_dir / 'mask_merged.npy'
        if not mask_file.exists():
            continue
        
        # Find marker channel image
        c1_path = find_marker_channel(c0_path, marker)
        if not c1_path.exists():
            errors.append(f"No c1: {c0_path}")
            continue
        
        # Load mask and marker image
        mask = np.load(mask_file)
        marker_img = imread(c1_path)
        
        # Convert to grayscale if needed
        if marker_img.ndim == 3:
            marker_img = marker_img[:, :, 0].astype(np.float32)
        else:
            marker_img = marker_img.astype(np.float32)
        
        # Refine
        n_total, n_positive, refined_mask = refine_mask(mask, marker_img, min_area=100)
        
        # Save refined mask
        refined_out_dir = out_base / marker / row['participant_id'] / row['sample_name']
        refined_out_dir.mkdir(parents=True, exist_ok=True)
        np.save(refined_out_dir / 'mask_refined.npy', refined_mask)
        
        # Calculate ratio
        ratio = n_positive / n_total if n_total > 0 else np.nan
        
        results.append({
            'marker': marker,
            'participant_id': row['participant_id'],
            'sample_name': row['sample_name'],
            'c0_path': str(c0_path),
            'c1_path': str(c1_path),
            'out_dir': str(refined_out_dir),
            'n_total_cells': n_total,
            'n_marker_positive': n_positive,
            'marker_positive_ratio_refined': ratio
        })
        
    except Exception as e:
        errors.append(f"{row.get('c0_path', 'unknown')}: {e}")
        continue

# Save results
df_results = pd.DataFrame(results)
df_results.to_csv(out_base / 'file_mapping_refined.csv', index=False)

print(f"\n{'='*60}")
print(f"✓ Refined {len(df_results)} images")
print(f"  Errors: {len(errors)}")
print(f"  Output: {out_base / 'file_mapping_refined.csv'}")

# Save error log
if errors:
    with open(out_base / 'errors.log', 'w') as f:
        f.write('\n'.join(errors))
    print(f"  Error log: {out_base / 'errors.log'}")

# Summary stats
print(f"\nRefined ratio summary by marker:")
for marker in markers:
    df_m = df_results[df_results['marker'] == marker]
    if len(df_m) > 0:
        print(f"  {marker}: mean={df_m['marker_positive_ratio_refined'].mean():.3f}, "
              f"std={df_m['marker_positive_ratio_refined'].std():.3f}, n={len(df_m)}")

print(f"{'='*60}")
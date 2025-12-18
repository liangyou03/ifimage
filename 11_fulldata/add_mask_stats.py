"""
Add mask statistics to file_mapping.csv
Extracts: nuc_cell_count, marker_cell_count, merged_cell_count
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Paths
seg_base = Path('/ihome/jbwang/liy121/ifimage/11_fulldata/segmentation')
mapping_file = seg_base / 'file_mapping.csv'

print(f"Loading: {mapping_file}")
df = pd.read_csv(mapping_file)
print(f"Total entries: {len(df)}")

# Add new columns
df['nuc_cell_count'] = None
df['marker_cell_count'] = None
df['merged_cell_count'] = None
df['mask_exists'] = False

# Process each row
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Reading masks"):
    out_dir = Path(row['out_dir'])
    
    # Check if masks exist
    mask_nuc = out_dir / 'mask_nuc.npy'
    mask_marker = out_dir / 'mask_marker.npy'
    mask_merged = out_dir / 'mask_merged.npy'
    
    if not all([mask_nuc.exists(), mask_marker.exists(), mask_merged.exists()]):
        continue
    
    try:
        # Load masks and get max values (cell count)
        nuc = np.load(mask_nuc)
        marker = np.load(mask_marker)
        merged = np.load(mask_merged)
        
        df.at[idx, 'nuc_cell_count'] = int(nuc.max())
        df.at[idx, 'marker_cell_count'] = int(marker.max())
        df.at[idx, 'merged_cell_count'] = int(merged.max())
        df.at[idx, 'mask_exists'] = True
        
    except Exception as e:
        print(f"\nError at {out_dir}: {e}")
        continue

# Save updated CSV
output_file = seg_base / 'file_mapping_with_stats.csv'
df.to_csv(output_file, index=False)

# Summary
print(f"\n{'='*60}")
print(f"✓ Saved: {output_file}")
print(f"  Total entries: {len(df)}")
print(f"  With masks: {df['mask_exists'].sum()}")
print(f"  Missing masks: {(~df['mask_exists']).sum()}")
print(f"\nCell count statistics:")
for col in ['nuc_cell_count', 'marker_cell_count', 'merged_cell_count']:
    valid = df[df['mask_exists']][col]
    print(f"  {col}: mean={valid.mean():.1f}, median={valid.median():.1f}, max={valid.max():.0f}")
print(f"{'='*60}")
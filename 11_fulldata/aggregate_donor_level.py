"""
Aggregate segmentation results at donor/projid level
Merge with clinical metadata
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Paths
seg_base = Path('/ihome/jbwang/liy121/ifimage/11_fulldata/segmentation')
mapping_file = seg_base / 'file_mapping_with_stats.csv'
clinical_file = Path('/ihome/jbwang/liy121/ifimage/11_fulldata/ROSMAP_clinical_n69.csv')

print("Loading data...")
df_map = pd.read_csv(mapping_file)
df_clinical = pd.read_csv(clinical_file)

print(f"  Mapping entries: {len(df_map)}")
print(f"  Clinical donors: {len(df_clinical)}")

# Only keep rows with masks
df_map = df_map[df_map['mask_exists'] == True].copy()
print(f"  With masks: {len(df_map)}")

# Convert participant_id to projid (use string for safe merge)
df_map['projid'] = df_map['participant_id'].astype(str)

# Also ensure clinical projid is string
df_clinical['projid'] = df_clinical['projid'].astype(str)

# Aggregate by donor (projid) and marker
agg_dict = {
    'sample_name': 'count',  # number of images
    'nuc_cell_count': ['mean', 'std', 'sum'],
    'marker_cell_count': ['mean', 'std', 'sum'],
    'merged_cell_count': ['mean', 'std', 'sum']
}

print("\nAggregating by donor and marker...")
df_agg = df_map.groupby(['projid', 'marker']).agg(agg_dict).reset_index()

# Flatten column names
df_agg.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                   for col in df_agg.columns.values]
df_agg.rename(columns={'sample_name_count': 'n_images'}, inplace=True)

# Pivot to wide format (one row per donor, columns for each marker)
markers = ['GFAP', 'iba1', 'NeuN', 'Olig2', 'PECAM']
df_wide_list = []

for marker in markers:
    df_marker = df_agg[df_agg['marker'] == marker].copy()
    df_marker = df_marker.drop('marker', axis=1)
    # Add marker prefix to columns
    df_marker.columns = ['projid' if col == 'projid' else f'{marker}_{col}' 
                         for col in df_marker.columns]
    df_wide_list.append(df_marker)

# Merge all markers
df_donor = df_wide_list[0]
for df in df_wide_list[1:]:
    df_donor = df_donor.merge(df, on='projid', how='outer')

# Merge with clinical data
print("\nMerging with clinical metadata...")
df_final = df_clinical.merge(df_donor, on='projid', how='left')

# Calculate marker-positive proportions
for marker in markers:
    nuc_col = f'{marker}_nuc_cell_count_sum'
    marker_col = f'{marker}_marker_cell_count_sum'
    if nuc_col in df_final.columns and marker_col in df_final.columns:
        df_final[f'{marker}_marker_positive_ratio'] = (
            df_final[marker_col] / df_final[nuc_col]
        )

# Save
output_file = seg_base / 'donor_level_aggregation.csv'
df_final.to_csv(output_file, index=False)

print(f"\n{'='*60}")
print(f"✓ Saved: {output_file}")
print(f"  Total donors: {len(df_final)}")
print(f"  Donors with imaging data: {df_final['GFAP_n_images'].notna().sum()}")
print(f"\nSummary by marker:")
for marker in markers:
    col = f'{marker}_n_images'
    if col in df_final.columns:
        n = df_final[col].notna().sum()
        mean_imgs = df_final[col].mean()
        print(f"  {marker}: {n} donors, avg {mean_imgs:.1f} images/donor")
print(f"{'='*60}")
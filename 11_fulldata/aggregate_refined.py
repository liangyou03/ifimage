"""
Aggregate refined segmentation results at donor level
Merge with clinical data
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
seg_base = Path('/ihome/jbwang/liy121/ifimage/11_fulldata/segmentation_refined')
clinical_file = Path('/ihome/jbwang/liy121/ifimage/11_fulldata/ROSMAP_clinical_n69.csv')

print("Loading data...")
df_refined = pd.read_csv(seg_base / 'file_mapping_refined.csv')
df_clinical = pd.read_csv(clinical_file)

print(f"  Refined entries: {len(df_refined)}")
print(f"  Clinical donors: {len(df_clinical)}")

# Convert to string for safe merge
df_refined['projid'] = df_refined['participant_id'].astype(str)
df_clinical['projid'] = df_clinical['projid'].astype(str)

markers = ['GFAP', 'iba1', 'NeuN', 'Olig2', 'PECAM']

# Aggregate by donor and marker
agg_dict = {
    'sample_name': 'count',
    'n_total_cells': ['mean', 'std', 'sum'],
    'n_marker_positive': ['mean', 'std', 'sum'],
    'marker_positive_ratio_refined': ['mean', 'std']
}

print("\nAggregating by donor and marker...")
df_agg = df_refined.groupby(['projid', 'marker']).agg(agg_dict).reset_index()

# Flatten columns
df_agg.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                   for col in df_agg.columns.values]
df_agg.rename(columns={'sample_name_count': 'n_images'}, inplace=True)

# Pivot to wide format
df_wide_list = []
for marker in markers:
    df_marker = df_agg[df_agg['marker'] == marker].copy()
    df_marker = df_marker.drop('marker', axis=1)
    df_marker.columns = ['projid' if col == 'projid' else f'{marker}_{col}' 
                         for col in df_marker.columns]
    df_wide_list.append(df_marker)

df_donor = df_wide_list[0]
for df in df_wide_list[1:]:
    df_donor = df_donor.merge(df, on='projid', how='outer')

# Also calculate donor-level ratio from sums
for marker in markers:
    total_col = f'{marker}_n_total_cells_sum'
    pos_col = f'{marker}_n_marker_positive_sum'
    if total_col in df_donor.columns and pos_col in df_donor.columns:
        df_donor[f'{marker}_ratio_from_sum'] = df_donor[pos_col] / df_donor[total_col]

# Merge with clinical
print("Merging with clinical metadata...")
df_final = df_clinical.merge(df_donor, on='projid', how='left')

# Save
output_file = seg_base / 'donor_level_aggregation_refined.csv'
df_final.to_csv(output_file, index=False)

print(f"\n{'='*60}")
print(f"✓ Saved: {output_file}")
print(f"  Total donors: {len(df_final)}")
print(f"  Donors with refined data: {df_final[f'{markers[0]}_n_images'].notna().sum()}")

print(f"\nSummary by marker (refined ratio):")
for marker in markers:
    ratio_col = f'{marker}_marker_positive_ratio_refined_mean'
    if ratio_col in df_final.columns:
        valid = df_final[ratio_col].dropna()
        print(f"  {marker}: mean={valid.mean():.3f}, std={valid.std():.3f}, n={len(valid)}")

print(f"{'='*60}")
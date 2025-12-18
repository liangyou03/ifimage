"""
Merge NeuN results into main file_mapping and re-aggregate
Run this after redo_neun.py completes
"""

import pandas as pd
from pathlib import Path

seg_base = Path('/ihome/jbwang/liy121/ifimage/11_fulldata/segmentation')
clinical_file = Path('/ihome/jbwang/liy121/ifimage/11_fulldata/ROSMAP_clinical_n69.csv')

print("Loading files...")

# Load main mapping (without NeuN stats)
df_main = pd.read_csv(seg_base / 'file_mapping_with_stats.csv')
print(f"  Main mapping: {len(df_main)} rows")

# Load new NeuN mapping
df_neun = pd.read_csv(seg_base / 'NeuN' / 'file_mapping_neun.csv')
print(f"  NeuN mapping: {len(df_neun)} rows")

# Remove old NeuN entries from main
df_main = df_main[df_main['marker'] != 'NeuN']
print(f"  Main without NeuN: {len(df_main)} rows")

# Combine
df_combined = pd.concat([df_main, df_neun], ignore_index=True)
print(f"  Combined: {len(df_combined)} rows")

# Save updated mapping
df_combined.to_csv(seg_base / 'file_mapping_with_stats.csv', index=False)
print(f"✓ Saved: file_mapping_with_stats.csv")

# Re-aggregate at donor level
print("\nRe-aggregating at donor level...")

df_clinical = pd.read_csv(clinical_file)
df_clinical['projid'] = df_clinical['projid'].astype(str)

df_map = df_combined[df_combined['mask_exists'] == True].copy()
df_map['projid'] = df_map['participant_id'].astype(str)

# Aggregate
markers = ['GFAP', 'iba1', 'NeuN', 'Olig2', 'PECAM']
agg_dict = {
    'sample_name': 'count',
    'nuc_cell_count': ['mean', 'std', 'sum'],
    'marker_cell_count': ['mean', 'std', 'sum'],
    'merged_cell_count': ['mean', 'std', 'sum']
}

df_agg = df_map.groupby(['projid', 'marker']).agg(agg_dict).reset_index()
df_agg.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                   for col in df_agg.columns.values]
df_agg.rename(columns={'sample_name_count': 'n_images'}, inplace=True)

# Pivot wide
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

# Merge with clinical
df_final = df_clinical.merge(df_donor, on='projid', how='left')

# Marker-positive ratios
for marker in markers:
    nuc_col = f'{marker}_nuc_cell_count_sum'
    marker_col = f'{marker}_marker_cell_count_sum'
    if nuc_col in df_final.columns and marker_col in df_final.columns:
        df_final[f'{marker}_marker_positive_ratio'] = df_final[marker_col] / df_final[nuc_col]

# Save
df_final.to_csv(seg_base / 'donor_level_aggregation.csv', index=False)

print(f"\n{'='*60}")
print(f"✓ Saved: donor_level_aggregation.csv")
print(f"  Total donors: {len(df_final)}")
print(f"\nSummary by marker:")
for marker in markers:
    col = f'{marker}_n_images'
    if col in df_final.columns:
        n = df_final[col].notna().sum()
        mean_imgs = df_final[col].mean()
        print(f"  {marker}: {n} donors, avg {mean_imgs:.1f} images/donor")
print(f"{'='*60}")
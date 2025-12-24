"""
Clinical Association Analysis - Genome Biology Style
For REFINED segmentation data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Genome Biology Figure Style Settings
# ============================================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# Color palette
COLORS = {
    'GFAP': '#E64B35',
    'iba1': '#4DBBD5',
    'NeuN': '#00A087',
    'Olig2': '#3C5488',
    'PECAM': '#F39B7F',
}

MARKER_DESC = {
    'GFAP': 'Astrocytes',
    'iba1': 'Microglia',
    'NeuN': 'Neurons',
    'Olig2': 'Oligodendrocytes',
    'PECAM': 'Endothelial'
}

# Load data
df = pd.read_csv('/ihome/jbwang/liy121/ifimage/11_fulldata/segmentation_refined/donor_level_aggregation_refined.csv')
out_dir = Path('/ihome/jbwang/liy121/ifimage/11_fulldata/analysis_refined')
out_dir.mkdir(exist_ok=True)

print(f"Loaded {len(df)} donors")
print(f"Columns: {list(df.columns)}")

markers = ['GFAP', 'iba1', 'NeuN', 'Olig2', 'PECAM']

# Check which ratio column exists
sample_marker = markers[0]
if f'{sample_marker}_marker_positive_ratio_refined_mean' in df.columns:
    RATIO_COL = lambda m: f'{m}_marker_positive_ratio_refined_mean'
    TOTAL_COL = lambda m: f'{m}_n_total_cells_sum'
    POSITIVE_COL = lambda m: f'{m}_n_marker_positive_sum'
    print("Using refined ratio columns")
elif f'{sample_marker}_ratio_from_sum' in df.columns:
    RATIO_COL = lambda m: f'{m}_ratio_from_sum'
    TOTAL_COL = lambda m: f'{m}_n_total_cells_sum'
    POSITIVE_COL = lambda m: f'{m}_n_marker_positive_sum'
    print("Using ratio_from_sum columns")
else:
    raise ValueError("Cannot find ratio columns in data")

# ============================================================
# Helper Functions
# ============================================================
def add_correlation_text(ax, x, y, pos='top'):
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 3:
        return
    r, p = stats.spearmanr(x[mask], y[mask])
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    text = f'ρ = {r:.2f}{sig}'
    if pos == 'top':
        ax.text(0.95, 0.95, text, transform=ax.transAxes, ha='right', va='top', fontsize=7)
    else:
        ax.text(0.95, 0.05, text, transform=ax.transAxes, ha='right', va='bottom', fontsize=7)

def add_regression_line(ax, x, y, color='black'):
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 3:
        return
    z = np.polyfit(x[mask], y[mask], 1)
    p = np.poly1d(z)
    x_line = np.linspace(x[mask].min(), x[mask].max(), 100)
    ax.plot(x_line, p(x_line), '--', color=color, alpha=0.7, linewidth=1)

# ============================================================
# Figure 1: Correlation Heatmap
# ============================================================
print("\nGenerating Figure 1: Correlation Heatmap...")

clinical_vars = ['cogdx', 'braaksc', 'ceradsc', 'cts_mmse30_lv', 'age_death', 'pmi']
clinical_labels = ['Cognitive\nDiagnosis', 'Braak\nStage', 'CERAD\nScore', 'MMSE\n(last visit)', 'Age at\nDeath', 'PMI']

corr_matrix = np.zeros((len(markers), len(clinical_vars)))
pval_matrix = np.zeros((len(markers), len(clinical_vars)))

for i, marker in enumerate(markers):
    marker_col = RATIO_COL(marker)
    for j, clin in enumerate(clinical_vars):
        if marker_col not in df.columns or clin not in df.columns:
            continue
        mask = df[[marker_col, clin]].notna().all(axis=1)
        if mask.sum() >= 3:
            r, p = stats.spearmanr(df.loc[mask, marker_col], df.loc[mask, clin])
            corr_matrix[i, j] = r
            pval_matrix[i, j] = p

fig, ax = plt.subplots(figsize=(4.5, 3.5))

annot = np.empty_like(corr_matrix, dtype=object)
for i in range(corr_matrix.shape[0]):
    for j in range(corr_matrix.shape[1]):
        r = corr_matrix[i, j]
        p = pval_matrix[i, j]
        if p < 0.001:
            annot[i, j] = f'{r:.2f}***'
        elif p < 0.01:
            annot[i, j] = f'{r:.2f}**'
        elif p < 0.05:
            annot[i, j] = f'{r:.2f}*'
        else:
            annot[i, j] = f'{r:.2f}'

im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-0.5, vmax=0.5, aspect='auto')

for i in range(len(markers)):
    for j in range(len(clinical_vars)):
        color = 'white' if abs(corr_matrix[i, j]) > 0.3 else 'black'
        ax.text(j, i, annot[i, j], ha='center', va='center', fontsize=6, color=color)

ax.set_xticks(range(len(clinical_vars)))
ax.set_xticklabels(clinical_labels, rotation=45, ha='right')
ax.set_yticks(range(len(markers)))
ax.set_yticklabels([f'{m} ({MARKER_DESC[m]})' for m in markers])

cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('Spearman ρ', fontsize=7)
cbar.ax.tick_params(labelsize=6)

ax.set_title('a', fontsize=10, fontweight='bold', loc='left', x=-0.15)
plt.tight_layout()
plt.savefig(out_dir / 'fig1_correlation_heatmap.pdf', bbox_inches='tight')
plt.savefig(out_dir / 'fig1_correlation_heatmap.png', bbox_inches='tight')
plt.close()
print("✓ Saved: fig1_correlation_heatmap.pdf/png")

# ============================================================
# Figure 2: Scatter Plots - Marker Ratio vs Braak Stage
# ============================================================
print("\nGenerating Figure 2: Marker Ratio vs Braak Stage...")

fig, axes = plt.subplots(1, 5, figsize=(10, 2.2))

for i, marker in enumerate(markers):
    ax = axes[i]
    marker_col = RATIO_COL(marker)
    
    if marker_col not in df.columns:
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
        continue
    
    mask = df[[marker_col, 'braaksc']].notna().all(axis=1)
    x = df.loc[mask, 'braaksc'].values
    y = df.loc[mask, marker_col].values
    
    ax.scatter(x, y, c=COLORS[marker], alpha=0.6, s=25, edgecolors='white', linewidth=0.3)
    add_regression_line(ax, x, y, color='black')
    add_correlation_text(ax, x, y)
    
    ax.set_xlabel('Braak stage')
    if i == 0:
        ax.set_ylabel('Marker+ ratio')
    ax.set_title(f'{marker}', fontsize=8, fontweight='bold')
    ax.set_xticks([0, 1, 2, 3, 4, 5, 6])

plt.suptitle('b', fontsize=10, fontweight='bold', x=0.02, y=1.02)
plt.tight_layout()
plt.savefig(out_dir / 'fig2_scatter_braak.pdf', bbox_inches='tight')
plt.savefig(out_dir / 'fig2_scatter_braak.png', bbox_inches='tight')
plt.close()
print("✓ Saved: fig2_scatter_braak.pdf/png")

# ============================================================
# Figure 3: Scatter Plots - Marker Ratio vs CERAD Score
# ============================================================
print("\nGenerating Figure 3: Marker Ratio vs CERAD Score...")

fig, axes = plt.subplots(1, 5, figsize=(10, 2.2))

for i, marker in enumerate(markers):
    ax = axes[i]
    marker_col = RATIO_COL(marker)
    
    if marker_col not in df.columns:
        continue
    
    mask = df[[marker_col, 'ceradsc']].notna().all(axis=1)
    x = df.loc[mask, 'ceradsc'].values
    y = df.loc[mask, marker_col].values
    
    ax.scatter(x, y, c=COLORS[marker], alpha=0.6, s=25, edgecolors='white', linewidth=0.3)
    add_regression_line(ax, x, y, color='black')
    add_correlation_text(ax, x, y)
    
    ax.set_xlabel('CERAD score')
    if i == 0:
        ax.set_ylabel('Marker+ ratio')
    ax.set_title(f'{marker}', fontsize=8, fontweight='bold')
    ax.set_xticks([1, 2, 3, 4])

plt.suptitle('c', fontsize=10, fontweight='bold', x=0.02, y=1.02)
plt.tight_layout()
plt.savefig(out_dir / 'fig3_scatter_cerad.pdf', bbox_inches='tight')
plt.savefig(out_dir / 'fig3_scatter_cerad.png', bbox_inches='tight')
plt.close()
print("✓ Saved: fig3_scatter_cerad.pdf/png")

# ============================================================
# Figure 4: Scatter Plots - Marker Ratio vs MMSE
# ============================================================
print("\nGenerating Figure 4: Marker Ratio vs MMSE...")

fig, axes = plt.subplots(1, 5, figsize=(10, 2.2))

for i, marker in enumerate(markers):
    ax = axes[i]
    marker_col = RATIO_COL(marker)
    
    if marker_col not in df.columns:
        continue
    
    mask = df[[marker_col, 'cts_mmse30_lv']].notna().all(axis=1)
    x = df.loc[mask, 'cts_mmse30_lv'].values
    y = df.loc[mask, marker_col].values
    
    ax.scatter(x, y, c=COLORS[marker], alpha=0.6, s=25, edgecolors='white', linewidth=0.3)
    add_regression_line(ax, x, y, color='black')
    add_correlation_text(ax, x, y)
    
    ax.set_xlabel('MMSE score')
    if i == 0:
        ax.set_ylabel('Marker+ ratio')
    ax.set_title(f'{marker}', fontsize=8, fontweight='bold')

plt.suptitle('d', fontsize=10, fontweight='bold', x=0.02, y=1.02)
plt.tight_layout()
plt.savefig(out_dir / 'fig4_scatter_mmse.pdf', bbox_inches='tight')
plt.savefig(out_dir / 'fig4_scatter_mmse.png', bbox_inches='tight')
plt.close()
print("✓ Saved: fig4_scatter_mmse.pdf/png")

# ============================================================
# Figure 5: Scatter Plots - Marker Ratio vs Pathology
# ============================================================
print("\nGenerating Figure 5: Marker Ratio vs Pathology...")

pathology_vars = ['plaq_d', 'plaq_n', 'nft', 'gpath']
pathology_labels = ['Diffuse plaques', 'Neuritic plaques', 'Neurofibrillary tangles', 'Global pathology']

fig, axes = plt.subplots(len(markers), len(pathology_vars), figsize=(9, 11))

for i, marker in enumerate(markers):
    marker_col = RATIO_COL(marker)
    for j, (path_var, path_label) in enumerate(zip(pathology_vars, pathology_labels)):
        ax = axes[i, j]
        
        if marker_col not in df.columns or path_var not in df.columns:
            continue
        
        mask = df[[marker_col, path_var]].notna().all(axis=1)
        x = df.loc[mask, path_var].values
        y = df.loc[mask, marker_col].values
        
        ax.scatter(x, y, c=COLORS[marker], alpha=0.6, s=20, edgecolors='white', linewidth=0.3)
        add_regression_line(ax, x, y, color='black')
        add_correlation_text(ax, x, y)
        
        if i == len(markers) - 1:
            ax.set_xlabel(path_label, fontsize=7)
        if j == 0:
            ax.set_ylabel(f'{marker}\nMarker+ ratio', fontsize=7)
        if i == 0:
            ax.set_title(path_label, fontsize=8)

plt.suptitle('e', fontsize=10, fontweight='bold', x=0.02, y=0.995)
plt.tight_layout()
plt.savefig(out_dir / 'fig5_scatter_pathology.pdf', bbox_inches='tight')
plt.savefig(out_dir / 'fig5_scatter_pathology.png', bbox_inches='tight')
plt.close()
print("✓ Saved: fig5_scatter_pathology.pdf/png")

# ============================================================
# Figure 6: Box Plots by Cognitive Diagnosis
# ============================================================
print("\nGenerating Figure 6: Box Plots by Cognitive Diagnosis...")

df['cogdx_group'] = df['cogdx'].map({1: 'NCI', 2: 'MCI', 3: 'MCI', 4: 'AD', 5: 'AD', 6: 'Other'})
df_plot = df[df['cogdx_group'].isin(['NCI', 'MCI', 'AD'])].copy()

fig, axes = plt.subplots(1, 5, figsize=(10, 2.5))

for i, marker in enumerate(markers):
    ax = axes[i]
    marker_col = RATIO_COL(marker)
    
    if marker_col not in df.columns:
        continue
    
    box_data = [df_plot[df_plot['cogdx_group'] == g][marker_col].dropna() for g in ['NCI', 'MCI', 'AD']]
    
    bp = ax.boxplot(box_data, positions=[0, 1, 2], widths=0.6, patch_artist=True,
                    boxprops=dict(facecolor=COLORS[marker], alpha=0.3),
                    medianprops=dict(color='black', linewidth=1.5),
                    whiskerprops=dict(color='black'),
                    capprops=dict(color='black'),
                    flierprops=dict(marker='o', markerfacecolor='gray', markersize=3, alpha=0.5))
    
    for j, (group, data) in enumerate(zip(['NCI', 'MCI', 'AD'], box_data)):
        jitter = np.random.uniform(-0.15, 0.15, len(data))
        ax.scatter(np.full(len(data), j) + jitter, data, c=COLORS[marker], 
                   alpha=0.6, s=15, edgecolors='white', linewidth=0.3, zorder=3)
    
    if all(len(d) > 0 for d in box_data):
        h, p = stats.kruskal(*box_data)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        ax.text(0.5, 0.95, f'p = {p:.3f} ({sig})', transform=ax.transAxes, 
                ha='center', va='top', fontsize=6)
    
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['NCI', 'MCI', 'AD'])
    ax.set_title(f'{marker}', fontsize=8, fontweight='bold')
    if i == 0:
        ax.set_ylabel('Marker+ ratio')

plt.suptitle('f', fontsize=10, fontweight='bold', x=0.02, y=1.02)
plt.tight_layout()
plt.savefig(out_dir / 'fig6_boxplot_cogdx.pdf', bbox_inches='tight')
plt.savefig(out_dir / 'fig6_boxplot_cogdx.png', bbox_inches='tight')
plt.close()
print("✓ Saved: fig6_boxplot_cogdx.pdf/png")

# ============================================================
# Figure 7: Marker-Marker Correlations
# ============================================================
print("\nGenerating Figure 7: Marker-Marker Correlations...")

marker_cols = [RATIO_COL(m) for m in markers if RATIO_COL(m) in df.columns]
df_markers = df[marker_cols].dropna()

fig, ax = plt.subplots(figsize=(4, 3.5))

corr_mm = df_markers.corr(method='spearman')

im = ax.imshow(corr_mm.values, cmap='RdBu_r', vmin=-1, vmax=1)

for i in range(len(corr_mm)):
    for j in range(len(corr_mm)):
        color = 'white' if abs(corr_mm.iloc[i, j]) > 0.5 else 'black'
        ax.text(j, i, f'{corr_mm.iloc[i, j]:.2f}', ha='center', va='center', fontsize=7, color=color)

ax.set_xticks(range(len(markers)))
ax.set_xticklabels(markers, rotation=45, ha='right')
ax.set_yticks(range(len(markers)))
ax.set_yticklabels(markers)

cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('Spearman ρ', fontsize=7)
cbar.ax.tick_params(labelsize=6)

ax.set_title('g', fontsize=10, fontweight='bold', loc='left', x=-0.2)
plt.tight_layout()
plt.savefig(out_dir / 'fig7_marker_correlation.pdf', bbox_inches='tight')
plt.savefig(out_dir / 'fig7_marker_correlation.png', bbox_inches='tight')
plt.close()
print("✓ Saved: fig7_marker_correlation.pdf/png")

# ============================================================
# Figure 8: Cell Count Scatter Plots
# ============================================================
print("\nGenerating Figure 8: Cell Count vs Braak...")

fig, axes = plt.subplots(2, 5, figsize=(10, 4.5))

# Row 1: Total cells vs Braak
for i, marker in enumerate(markers):
    ax = axes[0, i]
    col = TOTAL_COL(marker)
    
    if col not in df.columns:
        continue
    
    mask = df[[col, 'braaksc']].notna().all(axis=1)
    x = df.loc[mask, 'braaksc'].values
    y = df.loc[mask, col].values
    
    ax.scatter(x, y, c=COLORS[marker], alpha=0.6, s=20, edgecolors='white', linewidth=0.3)
    add_regression_line(ax, x, y, color='black')
    add_correlation_text(ax, x, y)
    
    ax.set_xlabel('Braak stage')
    if i == 0:
        ax.set_ylabel('Total cells')
    ax.set_title(f'{marker}', fontsize=8, fontweight='bold')

# Row 2: Marker+ cells vs Braak
for i, marker in enumerate(markers):
    ax = axes[1, i]
    col = POSITIVE_COL(marker)
    
    if col not in df.columns:
        continue
    
    mask = df[[col, 'braaksc']].notna().all(axis=1)
    x = df.loc[mask, 'braaksc'].values
    y = df.loc[mask, col].values
    
    ax.scatter(x, y, c=COLORS[marker], alpha=0.6, s=20, edgecolors='white', linewidth=0.3)
    add_regression_line(ax, x, y, color='black')
    add_correlation_text(ax, x, y)
    
    ax.set_xlabel('Braak stage')
    if i == 0:
        ax.set_ylabel('Marker+ cells')

plt.suptitle('h', fontsize=10, fontweight='bold', x=0.02, y=0.98)
plt.tight_layout()
plt.savefig(out_dir / 'fig8_cellcount_braak.pdf', bbox_inches='tight')
plt.savefig(out_dir / 'fig8_cellcount_braak.png', bbox_inches='tight')
plt.close()
print("✓ Saved: fig8_cellcount_braak.pdf/png")

# ============================================================
# Figure 10: Stacked Bar Plot - Cell Type Composition
# ============================================================
print("\nGenerating Figure 10: Stacked Bar Plot...")

# Use marker positive counts
stack_cols = [POSITIVE_COL(m) for m in markers]
df_stack = df[['projid'] + stack_cols].copy()
df_stack = df_stack.dropna()
df_stack.columns = ['projid'] + markers

# Calculate proportions
df_prop = df_stack[markers].div(df_stack[markers].sum(axis=1), axis=0)
df_prop['projid'] = df_stack['projid'].values

# Sort by NeuN
df_prop = df_prop.sort_values('NeuN', ascending=False).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(12, 4))

stack_order = ['NeuN', 'iba1', 'PECAM', 'GFAP', 'Olig2']
stack_colors = {
    'NeuN': '#4A90D9',
    'iba1': '#E84C8A',
    'PECAM': '#7CB342',
    'GFAP': '#FFB74D',
    'Olig2': '#9CCC65'
}
stack_labels = {
    'NeuN': 'Neurons',
    'iba1': 'Microglia',
    'PECAM': 'Endothelial',
    'GFAP': 'Astrocytes',
    'Olig2': 'Oligodendrocytes'
}

x = np.arange(len(df_prop))
bottom = np.zeros(len(df_prop))

for marker in stack_order:
    values = df_prop[marker].values
    ax.bar(x, values, bottom=bottom, width=1.0, label=stack_labels[marker], 
           color=stack_colors[marker], edgecolor='white', linewidth=0.3)
    bottom += values

ax.set_xlim(-0.5, len(df_prop) - 0.5)
ax.set_ylim(0, 1)
ax.set_xlabel('Donors (sorted by neuronal proportion)', fontsize=8)
ax.set_ylabel('Proportion', fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(df_prop['projid'].astype(str), rotation=90, fontsize=5)

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=5, fontsize=7, frameon=False)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(out_dir / 'fig10_stacked_barplot.pdf', bbox_inches='tight')
plt.savefig(out_dir / 'fig10_stacked_barplot.png', bbox_inches='tight')
plt.close()
print("✓ Saved: fig10_stacked_barplot.pdf/png")

# ============================================================
# Save Statistics Table
# ============================================================
print("\nGenerating statistics tables...")

results = []
all_clinical = ['cogdx', 'braaksc', 'ceradsc', 'cts_mmse30_lv', 'age_death', 'pmi', 'plaq_d', 'plaq_n', 'nft', 'gpath']

for marker in markers:
    marker_col = RATIO_COL(marker)
    if marker_col not in df.columns:
        continue
    for clin in all_clinical:
        if clin not in df.columns:
            continue
        mask = df[[marker_col, clin]].notna().all(axis=1)
        if mask.sum() >= 3:
            r, p = stats.spearmanr(df.loc[mask, marker_col], df.loc[mask, clin])
            results.append({
                'Marker': marker,
                'Clinical_Variable': clin,
                'Spearman_rho': r,
                'p_value': p,
                'n': mask.sum(),
                'Significant': p < 0.05
            })

df_results = pd.DataFrame(results)
df_results.to_csv(out_dir / 'full_correlation_table.csv', index=False)
print("✓ Saved: full_correlation_table.csv")

# ============================================================
# Summary
# ============================================================
print("\n" + "="*60)
print("ANALYSIS COMPLETE - Genome Biology Style (Refined Data)")
print("="*60)
print(f"\nOutput directory: {out_dir}")
print("\nFigures generated:")
for i in [1,2,3,4,5,6,7,8,10]:
    print(f"  - fig{i}_*.pdf/png")
print("\nTables generated:")
print("  - full_correlation_table.csv")
print("="*60)
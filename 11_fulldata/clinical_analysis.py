"""
Clinical Association Analysis - Genome Biology Style
Comprehensive scatter plots for all relevant metrics
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
    'font.family': 'Arial',
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
    'pdf.fonttype': 42,  # TrueType fonts for editing
    'ps.fonttype': 42,
})

# Color palette - professional, colorblind-friendly
COLORS = {
    'GFAP': '#E64B35',    # Red - Astrocytes
    'iba1': '#4DBBD5',    # Cyan - Microglia
    'NeuN': '#00A087',    # Teal - Neurons
    'Olig2': '#3C5488',   # Blue - Oligodendrocytes
    'PECAM': '#F39B7F',   # Salmon - Endothelial
}

MARKER_DESC = {
    'GFAP': 'Astrocytes',
    'iba1': 'Microglia',
    'NeuN': 'Neurons',
    'Olig2': 'Oligodendrocytes',
    'PECAM': 'Endothelial'
}

# Load data
df = pd.read_csv('/ihome/jbwang/liy121/ifimage/11_fulldata/segmentation/donor_level_aggregation.csv')
out_dir = Path('/ihome/jbwang/liy121/ifimage/11_fulldata/analysis')
out_dir.mkdir(exist_ok=True)

print(f"Loaded {len(df)} donors")

markers = ['GFAP', 'iba1', 'NeuN', 'Olig2', 'PECAM']

# ============================================================
# Helper Functions
# ============================================================
def add_correlation_text(ax, x, y, pos='top'):
    """Add Spearman correlation to plot"""
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 3:
        return
    r, p = stats.spearmanr(x[mask], y[mask])
    if p < 0.001:
        sig = '***'
    elif p < 0.01:
        sig = '**'
    elif p < 0.05:
        sig = '*'
    else:
        sig = ''
    
    text = f'ρ = {r:.2f}{sig}'
    if pos == 'top':
        ax.text(0.95, 0.95, text, transform=ax.transAxes, ha='right', va='top', fontsize=7)
    else:
        ax.text(0.95, 0.05, text, transform=ax.transAxes, ha='right', va='bottom', fontsize=7)

def add_regression_line(ax, x, y, color='black'):
    """Add linear regression line"""
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
    marker_col = f'{marker}_marker_positive_ratio'
    for j, clin in enumerate(clinical_vars):
        mask = df[[marker_col, clin]].notna().all(axis=1)
        if mask.sum() >= 3:
            r, p = stats.spearmanr(df.loc[mask, marker_col], df.loc[mask, clin])
            corr_matrix[i, j] = r
            pval_matrix[i, j] = p

fig, ax = plt.subplots(figsize=(4.5, 3.5))

# Create annotation matrix
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

# Add text annotations
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
    marker_col = f'{marker}_marker_positive_ratio'
    
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
    marker_col = f'{marker}_marker_positive_ratio'
    
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
    marker_col = f'{marker}_marker_positive_ratio'
    
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
# Figure 5: Scatter Plots - Marker Ratio vs Pathology Measures
# ============================================================
print("\nGenerating Figure 5: Marker Ratio vs Pathology...")

pathology_vars = ['plaq_d', 'plaq_n', 'nft', 'gpath']
pathology_labels = ['Diffuse plaques', 'Neuritic plaques', 'Neurofibrillary tangles', 'Global pathology']

fig, axes = plt.subplots(len(markers), len(pathology_vars), figsize=(9, 11))

for i, marker in enumerate(markers):
    marker_col = f'{marker}_marker_positive_ratio'
    for j, (path_var, path_label) in enumerate(zip(pathology_vars, pathology_labels)):
        ax = axes[i, j]
        
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
    marker_col = f'{marker}_marker_positive_ratio'
    
    # Box plot with individual points
    box_data = [df_plot[df_plot['cogdx_group'] == g][marker_col].dropna() for g in ['NCI', 'MCI', 'AD']]
    
    bp = ax.boxplot(box_data, positions=[0, 1, 2], widths=0.6, patch_artist=True,
                    boxprops=dict(facecolor=COLORS[marker], alpha=0.3),
                    medianprops=dict(color='black', linewidth=1.5),
                    whiskerprops=dict(color='black'),
                    capprops=dict(color='black'),
                    flierprops=dict(marker='o', markerfacecolor='gray', markersize=3, alpha=0.5))
    
    # Add individual points with jitter
    for j, (group, data) in enumerate(zip(['NCI', 'MCI', 'AD'], box_data)):
        jitter = np.random.uniform(-0.15, 0.15, len(data))
        ax.scatter(np.full(len(data), j) + jitter, data, c=COLORS[marker], 
                   alpha=0.6, s=15, edgecolors='white', linewidth=0.3, zorder=3)
    
    # Kruskal-Wallis test
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

marker_cols = [f'{m}_marker_positive_ratio' for m in markers]
df_markers = df[marker_cols].dropna()

fig, ax = plt.subplots(figsize=(4, 3.5))

corr_mm = df_markers.corr(method='spearman')
mask_upper = np.triu(np.ones_like(corr_mm, dtype=bool), k=1)

im = ax.imshow(corr_mm.values, cmap='RdBu_r', vmin=-1, vmax=1)

for i in range(len(markers)):
    for j in range(len(markers)):
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
print("\nGenerating Figure 8: Cell Count vs Pathology...")

fig, axes = plt.subplots(2, 5, figsize=(10, 4.5))

# Row 1: Nuc cell count vs Braak
for i, marker in enumerate(markers):
    ax = axes[0, i]
    col = f'{marker}_nuc_cell_count_mean'
    
    mask = df[[col, 'braaksc']].notna().all(axis=1)
    x = df.loc[mask, 'braaksc'].values
    y = df.loc[mask, col].values
    
    ax.scatter(x, y, c=COLORS[marker], alpha=0.6, s=20, edgecolors='white', linewidth=0.3)
    add_regression_line(ax, x, y, color='black')
    add_correlation_text(ax, x, y)
    
    ax.set_xlabel('Braak stage')
    if i == 0:
        ax.set_ylabel('Mean nuclei count')
    ax.set_title(f'{marker}', fontsize=8, fontweight='bold')

# Row 2: Marker cell count vs Braak
for i, marker in enumerate(markers):
    ax = axes[1, i]
    col = f'{marker}_marker_cell_count_mean'
    
    mask = df[[col, 'braaksc']].notna().all(axis=1)
    x = df.loc[mask, 'braaksc'].values
    y = df.loc[mask, col].values
    
    ax.scatter(x, y, c=COLORS[marker], alpha=0.6, s=20, edgecolors='white', linewidth=0.3)
    add_regression_line(ax, x, y, color='black')
    add_correlation_text(ax, x, y)
    
    ax.set_xlabel('Braak stage')
    if i == 0:
        ax.set_ylabel('Mean marker+ count')

plt.suptitle('h', fontsize=10, fontweight='bold', x=0.02, y=0.98)
plt.tight_layout()
plt.savefig(out_dir / 'fig8_cellcount_braak.pdf', bbox_inches='tight')
plt.savefig(out_dir / 'fig8_cellcount_braak.png', bbox_inches='tight')
plt.close()
print("✓ Saved: fig8_cellcount_braak.pdf/png")

# ============================================================
# Figure 9: Combined Multi-Panel Figure
# ============================================================
print("\nGenerating Figure 9: Combined Multi-Panel...")

fig = plt.figure(figsize=(7.5, 9))

# Panel a: Heatmap
ax1 = fig.add_subplot(3, 2, 1)
im = ax1.imshow(corr_matrix[:, :3], cmap='RdBu_r', vmin=-0.5, vmax=0.5, aspect='auto')
for i in range(len(markers)):
    for j in range(3):
        color = 'white' if abs(corr_matrix[i, j]) > 0.3 else 'black'
        ax1.text(j, i, annot[i, j], ha='center', va='center', fontsize=6, color=color)
ax1.set_xticks(range(3))
ax1.set_xticklabels(['cogdx', 'braaksc', 'ceradsc'], rotation=45, ha='right')
ax1.set_yticks(range(len(markers)))
ax1.set_yticklabels(markers)
ax1.set_title('a', fontsize=10, fontweight='bold', loc='left')

# Panel b: Marker correlation
ax2 = fig.add_subplot(3, 2, 2)
im2 = ax2.imshow(corr_mm.values, cmap='RdBu_r', vmin=-1, vmax=1)
for i in range(len(markers)):
    for j in range(len(markers)):
        color = 'white' if abs(corr_mm.iloc[i, j]) > 0.5 else 'black'
        ax2.text(j, i, f'{corr_mm.iloc[i, j]:.2f}', ha='center', va='center', fontsize=5, color=color)
ax2.set_xticks(range(len(markers)))
ax2.set_xticklabels(markers, rotation=45, ha='right')
ax2.set_yticks(range(len(markers)))
ax2.set_yticklabels(markers)
ax2.set_title('b', fontsize=10, fontweight='bold', loc='left')

# Panels c-g: Scatter plots for each marker vs Braak
for i, marker in enumerate(markers):
    ax = fig.add_subplot(3, 5, 6 + i)
    marker_col = f'{marker}_marker_positive_ratio'
    
    mask = df[[marker_col, 'braaksc']].notna().all(axis=1)
    x = df.loc[mask, 'braaksc'].values
    y = df.loc[mask, marker_col].values
    
    ax.scatter(x, y, c=COLORS[marker], alpha=0.6, s=15, edgecolors='white', linewidth=0.3)
    add_regression_line(ax, x, y, color='black')
    add_correlation_text(ax, x, y)
    
    ax.set_xlabel('Braak', fontsize=6)
    if i == 0:
        ax.set_ylabel('Marker+', fontsize=6)
    ax.set_title(marker, fontsize=7, fontweight='bold')
    ax.tick_params(labelsize=5)

# Panels h-l: Box plots
for i, marker in enumerate(markers):
    ax = fig.add_subplot(3, 5, 11 + i)
    marker_col = f'{marker}_marker_positive_ratio'
    
    box_data = [df_plot[df_plot['cogdx_group'] == g][marker_col].dropna() for g in ['NCI', 'MCI', 'AD']]
    
    bp = ax.boxplot(box_data, positions=[0, 1, 2], widths=0.5, patch_artist=True,
                    boxprops=dict(facecolor=COLORS[marker], alpha=0.4),
                    medianprops=dict(color='black', linewidth=1),
                    whiskerprops=dict(color='black', linewidth=0.5),
                    capprops=dict(color='black', linewidth=0.5),
                    flierprops=dict(marker='.', markersize=2))
    
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['NCI', 'MCI', 'AD'], fontsize=5)
    if i == 0:
        ax.set_ylabel('Marker+', fontsize=6)
    ax.set_title(marker, fontsize=7, fontweight='bold')
    ax.tick_params(labelsize=5)

fig.text(0.01, 0.66, 'c', fontsize=10, fontweight='bold')
fig.text(0.01, 0.33, 'd', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(out_dir / 'fig9_combined_panel.pdf', bbox_inches='tight')
plt.savefig(out_dir / 'fig9_combined_panel.png', bbox_inches='tight')
plt.close()
print("✓ Saved: fig9_combined_panel.pdf/png")

# ============================================================
# Figure 10: Stacked Bar Plot - Cell Type Composition
# ============================================================
print("\nGenerating Figure 10: Stacked Bar Plot...")

# Prepare data for stacking
df_stack = df[['projid'] + [f'{m}_marker_cell_count_sum' for m in markers]].copy()
df_stack = df_stack.dropna()

# Rename columns
df_stack.columns = ['projid'] + markers

# Calculate proportions (normalize to 100%)
df_prop = df_stack[markers].div(df_stack[markers].sum(axis=1), axis=0)
df_prop['projid'] = df_stack['projid'].values

# Sort by one marker (e.g., NeuN) for better visualization
df_prop = df_prop.sort_values('NeuN', ascending=False).reset_index(drop=True)

# Plot
fig, ax = plt.subplots(figsize=(12, 4))

# Stack order (bottom to top)
stack_order = ['NeuN', 'Microglia', 'Endothelial', 'Astrocytes', 'Oligodendrocytes']
marker_map = {'NeuN': 'NeuN', 'Microglia': 'iba1', 'Endothelial': 'PECAM', 
              'Astrocytes': 'GFAP', 'Oligodendrocytes': 'Olig2'}
stack_colors = {
    'NeuN': '#4A90D9',        # Blue - Neurons
    'Microglia': '#E84C8A',   # Pink - Microglia
    'Endothelial': '#7CB342', # Green - Endothelial
    'Astrocytes': '#FFB74D',  # Orange - Astrocytes
    'Oligodendrocytes': '#9CCC65'  # Light green - Oligodendrocytes
}

x = np.arange(len(df_prop))
bottom = np.zeros(len(df_prop))

for cell_type in stack_order:
    marker = marker_map[cell_type]
    values = df_prop[marker].values
    ax.bar(x, values, bottom=bottom, width=1.0, label=cell_type, 
           color=stack_colors[cell_type], edgecolor='white', linewidth=0.3)
    bottom += values

ax.set_xlim(-0.5, len(df_prop) - 0.5)
ax.set_ylim(0, 1)
ax.set_xlabel('Donors (sorted by neuronal proportion)', fontsize=8)
ax.set_ylabel('Proportion', fontsize=8)

# X-axis labels (projid)
ax.set_xticks(x)
ax.set_xticklabels(df_prop['projid'].astype(str), rotation=90, fontsize=5)

# Legend
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

# Full correlation table
results = []
all_clinical = ['cogdx', 'braaksc', 'ceradsc', 'cts_mmse30_lv', 'age_death', 'pmi', 'plaq_d', 'plaq_n', 'nft', 'gpath']

for marker in markers:
    marker_col = f'{marker}_marker_positive_ratio'
    for clin in all_clinical:
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
print("ANALYSIS COMPLETE - Genome Biology Style")
print("="*60)
print(f"\nOutput directory: {out_dir}")
print("\nFigures generated:")
print("  - fig1_correlation_heatmap.pdf/png")
print("  - fig2_scatter_braak.pdf/png")
print("  - fig3_scatter_cerad.pdf/png")
print("  - fig4_scatter_mmse.pdf/png")
print("  - fig5_scatter_pathology.pdf/png")
print("  - fig6_boxplot_cogdx.pdf/png")
print("  - fig7_marker_correlation.pdf/png")
print("  - fig8_cellcount_braak.pdf/png")
print("  - fig9_combined_panel.pdf/png")
print("  - fig10_stacked_barplot.pdf/png")
print("\nTables generated:")
print("  - full_correlation_table.csv")
print("="*60)
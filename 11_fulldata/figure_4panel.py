"""
Combined Multi-Panel Figure - v5
Better colors, larger fonts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Style Settings - Larger fonts
# ============================================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# Lighter, more vibrant color palette
COLORS = {
    'NeuN': '#5DA5DA',    # Pleasant blue - Neurons
    'iba1': '#FAA43A',    # Warm orange - Microglia  
    'PECAM': '#60BD68',   # Fresh green - Endothelial
    'GFAP': '#B276B2',    # Soft purple - Astrocytes
    'Olig2': '#F17CB0',   # Pink - Oligodendrocytes (replaced gray)
}

LABELS = {
    'NeuN': 'Neurons',
    'iba1': 'Microglia',
    'PECAM': 'Endothelial',
    'GFAP': 'Astrocytes',
    'Olig2': 'Oligodendrocytes',
}

# Load data
df = pd.read_csv('/ihome/jbwang/liy121/ifimage/11_fulldata/segmentation_refined/donor_level_aggregation_refined.csv')
out_dir = Path('/ihome/jbwang/liy121/ifimage/11_fulldata/analysis_refined')

markers = ['GFAP', 'iba1', 'NeuN', 'Olig2', 'PECAM']

# Determine ratio column
if 'GFAP_ratio_from_sum' in df.columns:
    RATIO_COL = lambda m: f'{m}_ratio_from_sum'
else:
    RATIO_COL = lambda m: f'{m}_marker_positive_ratio_refined_mean'

print(f"Loaded {len(df)} donors")

# ============================================================
# Helper Functions
# ============================================================
def add_corr(ax, x, y):
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 3:
        return
    r, p = stats.spearmanr(x[mask], y[mask])
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    ax.text(0.95, 0.95, f'ρ={r:.2f}{sig}', transform=ax.transAxes, 
            ha='right', va='top', fontsize=7)

def add_regline(ax, x, y):
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 3:
        return
    z = np.polyfit(x[mask], y[mask], 1)
    p = np.poly1d(z)
    x_line = np.linspace(x[mask].min(), x[mask].max(), 100)
    ax.plot(x_line, p(x_line), '--', color='#444444', alpha=0.6, linewidth=0.8)

# ============================================================
# Create Figure
# ============================================================
fig = plt.figure(figsize=(10, 8.5))

gs = fig.add_gridspec(4, 5, height_ratios=[1.3, 0.85, 0.85, 0.85], 
                      hspace=0.4, wspace=0.35,
                      left=0.06, right=0.98, top=0.92, bottom=0.06)

# ============================================================
# Panel A: Stacked Bar Plot (NORMALIZED to 1)
# ============================================================
ax_a = fig.add_subplot(gs[0, :])

ratio_cols = [RATIO_COL(m) for m in markers]
df_stack = df[['projid'] + ratio_cols].dropna().copy()
df_stack.columns = ['projid'] + markers

# NORMALIZE each row to sum to 1
row_sums = df_stack[markers].sum(axis=1)
for m in markers:
    df_stack[m] = df_stack[m] / row_sums

# Sort by NeuN descending
df_stack = df_stack.sort_values('NeuN', ascending=False).reset_index(drop=True)

stack_order = ['NeuN', 'iba1', 'PECAM', 'GFAP', 'Olig2']
x = np.arange(len(df_stack))
bottom = np.zeros(len(df_stack))

for marker in stack_order:
    values = df_stack[marker].values
    ax_a.bar(x, values, bottom=bottom, width=1.0,
             color=COLORS[marker], edgecolor='none')
    bottom += values

ax_a.set_xlim(-0.5, len(df_stack) - 0.5)
ax_a.set_ylim(0, 1)
ax_a.set_ylabel('Proportion')
ax_a.set_xticks([])
ax_a.set_xlabel(f'Donors (n={len(df_stack)}, sorted by neuronal proportion)')
ax_a.set_yticks([0, 0.5, 1])

# Legend
handles = [Patch(facecolor=COLORS[m], label=LABELS[m]) for m in stack_order]
ax_a.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.18),
            ncol=5, frameon=False, fontsize=8, handlelength=1.2, handletextpad=0.4,
            columnspacing=1)
ax_a.text(-0.06, 1.0, 'a', transform=ax_a.transAxes, fontsize=14, fontweight='bold', va='top')

# ============================================================
# Panel B: Scatter plots vs Braak Stage
# ============================================================
for i, marker in enumerate(markers):
    ax = fig.add_subplot(gs[1, i])
    col = RATIO_COL(marker)
    
    mask = df[[col, 'braaksc']].notna().all(axis=1)
    x_data = df.loc[mask, 'braaksc'].values
    y_data = df.loc[mask, col].values
    
    ax.scatter(x_data, y_data, c=COLORS[marker], alpha=0.7, s=20, 
               edgecolors='white', linewidth=0.4)
    add_regline(ax, x_data, y_data)
    add_corr(ax, x_data, y_data)
    
    ax.set_title(marker, fontsize=9, fontweight='bold')
    ax.set_xticks([0, 3, 6])
    ax.set_ylim(bottom=0)
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylabel('Ratio')
    
    if i == 2:
        ax.set_xlabel('Braak stage')

fig.text(0.005, 0.63, 'b', fontsize=14, fontweight='bold')

# ============================================================
# Panel C: Scatter plots vs CERAD Score
# ============================================================
for i, marker in enumerate(markers):
    ax = fig.add_subplot(gs[2, i])
    col = RATIO_COL(marker)
    
    mask = df[[col, 'ceradsc']].notna().all(axis=1)
    x_data = df.loc[mask, 'ceradsc'].values
    y_data = df.loc[mask, col].values
    
    ax.scatter(x_data, y_data, c=COLORS[marker], alpha=0.7, s=20,
               edgecolors='white', linewidth=0.4)
    add_regline(ax, x_data, y_data)
    add_corr(ax, x_data, y_data)
    
    ax.set_xticks([1, 2, 3, 4])
    ax.set_ylim(bottom=0)
    ax.set_xlim(0.5, 4.5)
    ax.set_ylabel('Ratio')
    
    if i == 2:
        ax.set_xlabel('CERAD score')

fig.text(0.005, 0.41, 'c', fontsize=14, fontweight='bold')

# ============================================================
# Panel D: Box plots by Cognitive Diagnosis
# ============================================================
df['cogdx_group'] = df['cogdx'].map({1: 'NCI', 2: 'MCI', 3: 'MCI', 4: 'AD', 5: 'AD'})
df_plot = df[df['cogdx_group'].isin(['NCI', 'MCI', 'AD'])].copy()

for i, marker in enumerate(markers):
    ax = fig.add_subplot(gs[3, i])
    col = RATIO_COL(marker)
    
    box_data = [df_plot[df_plot['cogdx_group'] == g][col].dropna() 
                for g in ['NCI', 'MCI', 'AD']]
    
    bp = ax.boxplot(box_data, positions=[0, 1, 2], widths=0.5, patch_artist=True,
                    boxprops=dict(facecolor=COLORS[marker], alpha=0.6, linewidth=0.6),
                    medianprops=dict(color='black', linewidth=1.2),
                    whiskerprops=dict(color='#444444', linewidth=0.6),
                    capprops=dict(color='#444444', linewidth=0.6),
                    flierprops=dict(marker=''))
    
    # Jittered points
    np.random.seed(42 + i)
    for j, data in enumerate(box_data):
        if len(data) > 0:
            jitter = np.random.uniform(-0.15, 0.15, len(data))
            ax.scatter(np.full(len(data), j) + jitter, data, 
                      c=COLORS[marker], alpha=0.7, s=15, 
                      edgecolors='white', linewidth=0.4, zorder=3)
    
    # Kruskal-Wallis p-value
    if all(len(d) > 0 for d in box_data):
        h, p = stats.kruskal(*box_data)
        if p < 0.001:
            ptext = 'p<0.001'
        else:
            ptext = f'p={p:.3f}'
        ax.text(0.5, 0.97, ptext, transform=ax.transAxes, 
                ha='center', va='top', fontsize=7)
    
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['NCI', 'MCI', 'AD'])
    ax.set_ylim(bottom=0)
    ax.set_ylabel('Ratio')
    
    if i == 2:
        ax.set_xlabel('Cognitive diagnosis')

fig.text(0.005, 0.2, 'd', fontsize=14, fontweight='bold')

# ============================================================
# Save
# ============================================================
plt.savefig(out_dir / 'figure_combined_v5.pdf', bbox_inches='tight', dpi=300)
plt.savefig(out_dir / 'figure_combined_v5.png', bbox_inches='tight', dpi=300)
plt.close()

print(f"\n✓ Saved: {out_dir / 'figure_combined_v5.pdf'}")
print(f"✓ Saved: {out_dir / 'figure_combined_v5.png'}")
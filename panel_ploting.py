import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle, FancyBboxPatch

# 1. 数据准备
# ------------------------------------------------------------------------------
df = pd.read_csv("benchmark_table_wide.csv")
if 'summary__overall_rank' in df.columns:
    df = df.sort_values('summary__overall_rank', ascending=True)

# 归一化函数
def normalize_minmax(series, invert=False):
    if series.isnull().all(): return series
    mn, mx = series.min(), series.max()
    if mn == mx: return series.apply(lambda x: 1.0 if pd.notnull(x) else np.nan)
    return (mx - series) / (mx - mn) if invert else (series - mn) / (mx - mn)

# --- 定义列的分组逻辑 (Key Upgrade) ---
# 格式: 'Group Name': [list of columns]

# Accuracy Groups
acc_groups = {
    'Brain Dataset': ['acc__cell_brain__precision_iou50', 'acc__cell_brain__recall_iou50', 'acc__cell_brain__boundary_f1'],
    'Heart Dataset': ['acc__heart_cell__recall_iou50']
}
# Flatten acc columns for indexing
acc_cols_flat = [c for group in acc_groups.values() for c in group]
# Short labels for the columns themselves
acc_labels_map = {
    'acc__cell_brain__precision_iou50': 'Precision',
    'acc__cell_brain__recall_iou50': 'Recall',
    'acc__cell_brain__boundary_f1': 'Boundary F1',
    'acc__heart_cell__recall_iou50': 'Recall'
}

# Usability Groups
usab_groups = {
    'Ranking': ['usab__rank', 'usab__time_rank'],
    'Properties': ['usab__code_behavior', 'usab__dependence']
}
usab_cols_flat = [c for group in usab_groups.values() for c in group]
usab_labels_map = {
    'usab__rank': 'Overall', 'usab__time_rank': 'Time',
    'usab__code_behavior': 'Code Behav.', 'usab__dependence': 'Dependence'
}

# Data Processing
df_sum = df[['summary__overall_rank', 'summary__accuracy_rank', 'summary__usability_rank']].copy()
for c in df_sum.columns: df_sum[c] = normalize_minmax(df_sum[c], invert=True)

df_acc = df[acc_cols_flat].copy() # Values are 0-1

df_usa = df[usab_cols_flat].copy()
for c in usab_cols_flat:
    if 'rank' in c: df_usa[c] = normalize_minmax(df_usa[c], invert=True)

# 2. 绘图设置
# ------------------------------------------------------------------------------
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

num_rows = len(df)
row_h = 0.55
fig_h = num_rows * row_h + 2.5 # 增加顶部空间给二级表头
fig_w = 16

fig = plt.figure(figsize=(fig_w, fig_h), dpi=300)
# 布局: Name | Sum | Acc | Usab
gs = gridspec.GridSpec(1, 4, width_ratios=[0.8, 1.2, 3.5, 2.0], wspace=0.08, 
                       left=0.05, right=0.98, top=0.85, bottom=0.05)

ax_names = plt.subplot(gs[0])
ax_sum = plt.subplot(gs[1])
ax_acc = plt.subplot(gs[2])
ax_usa = plt.subplot(gs[3])

y_centers = np.arange(num_rows)[::-1]
algo_names = df['algorithm'].values
groups = df['group'].values

# --- Helper: Draw Group Headers ---
def draw_grouped_headers(ax, group_dict, label_map, num_rows):
    current_x = 0
    bg_colors = ['#f9f9f9', '#ffffff'] # Alternating background for groups
    
    for i, (group_name, cols) in enumerate(group_dict.items()):
        n_cols = len(cols)
        x_start = current_x
        x_end = current_x + n_cols - 1
        x_center = (x_start + x_end) / 2
        
        # 1. Background Shading for the Group (纵向背景条)
        # Extend from top to bottom
        if i % 2 == 0: # Only shade alternating groups for subtle contrast
            rect_bg = Rectangle((x_start - 0.5, -0.5), n_cols, num_rows, 
                                facecolor='#f4f7f9', edgecolor='none', zorder=0, alpha=0.5)
            ax.add_patch(rect_bg)
        
        # 2. Group Header Line (Super Header)
        # Line above the diagonal labels
        ax.plot([x_start, x_end], [num_rows + 0.8, num_rows + 0.8], color='black', linewidth=1.5, clip_on=False)
        
        # 3. Group Header Text
        ax.text(x_center, num_rows + 1.0, group_name.upper(), ha='center', va='bottom', 
                fontsize=10, weight='bold', color='black')
        
        # 4. Individual Column Headers (Diagonal)
        for j, col in enumerate(cols):
            lbl = label_map.get(col, col)
            ax.text(current_x + j, num_rows + 0.1, lbl, rotation=45, ha='left', va='bottom', 
                    fontsize=9, color='#555555')
            
        current_x += n_cols

# 3. Panel Drawing
# ------------------------------------------------------------------------------

# --- Panel 0: Names ---
ax_names.set_ylim(-0.5, num_rows - 0.5)
ax_names.set_xlim(0, 1)
ax_names.axis('off')
group_colors = {g: c for g, c in zip(sorted(list(set(groups))), ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3'])}

for i in range(num_rows):
    y = y_centers[i]
    # Group Strip
    rect = Rectangle((0.92, y - 0.25), 0.08, 0.5, facecolor=group_colors.get(groups[i], 'grey'))
    ax_names.add_patch(rect)
    # Name
    ax_names.text(0.88, y, algo_names[i], ha='right', va='center', fontsize=11, weight='bold')

# --- Panel 1: Summary (Bubbles) ---
ax_sum.set_xlim(-0.5, 2.5)
ax_sum.set_ylim(-0.5, num_rows - 0.5)
ax_sum.axis('off')
# Simple headers for summary
for x, lbl in enumerate(['Overall', 'Accuracy', 'Usability']):
    ax_sum.text(x, num_rows + 0.1, lbl, rotation=45, ha='left', va='bottom', fontsize=9, weight='bold')

for i in range(num_rows):
    y = y_centers[i]
    ax_sum.axhline(y, color='#f0f0f0', lw=1, zorder=0)
    for j, col in enumerate(df_sum.columns):
        val = df_sum.iloc[i, j]
        if pd.notna(val):
            c = [plt.cm.Greys, plt.cm.Blues, plt.cm.Greens][j](0.3 + 0.7*val)
            s = 50 + 350 * (val**1.5)
            ax_sum.scatter(j, y, s=s, color=c, edgecolors='none', zorder=3)

# --- Panel 2: Accuracy (Grouped Bars) ---
ax_acc.set_xlim(-0.5, len(acc_cols_flat) - 0.5)
ax_acc.set_ylim(-0.5, num_rows - 0.5)
ax_acc.axis('off')

# Call the grouped header function
draw_grouped_headers(ax_acc, acc_groups, acc_labels_map, num_rows)

for i in range(num_rows):
    y = y_centers[i]
    ax_acc.axhline(y, color='#e0e0e0', lw=0.5, zorder=1) # Row grid
    
    for j, col in enumerate(acc_cols_flat):
        val = df_acc.iloc[i, j] # already 0-1
        
        # Track
        ax_acc.add_patch(Rectangle((j-0.4, y-0.25), 0.8, 0.5, facecolor='#eaeaea', edgecolor='none', zorder=2))
        
        if pd.notna(val):
            # Bar
            c = plt.cm.Blues(0.3 + 0.7*val)
            ax_acc.add_patch(Rectangle((j-0.4, y-0.25), 0.8*val, 0.5, facecolor=c, edgecolor='none', zorder=3))
            # Number (only if bar is long enough)
            if val > 0.3:
                ax_acc.text(j-0.4 + 0.8*val - 0.02, y, f"{val:.2f}".lstrip('0'), 
                            ha='right', va='center', color='white', fontsize=7, weight='bold', zorder=4)

# --- Panel 3: Usability (Grouped Tiles) ---
ax_usa.set_xlim(-0.5, len(usab_cols_flat) - 0.5)
ax_usa.set_ylim(-0.5, num_rows - 0.5)
ax_usa.axis('off')

# Divider Line
ax_usa.plot([-0.55, -0.55], [-0.5, num_rows+1], color='#cccccc', lw=1)

draw_grouped_headers(ax_usa, usab_groups, usab_labels_map, num_rows)

for i in range(num_rows):
    y = y_centers[i]
    for j, col in enumerate(usab_cols_flat):
        val = df_usa.iloc[i, j]
        if pd.notna(val):
            c = plt.cm.Greens(0.2 + 0.8*val)
            patch = FancyBboxPatch((j-0.4, y-0.25), 0.8, 0.5, 
                                   boxstyle="round,pad=0.0,rounding_size=0.1", 
                                   facecolor=c, edgecolor='none', zorder=3)
            ax_usa.add_patch(patch)
        else:
             patch = FancyBboxPatch((j-0.4, y-0.25), 0.8, 0.5, 
                                   boxstyle="round,pad=0.0,rounding_size=0.1", 
                                   facecolor='none', edgecolor='#cccccc', zorder=3)

plt.subplots_adjust(top=0.8)
plt.savefig('benchmark_dashboard_grouped.png', dpi=300, bbox_inches='tight')
plt.show()
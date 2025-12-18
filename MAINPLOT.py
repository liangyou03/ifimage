import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set font
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif']

# ============================================
# 1) Data Input
# ============================================
data = [
    ["Watershed",     8,1,5,   7,2,5,   6,2,5,   6,2,5,   2,0,50],
    ["StarDist",      7,7,6,   7,7,6,   7,7,6,   7,7,6,   2,1,120],
    ["SplineDist",    6,6,5,   6,6,5,   6,6,5,   6,6,5,   2,1,140],
    ["Omnipose",      2,5,5,   2,5,5,   2,5,5,   2,5,5,   3,1,180],
    ["LACSS",         3,4,4,   3,4,4,   3,4,4,   3,4,4,   3,1,200],
    ["MESMER",        4,3,4,   4,3,4,   4,3,4,   4,3,4,   3,1,300],
    ["CellposeSAM",   1,2,3,   1,2,3,   1,2,3,   1,2,3,   4,1,600],
    ["CellSAM",       5,9,7,   5,9,7,   5,9,7,   5,9,7,   4,1,500],
    ["MicroSAM",      4,10,8,  4,10,8,  4,10,8,  4,10,8,  5,1,700],
    ["YourMethod",    3,4,4,   3,4,4,   3,4,4,   3,4,4,   1,0,80],
]

cols = [
    "Method",
    "Brain_Nuclei_Accuracy","Brain_Nuclei_Runtime","Brain_Nuclei_Memory",
    "Brain_Cell_Accuracy","Brain_Cell_Runtime","Brain_Cell_Memory",
    "Heart_Nuclei_Accuracy","Heart_Nuclei_Runtime","Heart_Nuclei_Memory",
    "Heart_Cell_Accuracy","Heart_Cell_Runtime","Heart_Cell_Memory",
    "Install_Difficulty","GPU_Requirement","Model_Size_MB"
]

df = pd.DataFrame(data, columns=cols)

# ============================================
# 2) Rank Calculation & Aggregation
# ============================================
# Helper to rank a series (1 = Best)
def get_rank(series, metric_type="low_is_good"):
    if metric_type == "high_is_good":
        return series.rank(ascending=False, method="min")
    else:
        return series.rank(ascending=True, method="min")

# 2a. Calculate ranks for all raw columns first
rank_df = df.copy()
for c in cols[1:]:
    if "Accuracy" in c:
        rank_df[c] = get_rank(df[c], "high_is_good")
    else:
        rank_df[c] = get_rank(df[c], "low_is_good")

# 2b. Create the consolidated DataFrame for plotting
plot_df = pd.DataFrame()
plot_df["Method"] = df["Method"]

# -- Accuracy Columns (Keep separate) --
acc_cols = [c for c in cols if "Accuracy" in c]
for c in acc_cols:
    # Rename for display "Brain_Nuclei_Accuracy" -> "Brain Nuclei"
    new_name = c.replace("_Accuracy", "").replace("_", " ")
    plot_df[new_name] = rank_df[c]

# -- Aggregate Runtime --
run_cols = [c for c in cols if "Runtime" in c]
# Average the ranks, then rank the average to get a clean 1-10 integer
avg_run_rank = rank_df[run_cols].mean(axis=1)
plot_df["Time"] = avg_run_rank.rank(ascending=True, method="min").astype(int)

# -- Aggregate Memory --
mem_cols = [c for c in cols if "Memory" in c]
avg_mem_rank = rank_df[mem_cols].mean(axis=1)
plot_df["Memory"] = avg_mem_rank.rank(ascending=True, method="min").astype(int)

# -- Other Practicality Metrics --
plot_df["Install"] = rank_df["Install_Difficulty"].astype(int)
plot_df["GPU"] = rank_df["GPU_Requirement"].astype(int)
plot_df["Size"] = rank_df["Model_Size_MB"].astype(int)

# ============================================
# 3) Sorting & Ordering
# ============================================
# Define column order for heatmap
final_numeric_cols = [
    "Brain Nuclei", "Brain Cell", "Heart Nuclei", "Heart Cell", # Accuracy
    "Time", "Memory", "Install", "GPU", "Size"                  # Practicality
]

# Sort rows by average rank of these new columns (Best on top)
plot_df["Avg_Rank"] = plot_df[final_numeric_cols].mean(axis=1)
plot_df = plot_df.sort_values("Avg_Rank", ascending=True)

heatmap_data = plot_df.set_index("Method")[final_numeric_cols]

# ============================================
# 4) Visualization
# ============================================
fig, ax = plt.subplots(figsize=(12, 6))

# Colormap: Blue(1)=Best -> Red(10)=Worst
cmap = sns.color_palette("RdYlBu_r", as_cmap=True)

sns.heatmap(heatmap_data, ax=ax, cmap=cmap, annot=True, fmt=".0f", 
            linewidths=1.5, linecolor='white', cbar=False,
            annot_kws={"size": 11, "weight": "bold"})

# --- Grouping & Labels ---
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_title("Method Ranking (1 = Best)", fontsize=14, fontweight='bold', loc='left', pad=30)

# X-axis tick formatting
ax.set_xticklabels(final_numeric_cols, rotation=0, fontsize=10)

# Add Group Brackets
groups = [
    ("Segmentation Accuracy", 0, 4),
    ("Practicality", 4, 9)
]

for name, start, end in groups:
    center = (start + end) / 2
    ax.text(center, -0.6, name, ha="center", va="bottom", 
            fontsize=12, fontweight="bold", color="#333333")
    ax.plot([start+0.1, end-0.1], [-0.4, -0.4], color="black", clip_on=False, linewidth=1.5)
    
    # Separator line
    if end < 9:
        ax.axvline(x=end, color='white', linewidth=4)

# Highlight "YourMethod"
yticks = ax.get_yticklabels()
for label in yticks:
    if label.get_text() == "YourMethod":
        label.set_fontweight("bold")
        label.set_color("#D62728")
        label.set_fontsize(12)
    else:
        label.set_color("#333333")
        label.set_fontsize(11)

plt.tight_layout()
plt.subplots_adjust(top=0.85)

plt.show()
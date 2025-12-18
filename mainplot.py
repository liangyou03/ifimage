import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors

# 1. 设置字体 (Sans-serif)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif']

# ============================================
# 2. 数据准备
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
    ["ImPartial",    3,4,4,   3,4,4,   3,4,4,   3,4,4,   1,0,80],
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
# 3. 计算排名 (1 = 最佳)
# ============================================
def get_rank(series, metric_type="low_is_good"):
    if metric_type == "high_is_good":
        return series.rank(ascending=False, method="min")
    else:
        return series.rank(ascending=True, method="min")

rank_df = df.copy()
for c in cols[1:]:
    if "Accuracy" in c:
        rank_df[c] = get_rank(df[c], "high_is_good")
    else:
        rank_df[c] = get_rank(df[c], "low_is_good")

# 聚合列
plot_df = pd.DataFrame()
plot_df["Method"] = df["Method"]

# 精度列 (展开)
acc_cols = [c for c in cols if "Accuracy" in c]
for c in acc_cols:
    new_name = c.replace("_Accuracy", "").replace("_", " ")
    plot_df[new_name] = rank_df[c]

# 实用性列 (聚合)
run_cols = [c for c in cols if "Runtime" in c]
plot_df["Time"] = rank_df[run_cols].mean(axis=1).rank(method="min").astype(int)

mem_cols = [c for c in cols if "Memory" in c]
plot_df["Memory"] = rank_df[mem_cols].mean(axis=1).rank(method="min").astype(int)

plot_df["Install"] = rank_df["Install_Difficulty"].astype(int)
plot_df["GPU"] = rank_df["GPU_Requirement"].astype(int)
plot_df["Size"] = rank_df["Model_Size_MB"].astype(int)

# 最终列顺序
final_cols = ["Brain Nuclei", "Brain Cell", "Heart Nuclei", "Heart Cell", 
              "Time", "Memory", "Install", "GPU", "Size"]

# 排序：按平均排名 (Average Rank) 升序
plot_df["Avg_Rank"] = plot_df[final_cols].mean(axis=1)
plot_df = plot_df.sort_values("Avg_Rank", ascending=True)
heatmap_data = plot_df.set_index("Method")[final_cols]

# ============================================
# 4. 绘图 (带离散图例)
# ============================================
fig, ax = plt.subplots(figsize=(12, 6.5))

# 定义离散色板 (10个等级)
# Blue (Rank 1) -> Red (Rank 10)
# 使用 get_cmap 获取连续色板，再取 10 个离散值
cmap = plt.cm.get_cmap("RdYlBu_r", 10)

# 绘制热图
sns.heatmap(heatmap_data, ax=ax, cmap=cmap, annot=True, fmt=".0f", 
            linewidths=1.5, linecolor='white', 
            cbar=True, # 开启 Colorbar
            cbar_kws={
                'label': 'Rank (1 = Best)', 
                'orientation': 'horizontal', # 水平放置
                'fraction': 0.05, 
                'pad': 0.15,
                'ticks': np.arange(1.5, 10.5, 1) # 调整刻度位置使其居中
            },
            annot_kws={"size": 11, "weight": "bold"})

# 修改 Colorbar 的刻度标签为 1..10
cbar = ax.collections[0].colorbar
cbar.set_ticks([1.45 + 0.9*i for i in range(10)]) # 微调位置
cbar.set_ticklabels([str(i) for i in range(1, 11)])
cbar.ax.tick_params(labelsize=10)
cbar.set_label("Performance Rank (1 = Best, 10 = Worst)", weight='bold', fontsize=11)

# --- 细节调整 ---
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_title("Benchmark Comparison: Accuracy vs. Practicality", fontsize=15, fontweight='bold', loc='left', pad=40)

# X轴标签
ax.set_xticklabels(final_cols, rotation=0, fontsize=10)

# 分组括号
groups = [
    ("Segmentation Accuracy", 0, 4),
    ("Practicality", 4, 9)
]

for name, start, end in groups:
    center = (start + end) / 2
    ax.text(center, -0.6, name, ha="center", va="bottom", fontsize=12, fontweight="bold", color="#333333")
    ax.plot([start+0.1, end-0.1], [-0.4, -0.4], color="black", clip_on=False, linewidth=1.5)
    if end < 9:
        ax.axvline(x=end, color='white', linewidth=4)

# 高亮 YourMethod
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
plt.subplots_adjust(top=0.82, bottom=0.2) # 给顶部标题和底部图例留出空间

# ============================================
# 5. 保存图片
# ============================================
plt.savefig("benchmark_final.pdf", dpi=300, bbox_inches='tight')
#plt.savefig("benchmark_final.png", dpi=300, bbox_inches='tight')
plt.show()
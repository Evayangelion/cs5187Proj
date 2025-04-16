import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 配置
sns.set(style="whitegrid", font_scale=1.1)
plt.rcParams["figure.dpi"] = 150
os.makedirs("output", exist_ok=True)

# 读取数据
df = pd.read_csv("../output/lightgcn_grid_search.csv")

# ========== 折线图：不同 embedding_dim 下指标趋势 ==========
for metric in ["HR@10", "NDCG@10", "MRR@10"]:
    plt.figure(figsize=(10, 6))
    for dim in sorted(df["embedding_dim"].unique()):
        subset = df[df["embedding_dim"] == dim]
        subset_sorted = subset.sort_values(["n_layers", "learning_rate"])
        labels = subset_sorted[["n_layers", "learning_rate"]].astype(str).agg(
            lambda x: f"L{x['n_layers']}-lr{x['learning_rate']}", axis=1
        )
        plt.plot(labels, subset_sorted[metric], marker="o", label=f"dim={dim}")

    plt.title(f"LightGCN - {metric} vs (embedding_dim)")
    plt.xlabel("n_layers & learning_rate")
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    plt.legend(title="Embedding Dim")
    plt.tight_layout()
    plt.savefig(f"output/lgn_line_{metric}.png")
    plt.close()

# ========== 热力图：HR@10 热力图（按 embedding_dim × n_layers） ==========
for lr_val in df["learning_rate"].unique():
    pivot = df[df["learning_rate"] == lr_val].pivot(
        index="embedding_dim", columns="n_layers", values="HR@10"
    )
    plt.figure(figsize=(6, 4))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu")
    plt.title(f"HR@10 Heatmap (lr={lr_val})")
    plt.ylabel("Embedding Dim")
    plt.xlabel("n_layers")
    plt.tight_layout()
    plt.savefig(f"output/lgn_hr10_heatmap_lr{lr_val}.png")
    plt.close()

# ========== 耗时折线图 ==========
# 标签列
df["label"] = df[["embedding_dim", "n_layers", "learning_rate"]].astype(str).apply(
    lambda x: f"d{x['embedding_dim']}-L{x['n_layers']}-lr{x['learning_rate']}", axis=1
)

# 绘图
plt.figure(figsize=(10, 6))
df_sorted = df.sort_values("time")
plt.plot(df_sorted["label"], df_sorted["time"], marker='o')
plt.title("LightGCN Training Time per Configuration")
plt.xticks(rotation=45)
plt.ylabel("Time (s)")
plt.xlabel("Config (dim-L-layer-lr)")
plt.tight_layout()
plt.savefig("output/lgn_time.png")

print("✅ 所有图表已生成并保存在 output/")
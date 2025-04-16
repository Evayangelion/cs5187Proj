import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 配置
plt.rcParams["figure.dpi"] = 150
os.makedirs("output", exist_ok=True)

# 读取 CSV，只选取需要的列
df = pd.read_csv("../results_lastfm.csv", usecols=["model", "hr_all", "ndcg_all", "mrr_all"])

# 标题映射
metrics = {
    "hr_all": "HR @10",
    "ndcg_all": "NDCG @10",
    "mrr_all": "MRR @10"
}

# 绘图
for col, title in metrics.items():
    plt.figure(figsize=(8, 5))
    order = df.sort_values(by=col, ascending=False)["model"]
    ax = sns.barplot(data=df, x="model", y=col, palette="Set3", order=order)
    plt.title(f" LastFM - {title}")
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.ylim(0, df[col].max() * 1.1)

    # 添加每个柱子的数值标签
    for i, v in enumerate(df.set_index("model").loc[order][col]):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(f"output/lastfm_{col}.png")
    plt.close()

    print("✅ 所有图像已保存至 output/")

    # 折线图：横坐标为模型，三条线分别为 HR@10、NDCG@10、MRR@10
plt.figure(figsize=(10, 6))

# 重构数据格式：每个指标一条线，横坐标是模型
df_line = df.set_index("model")[["hr_all", "ndcg_all", "mrr_all"]].rename(columns=metrics)
df_line = df_line.reset_index()

# 按照模型顺序绘图
for metric in metrics.values():
    plt.plot(df_line["model"], df_line[metric], marker='o', label=metric, linewidth=2)

plt.title(f"LastFM - Model Comparison")
plt.xlabel("Model")
plt.ylabel("Score")
plt.ylim(0, df_line.drop(columns="model").values.max() * 1.1)
plt.xticks(rotation=45)
plt.legend(title="Metric")
plt.tight_layout()
plt.savefig("output/lastfm_model_lines_by_metric.png")
plt.close()

print("✅ 横轴为模型的折线图已保存至 output/")
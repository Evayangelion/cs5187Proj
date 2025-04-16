import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.svd import SVDRecommender
from utils.dataset import load_movielens
from utils.metrics import evaluate_model
import csv
import os
import itertools

os.makedirs("output", exist_ok=True)

# 数据准备
train, test, all_items, all_users = load_movielens(split='temporal')
num_users = len(all_users)
num_items = len(all_items)

# 扩展后的超参数搜索空间
embedding_dims = [32, 64, 128]
learning_rates = [0.001, 0.0005]
weight_decays = [0.0, 1e-4]  # 对应 regularization

# 网格搜索
param_grid = list(itertools.product(embedding_dims, learning_rates, weight_decays))
results = []

for emb_dim, lr, wd in param_grid:
    print(f"🧪 Running SVD: emb={emb_dim}, lr={lr}, reg={wd}")
    model = SVDRecommender(
        num_users, num_items,
        emb_size=emb_dim,
        lr=lr,
        weight_decay=wd,
        epochs=200,
        batch_size=1024
    )
    model.fit(train)
    hr, ndcg, mrr = evaluate_model(model, test, all_items)
    results.append((emb_dim, lr, wd, hr, ndcg, mrr))

# 写入结果
with open("output/svd_grid_search.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["embedding_dim", "learning_rate", "weight_decay", "HR@10", "NDCG@10", "MRR@10"])
    writer.writerows(results)

print("✅ SVD 网格搜索实验完成，结果保存在 output/svd_grid_search.csv")

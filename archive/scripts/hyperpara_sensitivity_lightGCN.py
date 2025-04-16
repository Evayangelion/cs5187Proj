import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.lightGCN import LightGCNRecommender
from utils.dataset import load_movielens
from utils.lastfm_dataset import load_lastfm
from utils.metrics import evaluate_model
import csv
import os
import itertools

os.makedirs("output", exist_ok=True)

# 数据准备

train, test, all_items, all_users = load_lastfm(split='leave-one-out')
num_users = len(all_users)
num_items = len(all_items)

# 超参数搜索空间（embedding_dim × n_layers × weight_decay × lr）
embedding_dims = [128, 256]
n_layers = [1, 2, 3]
weight_decays = [0.0, 1e-4]
learning_rates = [0.01, 0.001, 0.0005]

param_grid = list(itertools.product(embedding_dims, n_layers, weight_decays, learning_rates))
results = []

for dim, layers, wd, lr in param_grid:
    print(f"🧪 Running LightGCN: dim={dim}, layers={layers}, reg={wd}, lr={lr}")
    model = LightGCNRecommender(
        num_users, num_items,
        embedding_dim=dim,
        n_layers=layers,
        weight_decay=wd,
        lr=lr,
        epochs=100
    )
    model.fit(train)
    hr, ndcg, mrr = evaluate_model(model, test, all_items)
    results.append((dim, layers, wd, lr, hr, ndcg, mrr))

with open("output/lightgcn_grid_search.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["embedding_dim", "n_layers", "weight_decay", "learning_rate", "HR@10", "NDCG@10", "MRR@10"])
    writer.writerows(results)

print("✅ LightGCN 网格搜索完成，结果写入 output/lightgcn_grid_search.csv")

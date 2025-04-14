import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# ================= scripts/hyperpara_sensitivity_bert4rec.py =================
import itertools
import logging
import torch
import csv
from models.bert4rec import BERT4RecRecommender
from utils.dataset import load_movielens
from utils.metrics import evaluate_model

logging.basicConfig(format='[%(asctime)s] %(levelname)s - %(message)s', level=logging.INFO)

# 超参数网格
hidden_dims = [256, 512]
num_layers = [2, 3, 4]
lrs = [0.01, 0.001, 0.0005, 0.00005]

# 数据加载
dataset_name = 'temporal'
train, test, all_items, all_users = load_movielens(split=dataset_name)
num_users = len(all_users)
num_items = len(all_items)

results = []

# 搜索组合
for hidden_dim, layers, lr in itertools.product(hidden_dims, num_layers, lrs):
    logging.info(f"🧪 Running BERT4Rec: dim={hidden_dim}, layers={layers}, lr={lr}")
    model = BERT4RecRecommender(
        num_users=num_users,
        num_items=num_items,
        hidden_dim=hidden_dim,
        num_layers=layers,
        epochs=200,
        batch_size=128,
        lr=lr,
        dropout=0.1,
        max_len=100
    )
    # 不再重写 encoder，使用内部结构的 warmup scheduler
    model.fit(train)
    hr, ndcg, mrr = evaluate_model(model, test, all_items, k=10)
    results.append({
        'dim': hidden_dim,
        'layers': layers,
        'lr': lr,
        'HR@10': hr,
        'NDCG@10': ndcg,
        'MRR@10': mrr
    })
    logging.info(f"✅ Result: HR@10={hr:.4f}, NDCG@10={ndcg:.4f}, MRR@10={mrr:.4f}\n")

# 输出 CSV 文件
csv_path = "bert4rec_hyperparam_results.csv"
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['dim', 'layers', 'lr', 'HR@10', 'NDCG@10', 'MRR@10'])
    writer.writeheader()
    writer.writerows(results)

# 找出表现最好的参数
best = max(results, key=lambda x: x['HR@10'])
logging.info("\n🏆 最佳超参数组合:")
logging.info(f"dim={best['dim']}, layers={best['layers']}, lr={best['lr']}, HR@10={best['HR@10']:.4f}, NDCG={best['NDCG@10']:.4f}, MRR={best['MRR@10']:.4f}")

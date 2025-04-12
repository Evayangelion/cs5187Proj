from models.itemcf import ItemCF
from models.svd import SVDRecommender
from models.popularity import PopularityRecommender
from models.lightGCN import LightGCNRecommender

from utils.dataset import load_movielens
from utils.metrics import evaluate_model
import torch

import logging
import time

logging.basicConfig(
    format='[%(asctime)s] %(levelname)s - %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S'
)
if __name__ == '__main__':
    print(torch.cuda.is_available())  # True 表示 GPU 可用

    train, test, all_items, all_users = load_movielens()
    num_users = len(all_users)
    num_items = len(all_items)

    models = {
        'LightGCN': LightGCNRecommender(num_users, num_items, epochs=200),
        'ItemCF': ItemCF(),
        'SVD': SVDRecommender(num_users, num_items, epochs=200, batch_size=1024),
        'Popularity': PopularityRecommender()
    }

    results = {}

    for name, model in models.items():
        logging.info(f"⏳ 开始训练模型: {name}")
        start_time = time.time()
    
        model.fit(train)

        logging.info(f"✅ {name} 训练完成，开始评估")
        if name in ['SVD', 'LightGCN']:
            hr, ndcg, mrr = evaluate_model(model, test, all_items, k=10)
        else:
            hr, ndcg, mrr = evaluate_model(model, test, k=10)

        end_time = time.time()
        
        logging.info(f"📊 {name} HR@10 = {hr:.4f}，耗时 {(end_time - start_time):.2f} 秒")

        results[name] = {
            'HR@10': hr,
            'NDCG@10': ndcg,
            'MRR@10': mrr,
            'Time(s)': end_time - start_time
        }
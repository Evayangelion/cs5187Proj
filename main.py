from models.itemcf import ItemCF
from models.svd import SVDRecommender
from models.popularity import PopularityRecommender
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
        'ItemCF': ItemCF(),
        'SVD': SVDRecommender(num_users, num_items),
        'Popularity': PopularityRecommender()
    }

for name, model in models.items():
    logging.info(f"⏳ 开始训练模型: {name}")
    start_time = time.time()
    
    model.fit(train)

    logging.info(f"✅ {name} 训练完成，开始评估")
    if name == 'SVD':
        hr = evaluate_model(model, test, all_items)
    else:
        hr = evaluate_model(model, test)

    end_time = time.time()
    logging.info(f"📊 {name} HR@10 = {hr:.4f}，耗时 {(end_time - start_time):.2f} 秒")
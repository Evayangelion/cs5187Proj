# ==================== main.py (with BERT4Rec + temporal + 冷启动支持) ====================
from models.itemcf import ItemCF
from models.svd import SVDRecommender
from models.svdpp import SVDPPRecommender
from models.gru4rec import GRU4RecRecommender
from models.popularity import PopularityRecommender
from models.lightGCN import LightGCNRecommender
from models.bert4rec import BERT4RecRecommender

from utils.dataset import load_movielens
from utils.metrics import evaluate_model
import torch
import logging
import time
import csv
import os
import torch._dynamo
torch._dynamo.config.suppress_errors = True

logging.basicConfig(format='[%(asctime)s] %(levelname)s - %(message)s', level=logging.INFO)

if __name__ == '__main__':
    print("CUDA available:", torch.cuda.is_available())

    # 加载数据（temporal split）
    train, test, all_items, all_users = load_movielens(split='temporal')
    num_users = len(all_users)
    num_items = len(all_items)

    # 冷启动用户 = test 用户但 train 中无记录
    train_users = {u for u, _, _ in train}
    cold_users = [u for u, _ in test if u not in train_users]
    warm_users = [u for u, _ in test if u in train_users]

    models = {
        #'BERT4Rec': BERT4RecRecommender(num_users, num_items, epochs=100),
        'GRU4Rec': GRU4RecRecommender(num_users, num_items, epochs=100),
        'SVD++': SVDPPRecommender(num_users, num_items, emb_dim=256, epochs=50),
        'LightGCN': LightGCNRecommender(num_users, num_items, epochs=200),
        'ItemCF': ItemCF(),   
        'SVD': SVDRecommender(num_users, num_items, emb_size=256, lr=0.01, weight_decay=1e-4, epochs=200),
        'Popularity': PopularityRecommender(),
    }


    results_file = 'results.csv'
    write_header = not os.path.exists(results_file)
    
    for name, model in models.items():
        logging.info(f"\n⏳ 开始训练模型: {name}")
        start_time = time.time()
        model.fit(train)
        logging.info(f"✅ {name} 训练完成，开始整体评估")
    
        hr, ndcg, mrr = evaluate_model(model, test, all_items, k=10)
    
        hr_cold = ndcg_cold = mrr_cold = 0.0
        hr_warm = ndcg_warm = mrr_warm = 0.0
    
        if name != 'Popularity':
            cold_test = [(u, i) for u, i in test if u in cold_users]
            warm_test = [(u, i) for u, i in test if u in warm_users]
    
            logging.info(f"📉 {name} 冷启动评估: 🔹{len(cold_test)} 用户")
            if len(cold_test) > 0:
                hr_cold, ndcg_cold, mrr_cold = evaluate_model(model, cold_test, all_items, k=10)
    
            logging.info(f"📈 {name} 非冷启动评估: 🔸{len(warm_test)} 用户")
            if len(warm_test) > 0:
                hr_warm, ndcg_warm, mrr_warm = evaluate_model(model, warm_test, all_items, k=10)
    
        elapsed = time.time() - start_time
        logging.info(f"⏱️ {name} 总耗时: {elapsed:.2f}s")
    
        # 每个模型写入结果
        with open(results_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(['model', 'hr_all', 'ndcg_all', 'mrr_all', 
                                'hr_cold', 'ndcg_cold', 'mrr_cold', 
                                'hr_warm', 'ndcg_warm', 'mrr_warm', 'time'])
                write_header = False  # 写完一次表头就不再写了
    
            writer.writerow([name, f"{hr:.4f}", f"{ndcg:.4f}", f"{mrr:.4f}",
                            f"{hr_cold:.4f}", f"{ndcg_cold:.4f}", f"{mrr_cold:.4f}",
                            f"{hr_warm:.4f}", f"{ndcg_warm:.4f}", f"{mrr_warm:.4f}",
                            f"{elapsed:.2f}"])
        

        
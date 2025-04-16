# ============== utils/cold_start.py =================
from collections import defaultdict
from utils.metrics import evaluate_model
import logging


def split_cold_start_users(train_data, test_data, threshold=5):
    """
    将测试用户分为冷启动（训练行为数 <= threshold）和非冷启动用户
    """
    user_hist = defaultdict(int)
    for u, i, _ in train_data:
        user_hist[u] += 1

    cold_users = []
    warm_users = []
    for u, i in test_data:
        if user_hist[u] <= threshold:
            cold_users.append((u, i))
        else:
            warm_users.append((u, i))

    return cold_users, warm_users


def evaluate_cold_start(model, train_data, test_data, all_items, k=10):
    cold_users, warm_users = split_cold_start_users(train_data, test_data)

    logging.info(f"🧊 冷启动用户数: {len(cold_users)}, 非冷启动用户数: {len(warm_users)}")

    hr_c, ndcg_c, mrr_c = evaluate_model(model, cold_users, all_items, k)
    hr_w, ndcg_w, mrr_w = evaluate_model(model, warm_users, all_items, k)

    print("\n🧊 冷启动 vs 非冷启动评估结果")
    print("| 用户类型 | HR@10 | NDCG@10 | MRR@10 | Count |")
    print("|-----------|--------|----------|---------|--------|")
    print(f"| Cold      | {hr_c:.4f} | {ndcg_c:.4f}  | {mrr_c:.4f}  | {len(cold_users)}")
    print(f"| Warm      | {hr_w:.4f} | {ndcg_w:.4f}  | {mrr_w:.4f}  | {len(warm_users)}")

    return {
        "cold": {"HR@10": hr_c, "NDCG@10": ndcg_c, "MRR@10": mrr_c},
        "warm": {"HR@10": hr_w, "NDCG@10": ndcg_w, "MRR@10": mrr_w},
    }

# ==================== utils/metrics.py (patched for BERT4Rec) ====================
import torch
import logging
import time


def hit_rate(recommended, ground_truth):
    return int(ground_truth in recommended)


def evaluate_model(model, test_data, all_items=None, k=10, max_users=None):
    hits, ndcgs, mrrs = [], [], []
    start_time = time.time()

    for idx, (user, true_item) in enumerate(test_data):
        if max_users and idx >= max_users:
            break

        # logging progress every 100 users
        if idx % 100000 == 0:
            logging.info(f"🧪 已评估用户数: {idx}")

        if hasattr(model, 'recommend') and all_items is not None:
            try:
                recs = model.recommend(user, all_items, k)
            except TypeError:
                recs = model.recommend(user, k)
        else:
            recs = model.recommend(user, k)

        if true_item in recs:
            rank = recs.index(true_item) + 1
            hits.append(1)
            ndcgs.append(1.0 / torch.log2(torch.tensor(rank + 1.0)).item())
            mrrs.append(1.0 / rank)
        else:
            hits.append(0)
            ndcgs.append(0.0)
            mrrs.append(0.0)

    avg_hr = sum(hits) / len(hits) if hits else 0
    avg_ndcg = sum(ndcgs) / len(ndcgs) if ndcgs else 0
    avg_mrr = sum(mrrs) / len(mrrs) if mrrs else 0

    logging.info(f"📊 HR@{k} = {avg_hr:.4f}, NDCG@{k} = {avg_ndcg:.4f}, MRR@{k} = {avg_mrr:.4f}，耗时 {(time.time() - start_time):.2f} 秒")
    return avg_hr, avg_ndcg, avg_mrr

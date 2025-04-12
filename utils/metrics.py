# ==================== utils/metrics.py ====================
import torch
import logging
import time

def hit_rate(recommended, ground_truth):
    return int(ground_truth in recommended)

def evaluate_model(model, test_data, all_items=None, k=10):
    hits, ndcgs, mrrs = [], [], []
    start_time = time.time()

    item_idx_cache = []
    if hasattr(model, 'item_map') and all_items is not None:
        item_idx_cache = [model.item_map[i] for i in all_items if i in model.item_map]
        item_tensor_cache = torch.tensor(item_idx_cache, device=model.device)

    for user, true_item in test_data:
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

    avg_hr = sum(hits) / len(hits)
    avg_ndcg = sum(ndcgs) / len(ndcgs)
    avg_mrr = sum(mrrs) / len(mrrs)

    logging.info(f"📊 HR@{k} = {avg_hr:.4f}, NDCG@{k} = {avg_ndcg:.4f}, MRR@{k} = {avg_mrr:.4f}，耗时 {(time.time() - start_time):.2f} 秒")
    return avg_hr, avg_ndcg, avg_mrr

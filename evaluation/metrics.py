import numpy as np

def hit_rate_at_k(test_dict, reco_dict, k=10):
    """
    Calculate Hit Rate @ K.
    
    Args:
        test_dict (dict): Dictionary of user-items in test set {user_id: [item_ids]}.
        reco_dict (dict): Dictionary of recommended user-items {user_id: [(item_id, score)]}.
        k (int): Length of recommendation list.
    
    Returns:
        float: HR@K metric value.
    """
    hits = 0
    total_users = len(test_dict)
    
    if total_users == 0:
        return 0
        
    for user_id, test_items in test_dict.items():
        if user_id not in reco_dict:
            continue
        recommended_items = [item[0] for item in reco_dict[user_id][:k]]
        if any(item in recommended_items for item in test_items):
            hits += 1
    
    return hits / total_users

def ndcg_at_k(test_dict, reco_dict, k=10):
    """
    Calculate Normalized Discounted Cumulative Gain @ K.
    
    Args:
        test_dict (dict): Dictionary of user-items in test set {user_id: [item_ids]}.
        reco_dict (dict): Dictionary of recommended user-items {user_id: [(item_id, score)]}.
        k (int): Length of recommendation list.
    
    Returns:
        float: NDCG@K metric value.
    """
    ndcg_sum = 0
    total_users = len(test_dict)
    
    if total_users == 0:
        return 0
    
    for user_id, test_items in test_dict.items():
        if user_id not in reco_dict:
            continue
        recommended_items = [item[0] for item in reco_dict[user_id][:k]]
        dcg = 0
        for i, item in enumerate(recommended_items):
            if item in test_items:
                dcg += 1 / np.log2(i + 2)
        idcg = sum(1 / np.log2(i + 2) for i in range(min(len(test_items), k)))
        if idcg > 0:
            ndcg_sum += dcg / idcg
    
    return ndcg_sum / total_users

def mrr_at_k(test_dict, reco_dict, k=10):
    """
    Calculate Mean Reciprocal Rank @ K.
    
    Args:
        test_dict (dict): Dictionary of user-items in test set {user_id: [item_ids]}.
        reco_dict (dict): Dictionary of recommended user-items {user_id: [(item_id, score)]}.
        k (int): Length of recommendation list.
    
    Returns:
        float: MRR@K metric value.
    """
    mrr_sum = 0
    total_users = len(test_dict)
    
    if total_users == 0:
        return 0
    
    for user_id, test_items in test_dict.items():
        if user_id not in reco_dict:
            continue
        recommended_items = [item[0] for item in reco_dict[user_id][:k]]
        for i, item in enumerate(recommended_items):
            if item in test_items:
                mrr_sum += 1 / (i + 1)
                break
    
    return mrr_sum / total_users
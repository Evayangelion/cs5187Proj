def hit_rate(recommended, ground_truth):
    return int(ground_truth in recommended)

def evaluate_model(model, test_data, all_items=None, k=10):
    hits = 0
    for user, true_item in test_data:
        if hasattr(model, 'recommend') and all_items is not None:
            recs = model.recommend(user, all_items, k)
        else:
            recs = model.recommend(user, k)
        hits += hit_rate(recs, true_item)
    return hits / len(test_data)
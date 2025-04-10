from models.itemcf import ItemCF
from models.svd import SVDRecommender
from models.popularity import PopularityRecommender
from utils.dataset import load_movielens
from utils.metrics import evaluate_model

if __name__ == '__main__':
    train, test, all_items, all_users = load_movielens()

    num_users = len(all_users)
    num_items = len(all_items)

    models = {
        'ItemCF': ItemCF(),
        'SVD': SVDRecommender(num_users, num_items),
        'Popularity': PopularityRecommender()
    }

    for name, model in models.items():
        model.fit(train)
        hr = evaluate_model(model, test, all_items)
        print(f"{name} HR@10: {hr:.4f}")

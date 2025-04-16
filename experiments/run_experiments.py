import yaml
import numpy as np
import torch
from algorithms.user_cf import UserCF
from algorithms.item_cf import ItemCF
from algorithms.svd import SVDRecommender
from algorithms.lightgcn import LightGCNRecommender
from algorithms.gru4rec import GRU4Rec
from data.dataloader import RecDataset
from evaluation.metrics import hit_rate_at_k, ndcg_at_k, mrr_at_k
from visualization.plot_results import (
    plot_metrics, plot_user_similarity_distribution, 
    plot_item_similarity_distribution, plot_svd_visualizations
)

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_experiment(config_path, dataset_path):
    """
    Run experiment for a given configuration and dataset.
    
    Args:
        config_path (str): Path to configuration file.
        dataset_path (str): Path to dataset file.
    
    Returns:
        dict: Results dictionary with metrics and additional analysis.
    """
    config = load_config(config_path)
    dataset = RecDataset(dataset_path, config['dataset_name'], config.get('threshold', 4.0))
    
    results = {}
    if config['algorithm'] == 'svd':
        best_hr = 0
        best_results = None
        best_n_factors = None
        for n_factors in config['params']['n_factors_options']:
            model = SVDRecommender(n_factors=n_factors)
            model.fit(dataset.train_matrix)
            reco_dict = model.recommend_for_all_users(dataset.test_dict, n=config['n_recommendations'])
            method_results = {}
            for k in config['k_values']:
                method_results[k] = {
                    'HR@K': hit_rate_at_k(dataset.test_dict, reco_dict, k),
                    'NDCG@K': ndcg_at_k(dataset.test_dict, reco_dict, k),
                    'MRR@K': mrr_at_k(dataset.test_dict, reco_dict, k)
                }
            if method_results[10]['HR@K'] > best_hr:
                best_hr = method_results[10]['HR@K']
                best_results = method_results
                best_n_factors = n_factors
                best_model = model
        results['standard'] = best_results
        model = best_model
        n_factors = best_n_factors
    else:
        if config['algorithm'] == 'user_cf':
            model = UserCF(**config['params'])
            train_data = dataset.train_matrix
        elif config['algorithm'] == 'item_cf':
            model = ItemCF(**config['params'])
            train_data = dataset.train_matrix
        elif config['algorithm'] == 'lightgcn':
            model = LightGCNRecommender(**config['params'])
            train_data = dataset.train_data
        elif config['algorithm'] == 'gru4rec':
            model = GRU4Rec(**config['params'])
            train_data = dataset.train_matrix
        else:
            raise ValueError(f"Unknown algorithm: {config['algorithm']}")
        
        model.fit(train_data)
        methods = config.get('methods', ['standard'])
        for method in methods:
            reco_dict = model.recommend_for_all_users(dataset.test_dict, n=config['n_recommendations'], method=method)
            method_results = {}
            for k in config['k_values']:
                method_results[k] = {
                    'HR@K': hit_rate_at_k(dataset.test_dict, reco_dict, k),
                    'NDCG@K': ndcg_at_k(dataset.test_dict, reco_dict, k),
                    'MRR@K': mrr_at_k(dataset.test_dict, reco_dict, k)
                }
            results[method] = method_results
    
    # Cold-start analysis
    cold_start_results = {}
    cold_start_users = sum(1 for user_id in dataset.test_dict if user_id not in model.user_indices)
    test_movies = set()
    for items in dataset.test_dict.values():
        test_movies.update(items)
    cold_start_movies = sum(1 for movie_id in test_movies if movie_id not in model.movie_indices)
    
    cold_metrics = {"HR@10": 0, "NDCG@10": 0, "MRR@10": 0}
    non_cold_metrics = {"HR@10": 0, "NDCG@10": 0, "MRR@10": 0}
    cold_users = 0
    non_cold_users = 0
    
    for method in results:
        reco_dict = model.recommend_for_all_users(dataset.test_dict, n=config['n_recommendations'], method=method)
        for user_id, test_items in dataset.test_dict.items():
            if user_id not in reco_dict:
                continue
            is_cold = user_id not in model.user_indices
            recommended_items = [item[0] for item in reco_dict[user_id][:10]]
            hit = 1 if any(item in recommended_items for item in test_items) else 0
            dcg = 0
            for i, item in enumerate(recommended_items):
                if item in test_items:
                    dcg += 1 / np.log2(i + 2)
            idcg = sum(1 / np.log2(i + 2) for i in range(min(len(test_items), 10)))
            ndcg = dcg / idcg if idcg > 0 else 0
            mrr = 0
            for i, item in enumerate(recommended_items):
                if item in test_items:
                    mrr = 1 / (i + 1)
                    break
            if is_cold:
                cold_metrics["HR@10"] += hit
                cold_metrics["NDCG@10"] += ndcg
                cold_metrics["MRR@10"] += mrr
                cold_users += 1
            else:
                non_cold_metrics["HR@10"] += hit
                non_cold_metrics["NDCG@10"] += ndcg
                non_cold_metrics["MRR@10"] += mrr
                non_cold_users += 1
        
        if cold_users > 0:
            for metric in cold_metrics:
                cold_metrics[metric] /= cold_users
        if non_cold_users > 0:
            for metric in non_cold_metrics:
                non_cold_metrics[metric] /= non_cold_users
        
        cold_start_results[method] = {
            'cold_users': cold_users,
            'cold_movies': cold_start_movies,
            'cold_metrics': cold_metrics,
            'non_cold_metrics': non_cold_metrics
        }
    
    # Save visualizations
    for method in results:
        plot_metrics(
            results[method],
            output_path=f'figures/{config["algorithm"]}_{method}_metrics.png'
        )
    if config['algorithm'] == 'user_cf':
        plot_user_similarity_distribution(
            model.user_sim,
            output_path=f'figures/{config["algorithm"]}_user_similarity.png'
        )
    elif config['algorithm'] == 'item_cf':
        plot_item_similarity_distribution(
            model.item_sim,
            output_path=f'figures/{config["algorithm"]}_item_similarity.png'
        )
    elif config['algorithm'] == 'svd':
        plot_svd_visualizations(
            model.U, model.Vt, model.sigma, n_factors,
            output_path_prefix=f'figures/{config["algorithm"]}'
        )
    
    return {
        'metrics': results,
        'cold_start': cold_start_results
    }

if __name__ == '__main__':
    config_paths = [
        'experiments/configs/user_cf_config.yaml',
        'experiments/configs/item_cf_config.yaml',
        'experiments/configs/svd_config.yaml',
        'experiments/configs/lightgcn_config.yaml',
    ]
    dataset_path = 'datasets/movielens/ratings.csv'
    
    for config_path in config_paths:
        results = run_experiment(config_path, dataset_path)
        print(f"Results for {config_path}:")
        for method, method_results in results['metrics'].items():
            print(f"\nMethod: {method}")
            for k, metrics in method_results.items():
                print(f"K={k}: HR@{k}={metrics['HR@K']:.4f}, NDCG@{k}={metrics['NDCG@K']:.4f}, MRR@{k}={metrics['MRR@K']:.4f}")
            print(f"\nCold Start Analysis ({method}):")
            print(f"Cold start users: {results['cold_start'][method]['cold_users']}")
            print(f"Cold start movies: {results['cold_start'][method]['cold_movies']}")
            print("Cold start metrics:")
            for metric, value in results['cold_start'][method]['cold_metrics'].items():
                print(f"  {metric}: {value:.4f}")
            print("Non-cold start metrics:")
            for metric, value in results['cold_start'][method]['non_cold_metrics'].items():
                print(f"  {metric}: {value:.4f}")
import matplotlib.pyplot as plt
import numpy as np

def plot_metrics(results_dict, output_path='figures/metrics_comparison.png'):
    """
    Plot comparison of metrics across K values.
    
    Args:
        results_dict (dict): Dictionary of results {k: {'HR@K': value, 'NDCG@K': value, 'MRR@K': value}}.
        output_path (str): Path to save the figure.
    """
    plt.figure(figsize=(12, 8))
    metrics = ['HR@K', 'NDCG@K', 'MRR@K']
    colors = ['blue', 'green', 'red']
    k_values = sorted(results_dict.keys())

    for i, metric in enumerate(metrics):
        values = [results_dict[k][metric] for k in k_values]
        plt.plot(k_values, values, marker='o', color=colors[i], label=metric)

    plt.xlabel('K value')
    plt.ylabel('Metric value')
    plt.title('Recommendation System Evaluation Metrics for Different K Values')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def plot_user_similarity_distribution(user_sim, output_path='figures/user_similarity_distribution.png'):
    """
    Plot distribution of user similarities.
    
    Args:
        user_sim (np.ndarray): User similarity matrix.
        output_path (str): Path to save the figure.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(user_sim.flatten(), bins=50, alpha=0.7)
    plt.title('User Similarity Distribution')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def plot_item_similarity_distribution(item_sim, output_path='figures/item_similarity_distribution.png'):
    """
    Plot distribution of item similarities.
    
    Args:
        item_sim (np.ndarray): Item similarity matrix.
        output_path (str): Path to save the figure.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(item_sim.flatten(), bins=50, alpha=0.7)
    plt.title('Item Similarity Distribution')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def plot_svd_visualizations(U, Vt, sigma, n_factors, output_path_prefix='figures/svd'):
    """
    Plot SVD-specific visualizations: singular values and latent space.
    
    Args:
        U (np.ndarray): User latent factors.
        Vt (np.ndarray): Item latent factors.
        sigma (np.ndarray): Singular values.
        n_factors (int): Number of latent factors.
        output_path_prefix (str): Prefix for output file paths.
    """
    # Singular value distribution
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(np.diag(sigma))), np.diag(sigma))
    plt.title(f'SVD Singular Value Distribution (Number of Factors={n_factors})')
    plt.xlabel('Factor Index')
    plt.ylabel('Singular Value')
    plt.grid(True)
    plt.savefig(f'{output_path_prefix}_singular_values.png')
    plt.close()

    # Latent space visualization
    plt.figure(figsize=(12, 10))
    n_users_to_plot = min(50, U.shape[0])
    n_movies_to_plot = min(100, Vt.shape[1])
    user_indices = np.random.choice(U.shape[0], n_users_to_plot, replace=False)
    movie_indices = np.random.choice(Vt.shape[1], n_movies_to_plot, replace=False)
    plt.scatter(U[user_indices, 0], U[user_indices, 1], c='blue', label='Users', alpha=0.7, s=50)
    plt.scatter(Vt.T[movie_indices, 0], Vt.T[movie_indices, 1], c='red', label='Movies', alpha=0.7, s=30)
    plt.title('Distribution of Users and Movies in the First Two Latent Factors')
    plt.xlabel('Latent Factor 1')
    plt.ylabel('Latent Factor 2')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_path_prefix}_latent_space.png')
    plt.close()

def plot_algorithm_comparison(results_std, results_alt, output_path='figures/algorithm_comparison.png'):
    """
    Plot comparison of standard vs alternative algorithm.
    
    Args:
        results_std (dict): Results for standard algorithm.
        results_alt (dict): Results for alternative algorithm.
        output_path (str): Path to save the figure.
    """
    plt.figure(figsize=(18, 6))
    metrics = ['HR@K', 'NDCG@K', 'MRR@K']
    k_values = sorted(results_std.keys())

    for i, metric in enumerate(metrics):
        plt.subplot(1, 3, i+1)
        std_values = [results_std[k][metric] for k in k_values]
        alt_values = [results_alt[k][metric] for k in k_values]
        plt.plot(k_values, std_values, marker='o', label='Standard Algorithm')
        plt.plot(k_values, alt_values, marker='s', label='Alternative Algorithm')
        plt.xlabel('K value')
        plt.ylabel(metric)
        plt.title(f'{metric} Comparison')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_user_activity_impact(cold_start_results, output_path='figures/user_activity_impact.png'):
    """
    Plot HR@10 for different user activity levels.
    
    Args:
        cold_start_results (dict): Cold start analysis results.
        output_path (str): Path to save the figure.
    """
    plt.figure(figsize=(10, 6))
    activity_levels = ['Low Activity', 'Medium Activity', 'High Activity']
    hr_values = [
        cold_start_results['low_activity_hr@10'],
        cold_start_results['medium_activity_hr@10'],
        cold_start_results['high_activity_hr@10']
    ]
    plt.bar(activity_levels, hr_values, color=['#ff9999', '#66b3ff', '#99ff99'])
    plt.xlabel('User Activity Level')
    plt.ylabel('HR@10')
    plt.title('Recommendation Performance by User Activity Level')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(output_path)
    plt.close()
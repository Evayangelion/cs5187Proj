
# Recommender System Project

## Overview

This project implements and evaluates recommender system algorithms, including User-based Collaborative Filtering (UserCF), Item-based Collaborative Filtering (ItemCF), SVD-based Recommender (SVD), and LightGCN, with a placeholder for GRU4Rec. It includes data preprocessing, evaluation metrics (HR@K, NDCG@K, MRR@K), experiment scripts, visualizations, and cold-start analysis.

```bash
current_project/
│
├── algorithms/                   # Algorithm implementations
│   ├── __init__.py
│   ├── user_cf.py              # User-based Collaborative Filtering
│   ├── item_cf.py              # Item-based Collaborative Filtering
│   ├── svd.py                  # SVD-based Recommender
│   ├── lightgcn.py             # LightGCN Recommender
│   ├── gru4rec.py              # Placeholder for GRU4Rec
│
├── data/                        # Data loading and preprocessing
│   ├── __init__.py
│   ├── dataloader.py           # Dataset loading utilities
│   ├── preprocess.py           # Preprocessing scripts
│
├── evaluation/                  # Evaluation metrics
│   ├── __init__.py
│   ├── metrics.py              # HR@K, NDCG@K, MRR@K implementations
│
├── experiments/                 # Experiment scripts and configs
│   ├── run_experiments.py      # Main experiment runner
│   ├── configs/                # Hyperparameter configurations
│   │   ├── user_cf_config.yaml
│   │   ├── item_cf_config.yaml
│   │   ├── svd_config.yaml
│   │   ├── lightgcn_config.yaml
│   │   ├── gru4rec_config.yaml
│
├── notebooks/                   # Exploratory data analysis
│   ├── eda_movielens.ipynb     # EDA for MovieLens dataset
│
├── visualization/               # Visualization scripts
│   ├── __init__.py
│   ├── plot_results.py         # Scripts to generate figures
│
├── datasets/                    # Dataset files (not included in repo, downloaded separately)
│   ├── movielens/
│   │   ├── ratings.csv
│   ├── lastfm/
│   ├── yelp/
│
├── figures/                     # Output directory for visualizations
│
├── README.md                    # Project documentation
├── requirements.txt             # Dependencies
└── environment.yml              # Conda environment (optional)
```

## Setup

1. Clone the repository.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
    ```


3. Download the MovieLens dataset and place `ratings.csv` in `datasets/movielens/`.

## Running Experiments

1. Update configuration files in `experiments/configs/` with desired hyperparameters.
2. Run experiments:

   ```bash
   python experiments/run_experiments.py
   ```

## Project Structure

- `algorithms/`: Algorithm implementations (UserCF, ItemCF, SVD, LightGCN, etc.).
- `data/`: Data loading and preprocessing utilities.
- `evaluation/`: Evaluation metrics (HR@K, NDCG@K, MRR@K).
- `experiments/`: Experiment scripts and configuration files.
- `notebooks/`: Exploratory data analysis (e.g., MovieLens EDA).
- `visualization/`: Visualization scripts for metrics and distributions.
- `datasets/`: Directory for datasets (e.g., MovieLens).
- `figures/`: Output directory for visualizations.

## Reproducing Results

Run `experiments/run_experiments.py` with the provided config files to reproduce results. Results are saved as figures in `figures/` and printed to the console.

## Notes

- ItemCF supports two methods: 'standard' (weighted rating prediction) and 'similarity' (similarity-based scoring).
- SVD tests multiple latent factor counts and selects the best based on HR@10.
- LightGCN uses a graph-based approach with PyTorch and requires CUDA for GPU acceleration (falls back to CPU if unavailable).
- Cold-start analysis is included for all algorithms, evaluating performance for cold and non-cold start users.
- Visualization includes metric plots, similarity distributions (UserCF, ItemCF), singular value and latent space plots (SVD), and standard metric plots for LightGCN.
- The `lightgcn.py` file includes three classes: `LightGCNDataset` for data preprocessing, `LightGCN` for the model, and `LightGCNRecommender` for the recommender interface, ensuring consistency with other algorithms.
- The `dataloader.py` file is updated to return raw `train_data` alongside `train_matrix` and `test_dict`, supporting LightGCN’s need for raw data to build the graph.
- The experiment runner (`run_experiments.py`) now supports LightGCN, passing raw `train_data` instead of `train_matrix`.
- Cold-start analysis is integrated for all algorithms, reporting metrics for cold and non-cold start users.
- Visualization for LightGCN reuses the generic `plot_metrics` function, as the original code didn’t include unique visualizations beyond metrics.
- The GRU4Rec placeholder remains; please share its code to complete the integration.
- Added `torch` to `requirements.txt`. The `cpuonly` package is included for CPU users; remove it if using a GPU.
- The `figures/` directory should be created to store visualization outputs.
- The original LightGCN code saves a model checkpoint (`lightgcn_best_model.pth`). This is omitted in the modular version to avoid file I/O complexity, but I can add it back if needed.
- The LightGCN code uses a fixed random seed (42) for reproducibility. I’ve kept this in the dataset splitting but omitted global seed setting to avoid affecting other algorithms.

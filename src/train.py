import os
import argparse
import yaml
import logging
import joblib
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from core.dataset import ClusteringDatasetLoader
from core.reducer import DimensionalityReducer
from core.clusterer import ClusteringEngine

def setup_logging(log_path: str):
    """Configures the training history logger."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

def update_leaderboard(csv_path: str, exp_id: str, exp_config: dict, ari: float, nmi: float):
    """
    Appends or updates the experiment metrics inside the tracking CSV safely,
    matching the exact notebook naming schema to prevent redundancies.
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    algo_map = {
        'kmeans': 'K-Means',
        'hierarchical': 'Hierarchical',
        'gmm': 'GMM'
    }

    algo_name = algo_map.get(exp_config['clustering'].lower(), exp_config['clustering'])

    embed_type = 'TF-IDF' if exp_config['embedding'].lower() == 'tfidf' else 'BERT'

    if exp_config['reduction'].lower() != 'none':
        display_name = f"Exp {exp_id}: {algo_name} ({embed_type} + {exp_config['reduction'].upper()})"
    else:
        display_name = f"Exp {exp_id}: {algo_name} ({embed_type})"


    new_row = {
        'Experiment': display_name,
        'Embedding': 'TF-IDF (Sparse)' if embed_type == 'TF-IDF' else 'BERT (Dense)',
        'Reduction': f"{exp_config['reduction'].upper()} ({exp_config.get('reduction_dim', '')}D)" if exp_config['reduction'].lower() != 'none' else 'None',
        'Clusters (K)': exp_config['n_clusters'],
        'ARI Score': round(ari, 4),
        'NMI Score': round(nmi, 4)
    }

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        
        if 'Experiment' in df.columns:
            df = df[df['Experiment'] != display_name]
            
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df = pd.DataFrame([new_row])

    if 'ARI Score' in df.columns:
        df = df.sort_values(by='ARI Score', ascending=False)

    df.to_csv(csv_path, index=False)
    print(f"Leaderboard synchronized and sorted by NMI. Top model is on top!")


def main():
    parser = argparse.ArgumentParser(description="Advanced Text Clustering Training & Ablation Pipeline")
    parser.add_argument(
        "--exp", 
        type=str, 
        required=True, 
        help="The Experiment ID to execute from config.yaml (e.g., A-06, C-03)"
    )
    args = parser.parse_args()
    exp_id = args.exp.upper()

    config_path = "config/config.yaml"
    if not os.path.exists(config_path):
        print(f"Error: Central config file not found at {config_path}")
        return
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if exp_id not in config["experiments"]:
        print(f"Error: Experiment ID '{exp_id}' is not defined inside config.yaml")
        return
    
    exp_config = config["experiments"][exp_id]
    paths = config["paths"]

    setup_logging(paths["log_file"])
    logging.info(f"Execution Started for Experiment: {exp_id} Configuration: {exp_config}")
    print(f"\nStarting Execution Sequence for Experiment: [ {exp_id} ]")

    try:
        data_loader = ClusteringDatasetLoader(config)
        X_train, X_val, y_train, y_val = data_loader.load_data(exp_id, exp_config)

        reducer_manager = DimensionalityReducer(exp_config)
        X_train_red, X_val_red = reducer_manager.fit_transform_pipeline(X_train, X_val)

        clustering_engine = ClusteringEngine(exp_config)
        train_preds, val_preds = clustering_engine.fit_predict_pipeline(X_train_red, X_val_red)

        ari_score = adjusted_rand_score(y_val, val_preds)
        nmi_score = normalized_mutual_info_score(y_val, val_preds)

        print(f"\n --- [ {exp_id} ] Validation Metrics ---")
        print(f"🔹 Adjusted Rand Index (ARI):       {ari_score:.4f}")
        print(f"🔹 Normalized Mutual Info (NMI):    {nmi_score:.4f}\n")

        logging.info(f"Experiment {exp_id} Success - ARI: {ari_score:.4f}, NMI: {nmi_score:.4f}")

        update_leaderboard(paths["leaderboard_csv"], exp_id, exp_config, ari_score, nmi_score)

        models_dir = paths["models_dir"]
        os.makedirs(models_dir, exist_ok=True)

        reducer_artifact_path = os.path.join(models_dir, f"{exp_id.lower()}_reducer.pkl")
        clusterer_artifact_path = os.path.join(models_dir, f"{exp_id.lower()}_clusterer.pkl")

        joblib.dump(reducer_manager.get_reducer(), reducer_artifact_path)
        joblib.dump(clustering_engine.get_clusterer(), clusterer_artifact_path)

        print(f"Production Artifacts saved successfully:")
        print(f"   -> Reducer Matrix Mapper: {reducer_artifact_path}")
        print(f"   -> Clustering Estimator : {clusterer_artifact_path}\n")
        logging.info(f"Saved artifacts for {exp_id} to {models_dir}")

    except Exception as e:
        error_msg = f"Catastrophic failure during Exp {exp_id} runtime: {str(e)}"
        print(error_msg)
        logging.error(error_msg, exc_info=True)


if __name__ == "__main__":
    main()











    
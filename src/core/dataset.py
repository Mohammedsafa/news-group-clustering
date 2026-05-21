import os
import joblib
import pandas as pd
from typing import Tuple, Any

class ClusteringDatasetLoader:
    """
    Responsible for routing and loading the correct precomputed matrices 
    and ground truth labels based on the active Ablation config.
    """

    def __init__(self, config_dict: dict):
        """
        Args:
            config_dict (dict): The dictionary containing the paths and specific experiment parameters.
        """
        self.config = config_dict
        self.data_dir = self.config['paths']['data_dir']


    def load_data(self, exp_id: str, exp_config: dict) -> Tuple[Any, Any, pd.Series, pd.Series]:
        """
        Loads the precomputed train/validation matrices and their matching labels.
        
        Args:
            exp_id (str): The experiment identifier (e.g., 'A-06')
            exp_config (dict): The sub-dictionary holding 'embedding' type ('bert' or 'tfidf')
            
        Returns:
            Tuple: (X_train, X_val, y_train, y_val)
        """

        embedding_type = exp_config['embedding'].lower()

        if embedding_type == "bert":
            train_matrix_path = os.path.join(self.data_dir, "train_bert_matrix.pkl")
            val_matrix_path = os.path.join(self.data_dir, "val_bert_matrix.pkl")
        elif embedding_type == "tfidf":
            train_matrix_path = os.path.join(self.data_dir, "train_tfidf_matrix.pkl")
            val_matrix_path = os.path.join(self.data_dir, "val_tfidf_matrix.pkl")
        else:
            raise ValueError(f"[-] Unsupported embedding type '{embedding_type}' found in Exp {exp_id}")
        
        if not os.path.exists(train_matrix_path) or not os.path.exists(val_matrix_path):
            raise FileNotFoundError(f"[-] Precomputed matrices not found for {embedding_type} in {self.data_dir}")
        

        print(f"Loading {embedding_type.upper()} embeddings for Experiment {exp_id}...")
        X_train = joblib.load(train_matrix_path)
        X_val = joblib.load(val_matrix_path)

        train_csv_path = os.path.join(self.data_dir, "train_processed.csv")
        val_csv_path = os.path.join(self.data_dir, "val_processed.csv")

        train_df = pd.read_csv(train_csv_path)
        val_df = pd.read_csv(val_csv_path)

        y_train = train_df['target']
        y_val = val_df['target']

        print(f"Successfully loaded matrices. Train shape: {X_train.shape}, Val shape: {X_val.shape}")
        return X_train, X_val, y_train, y_val
    
    
        

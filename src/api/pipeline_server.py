import os
import joblib
import yaml
import numpy as np
from typing import Any
from sentence_transformers import SentenceTransformer
import scipy.sparse as sp



class LiveInferenceServer:
    """
    Server component responsible for memory-caching model artifacts 
    and executing top-down vectorization, contraction, and cluster assignments.
    Uses the locally downloaded SentenceTransformer model.
    """

    def __init__(self):
        self.config = None
        self.active_exp_id = None
        self.active_config = None
        
        self.vectorizer = None 
        self.bert_model = None
        self.reducer = None
        self.clusterer = None

        self._load_global_config()

    def _load_global_config(self):
        config_path = "config/config.yaml"
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        else:
            raise FileNotFoundError(f"[-] Central config missing at {config_path}")
        
    def load_experiment_artifacts(self, exp_id: str):
        """
        Dynamically loads or switches the trained models in memory based on the requested Exp ID.
        """
        exp_id = exp_id.upper()
        if exp_id not in self.config["experiments"]:
            raise ValueError(f"[-] Experiment {exp_id} is not registered in config.yaml")
        
        if self.active_exp_id == exp_id:
            return

        print(f"Inference Server switching contexts to [ {exp_id} ]...")
        self.active_exp_id = exp_id
        self.active_config = self.config["experiments"][exp_id]
        embedding_type = self.active_config["embedding"].lower()

        if embedding_type == "tfidf":
            tfidf_path = os.path.join(self.config["paths"]["data_dir"], "tfidf_vectorizer.pkl")
            if os.path.exists(tfidf_path):
                self.vectorizer = joblib.load(tfidf_path)
            else:
                raise FileNotFoundError("[-] Trained TF-IDF Vectorizer artifact not found.")
        
        elif embedding_type == "bert":
            local_model_path = "models/all-MiniLM-L6-v2"
            if os.path.exists(local_model_path):
                print(f"Loading Local SentenceTransformer Weights from: {local_model_path}")
                self.bert_model = SentenceTransformer(local_model_path)
            else:
                print("Local folder not found directly, trying fallback cache...")
                self.bert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        models_dir = self.config["paths"]["models_dir"]
        reducer_path = os.path.join(models_dir, f"{exp_id.lower()}_reducer.pkl")
        clusterer_path = os.path.join(models_dir, f"{exp_id.lower()}_clusterer.pkl")

        if not os.path.exists(reducer_path) or not os.path.exists(clusterer_path):
            raise FileNotFoundError(f"[-] Trained artifacts for {exp_id} missing. Run python src/train.py --exp {exp_id} first.")

        self.reducer = joblib.load(reducer_path)
        self.clusterer = joblib.load(clusterer_path)
        print(f"Context [ {exp_id} ] loaded into RAM successfully.")

    def _get_bert_embedding(self, text: str) -> np.ndarray:
        """Helper to compute embeddings using the local SentenceTransformer."""
        return self.bert_model.encode(text)[None, :]
    
    def predict_single(self, text: str) -> int:
        """
        Processes a single string text through the active configuration pipeline 
        and returns the cluster index.
        """
        embedding_type = self.active_config["embedding"].lower()

        if embedding_type == "tfidf":
            features = self.vectorizer.transform([text])
            if self.active_config["reduction"].lower() in ["umap", "pca"] or self.active_config["clustering"].lower() == "hierarchical":
                features = features.toarray()
        else:
            features = self._get_bert_embedding(text)

        features_reduced = self.reducer.transform(features)

        if self.active_config["clustering"].lower() == "hierarchical":
            raise AttributeError("[-] Hierarchical Clustering (Agglomerative) does not support online inference for isolated inputs.")
        
        cluster_assignment = self.clusterer.predict(features_reduced)
        
        return int(cluster_assignment[0])
    
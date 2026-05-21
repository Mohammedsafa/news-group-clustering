import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import scipy.sparse as sp
from typing import Tuple, Any

class ClusteringEngine:
    """
    Unified cluster execution layer supporting hard partitioning, 
    hierarchical trees, and soft probabilistic mixture models.
    """

    def __init__(self, exp_config: dict):
        """
        Args:
            exp_config (dict): The sub-dictionary of the active experiment from config.yaml
        """
        self.config = exp_config
        self.algorithm = self.config.get('clustering', 'kmeans').lower()
        self.n_clusters = self.config.get('n_clusters', 20)
        self.clusterer_object = None

    def fit_predict_pipeline(self, X_train: Any, X_val: Any) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fits the chosen clustering algorithm on the training matrices 
        and outputs label assignments for both train and validation splits.
        
        Args:
            X_train: The compressed/raw training feature space.
            X_val: The compressed/raw validation feature space.
            
        Returns:
            Tuple: (train_preds, val_preds)
        """

        train_size = X_train.shape[0]

        if self.algorithm == "kmeans":
            print(f"[Clustering Layer] Initializing K-Means with K={self.n_clusters}...")
            self.clusterer_object = KMeans(n_clusters=self.n_clusters, init='k-means++', random_state=42)
            
            self.clusterer_object.fit(X_train)
            train_preds = self.clusterer_object.labels_
            val_preds = self.clusterer_object.predict(X_val)

        elif self.algorithm == "hierarchical":
            print(f"[Clustering Layer] Initializing Hierarchical (Agglomerative) with K={self.n_clusters}...")
            self.clusterer_object = AgglomerativeClustering(n_clusters=self.n_clusters, linkage='ward')
            
            if sp.issparse(X_train):
                X_combined = sp.vstack((X_train, X_val)).toarray()
            else:
                X_combined = np.concatenate((X_train, X_val), axis=0)
                
            print("Training Agglomerative Clustering on Combined Matrix (Train + Val)...")
            combined_preds = self.clusterer_object.fit_predict(X_combined)
            
            train_preds = combined_preds[:train_size]
            val_preds = combined_preds[train_size:]

        elif self.algorithm == "gmm":
            print(f"[Clustering Layer] Initializing Gaussian Mixture Model with Components={self.n_clusters}...")
            self.clusterer_object = GaussianMixture(n_components=self.n_clusters, random_state=42)
            
            self.clusterer_object.fit(X_train)
            train_preds = self.clusterer_object.predict(X_train)
            val_preds = self.clusterer_object.predict(X_val)

        else:
            raise ValueError(f"[-] Unsupported clustering algorithm '{self.algorithm}' provided.")

        return train_preds, val_preds
    
    def get_clusterer(self) -> Any:
        """
        Returns the trained clusterer model instance for backend serialization.
        """
        return self.clusterer_object
    
    

    
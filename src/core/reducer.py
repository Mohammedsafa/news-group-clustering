from sklearn.decomposition import PCA, TruncatedSVD
import umap
from typing import Tuple, Any


class IdentityTransformer:
    """A dummy transformer that returns the data untouched if no reduction is requested."""
    def fit_transform(self, X, y=None):
        return X
    def transform(self, X):
        return X
    

class DimensionalityReducer:
    """
    Factory-style class to handle linear and non-linear manifold compression 
    based on the experiment configuration.
    """

    def __init__(self, exp_config: dict):
        """
        Args:
            exp_config (dict): The sub-dictionary of the active experiment from config.yaml
        """
        self.config = exp_config
        self.reduction_type = self.config.get('reduction', 'none').lower()
        self.n_components = self.config.get('reduction_dim', None)
        self.reducer_object = None

    def fit_transform_pipeline(self, X_train: Any, X_val: Any) -> Tuple[Any, Any]:
        """
        Fits the selected reducer on the training matrix and transforms both train and validation splits.
        
        Args:
            X_train: Precomputed training matrix (sparse or dense).
            X_val: Precomputed validation matrix.
            
        Returns:
            Tuple: (X_train_reduced, X_val_reduced)
        """

        if self.reduction_type == "none":
            self.reducer_object = IdentityTransformer()
            print("[Reduction Layer] Set to 'NONE' - Passing raw embeddings through.")
        
        elif self.reduction_type == "pca":
            self.reducer_object = PCA(n_components=self.n_components, random_state=42)
            print(f"[Reduction Layer] Initialized PCA to contract space down to {self.n_components}D...")
            
        elif self.reduction_type == "svd":
            self.reducer_object = TruncatedSVD(n_components=self.n_components, random_state=42)
            print(f"[Reduction Layer] Initialized TruncatedSVD to contract space down to {self.n_components}D...")
        
        elif self.reduction_type == "umap":
            self.reducer_object = umap.UMAP(
                n_components=self.n_components, 
                n_neighbors=15, 
                min_dist=0.1, 
                random_state=42
            )
            print(f"[Reduction Layer] Initialized UMAP to contract space down to {self.n_components}D Manifold...")
            
        else:
            raise ValueError(f"[-] Unsupported reduction type '{self.reduction_type}' provided.")
        
        X_train_reduced = self.reducer_object.fit_transform(X_train)
        X_val_reduced = self.reducer_object.transform(X_val)
        
        return X_train_reduced, X_val_reduced
    
    def get_reducer(self) -> Any:
        """
        Returns the fitted reducer instance to be dumped as a production artifact.
        """
        return self.reducer_object
    
    

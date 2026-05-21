from pydantic import BaseModel, Field
from typing import List, Optional

class ClusterRequest(BaseModel):
    """
    Schema for validating incoming inference requests.
    Enforces that the user must provide text to be clustered.
    """

    text: str = Field(
        ..., 
        min_length=10, 
        description="The raw news text or document to be classified into a cluster.",
        examples=["The space shuttle Discovery blasted off today from Kennedy Space Center to repair the satellite."]
    )
    model_experiment: Optional[str] = Field(
        default="A-06", 
        description="The experiment ID to use for inference (defaults to the champion model A-06)."
    )

class ClusterResponse(BaseModel):
    """
    Schema for structuring the API outgoing response.
    Provides clean, predictable JSON output for production clients.
    """
    text_preview: str = Field(..., description="A short preview of the input text processed.")
    cluster_id: int = Field(..., description="The predicted cluster assignment (0-19).")
    experiment_used: str = Field(..., description="The active experiment registry key used for inference.")
    status: str = Field(default="success", description="API execution status flag.")

class BulkClusterRequest(BaseModel):
    """
    Schema to support batch processing if the user wants to cluster multiple documents at once.
    """
    texts: List[str] = Field(..., min_items=1, description="List of news texts to process in a single batch.")
    model_experiment: Optional[str] = Field(default="A-06")

class BulkClusterResponse(BaseModel):
    """
    Schema for returning batch prediction outputs.
    """
    predictions: List[ClusterResponse] = Field(..., description="List of individual cluster predictions.")



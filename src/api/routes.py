import os
import pandas as pd
from fastapi import FastAPI, APIRouter, HTTPException, Depends
from fastapi.responses import RedirectResponse
import json

from src.api.schemas import ClusterRequest, ClusterResponse, BulkClusterRequest, BulkClusterResponse
from src.api.pipeline_server import LiveInferenceServer


app = FastAPI(
    title="Advanced News Text Clustering Engine",
    description="Production MLOps API delivering high-performance real-time and batch document partitioning utilizing BERT & UMAP manifolds.",
    version="1.0.0"
)

router = APIRouter(prefix="/api/v1", tags=["Clustering Core"])

inference_server = LiveInferenceServer()

CLUSTER_MAPPING = {
    0: 'rec.motorcycles', 1: 'comp.sys.ibm.pc.hardware', 2: 'rec.sport.hockey', 
    3: 'rec.autos', 4: 'soc.religion.christian', 5: 'comp.windows.x', 
    6: 'misc.forsale', 7: 'comp.graphics', 8: 'talk.politics.mideast', 
    9: 'sci.med', 10: 'comp.sys.ibm.pc.hardware', 11: 'talk.politics.guns', 
    12: 'sci.electronics', 13: 'comp.os.ms-windows.misc', 14: 'talk.politics.misc', 
    15: 'sci.space', 16: 'rec.sport.baseball', 17: 'comp.graphics', 
    18: 'sci.crypt', 19: 'comp.os.ms-windows.misc'
}


def get_inference_server() -> LiveInferenceServer:
    """Dependency injection provider for the memory-cached inference runner."""
    return inference_server


@app.get("/", include_in_schema=False)
def root_redirect():
    """Redirects the root URL directly to the interactive Swagger Documentation."""
    return RedirectResponse(url="/docs")


@router.post("/predict", response_model=ClusterResponse)
async def predict_cluster(request: ClusterRequest, server: LiveInferenceServer = Depends(get_inference_server)):
    try:
        server.load_experiment_artifacts(request.model_experiment)
        cluster_id = server.predict_single(request.text)
        preview = request.text[:60] + "..." if len(request.text) > 60 else request.text
        
        cluster_name = CLUSTER_MAPPING.get(cluster_id, f"Cluster {cluster_id} (Unknown)")
        
        return ClusterResponse(
            text_preview=preview,
            cluster_id=cluster_id,
            cluster_name=cluster_name,
            experiment_used=server.active_exp_id
        )
    except AttributeError as ae:
        raise HTTPException(status_code=400, detail=str(ae))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Engine failure: {str(e)}")

@router.post("/bulk-predict", response_model=BulkClusterResponse)
async def predict_bulk_clusters(request: BulkClusterRequest, server: LiveInferenceServer = Depends(get_inference_server)):
    try:
        server.load_experiment_artifacts(request.model_experiment)
        responses = []
        for text in request.texts:
            cluster_id = server.predict_single(text)
            preview = text[:60] + "..." if len(text) > 60 else text
            cluster_name = CLUSTER_MAPPING.get(cluster_id, f"Cluster {cluster_id}")
            
            responses.append(ClusterResponse(
                text_preview=preview,
                cluster_id=cluster_id,
                cluster_name=cluster_name,
                experiment_used=server.active_exp_id
            ))
        return BulkClusterResponse(predictions=responses)
    except AttributeError as ae:
        raise HTTPException(status_code=400, detail=str(ae))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch Inference failure: {str(e)}")
    

@router.get("/leaderboard")
async def get_live_leaderboard():
    csv_path = "outputs/clustering_leaderboard.csv"
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail="Leaderboard is empty.")
    df = pd.read_csv(csv_path)
    return df.to_dict(orient="records")

app.include_router(router)


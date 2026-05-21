# news-group-clustering

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: FastAPI](https://img.shields.io/badge/FastAPI-0.136.1-009688.svg?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![ML: Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-1.8.0-F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Embeddings: Sentence--Transformers](https://img.shields.io/badge/Sentence--Transformers-5.4.1-red.svg)](https://huggingface.co/sentence-transformers)
[![UMAP: umap--learn](https://img.shields.io/badge/UMAP--Learn-0.5.12-orange.svg)](https://umap-learn.readthedocs.io/)

An advanced, production-grade **End-to-End News Text Clustering Engine**. This repository delivers a modular MLOps pipeline designed to vectorize unstructured textual documents, reduce high-dimensional manifolds, and partition news vectors into semantic clusters using optimized BERT representations.

---

## 📂 1. Project Deliverables

This repository provides a fully decoupled, enterprise-ready machine learning lifecycle:

1. **Central Entrypoint Orchestrator (`app.py`)**: The main execution hub that boots the production Uvicorn server and links core environment boundaries.
2. **Production API Layer (`src/api/routes.py`)**: High-performance FastAPI endpoints delivering real-time single inference (`/predict`), batch document processing (`/bulk-predict`), and training performance ledger tracking (`/leaderboard`).
3. **Inference Cache Server (`src/api/pipeline_server.py`)**: An in-memory lifecycle manager leveraging the Singleton pattern to look up and hot-swap local serialized checkpoints without system reboots.
4. **Offline Training Pipeline (`src/train.py`)**: A centralized training execution routine that iterates over multi-scale experiments, manages automated dimension tracking, and updates a local performance ledger.

---

## 📊 2. Data & Dimensionality Transformation Pipeline

The system processes text documents through a rigorous, mathematical pipeline to overcome the "Curse of Dimensionality":

* **Deep Vectorization:** Raw texts are passed through a locally cached `all-MiniLM-L6-v2` Sentence-Transformer model, translating text semantics into dense **384-dimensional** embedding vectors.
* **Manifold Learning & Reduction:** High-dimensional dense vectors are compressed via **UMAP (Uniform Manifold Approximation and Projection)** down to a stable, lower-dimensional space (e.g., **10-D** in Exp A-06) applying custom-tuned neighborhood settings (`n_neighbors=15`) to preserve local structural topologies.
* **Iterative Partitioning:** Deployment of **K-Means Clustering** ($K=20$) configured with deterministic initialization rules (`random_state=42`) to segment data points into dense, highly separated neighborhoods.

---

## 🧠 3. Evaluation & Production Label Resolution

To ensure strict tracking and clinical alignment of Unsupervised models, the project implements:

* **Pairwise Metrics Tracking:** Models are validated against Ground Truth configurations using **Adjusted Rand Index (ARI)** and **Normalized Mutual Information (NMI)** metrics, safeguarding partition quality regardless of cluster identifier indexing.
* **Empirical Majority-Vote Mapping:** Real-time cluster IDs are resolved via a verified majority-vote mapping dictionary generated directly from training distributions, translating raw integers into actual human-readable 20-Newsgroup categories (e.g., `sci.space`, `comp.sys.ibm.pc.hardware`).

---

## 🛠️ 4. Tech Stack & Architecture

* **Runtime Environment:** Python 3.12.2 (Managed via `pyenv`)
* **API Framework:** FastAPI with `uvicorn` ASGI Server
* **Dimensionality Reduction:** `umap-learn`
* **Clustering & Analytics:** `scikit-learn` & `pandas`
* **Embeddings:** `sentence-transformers`
* **Configuration Utility:** PyYAML (Structured `config.yaml` processing)

---

## 📁 5. Repository Structure

```text
.
├── app.py                  # Main central entrypoint (Uvicorn Orchestrator)
├── config/
│   └── config.yaml         # Central pipeline hyperparameters & directory paths
├── data/
│   ├── processed/          # Pickled BERT/TF-IDF matrices and data splits
│   └── raw/                # Original newsgroups_raw.csv source ledger
├── models/
│   ├── all-MiniLM-L6-v2/   # Local standalone HuggingFace weights cache
│   └── saved_models/       # Serialized (.pkl) Reducer and Clusterer checkpoints
├── notebooks/              # Laboratory exploration & prototyping notebooks
│   ├── 01_eda_cleaning.ipynb
│   ├── 02_embeddings.ipynb
│   └── 03_models.ipynb     # Empirical analysis and majority-voting trials
├── outputs/
│   ├── clustering_leaderboard.csv # Central registry of tracking metrics
│   └── training_history.log       # Chronological training performance telemetry
├── src/
│   ├── api/                # Core production API endpoints and schemas
│   │   ├── __init__.py
│   │   ├── pipeline_server.py # Memory-cached inference worker
│   │   ├── routes.py       # FastAPI application endpoints
│   │   └── schemas.py      # Strict Pydantic operational contracts
│   ├── core/               # Low-level infrastructure algorithms
│   │   ├── clusterer.py    # K-Means initialization wrapper
│   │   ├── dataset.py      # Local disk-ingestion text handlers
│   │   └── reducer.py      # Abstracted UMAP/PCA framework interfaces
│   └── train.py            # Central automated training script execution
├── REQUIREMENTS.txt        # Virtual environment dependencies
├── LICENSE                 # MIT License Document
└── README.md               # Main project documentation hub

```


## ⚙️ 6. Setup & Installation

1. Environment Configuration
Clone the repository and ensure your virtual environment or pyenv is active:

```bash
pip install -r requirements.txt
```

2. Run Pipeline Training
To execute data ingestion, run multi-scale training configurations, and update the metrics leaderboard:

```bash
python src/train.py
```

3. Launching the Production Server
To boot up the FastAPI cluster server hub using the centralized entrypoint script:

```bash
python app.py
```

Once the logging verifies connection, navigate to the active Swagger UI client documentation:
`http://127.0.0.1:8000/docs`

---

## Authors

Course project implementation by:

 - Mohammed Sherif Safa



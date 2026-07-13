import json
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from datasets import DATASETS
from precompute import make_key

EMBEDDINGS_DIR = Path(os.environ.get("EMBEDDINGS_DIR", "data/embeddings"))

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


def _load_json(dataset_name: str) -> dict:
    path = EMBEDDINGS_DIR / f"{dataset_name}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"No embeddings for '{dataset_name}'")
    return json.loads(path.read_text())


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.get("/api/datasets")
def list_datasets():
    result = []
    for name, meta in DATASETS.items():
        path = EMBEDDINGS_DIR / f"{name}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        m = data.get("_meta", {})
        result.append({
            "name": name,
            "label": meta["label"],
            "n_points": m.get("n_points"),
            "has_labels": m.get("label_names") is not None,
        })
    return result


@app.get("/api/embeddings/{dataset_name}")
def get_embedding(dataset_name: str, n_neighbors: int, min_dist: float,
                  n_components: int, metric: str):
    if dataset_name not in DATASETS:
        raise HTTPException(status_code=404, detail=f"Unknown dataset '{dataset_name}'")
    data = _load_json(dataset_name)
    key = make_key(n_neighbors, min_dist, n_components, metric)
    if key not in data:
        raise HTTPException(status_code=404, detail=f"No embedding for key '{key}'")
    return data[key]

import json
import os
import re
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from typing import Literal

import colorcet as cc
from datasets import DATASETS
from precompute import make_key

def _label_colors(label_names: list | None) -> list | None:
    if not label_names:
        return None
    return [cc.glasbey_dark[i % len(cc.glasbey_dark)] for i in range(len(label_names))]

EMBEDDINGS_DIR = Path(os.environ.get("EMBEDDINGS_DIR", "data/embeddings"))
_STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


def _load_json(dataset_name: str) -> dict:
    path = EMBEDDINGS_DIR / f"{dataset_name}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"No embeddings for '{dataset_name}'")
    return json.loads(path.read_text())


@app.get("/")
def index():
    return FileResponse(_STATIC_DIR / "index.html")


@app.get("/api/datasets")
def list_datasets():
    result = []
    for name, meta in DATASETS.items():
        path = EMBEDDINGS_DIR / f"{name}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        m = data.get("_meta", {})
        label_names = m.get("label_names")
        result.append({
            "name": name,
            "label": meta["label"],
            "n_points": m.get("n_points"),
            "has_labels": label_names is not None,
            "label_colors": _label_colors(label_names),
        })
    return result


@app.get("/api/embeddings/{dataset_name}")
def get_embedding(
    dataset_name: str,
    n_neighbors: int = Query(..., ge=1),
    min_dist: float = Query(..., ge=0.0, le=2.0),
    n_components: int = Query(..., ge=2, le=3),
    metric: Literal['euclidean', 'cosine', 'manhattan', 'correlation'] = Query(...),
):
    if not re.match(r'^[a-zA-Z0-9_]+$', dataset_name):
        raise HTTPException(status_code=400, detail="Invalid dataset name")
    if dataset_name not in DATASETS:
        raise HTTPException(status_code=404, detail=f"Unknown dataset '{dataset_name}'")
    data = _load_json(dataset_name)
    key = make_key(n_neighbors, min_dist, n_components, metric)
    if key not in data:
        raise HTTPException(status_code=404, detail=f"No embedding for key '{key}'")
    return data[key]

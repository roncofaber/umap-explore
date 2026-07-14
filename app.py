import os
import re
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from typing import Literal

import colorcet as cc
import h5py
from datasets.meta import DATASETS_META
from utils import make_key


def _label_colors(label_names: list | None) -> list | None:
    if not label_names:
        return None
    return [cc.glasbey_dark[i % len(cc.glasbey_dark)] for i in range(len(label_names))]


def _decode(arr) -> list:
    """h5py may return byte strings; decode them."""
    return [s.decode() if isinstance(s, bytes) else s for s in arr.tolist()]


EMBEDDINGS_DIR = Path(os.environ.get("EMBEDDINGS_DIR", "data/embeddings"))
_STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


def _h5(dataset_name: str) -> Path:
    return EMBEDDINGS_DIR / f"{dataset_name}.h5"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def index():
    return FileResponse(_STATIC_DIR / "index.html")


@app.get("/api/datasets")
def list_datasets():
    result = []
    for name, meta in DATASETS_META.items():
        path = _h5(name)
        if not path.exists():
            continue
        with h5py.File(path, 'r') as f:
            if '_meta' not in f:
                continue
            m = f['_meta']
            n_points = int(m['n_points'][()])
            label_names = _decode(m['label_names'][()]) if 'label_names' in m else None
            feature_names = _decode(m['feature_names'][()]) if 'feature_names' in m else None
        result.append({
            "name": name,
            "label": meta["label"],
            "n_points": n_points,
            "n_features": meta.get("n_features"),
            "description": meta.get("description"),
            "has_labels": label_names is not None,
            "label_colors": _label_colors(label_names),
            "feature_names": feature_names,
        })
    return result


@app.get("/api/data/{dataset_name}")
def get_data(
    dataset_name: str,
    scale: Literal['scaled', 'raw'] = Query('raw'),
):
    if not re.match(r'^[a-zA-Z0-9_]+$', dataset_name):
        raise HTTPException(status_code=400, detail="Invalid dataset name")
    if dataset_name not in DATASETS_META:
        raise HTTPException(status_code=404, detail=f"Unknown dataset '{dataset_name}'")
    path = _h5(dataset_name)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"No data for '{dataset_name}'")
    with h5py.File(path, 'r') as f:
        m = f['_meta']
        if 'X_raw' not in m:
            raise HTTPException(status_code=404, detail="Feature data not yet available — run add_features_to_h5.py")
        X = m['X_scaled' if scale == 'scaled' else 'X_raw'][()].tolist()
        feature_names = _decode(m['feature_names'][()]) if 'feature_names' in m else None
        labels = m['labels'][()].tolist()
        label_names = _decode(m['label_names'][()]) if 'label_names' in m else None
    return {
        'X': X,
        'feature_names': feature_names,
        'labels': labels,
        'label_names': label_names,
    }


@app.get("/api/embeddings/{dataset_name}")
def get_embedding(
    dataset_name: str,
    method: Literal['umap', 'pca'] = Query('umap'),
    n_neighbors: int = Query(15, ge=1),
    min_dist: float = Query(0.1, ge=0.0, le=2.0),
    n_components: int = Query(2, ge=2, le=2),
    metric: Literal['euclidean', 'cosine', 'manhattan', 'correlation'] = Query('euclidean'),
    scale: Literal['scaled', 'raw'] = Query('scaled'),
):
    if not re.match(r'^[a-zA-Z0-9_]+$', dataset_name):
        raise HTTPException(status_code=400, detail="Invalid dataset name")
    if dataset_name not in DATASETS_META:
        raise HTTPException(status_code=404, detail=f"Unknown dataset '{dataset_name}'")
    path = _h5(dataset_name)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"No embeddings for '{dataset_name}'")
    key = f"pca_{n_components}_{scale}" if method == 'pca' else make_key(n_neighbors, min_dist, n_components, metric, scale)
    with h5py.File(path, 'r') as f:
        if key not in f:
            raise HTTPException(status_code=404, detail=f"No embedding for key '{key}'")
        grp = f[key]
        m = f['_meta']
        return {
            'x': grp['x'][()].tolist(),
            'y': grp['y'][()].tolist(),
            'z': grp['z'][()].tolist() if 'z' in grp else None,
            'labels': m['labels'][()].tolist(),
            'label_names': _decode(m['label_names'][()]) if 'label_names' in m else None,
        }

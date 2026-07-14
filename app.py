import os
import re
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from typing import Literal

import colorcet as cc
import h5py
import numpy as np
import hdbscan as hdbscan_lib
from datasets.meta import DATASETS_META, make_key


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


def _get_cluster_coords(dataset_name, method, n_neighbors, min_dist,
                        n_components, metric, scale, cluster_on, path):
    """Shared helper: return the coordinate matrix to cluster on."""
    if cluster_on == 'data':
        with h5py.File(path, 'r') as f:
            m = f['_meta']
            if 'X_raw' not in m:
                raise HTTPException(
                    status_code=404,
                    detail="Feature data not available — run: python precompute.py --action add-features",
                )
            X_key = 'X_scaled' if scale == 'scaled' else 'X_raw'
            return m[X_key][()]
    else:
        key = (f"pca_{n_components}_{scale}" if method == 'pca'
               else make_key(n_neighbors, min_dist, n_components, metric, scale))
        with h5py.File(path, 'r') as f:
            if key not in f:
                raise HTTPException(status_code=404, detail=f"No embedding for key '{key}'")
            return np.column_stack([f[key]['x'][()], f[key]['y'][()]])


def _hdbscan_params(min_cluster_size, min_samples, cluster_selection_method,
                    cluster_selection_epsilon, allow_single_cluster):
    return dict(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=cluster_selection_method,
        cluster_selection_epsilon=cluster_selection_epsilon,
        allow_single_cluster=allow_single_cluster,
    )


@app.get("/api/cluster/{dataset_name}")
def get_cluster(
    dataset_name: str,
    method: Literal['umap', 'pca'] = Query('umap'),
    n_neighbors: int = Query(15, ge=1),
    min_dist: float = Query(0.1, ge=0.0, le=2.0),
    n_components: int = Query(2, ge=2, le=2),
    metric: Literal['euclidean', 'cosine', 'manhattan', 'correlation'] = Query('euclidean'),
    scale: Literal['scaled', 'raw'] = Query('scaled'),
    min_cluster_size: int = Query(15, ge=2),
    min_samples: int = Query(5, ge=1),
    cluster_selection_method: Literal['eom', 'leaf'] = Query('eom'),
    cluster_selection_epsilon: float = Query(0.0, ge=0.0),
    allow_single_cluster: bool = Query(False),
    cluster_on: Literal['projection', 'data'] = Query('projection'),
):
    if not re.match(r'^[a-zA-Z0-9_]+$', dataset_name):
        raise HTTPException(status_code=400, detail="Invalid dataset name")
    if dataset_name not in DATASETS_META:
        raise HTTPException(status_code=404, detail=f"Unknown dataset '{dataset_name}'")
    path = _h5(dataset_name)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"No embeddings for '{dataset_name}'")

    coords = _get_cluster_coords(dataset_name, method, n_neighbors, min_dist,
                                  n_components, metric, scale, cluster_on, path)
    clusterer = hdbscan_lib.HDBSCAN(
        **_hdbscan_params(min_cluster_size, min_samples, cluster_selection_method,
                          cluster_selection_epsilon, allow_single_cluster)
    ).fit(coords)

    labels = clusterer.labels_.tolist()
    unique = sorted(set(l for l in labels if l >= 0))
    palette = cc.glasbey_dark
    colors = [palette[l % len(palette)] if l >= 0 else '#c0c8d8' for l in labels]
    n_clusters = len(unique)
    n_noise = labels.count(-1)

    return {
        'labels': labels,
        'colors': colors,
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'cluster_names': [f'cluster {i}' for i in unique],
        'cluster_colors': [palette[i % len(palette)] for i in unique],
    }


@app.get("/api/cluster/{dataset_name}/tree")
def get_cluster_tree(
    dataset_name: str,
    method: Literal['umap', 'pca'] = Query('umap'),
    n_neighbors: int = Query(15, ge=1),
    min_dist: float = Query(0.1, ge=0.0, le=2.0),
    n_components: int = Query(2, ge=2, le=2),
    metric: Literal['euclidean', 'cosine', 'manhattan', 'correlation'] = Query('euclidean'),
    scale: Literal['scaled', 'raw'] = Query('scaled'),
    min_cluster_size: int = Query(15, ge=2),
    min_samples: int = Query(5, ge=1),
    cluster_selection_method: Literal['eom', 'leaf'] = Query('eom'),
    cluster_selection_epsilon: float = Query(0.0, ge=0.0),
    allow_single_cluster: bool = Query(False),
    cluster_on: Literal['projection', 'data'] = Query('projection'),
):
    if not re.match(r'^[a-zA-Z0-9_]+$', dataset_name):
        raise HTTPException(status_code=400, detail="Invalid dataset name")
    if dataset_name not in DATASETS_META:
        raise HTTPException(status_code=404, detail=f"Unknown dataset '{dataset_name}'")
    path = _h5(dataset_name)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"No embeddings for '{dataset_name}'")

    coords = _get_cluster_coords(dataset_name, method, n_neighbors, min_dist,
                                  n_components, metric, scale, cluster_on, path)
    clusterer = hdbscan_lib.HDBSCAN(
        **_hdbscan_params(min_cluster_size, min_samples, cluster_selection_method,
                          cluster_selection_epsilon, allow_single_cluster)
    ).fit(coords)

    n = len(coords)
    palette = cc.glasbey_dark
    tree_df = clusterer.condensed_tree_.to_pandas()
    selected = set(int(x) for x in clusterer.condensed_tree_._select_clusters())
    sorted_selected = sorted(selected)
    node_to_label = {node: i for i, node in enumerate(sorted_selected)}

    # Build node info: every row where child_size > 1 is an internal cluster node.
    # The root is the node that appears as parent but never as a child (ID = n).
    root_id = int(tree_df['parent'].min())
    node_map = {root_id: {'id': root_id, 'parent': -1, 'birth_lambda': 0.0,
                           'size': n, 'death_lambda': 0.0}}

    for _, row in tree_df[tree_df['child_size'] > 1].iterrows():
        cid, pid = int(row['child']), int(row['parent'])
        if cid not in node_map:
            node_map[cid] = {'id': cid, 'parent': pid,
                              'birth_lambda': float(row['lambda_val']),
                              'size': int(row['child_size']), 'death_lambda': 0.0}

    for _, row in tree_df.iterrows():
        pid = int(row['parent'])
        if pid in node_map:
            node_map[pid]['death_lambda'] = max(
                node_map[pid]['death_lambda'], float(row['lambda_val']))

    nodes = []
    for info in node_map.values():
        is_sel = info['id'] in selected
        label = node_to_label.get(info['id'], -1)
        color = palette[label % len(palette)] if is_sel else '#c0c8d8'
        nodes.append({
            'id': info['id'], 'parent': info['parent'],
            'birth_lambda': info['birth_lambda'],
            'death_lambda': max(info['death_lambda'], info['birth_lambda'] + 1e-6),
            'size': info['size'], 'selected': is_sel,
            'label': label, 'color': color,
        })

    return {'nodes': nodes, 'n_points': n,
            'epsilon': cluster_selection_epsilon, 'n_clusters': len(selected)}


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
        # Use Procrustes-aligned coordinates when available (--action align),
        # falling back to originals so the app works before alignment is run.
        x_key = 'x_aligned' if 'x_aligned' in grp else 'x'
        y_key = 'y_aligned' if 'y_aligned' in grp else 'y'
        return {
            'x': grp[x_key][()].tolist(),
            'y': grp[y_key][()].tolist(),
            'z': grp['z'][()].tolist() if 'z' in grp else None,
            'labels': m['labels'][()].tolist(),
            'label_names': _decode(m['label_names'][()]) if 'label_names' in m else None,
            'explained_variance_ratio': grp['explained_variance_ratio'][()].tolist()
                                        if 'explained_variance_ratio' in grp else None,
            'aligned': x_key == 'x_aligned',
        }

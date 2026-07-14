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


def _condensed_tree_plot_data(clusterer, palette):
    """Build Plotly-ready data from the hdbscan condensed tree using the
    library's own get_plot_data() method."""
    ct = clusterer.condensed_tree_          # already a CondensedTree object
    pd = ct.get_plot_data(max_rectangle_per_icicle=20)

    sel  = sorted(int(x) for x in ct._select_clusters())
    n_cl = len(sel)

    selected_info = []
    for i, node in enumerate(sel):
        b = pd['cluster_bounds'].get(node, [0, 0, 0, 0])
        y_top = float(b[3]) if np.isfinite(float(b[3])) else float(b[2]) * 2 + 0.1
        selected_info.append({
            'label':  i,
            'color':  palette[i % len(palette)],
            'bounds': {
                'x_left':   float(b[0]), 'x_right': float(b[1]),
                'y_bottom': float(b[2]), 'y_top':   y_top,
            },
        })

    max_w = max((float(x) for x in pd['bar_widths']), default=1.0)
    return {
        'bars': {
            'centers':          [float(x) for x in pd['bar_centers']],
            'tops':             [float(x) for x in pd['bar_tops']],
            'bottoms':          [float(x) for x in pd['bar_bottoms']],
            'widths':           [float(x) for x in pd['bar_widths']],
            'sizes_normalized': [float(x) / max_w for x in pd['bar_widths']],
        },
        'lines': [
            {'x': [float(xs[0]), float(xs[1])], 'y': [float(ys[0]), float(ys[1])]}
            for xs, ys in zip(pd['line_xs'], pd['line_ys'])
        ],
        'selected_clusters': selected_info,
        'n_clusters': n_cl,
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
    palette = cc.glasbey_dark
    clusterer = hdbscan_lib.HDBSCAN(
        **_hdbscan_params(min_cluster_size, min_samples, cluster_selection_method,
                          cluster_selection_epsilon, allow_single_cluster)
    ).fit(coords)

    labels = clusterer.labels_.tolist()
    unique = sorted(set(l for l in labels if l >= 0))
    colors = [palette[l % len(palette)] if l >= 0 else '#c0c8d8' for l in labels]

    data = _condensed_tree_plot_data(clusterer, palette)
    data.update({
        'epsilon': cluster_selection_epsilon,
        'labels': labels,
        'colors': colors,
        'n_clusters': len(unique),
        'n_noise': labels.count(-1),
        'cluster_names': [f'cluster {i}' for i in unique],
        'cluster_colors': [palette[i % len(palette)] for i in unique],
    })
    return data


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

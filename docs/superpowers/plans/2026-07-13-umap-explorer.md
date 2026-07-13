# UMAP Explorer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an interactive UMAP parameter explorer — a FastAPI backend serving pre-computed embeddings and a vanilla JS + Plotly.js frontend with smooth animated transitions between embeddings.

**Architecture:** A one-time `precompute.py` script generates 288 embeddings per dataset (6×6×2×4 parameter grid, seed=42) and saves them as one JSON file per dataset with a `_meta` key for fast dataset listing. FastAPI serves `/api/datasets` and `/api/embeddings/{dataset}` from those files at runtime with no UMAP computation. The frontend fetches embeddings on demand and renders with Plotly.js, using `Plotly.animate()` for smooth transitions on parameter changes and `Plotly.react()` for full re-renders on dataset switches.

**Tech Stack:** Python 3.10+, FastAPI, uvicorn, umap-learn, scikit-learn, numpy, pytest, httpx; Plotly.js 2.35.2 (CDN)

## Global Constraints

- Python 3.10+
- All UMAP computations use `random_state=42`
- Embedding key format: `"{n_neighbors}_{min_dist}_{n_components}_{metric}"` (e.g., `"15_0.1_2_euclidean"`)
- Data directory: `data/embeddings/` (configurable via `EMBEDDINGS_DIR` env var for tests)
- No TypeScript, no npm/bundlers — vanilla JS only
- Plotly.js loaded from CDN: `https://cdn.plot.ly/plotly-2.35.2.min.js`
- Always run commands from the project root `umap-explore/`

---

## File Map

| File | Responsibility |
|------|---------------|
| `requirements.txt` | All Python dependencies |
| `precompute.py` | Offline script: iterate parameter grid, compute embeddings, write JSON |
| `app.py` | FastAPI app: serve static files + two API endpoints |
| `datasets/__init__.py` | `DATASETS` registry + `DatasetResult` TypedDict |
| `datasets/iris.py` | Iris loader |
| `datasets/mnist.py` | MNIST loader (3000 subsampled points) |
| `datasets/swiss_roll.py` | Swiss Roll loader (continuous labels) |
| `data/embeddings/` | One JSON per dataset (generated, not committed to git) |
| `static/index.html` | Two-column layout: controls panel + plot area |
| `static/style.css` | All styles |
| `static/app.js` | Controls wiring, fetch, Plotly rendering + animations |
| `tests/test_datasets.py` | Unit tests for loaders and registry |
| `tests/test_precompute.py` | Unit tests for key format and embedding output structure |
| `tests/test_api.py` | Integration tests for API endpoints |

---

### Task 1: Project scaffold

**Files:**
- Create: `requirements.txt`
- Create: `app.py`
- Create: `static/index.html` (placeholder)
- Create: `static/app.js` (placeholder)
- Create: `static/style.css` (placeholder)
- Create: `data/embeddings/.gitkeep`
- Create: `datasets/__init__.py` (empty)
- Create: `tests/__init__.py`

**Interfaces:**
- Produces: FastAPI `app` object in `app.py`, importable as `from app import app`

- [ ] **Step 1: Create requirements.txt**

```
fastapi==0.115.0
uvicorn[standard]==0.32.0
umap-learn==0.5.7
scikit-learn==1.5.2
numpy==1.26.4
httpx==0.27.2
pytest==8.3.3
```

- [ ] **Step 2: Install dependencies**

```bash
pip install -r requirements.txt
```

Expected: all packages install without errors.

- [ ] **Step 3: Create directory structure**

```bash
mkdir -p data/embeddings static datasets tests
touch data/embeddings/.gitkeep datasets/__init__.py tests/__init__.py
```

- [ ] **Step 4: Create placeholder static files**

`static/index.html`:
```html
<!DOCTYPE html>
<html><head><title>UMAP Explorer</title></head>
<body><h1>UMAP Explorer — coming soon</h1></body>
</html>
```

`static/app.js`:
```javascript
// placeholder
```

`static/style.css`:
```css
/* placeholder */
```

- [ ] **Step 5: Create app.py**

```python
import json
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

EMBEDDINGS_DIR = Path(os.environ.get("EMBEDDINGS_DIR", "data/embeddings"))

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def index():
    return FileResponse("static/index.html")
```

- [ ] **Step 6: Verify server starts**

```bash
uvicorn app:app --reload
```

Open http://localhost:8000 — expect "UMAP Explorer — coming soon". Stop with Ctrl+C.

- [ ] **Step 7: Initialize git and commit**

```bash
git init
git add .
git commit -m "feat: project scaffold"
```

---

### Task 2: Dataset loaders

**Files:**
- Create: `datasets/iris.py`
- Create: `datasets/mnist.py`
- Create: `datasets/swiss_roll.py`
- Modify: `datasets/__init__.py`
- Create: `tests/test_datasets.py`

**Interfaces:**
- Produces: `DATASETS` dict in `datasets/__init__.py`:
  ```python
  DATASETS: dict[str, dict] = {
      'iris': {'label': 'Iris', 'loader': load_iris},
      'mnist': {'label': 'MNIST Digits', 'loader': load_mnist},
      'swiss_roll': {'label': 'Swiss Roll', 'loader': load_swiss_roll},
  }
  ```
- Each loader returns:
  ```python
  {
      'X': np.ndarray,          # shape (n_points, n_features)
      'labels': list,            # int list (categorical) or float list (continuous)
      'label_names': list[str] | None,  # None for continuous
      'n_points': int,
  }
  ```

- [ ] **Step 1: Write failing tests**

`tests/test_datasets.py`:
```python
import numpy as np
from datasets import DATASETS


def test_iris_shape():
    data = DATASETS['iris']['loader']()
    assert data['X'].shape == (150, 4)
    assert len(data['labels']) == 150
    assert data['label_names'] == ['setosa', 'versicolor', 'virginica']
    assert data['n_points'] == 150


def test_mnist_shape():
    data = DATASETS['mnist']['loader']()
    assert data['X'].shape == (3000, 784)
    assert len(data['labels']) == 3000
    assert len(data['label_names']) == 10
    assert data['n_points'] == 3000


def test_swiss_roll_shape():
    data = DATASETS['swiss_roll']['loader']()
    assert data['X'].shape == (2000, 3)
    assert len(data['labels']) == 2000
    assert data['label_names'] is None
    assert data['n_points'] == 2000


def test_all_datasets_registered():
    assert set(DATASETS.keys()) == {'iris', 'mnist', 'swiss_roll'}
    for name, ds in DATASETS.items():
        assert 'label' in ds
        assert callable(ds['loader'])
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_datasets.py -v
```

Expected: ImportError or AttributeError — `DATASETS` not defined yet.

- [ ] **Step 3: Implement iris.py**

`datasets/iris.py`:
```python
import numpy as np
from sklearn.datasets import load_iris as _load_iris


def load_iris():
    raw = _load_iris()
    return {
        'X': raw.data,
        'labels': raw.target.tolist(),
        'label_names': list(raw.target_names),
        'n_points': len(raw.data),
    }
```

- [ ] **Step 4: Implement mnist.py**

`datasets/mnist.py`:
```python
import numpy as np
from sklearn.datasets import fetch_openml


def load_mnist():
    raw = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    rng = np.random.RandomState(42)
    idx = rng.choice(len(raw.data), size=3000, replace=False)
    X = raw.data[idx].astype(np.float32)
    labels = raw.target[idx].astype(int).tolist()
    return {
        'X': X,
        'labels': labels,
        'label_names': [str(i) for i in range(10)],
        'n_points': 3000,
    }
```

- [ ] **Step 5: Implement swiss_roll.py**

`datasets/swiss_roll.py`:
```python
from sklearn.datasets import make_swiss_roll


def load_swiss_roll():
    X, t = make_swiss_roll(n_samples=2000, random_state=42)
    return {
        'X': X,
        'labels': t.tolist(),
        'label_names': None,
        'n_points': 2000,
    }
```

- [ ] **Step 6: Populate datasets/__init__.py**

```python
from typing import TypedDict
import numpy as np
from datasets.iris import load_iris
from datasets.mnist import load_mnist
from datasets.swiss_roll import load_swiss_roll


class DatasetResult(TypedDict):
    X: np.ndarray
    labels: list
    label_names: list | None
    n_points: int


DATASETS: dict[str, dict] = {
    'iris': {
        'label': 'Iris',
        'loader': load_iris,
    },
    'mnist': {
        'label': 'MNIST Digits',
        'loader': load_mnist,
    },
    'swiss_roll': {
        'label': 'Swiss Roll',
        'loader': load_swiss_roll,
    },
}
```

- [ ] **Step 7: Run tests to verify they pass**

```bash
pytest tests/test_datasets.py -v
```

Expected: 4 tests pass. (`test_mnist_shape` downloads MNIST on first run — ~1 min.)

- [ ] **Step 8: Commit**

```bash
git add datasets/ tests/test_datasets.py
git commit -m "feat: add dataset loaders (iris, mnist, swiss_roll)"
```

---

### Task 3: Precompute script

**Files:**
- Create: `precompute.py`
- Create: `tests/test_precompute.py`

**Interfaces:**
- Consumes: `DATASETS` from `datasets/__init__.py`
- Produces: `data/embeddings/{dataset}.json` with structure:
  ```json
  {
    "_meta": {"n_points": 150, "label_names": ["setosa", "versicolor", "virginica"]},
    "15_0.1_2_euclidean": {
      "x": [...], "y": [...], "z": null,
      "labels": [...], "label_names": [...]
    }
  }
  ```
- Produces: `make_key(n_neighbors, min_dist, n_components, metric) -> str`, importable from `precompute`

**Parameter grid constants (defined in precompute.py):**
```
N_NEIGHBORS = [5, 10, 15, 20, 30, 50]
MIN_DIST    = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0]
N_COMPONENTS = [2, 3]
METRICS     = ['euclidean', 'cosine', 'manhattan', 'correlation']
```

- [ ] **Step 1: Write failing tests**

`tests/test_precompute.py`:
```python
import json
import sys
import subprocess
from precompute import make_key


def test_make_key_basic():
    assert make_key(15, 0.1, 2, 'euclidean') == '15_0.1_2_euclidean'


def test_make_key_zero_dist():
    assert make_key(5, 0.0, 3, 'cosine') == '5_0.0_3_cosine'


def test_make_key_two_decimal():
    assert make_key(15, 0.05, 2, 'euclidean') == '15_0.05_2_euclidean'


def test_embedding_output_structure(tmp_path):
    result = subprocess.run(
        [
            sys.executable, 'precompute.py',
            '--dataset', 'iris',
            '--output-dir', str(tmp_path),
            '--n-neighbors', '5',
            '--min-dist', '0.1',
            '--n-components', '2',
            '--metric', 'euclidean',
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr

    out_file = tmp_path / 'iris.json'
    assert out_file.exists()
    data = json.loads(out_file.read_text())

    assert '_meta' in data
    assert data['_meta']['n_points'] == 150
    assert data['_meta']['label_names'] == ['setosa', 'versicolor', 'virginica']

    key = '5_0.1_2_euclidean'
    assert key in data
    emb = data[key]
    assert set(emb.keys()) == {'x', 'y', 'z', 'labels', 'label_names'}
    assert emb['z'] is None
    assert len(emb['x']) == len(emb['y']) == 150
    assert len(emb['labels']) == 150
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_precompute.py -v
```

Expected: ImportError — `precompute` module not found.

- [ ] **Step 3: Implement precompute.py**

```python
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import umap

from datasets import DATASETS

N_NEIGHBORS = [5, 10, 15, 20, 30, 50]
MIN_DIST = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0]
N_COMPONENTS = [2, 3]
METRICS = ['euclidean', 'cosine', 'manhattan', 'correlation']


def make_key(n_neighbors, min_dist, n_components, metric):
    return f"{n_neighbors}_{min_dist}_{n_components}_{metric}"


def compute_embedding(X, n_neighbors, min_dist, n_components, metric):
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
        random_state=42,
    )
    return reducer.fit_transform(X)


def precompute_dataset(dataset_name, output_dir, n_neighbors_list, min_dist_list,
                       n_components_list, metrics_list):
    dataset = DATASETS[dataset_name]
    data = dataset['loader']()
    X = data['X']

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{dataset_name}.json"

    results = json.loads(out_file.read_text()) if out_file.exists() else {}
    results['_meta'] = {
        'n_points': data['n_points'],
        'label_names': data['label_names'],
    }

    combos = list(itertools.product(n_neighbors_list, min_dist_list, n_components_list, metrics_list))
    total = len(combos)

    for i, (nn, md, nc, metric) in enumerate(combos):
        key = make_key(nn, md, nc, metric)
        if key in results:
            print(f"[{i+1}/{total}] {key} — skipping (cached)")
            continue
        print(f"[{i+1}/{total}] {key} — computing...")
        embedding = compute_embedding(X, nn, md, nc, metric)
        results[key] = {
            'x': embedding[:, 0].tolist(),
            'y': embedding[:, 1].tolist(),
            'z': embedding[:, 2].tolist() if nc == 3 else None,
            'labels': data['labels'],
            'label_names': data['label_names'],
        }
        out_file.write_text(json.dumps(results))

    print(f"Done. {len(results) - 1} embeddings saved to {out_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=list(DATASETS.keys()))
    parser.add_argument('--output-dir', default='data/embeddings')
    parser.add_argument('--n-neighbors', type=int, nargs='+', default=N_NEIGHBORS)
    parser.add_argument('--min-dist', type=float, nargs='+', default=MIN_DIST)
    parser.add_argument('--n-components', type=int, nargs='+', default=N_COMPONENTS)
    parser.add_argument('--metric', nargs='+', default=METRICS)
    args = parser.parse_args()

    targets = [args.dataset] if args.dataset else list(DATASETS.keys())
    for name in targets:
        print(f"\n=== Precomputing {name} ===")
        precompute_dataset(
            name, args.output_dir,
            args.n_neighbors, args.min_dist, args.n_components, args.metric,
        )


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_precompute.py -v
```

Expected: 4 tests pass. `test_embedding_output_structure` runs UMAP on iris (~10s).

- [ ] **Step 5: Commit**

```bash
git add precompute.py tests/test_precompute.py
git commit -m "feat: add precompute script with incremental JSON output"
```

---

### Task 4: FastAPI API endpoints

**Files:**
- Modify: `app.py`
- Create: `tests/test_api.py`

**Interfaces:**
- Consumes: `DATASETS` from `datasets/__init__.py`; JSON files from `EMBEDDINGS_DIR`; `make_key` from `precompute`
- Produces:
  - `GET /api/datasets` → `list[{name, label, n_points, has_labels}]`
  - `GET /api/embeddings/{dataset}?n_neighbors=int&min_dist=float&n_components=int&metric=str` → `{x, y, z, labels, label_names}`

- [ ] **Step 1: Write failing tests**

`tests/test_api.py`:
```python
import json
import pytest
from fastapi.testclient import TestClient

FIXTURE_EMBEDDING = {
    "_meta": {
        "n_points": 3,
        "label_names": ["setosa", "versicolor", "virginica"],
    },
    "15_0.1_2_euclidean": {
        "x": [1.0, 2.0, 3.0],
        "y": [0.5, 1.5, 2.5],
        "z": None,
        "labels": [0, 1, 2],
        "label_names": ["setosa", "versicolor", "virginica"],
    },
}


@pytest.fixture(autouse=True)
def patch_embeddings_dir(tmp_path, monkeypatch):
    emb_dir = tmp_path / "embeddings"
    emb_dir.mkdir()
    (emb_dir / "iris.json").write_text(json.dumps(FIXTURE_EMBEDDING))
    import app
    monkeypatch.setattr(app, "EMBEDDINGS_DIR", emb_dir)


@pytest.fixture
def client():
    from app import app as fastapi_app
    return TestClient(fastapi_app)


def test_list_datasets(client):
    resp = client.get("/api/datasets")
    assert resp.status_code == 200
    data = resp.json()
    iris = next(d for d in data if d["name"] == "iris")
    assert iris["n_points"] == 3
    assert iris["has_labels"] is True


def test_get_embedding(client):
    resp = client.get(
        "/api/embeddings/iris?n_neighbors=15&min_dist=0.1&n_components=2&metric=euclidean"
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["x"] == [1.0, 2.0, 3.0]
    assert data["z"] is None
    assert data["labels"] == [0, 1, 2]


def test_embedding_key_not_found(client):
    resp = client.get(
        "/api/embeddings/iris?n_neighbors=99&min_dist=0.1&n_components=2&metric=euclidean"
    )
    assert resp.status_code == 404


def test_dataset_not_found(client):
    resp = client.get(
        "/api/embeddings/nonexistent?n_neighbors=15&min_dist=0.1&n_components=2&metric=euclidean"
    )
    assert resp.status_code == 404
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_api.py -v
```

Expected: tests fail — endpoints not implemented.

- [ ] **Step 3: Implement API endpoints in app.py**

Replace `app.py` with:
```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_api.py -v
```

Expected: 4 tests pass.

- [ ] **Step 5: Run full test suite**

```bash
pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add app.py tests/test_api.py
git commit -m "feat: add API endpoints for datasets and embeddings"
```

---

### Task 5: Frontend HTML/CSS

**Files:**
- Modify: `static/index.html`
- Modify: `static/style.css`

**Interfaces:**
- Produces: HTML element IDs consumed by `app.js`:
  - `#dataset-select` — dataset dropdown
  - `#n-neighbors-slider`, `#n-neighbors-value`
  - `#min-dist-slider`, `#min-dist-value`
  - `#n-components-2d`, `#n-components-3d` — toggle buttons
  - `#metric-select`
  - `#plot` — Plotly container
  - `#loading` — spinner (hidden by default, shown during fetch)
  - `#umap-explainer` — collapsible details element

- [ ] **Step 1: Write index.html**

`static/index.html`:
```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>UMAP Explorer</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <link rel="stylesheet" href="/static/style.css">
</head>
<body>
  <header>
    <h1>UMAP Explorer</h1>
  </header>

  <main>
    <aside id="controls">
      <details id="umap-explainer">
        <summary>What is UMAP?</summary>
        <p>UMAP (Uniform Manifold Approximation and Projection) finds a low-dimensional map of high-dimensional data that preserves its structure. It works by learning which points are neighbors in the original space, then arranging them so neighbors stay close in the map. Use the controls below to see how each parameter shapes the result.</p>
      </details>

      <div class="control-group">
        <label for="dataset-select">Dataset</label>
        <select id="dataset-select"></select>
      </div>

      <div class="control-group">
        <label for="n-neighbors-slider">
          n_neighbors: <span id="n-neighbors-value">15</span>
        </label>
        <input type="range" id="n-neighbors-slider" min="0" max="5" step="1" value="2">
        <p class="param-description">How many nearby points UMAP considers when learning data structure. Low values capture fine local detail; high values reveal the global shape.</p>
      </div>

      <div class="control-group">
        <label for="min-dist-slider">
          min_dist: <span id="min-dist-value">0.1</span>
        </label>
        <input type="range" id="min-dist-slider" min="0" max="5" step="1" value="2">
        <p class="param-description">Minimum distance between points in the layout. Lower values pack clusters tightly; higher values spread them out more evenly.</p>
      </div>

      <div class="control-group">
        <label>n_components</label>
        <div class="toggle-group">
          <button id="n-components-2d" class="toggle active">2D</button>
          <button id="n-components-3d" class="toggle">3D</button>
        </div>
        <p class="param-description">Number of output dimensions. 2D is easier to read; 3D preserves slightly more structure and is rotatable.</p>
      </div>

      <div class="control-group">
        <label for="metric-select">metric</label>
        <select id="metric-select">
          <option value="euclidean">euclidean</option>
          <option value="cosine">cosine</option>
          <option value="manhattan">manhattan</option>
          <option value="correlation">correlation</option>
        </select>
        <p class="param-description">How distances between points are measured in the original space. Different metrics can reveal different structure in your data.</p>
      </div>
    </aside>

    <section id="plot-container">
      <div id="loading">Loading...</div>
      <div id="plot"></div>
    </section>
  </main>

  <script src="/static/app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Write style.css**

`static/style.css`:
```css
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: system-ui, sans-serif;
  background: #f8f9fa;
  color: #212529;
  height: 100vh;
  display: flex;
  flex-direction: column;
}

header {
  padding: 0.75rem 1.5rem;
  background: #2c3e50;
  color: white;
  flex-shrink: 0;
}

header h1 { font-size: 1.25rem; font-weight: 600; }

main {
  flex: 1;
  display: flex;
  overflow: hidden;
}

#controls {
  width: 280px;
  min-width: 240px;
  background: white;
  border-right: 1px solid #dee2e6;
  padding: 1rem;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 1.25rem;
}

#umap-explainer summary {
  cursor: pointer;
  font-weight: 600;
  font-size: 0.875rem;
  color: #495057;
  user-select: none;
}

#umap-explainer p {
  margin-top: 0.5rem;
  font-size: 0.8rem;
  line-height: 1.5;
  color: #6c757d;
}

.control-group { display: flex; flex-direction: column; gap: 0.375rem; }

.control-group > label {
  font-size: 0.875rem;
  font-weight: 600;
  color: #495057;
}

.control-group select,
.control-group input[type="range"] { width: 100%; cursor: pointer; }

.param-description {
  font-size: 0.75rem;
  color: #6c757d;
  line-height: 1.4;
}

.toggle-group { display: flex; gap: 0.5rem; }

.toggle {
  flex: 1;
  padding: 0.375rem;
  border: 1px solid #ced4da;
  background: white;
  border-radius: 4px;
  cursor: pointer;
  font-size: 0.875rem;
  transition: background 0.15s, color 0.15s;
}

.toggle.active {
  background: #2c3e50;
  color: white;
  border-color: #2c3e50;
}

#plot-container {
  flex: 1;
  position: relative;
  overflow: hidden;
}

#plot { width: 100%; height: 100%; }

#loading {
  display: none;
  position: absolute;
  top: 1rem;
  right: 1rem;
  background: rgba(44, 62, 80, 0.85);
  color: white;
  padding: 0.375rem 0.75rem;
  border-radius: 4px;
  font-size: 0.8rem;
  z-index: 10;
}
```

- [ ] **Step 3: Verify layout renders**

```bash
uvicorn app:app --reload
```

Open http://localhost:8000. Expect: two-column layout, controls on left, empty area on right. No JS console errors (app.js is still a placeholder). Stop server.

- [ ] **Step 4: Commit**

```bash
git add static/index.html static/style.css
git commit -m "feat: add frontend HTML/CSS two-column layout"
```

---

### Task 6: Frontend JS — controls, fetch pipeline, and Plotly rendering

**Files:**
- Modify: `static/app.js`

**Interfaces:**
- Consumes: `/api/datasets`, `/api/embeddings/{dataset}`, HTML IDs from Task 5
- Produces: interactive app with animated transitions

Slider index → value mappings:
```
N_NEIGHBORS_STEPS = [5, 10, 15, 20, 30, 50]   // slider index 0–5, default index 2 (value 15)
MIN_DIST_STEPS    = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0]  // slider index 0–5, default index 2 (value 0.1)
```

Color rules:
- Categorical (`label_names !== null`): normalize label index to `[0,1]`, use `'Turbo'` colorscale
- Continuous (`label_names === null`): pass raw float values as `marker.color`, use `'Viridis'` colorscale

Transition rules:
- `state.isFirstRender === true` or dimension changed (2D↔3D): use `Plotly.react()` (full re-render)
- Otherwise: use `Plotly.animate()` with 400ms cubic-in-out easing
- After any `Plotly.react()` call, set `state.isFirstRender = false`
- `state.isFirstRender` is reset to `true` on dataset change

- [ ] **Step 1: Write app.js**

`static/app.js`:
```javascript
const N_NEIGHBORS_STEPS = [5, 10, 15, 20, 30, 50];
const MIN_DIST_STEPS = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0];

const state = {
  dataset: null,
  nNeighbors: 15,
  minDist: 0.1,
  nComponents: 2,
  metric: 'euclidean',
  isFirstRender: true,
  prevNComponents: null,
};

const els = {
  datasetSelect: document.getElementById('dataset-select'),
  nnSlider:      document.getElementById('n-neighbors-slider'),
  nnValue:       document.getElementById('n-neighbors-value'),
  mdSlider:      document.getElementById('min-dist-slider'),
  mdValue:       document.getElementById('min-dist-value'),
  btn2d:         document.getElementById('n-components-2d'),
  btn3d:         document.getElementById('n-components-3d'),
  metricSelect:  document.getElementById('metric-select'),
  plot:          document.getElementById('plot'),
  loading:       document.getElementById('loading'),
};

async function fetchEmbedding() {
  const params = new URLSearchParams({
    n_neighbors:  state.nNeighbors,
    min_dist:     state.minDist,
    n_components: state.nComponents,
    metric:       state.metric,
  });
  const resp = await fetch(`/api/embeddings/${state.dataset}?${params}`);
  if (!resp.ok) throw new Error(`API error ${resp.status}`);
  return resp.json();
}

function makeTrace(emb) {
  const isContinuous = emb.label_names === null;
  const markerColor = isContinuous
    ? emb.labels
    : emb.labels.map(l => l / Math.max(emb.label_names.length - 1, 1));
  const colorscale = isContinuous ? 'Viridis' : 'Turbo';
  const hoverText = isContinuous
    ? emb.labels.map(v => `value: ${v.toFixed(2)}`)
    : emb.labels.map(l => emb.label_names[l]);

  if (state.nComponents === 3) {
    return {
      type: 'scatter3d', mode: 'markers',
      x: emb.x, y: emb.y, z: emb.z,
      text: hoverText,
      hovertemplate: '%{text}<extra></extra>',
      marker: { size: 3, color: markerColor, colorscale, showscale: isContinuous, opacity: 0.8 },
    };
  }

  return {
    type: 'scatter', mode: 'markers',
    x: emb.x, y: emb.y,
    text: hoverText,
    hovertemplate: '%{text}<extra></extra>',
    marker: { size: 5, color: markerColor, colorscale, showscale: isContinuous, opacity: 0.8 },
  };
}

function makeLayout() {
  const base = {
    margin: { t: 20, r: 20, b: 20, l: 20 },
    paper_bgcolor: '#f8f9fa',
    plot_bgcolor: '#f8f9fa',
    showlegend: false,
    uirevision: state.nComponents,
  };
  if (state.nComponents === 3) {
    return {
      ...base,
      scene: {
        xaxis: { visible: false },
        yaxis: { visible: false },
        zaxis: { visible: false },
      },
    };
  }
  return { ...base, xaxis: { visible: false }, yaxis: { visible: false } };
}

function renderPlot(emb) {
  const trace = makeTrace(emb);
  const layout = makeLayout();
  const dimensionChanged = state.prevNComponents !== null
    && state.prevNComponents !== state.nComponents;
  state.prevNComponents = state.nComponents;

  if (state.isFirstRender || dimensionChanged) {
    Plotly.react(els.plot, [trace], layout, { responsive: true });
    state.isFirstRender = false;
    return;
  }

  const frameData = { x: emb.x, y: emb.y, 'marker.color': trace.marker.color };
  if (state.nComponents === 3) frameData.z = emb.z;

  Plotly.animate(
    els.plot,
    { data: [frameData], traces: [0] },
    { transition: { duration: 400, easing: 'cubic-in-out' }, frame: { duration: 400 } },
  );
}

async function fetchAndRender() {
  els.loading.style.display = 'block';
  try {
    const emb = await fetchEmbedding();
    renderPlot(emb);
  } catch (e) {
    console.error('Failed to load embedding:', e);
  } finally {
    els.loading.style.display = 'none';
  }
}

els.nnSlider.addEventListener('input', () => {
  state.nNeighbors = N_NEIGHBORS_STEPS[parseInt(els.nnSlider.value)];
  els.nnValue.textContent = state.nNeighbors;
  fetchAndRender();
});

els.mdSlider.addEventListener('input', () => {
  state.minDist = MIN_DIST_STEPS[parseInt(els.mdSlider.value)];
  els.mdValue.textContent = state.minDist;
  fetchAndRender();
});

els.btn2d.addEventListener('click', () => {
  if (state.nComponents === 2) return;
  state.nComponents = 2;
  els.btn2d.classList.add('active');
  els.btn3d.classList.remove('active');
  fetchAndRender();
});

els.btn3d.addEventListener('click', () => {
  if (state.nComponents === 3) return;
  state.nComponents = 3;
  els.btn3d.classList.add('active');
  els.btn2d.classList.remove('active');
  fetchAndRender();
});

els.metricSelect.addEventListener('change', () => {
  state.metric = els.metricSelect.value;
  fetchAndRender();
});

els.datasetSelect.addEventListener('change', () => {
  state.dataset = els.datasetSelect.value;
  state.isFirstRender = true;
  fetchAndRender();
});

async function init() {
  const datasets = await fetch('/api/datasets').then(r => r.json());
  datasets.forEach(ds => {
    const opt = document.createElement('option');
    opt.value = ds.name;
    opt.textContent = ds.label;
    els.datasetSelect.appendChild(opt);
  });

  els.nnSlider.value = N_NEIGHBORS_STEPS.indexOf(state.nNeighbors);
  els.nnValue.textContent = state.nNeighbors;
  els.mdSlider.value = MIN_DIST_STEPS.indexOf(state.minDist);
  els.mdValue.textContent = state.minDist;

  if (datasets.length > 0) {
    state.dataset = datasets[0].name;
    fetchAndRender();
  }
}

init();
```

- [ ] **Step 2: Precompute iris embeddings (needed to test the UI)**

```bash
python precompute.py --dataset iris
```

Expected: `data/embeddings/iris.json` created with 288 embeddings (~5-10 min).

- [ ] **Step 3: Start server and verify 2D rendering**

```bash
uvicorn app:app --reload
```

Open http://localhost:8000. Expect:
- Iris scatter plot renders on load, points colored by species
- Hovering a point shows the species name
- No console errors

- [ ] **Step 4: Verify smooth transitions**

Move the `n_neighbors` slider from one end to the other slowly.
Expected: points animate smoothly between positions with no flicker or full re-render flash.

- [ ] **Step 5: Verify 3D toggle**

Click "3D". Expected: 3D scatter renders, camera is draggable. Click "2D" — switches back cleanly.

- [ ] **Step 6: Verify dataset switch (once MNIST is precomputed)**

```bash
python precompute.py --dataset mnist  # ~30-60 min
```

Then in the running app, switch dataset dropdown to MNIST. Expected: full re-render (no animation between datasets), new point cloud appears.

- [ ] **Step 7: Commit**

```bash
git add static/app.js
git commit -m "feat: add JS controls, fetch pipeline, and animated Plotly rendering"
```

---

### Task 7: Full precomputation and deployment smoke test

**Files:** none (operational task)

- [ ] **Step 1: Precompute all remaining datasets**

```bash
python precompute.py --dataset swiss_roll
python precompute.py --dataset mnist  # if not already done
```

Expected: `data/embeddings/swiss_roll.json` and `data/embeddings/mnist.json` created.

- [ ] **Step 2: Run full test suite**

```bash
pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 3: Manual smoke test checklist**

Start server: `uvicorn app:app --host 0.0.0.0 --port 8000`

- [ ] All three datasets appear in the dropdown
- [ ] n_neighbors slider: moving from 5→50 shows smooth transitions; at low values local clusters are tighter
- [ ] min_dist slider: high values spread points; low values pack them
- [ ] Metric dropdown: switching euclidean→cosine gives visibly different structure on MNIST
- [ ] 2D/3D toggle: 3D is rotatable; toggling back to 2D works cleanly
- [ ] Swiss Roll: uses a continuous colormap (not discrete class colors)
- [ ] Loading indicator appears briefly during each fetch
- [ ] UMAP explainer section expands/collapses correctly
- [ ] No JS console errors throughout

- [ ] **Step 4: Final commit**

```bash
git add .
git commit -m "chore: verify full precomputation and smoke test complete"
```

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import umap
from sklearn.preprocessing import StandardScaler

from datasets import DATASETS

N_NEIGHBORS = [5, 10, 15, 20, 30, 50, 100]
MIN_DIST = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0]
N_COMPONENTS = [2]
METRICS = ['euclidean', 'cosine', 'manhattan', 'correlation']
SCALE = ['scaled', 'raw']


def make_key(n_neighbors, min_dist, n_components, metric, scale):
    return f"{n_neighbors}_{min_dist}_{n_components}_{metric}_{scale}"


def compute_embedding(X, n_neighbors, min_dist, n_components, metric):
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
    )
    return reducer.fit_transform(X)


def precompute_dataset(dataset_name, output_dir, n_neighbors_list, min_dist_list,
                       n_components_list, metrics_list, scales_list):
    dataset = DATASETS[dataset_name]
    data = dataset['loader']()
    X_raw = data['X'].astype(float)
    X_scaled = StandardScaler().fit_transform(X_raw)
    X_by_scale = {'scaled': X_scaled, 'raw': X_raw}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{dataset_name}.json"

    results = json.loads(out_file.read_text()) if out_file.exists() else {}
    results['_meta'] = {
        'n_points': data['n_points'],
        'label_names': data['label_names'],
    }

    combos = list(itertools.product(n_neighbors_list, min_dist_list, n_components_list, metrics_list, scales_list))
    total = len(combos)

    for i, (nn, md, nc, metric, scale) in enumerate(combos):
        key = make_key(nn, md, nc, metric, scale)
        if key in results:
            print(f"[{i+1}/{total}] {key} — skipping (cached)")
            continue
        print(f"[{i+1}/{total}] {key} — computing...")
        embedding = compute_embedding(X_by_scale[scale], nn, md, nc, metric)
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
    parser.add_argument('--scale', nargs='+', default=SCALE, choices=SCALE)
    args = parser.parse_args()

    targets = [args.dataset] if args.dataset else list(DATASETS.keys())
    for name in targets:
        print(f"\n=== Precomputing {name} ===")
        precompute_dataset(
            name, args.output_dir,
            args.n_neighbors, args.min_dist, args.n_components, args.metric, args.scale,
        )


if __name__ == '__main__':
    main()

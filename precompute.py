"""Precompute and manage UMAP/PCA embeddings stored in HDF5 format.

Usage
-----
# Compute embeddings (default action):
python precompute.py [--dataset NAME] [--output-dir DIR] [--n-neighbors ...] ...

# Align embeddings via Procrustes (rotation + reflection, no scaling):
python precompute.py --action align [--dataset NAME] [--output-dir DIR]
  Adds x_aligned / y_aligned datasets alongside the original x / y.
  Reference per (metric, scale) group: largest n_neighbors, min_dist=0.1.
  Safe: original coordinates are never modified.

# Convert legacy JSON embeddings to HDF5 (one-time migration):
python precompute.py --action convert-json [--output-dir DIR]

# Add raw/scaled feature matrices to existing HDF5 files:
python precompute.py --action add-features [--dataset NAME] [--output-dir DIR]
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import umap
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from datasets import DATASETS
from datasets.meta import make_key, make_tsne_key

N_NEIGHBORS     = [5, 10, 15, 20, 30, 50, 100]
MIN_DIST        = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0]
N_COMPONENTS    = [2]
METRICS         = ['euclidean', 'cosine', 'manhattan', 'correlation']
SCALE           = ['scaled', 'raw']
PERPLEXITY_STEPS = [5, 15, 30, 50, 100]


# ── Embedding computation ──────────────────────────────────────────────────────

def _compute_umap(X, n_neighbors, min_dist, n_components, metric):
    return umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
    ).fit_transform(X)


def precompute_dataset(dataset_name, output_dir, n_neighbors_list, min_dist_list,
                       n_components_list, metrics_list, scales_list):
    dataset = DATASETS[dataset_name]
    data = dataset['loader']()
    X_raw    = data['X'].astype(float)
    X_scaled = StandardScaler().fit_transform(X_raw)
    X_by_scale = {'scaled': X_scaled, 'raw': X_raw}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{dataset_name}.h5"

    with h5py.File(out_file, 'a') as f:
        if '_meta' in f:
            del f['_meta']
        meta = f.create_group('_meta')
        meta.create_dataset('n_points', data=data['n_points'])
        meta.create_dataset('labels', data=np.array(data['labels']))
        if data['label_names'] is not None:
            meta.create_dataset('label_names', data=data['label_names'],
                                dtype=h5py.string_dtype())

        combos = list(itertools.product(
            n_neighbors_list, min_dist_list, n_components_list,
            metrics_list, scales_list,
        ))
        for i, (nn, md, nc, metric, scale) in enumerate(combos):
            key = make_key(nn, md, nc, metric, scale)
            if key in f:
                print(f"[{i+1}/{len(combos)}] {key} — skipping (cached)")
                continue
            print(f"[{i+1}/{len(combos)}] {key} — computing...")
            embedding = _compute_umap(X_by_scale[scale], nn, md, nc, metric)
            grp = f.create_group(key)
            grp.create_dataset('x', data=embedding[:, 0], compression='gzip')
            grp.create_dataset('y', data=embedding[:, 1], compression='gzip')
            if nc == 3:
                grp.create_dataset('z', data=embedding[:, 2], compression='gzip')
            f.flush()

        for scale in scales_list:
            pca_key = f"pca_3_{scale}"
            if pca_key in f:
                if 'explained_variance_ratio' in f[pca_key] and 'z' in f[pca_key]:
                    print(f"PCA ({scale}) — skipping (cached)")
                    continue
                print(f"PCA ({scale}) — refreshing...")
                del f[pca_key]
            print(f"PCA ({scale}) — computing (3 components)...")
            n_pc = min(3, X_by_scale[scale].shape[1])
            pca = PCA(n_components=n_pc)
            embedding = pca.fit_transform(X_by_scale[scale])
            grp = f.create_group(pca_key)
            grp.create_dataset('x', data=embedding[:, 0], compression='gzip')
            grp.create_dataset('y', data=embedding[:, 1], compression='gzip')
            if n_pc >= 3:
                grp.create_dataset('z', data=embedding[:, 2], compression='gzip')
            grp.create_dataset('explained_variance_ratio',
                               data=pca.explained_variance_ratio_)
            f.flush()

    print(f"Done. Saved to {out_file}")


# ── t-SNE precomputation ──────────────────────────────────────────────────────

def precompute_tsne(dataset_name, output_dir, perplexity_list, metrics_list, scales_list):
    dataset = DATASETS[dataset_name]
    data = dataset['loader']()
    X_raw    = data['X'].astype(float)
    X_scaled = StandardScaler().fit_transform(X_raw)
    X_by_scale = {'scaled': X_scaled, 'raw': X_raw}

    output_dir = Path(output_dir)
    out_file = output_dir / f"{dataset_name}.h5"
    if not out_file.exists():
        print(f"{dataset_name}: no HDF5 file yet — run --action compute first")
        return

    combos = list(itertools.product(perplexity_list, metrics_list, scales_list))
    with h5py.File(out_file, 'a') as f:
        for i, (perp, metric, scale) in enumerate(combos):
            key = make_tsne_key(perp, metric, scale)
            if key in f:
                print(f"[{i+1}/{len(combos)}] {key} — skipping (cached)")
                continue
            print(f"[{i+1}/{len(combos)}] {key} — computing...")
            init = 'pca' if metric == 'euclidean' else 'random'
            embedding = TSNE(
                n_components=2, perplexity=perp, metric=metric,
                init=init, learning_rate='auto', n_iter=500, random_state=42,
            ).fit_transform(X_by_scale[scale])
            grp = f.create_group(key)
            grp.create_dataset('x', data=embedding[:, 0], compression='gzip')
            grp.create_dataset('y', data=embedding[:, 1], compression='gzip')
            f.flush()

    print(f"Done. Saved to {out_file}")


# ── Procrustes alignment ──────────────────────────────────────────────────────

def _align_dataset(dataset_name, output_dir: Path,
                   n_neighbors_list, min_dist_list, n_components_list,
                   metrics_list, scales_list):
    """
    Align ALL embeddings in the dataset (UMAP across all params, PCA, all metrics
    and scales) to a single universal reference using Orthogonal Procrustes
    (rotation + optional reflection, no scaling).

    Reference: UMAP with standard parameters — n_neighbors=15, min_dist=0.1,
    metric=euclidean, scale=scaled.  If that key is absent, falls back to the
    first available key.

    Aligned coordinates are stored as x_aligned / y_aligned alongside the
    original x / y — originals are never modified.
    """
    path = output_dir / f'{dataset_name}.h5'
    if not path.exists():
        print(f'{dataset_name}: no HDF5 file, skipping')
        return

    # Universal reference: standard UMAP parameters
    ref_nn = 15  if 15  in n_neighbors_list  else n_neighbors_list[0]
    ref_md = 0.1 if 0.1 in min_dist_list     else min_dist_list[0]
    ref_key = make_key(ref_nn, ref_md, 2, 'euclidean', 'scaled')

    # Collect every embedding key that should be aligned
    all_keys = [
        make_key(nn, md, nc, metric, scale)
        for nn, md, nc, metric, scale in itertools.product(
            n_neighbors_list, min_dist_list, n_components_list,
            metrics_list, scales_list)
    ] + [f'pca_3_{scale}' for scale in scales_list] + [
        make_tsne_key(perp, metric, scale)
        for perp, metric, scale in itertools.product(
            PERPLEXITY_STEPS, metrics_list, scales_list)
    ]

    with h5py.File(path, 'a') as f:
        # Find the reference; fall back to first available key if missing
        if ref_key not in f:
            fallback = next((k for k in all_keys if k in f), None)
            if fallback is None:
                print(f'{dataset_name}: no embeddings found, skipping')
                return
            print(f'{dataset_name}: reference {ref_key!r} missing, using {fallback!r}')
            ref_key = fallback

        ref   = np.column_stack([f[ref_key]['x'][()], f[ref_key]['y'][()]])
        ref_mu = ref.mean(axis=0)
        ref_c  = ref - ref_mu

        aligned_count = 0
        for key in all_keys:
            if key not in f:
                continue
            if key == ref_key:
                for ax in ('x_aligned', 'y_aligned'):
                    if ax in f[key]: del f[key][ax]
                f[key].create_dataset('x_aligned', data=f[key]['x'][()])
                f[key].create_dataset('y_aligned', data=f[key]['y'][()])
                aligned_count += 1
                continue

            tgt   = np.column_stack([f[key]['x'][()], f[key]['y'][()]])
            tgt_c = tgt - tgt.mean(axis=0)
            Q, _  = orthogonal_procrustes(tgt_c, ref_c)
            aligned = tgt_c @ Q + ref_mu

            for ax in ('x_aligned', 'y_aligned'):
                if ax in f[key]: del f[key][ax]
            f[key].create_dataset('x_aligned', data=aligned[:, 0])
            f[key].create_dataset('y_aligned', data=aligned[:, 1])
            aligned_count += 1

    print(f'{dataset_name}: aligned {aligned_count} embeddings to ref={ref_key}')


# ── JSON → HDF5 migration ──────────────────────────────────────────────────────

def _convert_json_to_h5(output_dir: Path):
    json_files = sorted(output_dir.glob('*.json'))
    if not json_files:
        print("No JSON files found.")
        sys.exit(1)

    for json_path in json_files:
        h5_path = json_path.with_suffix('.h5')
        print(f"Reading  {json_path.name} ...")
        raw = json.loads(json_path.read_text())

        labels = label_names = None
        for key, val in raw.items():
            if key != '_meta':
                labels     = val.get('labels')
                label_names = val.get('label_names')
                break

        meta = raw.get('_meta', {})
        print(f"Writing  {h5_path.name} ...")
        with h5py.File(h5_path, 'w') as f:
            m = f.create_group('_meta')
            m.create_dataset('n_points', data=meta.get('n_points', len(labels) if labels else 0))
            if labels is not None:
                m.create_dataset('labels', data=np.array(labels))
            if label_names is not None:
                m.create_dataset('label_names', data=label_names, dtype=h5py.string_dtype())
            count = 0
            for key, val in raw.items():
                if key == '_meta':
                    continue
                grp = f.create_group(key)
                grp.create_dataset('x', data=np.array(val['x']), compression='gzip')
                grp.create_dataset('y', data=np.array(val['y']), compression='gzip')
                if val.get('z') is not None:
                    grp.create_dataset('z', data=np.array(val['z']), compression='gzip')
                count += 1
        size_mb = h5_path.stat().st_size / 1e6
        print(f"Done     {count} embeddings, {size_mb:.1f} MB\n")

    print("All done. Remove JSON files with:  rm data/embeddings/*.json")


# ── Add feature matrices to HDF5 ──────────────────────────────────────────────

def _add_features(dataset_names, output_dir: Path):
    for name in dataset_names:
        path = output_dir / f'{name}.h5'
        if not path.exists():
            print(f'{name}: no HDF5 file, skipping')
            continue

        data = DATASETS[name]['loader']()
        X_raw    = data['X'].astype(np.float32)
        X_scaled = StandardScaler().fit_transform(X_raw).astype(np.float32)
        feat_names = (data.get('feature_names')
                      or [f'feature_{i}' for i in range(X_raw.shape[1])])

        with h5py.File(path, 'a') as f:
            meta = f['_meta']
            for key in ('X_raw', 'X_scaled', 'feature_names'):
                if key in meta:
                    del meta[key]
            meta.create_dataset('X_raw',        data=X_raw,    compression='gzip')
            meta.create_dataset('X_scaled',     data=X_scaled, compression='gzip')
            meta.create_dataset('feature_names', data=feat_names,
                                dtype=h5py.string_dtype())

        print(f'{name}: added {X_raw.shape[0]} × {X_raw.shape[1]} feature matrix')

    print('\nDone. Upload updated .h5 files to the fly volume.')


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--action',
                        choices=['compute', 'tsne', 'align', 'add-features',
                                 'all', 'convert-json'],
                        default='compute',
                        help='"all" runs compute → tsne → add-features → align')
    parser.add_argument('--dataset', choices=list(DATASETS.keys()))
    parser.add_argument('--output-dir', default='data/embeddings')
    # Compute-only options
    parser.add_argument('--n-neighbors', type=int,   nargs='+', default=N_NEIGHBORS)
    parser.add_argument('--min-dist',    type=float, nargs='+', default=MIN_DIST)
    parser.add_argument('--n-components',type=int,   nargs='+', default=N_COMPONENTS)
    parser.add_argument('--metric',                  nargs='+', default=METRICS)
    parser.add_argument('--scale',                   nargs='+', default=SCALE,
                        choices=SCALE)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    targets = [args.dataset] if args.dataset else list(DATASETS.keys())

    if args.action in ('compute', 'all'):
        for name in targets:
            print(f"\n=== Precomputing UMAP + PCA {name} ===")
            precompute_dataset(
                name, output_dir,
                args.n_neighbors, args.min_dist, args.n_components,
                args.metric, args.scale,
            )

    if args.action in ('tsne', 'all'):
        for name in targets:
            print(f"\n=== Precomputing t-SNE {name} ===")
            precompute_tsne(name, output_dir, PERPLEXITY_STEPS, args.metric, args.scale)

    if args.action in ('add-features', 'all'):
        _add_features(targets, output_dir)

    if args.action in ('align', 'all'):
        for name in targets:
            print(f"\n=== Aligning {name} ===")
            _align_dataset(
                name, output_dir,
                args.n_neighbors, args.min_dist, args.n_components,
                args.metric, args.scale,
            )

    if args.action == 'convert-json':
        _convert_json_to_h5(output_dir)


if __name__ == '__main__':
    main()

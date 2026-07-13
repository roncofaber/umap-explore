"""Patch existing HDF5 files to add raw/scaled feature data without re-running UMAP."""
from pathlib import Path
import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler
from datasets import DATASETS

EMBEDDINGS_DIR = Path('data/embeddings')


def patch(dataset_name: str):
    path = EMBEDDINGS_DIR / f'{dataset_name}.h5'
    if not path.exists():
        print(f'{dataset_name}: no HDF5 file, skipping')
        return

    data = DATASETS[dataset_name]['loader']()
    X_raw    = data['X'].astype(np.float32)
    X_scaled = StandardScaler().fit_transform(X_raw).astype(np.float32)
    feat_names = data.get('feature_names') or [f'feature_{i}' for i in range(X_raw.shape[1])]

    with h5py.File(path, 'a') as f:
        meta = f['_meta']
        for key in ('X_raw', 'X_scaled', 'feature_names'):
            if key in meta:
                del meta[key]
        meta.create_dataset('X_raw',    data=X_raw,    compression='gzip')
        meta.create_dataset('X_scaled', data=X_scaled, compression='gzip')
        meta.create_dataset('feature_names', data=feat_names, dtype=h5py.string_dtype())

    print(f'{dataset_name}: added {X_raw.shape[0]} × {X_raw.shape[1]} feature matrix')


for name in DATASETS:
    patch(name)

print('\nDone. Upload updated .h5 files to the fly volume.')

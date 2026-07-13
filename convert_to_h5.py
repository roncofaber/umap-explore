"""Convert pre-computed JSON embeddings to HDF5 format."""
import sys
from pathlib import Path

import h5py
import json
import numpy as np


def convert(json_path: Path, h5_path: Path):
    print(f"Reading  {json_path.name} ...")
    data = json.loads(json_path.read_text())

    # Pull labels from the first embedding (same for all, stored once in HDF5)
    labels = label_names = None
    for key, val in data.items():
        if key == '_meta':
            continue
        labels = val.get('labels')
        label_names = val.get('label_names')
        break

    meta = data.get('_meta', {})

    print(f"Writing  {h5_path.name} ...")
    with h5py.File(h5_path, 'w') as f:
        m = f.create_group('_meta')
        m.create_dataset('n_points', data=meta.get('n_points', len(labels) if labels else 0))
        if labels is not None:
            m.create_dataset('labels', data=np.array(labels))
        if label_names is not None:
            m.create_dataset('label_names', data=label_names, dtype=h5py.string_dtype())

        count = 0
        for key, val in data.items():
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


def main():
    embeddings_dir = Path('data/embeddings')
    json_files = sorted(embeddings_dir.glob('*.json'))

    if not json_files:
        print("No JSON files found in data/embeddings/")
        sys.exit(1)

    for json_path in json_files:
        convert(json_path, json_path.with_suffix('.h5'))

    print("All done. Remove JSON files with:")
    print("  rm data/embeddings/*.json")


if __name__ == '__main__':
    main()

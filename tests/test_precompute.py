import sys
import subprocess
import h5py
import numpy as np
from datasets.meta import make_key


def test_make_key_basic():
    assert make_key(15, 0.1, 2, 'euclidean', 'scaled') == '15_0.1_2_euclidean_scaled'


def test_make_key_zero_dist():
    assert make_key(5, 0.0, 3, 'cosine', 'raw') == '5_0.0_3_cosine_raw'


def test_make_key_two_decimal():
    assert make_key(15, 0.05, 2, 'euclidean', 'scaled') == '15_0.05_2_euclidean_scaled'


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
            '--scale', 'scaled',
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr

    out_file = tmp_path / 'iris.h5'
    assert out_file.exists()

    with h5py.File(out_file, 'r') as f:
        assert '_meta' in f
        assert int(f['_meta/n_points'][()]) == 150
        label_names = [s.decode() if isinstance(s, bytes) else s
                       for s in f['_meta/label_names'][()].tolist()]
        assert label_names == ['setosa', 'versicolor', 'virginica']
        assert len(f['_meta/labels'][()]) == 150

        key = '5_0.1_2_euclidean_scaled'
        assert key in f
        assert 'x' in f[key] and 'y' in f[key]
        assert len(f[key]['x']) == len(f[key]['y']) == 150
        assert 'z' not in f[key]

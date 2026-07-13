import json
import sys
import subprocess
from precompute import make_key


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

    out_file = tmp_path / 'iris.json'
    assert out_file.exists()
    data = json.loads(out_file.read_text())

    assert '_meta' in data
    assert data['_meta']['n_points'] == 150
    assert data['_meta']['label_names'] == ['setosa', 'versicolor', 'virginica']

    key = '5_0.1_2_euclidean_scaled'
    assert key in data
    emb = data[key]
    assert set(emb.keys()) == {'x', 'y', 'z', 'labels', 'label_names'}
    assert emb['z'] is None
    assert len(emb['x']) == len(emb['y']) == 150
    assert len(emb['labels']) == 150

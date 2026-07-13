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

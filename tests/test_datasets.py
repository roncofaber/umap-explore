import pytest
import numpy as np
from datasets import DATASETS


def test_iris_shape():
    data = DATASETS['iris']['loader']()
    assert data['X'].shape == (150, 4)
    assert data['label_names'] == ['setosa', 'versicolor', 'virginica']
    assert data['n_points'] == 150


def test_penguins_shape():
    data = DATASETS['penguins']['loader']()
    assert data['X'].shape[1] == 4
    assert data['label_names'] == ['Adelie', 'Chinstrap', 'Gentoo']
    assert data['n_points'] == data['X'].shape[0]


def test_digits_shape():
    data = DATASETS['digits']['loader']()
    assert data['X'].shape == (1797, 64)
    assert len(data['label_names']) == 10
    assert data['n_points'] == 1797


def test_olivetti_faces_shape():
    data = DATASETS['olivetti_faces']['loader']()
    assert data['X'].shape == (400, 4096)
    assert len(data['label_names']) == 40
    assert data['n_points'] == 400


def test_swiss_roll_shape():
    data = DATASETS['swiss_roll']['loader']()
    assert data['X'].shape == (2000, 3)
    assert data['label_names'] is None
    assert data['n_points'] == 2000


def test_breast_cancer_shape():
    data = DATASETS['breast_cancer']['loader']()
    assert data['X'].shape == (569, 30)
    assert data['label_names'] == ['malignant', 'benign']
    assert data['n_points'] == 569


def test_pbmc3k_shape():
    pytest.importorskip('scanpy', reason='scanpy not installed')
    data = DATASETS['pbmc3k']['loader']()
    assert data['X'].shape == (2638, 50)
    assert len(data['label_names']) == 8
    assert data['n_points'] == 2638


def test_all_datasets_registered():
    expected = {'iris', 'penguins', 'digits', 'olivetti_faces',
                'swiss_roll', 'breast_cancer', 'pbmc3k'}
    assert set(DATASETS.keys()) == expected
    for name, ds in DATASETS.items():
        assert 'label' in ds
        assert callable(ds['loader'])

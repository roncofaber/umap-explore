import numpy as np
from datasets import DATASETS


def test_iris_shape():
    data = DATASETS['iris']['loader']()
    assert data['X'].shape == (150, 4)
    assert len(data['labels']) == 150
    assert data['label_names'] == ['setosa', 'versicolor', 'virginica']
    assert data['n_points'] == 150


def test_digits_shape():
    data = DATASETS['digits']['loader']()
    assert data['X'].shape == (1797, 64)
    assert len(data['labels']) == 1797
    assert len(data['label_names']) == 10
    assert data['n_points'] == 1797


def test_swiss_roll_shape():
    data = DATASETS['swiss_roll']['loader']()
    assert data['X'].shape == (2000, 3)
    assert len(data['labels']) == 2000
    assert data['label_names'] is None
    assert data['n_points'] == 2000


def test_breast_cancer_shape():
    data = DATASETS['breast_cancer']['loader']()
    assert data['X'].shape == (569, 30)
    assert len(data['labels']) == 569
    assert data['label_names'] == ['malignant', 'benign']
    assert data['n_points'] == 569


def test_all_datasets_registered():
    assert set(DATASETS.keys()) == {'iris', 'digits', 'swiss_roll', 'breast_cancer'}
    for name, ds in DATASETS.items():
        assert 'label' in ds
        assert callable(ds['loader'])

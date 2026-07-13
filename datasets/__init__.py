from typing import TypedDict
import numpy as np
from datasets.iris import load_iris
from datasets.mnist import load_mnist
from datasets.swiss_roll import load_swiss_roll
from datasets.breast_cancer import load_breast_cancer


class DatasetResult(TypedDict):
    X: np.ndarray
    labels: list
    label_names: list | None
    n_points: int


DATASETS: dict[str, dict] = {
    'iris': {
        'label': 'Iris',
        'loader': load_iris,
        'n_features': 4,
        'description': 'Sepal and petal measurements for three iris species. A classic benchmark for classification and clustering.',
    },
    'mnist': {
        'label': 'MNIST Digits',
        'loader': load_mnist,
        'n_features': 784,
        'description': '28×28 pixel grayscale images of handwritten digits (0–9), subsampled to 3 000 points.',
    },
    'swiss_roll': {
        'label': 'Swiss Roll',
        'loader': load_swiss_roll,
        'n_features': 3,
        'description': 'A 2D manifold rolled into 3D space. A standard test for non-linear dimensionality reduction methods.',
    },
    'breast_cancer': {
        'label': 'Breast Cancer',
        'loader': load_breast_cancer,
        'n_features': 30,
        'description': 'Clinical measurements from 569 breast biopsies labeled malignant or benign. From the UCI ML repository.',
    },
}

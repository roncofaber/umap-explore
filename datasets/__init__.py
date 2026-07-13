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
    },
    'mnist': {
        'label': 'MNIST Digits',
        'loader': load_mnist,
    },
    'swiss_roll': {
        'label': 'Swiss Roll',
        'loader': load_swiss_roll,
    },
    'breast_cancer': {
        'label': 'Breast Cancer',
        'loader': load_breast_cancer,
    },
}

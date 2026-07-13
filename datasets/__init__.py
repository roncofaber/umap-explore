from typing import TypedDict
import numpy as np

from datasets.meta import DATASETS_META
from datasets.iris import load_iris
from datasets.digits import load_digits
from datasets.swiss_roll import load_swiss_roll
from datasets.breast_cancer import load_breast_cancer


class DatasetResult(TypedDict):
    X: np.ndarray
    labels: list
    label_names: list | None
    n_points: int


_LOADERS = {
    'iris': load_iris,
    'digits': load_digits,
    'swiss_roll': load_swiss_roll,
    'breast_cancer': load_breast_cancer,
}

DATASETS = {
    name: {**meta, 'loader': _LOADERS[name]}
    for name, meta in DATASETS_META.items()
}

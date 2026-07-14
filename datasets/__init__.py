from typing import TypedDict
import numpy as np

from datasets.meta import DATASETS_META
from datasets.iris import load_iris
from datasets.penguins import load_penguins
from datasets.digits import load_digits
from datasets.olivetti_faces import load_olivetti_faces
from datasets.swiss_roll import load_swiss_roll
from datasets.breast_cancer import load_breast_cancer
from datasets.pbmc3k import load_pbmc3k


class DatasetResult(TypedDict):
    X: np.ndarray
    labels: list
    label_names: list | None
    n_points: int


_LOADERS = {
    'iris':           load_iris,
    'penguins':       load_penguins,
    'digits':         load_digits,
    'olivetti_faces': load_olivetti_faces,
    'swiss_roll':     load_swiss_roll,
    'breast_cancer':  load_breast_cancer,
    'pbmc3k':         load_pbmc3k,
}

DATASETS = {
    name: {**meta, 'loader': _LOADERS[name]}
    for name, meta in DATASETS_META.items()
}

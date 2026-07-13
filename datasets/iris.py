import numpy as np
from sklearn.datasets import load_iris as _load_iris


def load_iris():
    raw = _load_iris()
    return {
        'X': raw.data,
        'labels': raw.target.tolist(),
        'label_names': list(raw.target_names),
        'n_points': len(raw.data),
    }

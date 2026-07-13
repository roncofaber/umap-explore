import numpy as np
from sklearn.datasets import fetch_openml


def load_mnist():
    raw = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    rng = np.random.RandomState(42)
    idx = rng.choice(len(raw.data), size=3000, replace=False)
    X = raw.data[idx].astype(np.float32)
    labels = raw.target[idx].astype(int).tolist()
    return {
        'X': X,
        'labels': labels,
        'label_names': [str(i) for i in range(10)],
        'n_points': 3000,
    }

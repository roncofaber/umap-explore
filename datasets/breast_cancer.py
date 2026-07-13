from sklearn.datasets import load_breast_cancer as _load


def load_breast_cancer():
    raw = _load()
    return {
        'X': raw.data,
        'labels': raw.target.tolist(),
        'label_names': list(raw.target_names),
        'feature_names': list(raw.feature_names),
        'n_points': len(raw.data),
    }

from sklearn.datasets import load_digits as _load


def load_digits():
    raw = _load()
    return {
        'X': raw.data.astype(float),
        'labels': raw.target.tolist(),
        'label_names': [str(i) for i in range(10)],
        'feature_names': [f'pixel_{i//8}_{i%8}' for i in range(64)],
        'n_points': len(raw.data),
    }

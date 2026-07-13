from sklearn.datasets import load_digits as _load


def load_digits():
    raw = _load()
    return {
        'X': raw.data.astype(float),
        'labels': raw.target.tolist(),
        'label_names': [str(i) for i in range(10)],
        'n_points': len(raw.data),
    }

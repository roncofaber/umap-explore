from sklearn.datasets import make_swiss_roll


def load_swiss_roll():
    X, t = make_swiss_roll(n_samples=2000, random_state=42)
    return {
        'X': X,
        'labels': t.tolist(),
        'label_names': None,
        'feature_names': ['x', 'y', 'z'],
        'n_points': 2000,
    }

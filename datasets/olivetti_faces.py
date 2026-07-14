from sklearn.datasets import fetch_olivetti_faces


def load_olivetti_faces():
    raw = fetch_olivetti_faces(shuffle=True, random_state=42)
    n, px = raw.data.shape          # 400, 4096
    side = int(px ** 0.5)           # 64

    return {
        'X': raw.data.astype(float),
        'labels': raw.target.tolist(),
        'label_names': [f'person {i:02d}' for i in range(40)],
        'feature_names': [f'pixel_{i//side}_{i%side}' for i in range(px)],
        'n_points': n,
    }

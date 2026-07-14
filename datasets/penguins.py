import csv
from pathlib import Path
import numpy as np

_CSV = Path(__file__).parent.parent / 'data' / 'raw' / 'penguins.csv'

_FEATURES = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']


def load_penguins():
    with open(_CSV) as f:
        rows = [r for r in csv.DictReader(f) if all(r[c] for c in _FEATURES)]

    X = np.array([[float(r[c]) for c in _FEATURES] for r in rows])
    species = [r['species'] for r in rows]
    unique = sorted(set(species))

    return {
        'X': X,
        'labels': [unique.index(s) for s in species],
        'label_names': unique,
        'feature_names': _FEATURES,
        'n_points': len(X),
    }

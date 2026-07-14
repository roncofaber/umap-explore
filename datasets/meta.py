DATASETS_META = {
    'iris': {
        'label': 'Iris',
        'n_features': 4,
        'description': 'Sepal and petal measurements for three iris species. A classic benchmark for classification and clustering.',
    },
    'digits': {
        'label': 'Digits',
        'n_features': 64,
        'description': '8×8 pixel grayscale images of handwritten digits (0–9). 1 797 samples built into scikit-learn.',
    },
    'swiss_roll': {
        'label': 'Swiss Roll',
        'n_features': 3,
        'description': 'A 2D manifold rolled into 3D space. A standard test for non-linear dimensionality reduction methods.',
    },
    'breast_cancer': {
        'label': 'Breast Cancer',
        'n_features': 30,
        'description': 'Clinical measurements from 569 breast biopsies labeled malignant or benign. From the UCI ML repository.',
    },
}


def make_key(n_neighbors, min_dist, n_components, metric, scale):
    """Return the HDF5 group key for a given parameter combination."""
    return f"{n_neighbors}_{min_dist}_{n_components}_{metric}_{scale}"

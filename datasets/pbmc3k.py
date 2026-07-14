"""PBMC3k single-cell RNA-seq dataset (Zheng et al. 2017).

Input: first 50 principal components of the log-normalised count matrix
       (standard Seurat/scanpy preprocessing pipeline).
Labels: 8 immune cell types from Louvain clustering.

Requires scanpy:  pip install scanpy
"""


def load_pbmc3k():
    try:
        import scanpy as sc
    except ImportError as e:
        raise ImportError(
            "scanpy is required for the PBMC3k dataset: pip install scanpy"
        ) from e

    adata = sc.datasets.pbmc3k_processed()
    X = adata.obsm['X_pca'].astype(float)          # 2638 × 50
    cell_types = adata.obs['louvain'].astype(str).tolist()
    unique = sorted(set(cell_types))

    return {
        'X': X,
        'labels': [unique.index(ct) for ct in cell_types],
        'label_names': unique,
        'feature_names': [f'PC{i+1}' for i in range(X.shape[1])],
        'n_points': len(X),
    }

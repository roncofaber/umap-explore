import pytest
import h5py
import numpy as np
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def patch_embeddings_dir(tmp_path, monkeypatch):
    emb_dir = tmp_path / "embeddings"
    emb_dir.mkdir()

    with h5py.File(emb_dir / "iris.h5", 'w') as f:
        meta = f.create_group('_meta')
        meta.create_dataset('n_points', data=3)
        meta.create_dataset('labels', data=np.array([0, 1, 2]))
        meta.create_dataset('label_names',
                            data=['setosa', 'versicolor', 'virginica'],
                            dtype=h5py.string_dtype())

        grp = f.create_group('15_0.1_2_euclidean_scaled')
        grp.create_dataset('x', data=np.array([1.0, 2.0, 3.0]))
        grp.create_dataset('y', data=np.array([0.5, 1.5, 2.5]))

        grp = f.create_group('pca_2_scaled')
        grp.create_dataset('x', data=np.array([0.1, 0.2, 0.3]))
        grp.create_dataset('y', data=np.array([0.4, 0.5, 0.6]))

    import app
    monkeypatch.setattr(app, 'EMBEDDINGS_DIR', emb_dir)


@pytest.fixture
def client():
    from app import app as fastapi_app
    return TestClient(fastapi_app)


def test_list_datasets(client):
    resp = client.get("/api/datasets")
    assert resp.status_code == 200
    data = resp.json()
    iris = next(d for d in data if d["name"] == "iris")
    assert iris["n_points"] == 3
    assert iris["has_labels"] is True


def test_get_embedding(client):
    resp = client.get(
        "/api/embeddings/iris?n_neighbors=15&min_dist=0.1&n_components=2&metric=euclidean&scale=scaled"
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["x"] == pytest.approx([1.0, 2.0, 3.0])
    assert data["z"] is None
    assert data["labels"] == [0, 1, 2]
    assert data["label_names"] == ["setosa", "versicolor", "virginica"]


def test_get_pca_embedding(client):
    resp = client.get("/api/embeddings/iris?method=pca&scale=scaled")
    assert resp.status_code == 200
    data = resp.json()
    assert data["x"] == pytest.approx([0.1, 0.2, 0.3])


def test_embedding_key_not_found(client):
    resp = client.get(
        "/api/embeddings/iris?n_neighbors=99&min_dist=0.1&n_components=2&metric=euclidean&scale=scaled"
    )
    assert resp.status_code == 404


def test_dataset_not_found(client):
    resp = client.get(
        "/api/embeddings/nonexistent?n_neighbors=15&min_dist=0.1&n_components=2&metric=euclidean&scale=scaled"
    )
    assert resp.status_code == 404

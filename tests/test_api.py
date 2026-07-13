import json
import pytest
from fastapi.testclient import TestClient

FIXTURE_EMBEDDING = {
    "_meta": {
        "n_points": 3,
        "label_names": ["setosa", "versicolor", "virginica"],
    },
    "15_0.1_2_euclidean": {
        "x": [1.0, 2.0, 3.0],
        "y": [0.5, 1.5, 2.5],
        "z": None,
        "labels": [0, 1, 2],
        "label_names": ["setosa", "versicolor", "virginica"],
    },
}


@pytest.fixture(autouse=True)
def patch_embeddings_dir(tmp_path, monkeypatch):
    emb_dir = tmp_path / "embeddings"
    emb_dir.mkdir()
    (emb_dir / "iris.json").write_text(json.dumps(FIXTURE_EMBEDDING))
    import app
    monkeypatch.setattr(app, "EMBEDDINGS_DIR", emb_dir)


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
        "/api/embeddings/iris?n_neighbors=15&min_dist=0.1&n_components=2&metric=euclidean"
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["x"] == [1.0, 2.0, 3.0]
    assert data["z"] is None
    assert data["labels"] == [0, 1, 2]
    assert data["label_names"] == ["setosa", "versicolor", "virginica"]


def test_embedding_key_not_found(client):
    resp = client.get(
        "/api/embeddings/iris?n_neighbors=99&min_dist=0.1&n_components=2&metric=euclidean"
    )
    assert resp.status_code == 404


def test_dataset_not_found(client):
    resp = client.get(
        "/api/embeddings/nonexistent?n_neighbors=15&min_dist=0.1&n_components=2&metric=euclidean"
    )
    assert resp.status_code == 404

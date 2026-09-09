from fastapi.testclient import TestClient

from pathoverse.api.main import app


client = TestClient(app)


def test_list_models():
    response = client.get(
        "/api/models"
    )

    assert response.status_code == 200

    data = response.json()

    assert len(data) == 3

    model_ids = {
        item["model_id"]
        for item in data
    }

    assert model_ids == {
        "vit-b-16",
        "gigapath-flash",
        "conch",
    }


def test_gigapath_model():
    response = client.get(
        "/api/models/gigapath-flash"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["key"] == "gigapath-flash"
    assert data["name"] == "GigaPath-Flash"
    assert data["embedding_dimension"] == 384
    assert data["loaded"] is False


def test_model_alias():
    response = client.get(
        "/api/models/vit-base"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["key"] == "vit-b-16"
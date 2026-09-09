from fastapi.testclient import TestClient

from pathoverse.api.main import app


client = TestClient(app)


def test_gigapath_classification():
    response = client.post(
        "/api/analysis/classification",
        json={
            "model": "gigapath-flash",
            "dataset": "patchcamelyon",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["model"] == "GigaPath-Flash"
    assert data["accuracy"] == 0.92

    assert (
        abs(
            data["auroc"]
            - 0.9748597756
        )
        < 1e-6
    )


def test_classification_alias():
    response = client.post(
        "/api/analysis/classification",
        json={
            "model": "vit-base",
            "dataset": "patchcamelyon",
        },
    )

    assert response.status_code == 200


def test_mil():
    response = client.post(
        "/api/analysis/mil",
        json={
            "slide_id": "CMU-1-Small-Region",
            "model": "gigapath-flash",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert (
        data["slide_id"]
        == "CMU-1-Small-Region"
    )

    assert data["model"] == "gigapath-flash"

    assert (
        data["foundation_model"]
        ["embedding_dimension"]
        == 384
    )

    assert data["prediction"]["class_id"] == 1

    assert (
        abs(
            data["prediction"]
            ["probability"]
            - 0.5028
        )
        < 1e-3
    )

    assert (
        data["attention"]["tile_count"]
        == 45
    )

    assert (
        abs(
            data["attention"]["sum"]
            - 1.0
        )
        < 1e-6
    )

    assert data["trained"] is False
    assert data["prototype"] is True


def test_heatmap():
    response = client.get(
        "/api/analysis/"
        "CMU-1-Small-Region/"
        "heatmap",
        params={
            "model": "gigapath-flash"
        },
    )

    assert response.status_code == 200

    assert response.headers[
        "content-type"
    ].startswith("image/png")

    assert len(response.content) > 1000
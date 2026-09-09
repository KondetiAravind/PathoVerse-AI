from fastapi.testclient import TestClient

from pathoverse.api.main import app


client = TestClient(app)


def test_retrieval():
    response = client.post(
        "/api/retrieval/search",
        json={
            "slide_id": "CMU-1-Small-Region",
            "tile_id": 0,
            "model": "gigapath-flash",
            "top_k": 5,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["query_tile_id"] == 0
    assert data["model"] == "gigapath-flash"
    assert len(data["results"]) == 5

    result_ids = {
        item["tile_id"]
        for item in data["results"]
    }

    assert 0 not in result_ids

    scores = [
        item["score"]
        for item in data["results"]
    ]

    assert scores == sorted(
        scores,
        reverse=True,
    )
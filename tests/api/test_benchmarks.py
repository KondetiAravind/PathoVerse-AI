from fastapi.testclient import TestClient

from pathoverse.api.main import app


client = TestClient(app)


def test_benchmarks():
    response = client.get(
        "/api/benchmarks"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["total_records"] == 10
    assert len(data["records"]) == 10


def test_benchmark_task():
    response = client.get(
        "/api/benchmarks/classification"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["total_records"] == 3

    model_ids = {
        record["model_id"]
        for record in data["records"]
    }

    assert model_ids == {
        "vit-b-16",
        "gigapath-flash",
        "conch",
    }
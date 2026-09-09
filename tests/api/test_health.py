from fastapi.testclient import TestClient

from pathoverse.api.main import app


client = TestClient(app)


def test_root():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["name"] == "PathoVerse AI"
    assert data["status"] == "running"


def test_health():
    response = client.get("/api/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["service"] == "pathoverse-api"


def test_readiness():
    response = client.get(
        "/api/health/ready"
    )

    assert response.status_code == 200
    assert response.json()["status"] == "ready"
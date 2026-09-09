from fastapi.testclient import TestClient

from pathoverse.api.main import app


client = TestClient(app)


SLIDE_ID = "CMU-1-Small-Region"


def test_list_slides():
    response = client.get(
        "/api/slides"
    )

    assert response.status_code == 200

    data = response.json()

    assert len(data) >= 1

    slide = next(
        item
        for item in data
        if item["slide_id"] == SLIDE_ID
    )

    assert slide["width"] == 2220
    assert slide["height"] == 2967


def test_slide_detail():
    response = client.get(
        f"/api/slides/{SLIDE_ID}"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["slide_id"] == SLIDE_ID
    assert data["width"] == 2220
    assert data["height"] == 2967
    assert data["mpp_x"] == 0.499
    assert data["mpp_y"] == 0.499
    assert data["tile_count"] == 45

    assert (
        data["tissue_ratio"] is not None
    )

    assert (
        0.30
        < data["tissue_ratio"]
        < 0.40
    )


def test_slide_tiles():
    response = client.get(
        f"/api/slides/{SLIDE_ID}/tiles"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["slide_id"] == SLIDE_ID
    assert data["total_tiles"] == 45
    assert len(data["tiles"]) == 45
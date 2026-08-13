import os

os.environ["ORBBEC_AUTOSTART"] = "0"

from camera.capture import CameraManager
from server import create_app


def make_client():
    manager = CameraManager()
    app = create_app(camera_manager=manager, autostart=False)
    app.config.update(TESTING=True)
    return app.test_client()


def test_index_and_health():
    client = make_client()
    assert client.get("/").status_code == 200
    assert client.get("/healthz").get_json() == {"ok": True}


def test_status_shape():
    payload = make_client().get("/api/status").get_json()
    assert payload["state"] == "disconnected"
    assert set(payload["streams"]) == {"rgb", "depth"}
    assert payload["device"]["pid"] == "0x0614"


def test_settings_update_and_validation():
    client = make_client()
    response = client.put(
        "/api/settings",
        json={"depth_max_mm": 7000, "jpeg_quality": 88},
    )
    assert response.status_code == 200
    assert response.get_json()["settings"]["depth_max_mm"] == 7000

    invalid = client.put("/api/settings", json={"depth_max_mm": 500})
    assert invalid.status_code == 400
    assert invalid.get_json()["ok"] is False


def test_unknown_stream_is_404():
    assert make_client().get("/stream/infrared.mjpg").status_code == 404

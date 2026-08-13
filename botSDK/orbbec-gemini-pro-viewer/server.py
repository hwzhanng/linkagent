from __future__ import annotations

import atexit
import os
from collections.abc import Iterator

from flask import Flask, Response, jsonify, render_template, request

from camera import CameraManager


def create_app(
    *,
    camera_manager: CameraManager | None = None,
    autostart: bool | None = None,
) -> Flask:
    app = Flask(__name__)
    manager = camera_manager or CameraManager()
    should_start = (
        os.environ.get("ORBBEC_AUTOSTART", "1") != "0"
        if autostart is None
        else autostart
    )
    if should_start:
        manager.start()
        atexit.register(manager.stop)

    app.extensions["camera_manager"] = manager

    @app.after_request
    def no_cache(response: Response) -> Response:
        if request.path.startswith(("/api/", "/stream/")):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
            response.headers["Pragma"] = "no-cache"
        return response

    @app.get("/")
    def index() -> str:
        return render_template("index.html")

    @app.get("/healthz")
    def healthz() -> Response:
        return jsonify({"ok": True})

    @app.get("/api/status")
    def status() -> Response:
        return jsonify(manager.status())

    @app.post("/api/retry")
    def retry() -> Response:
        manager.retry()
        return jsonify({"ok": True, "status": manager.status()})

    @app.route("/api/settings", methods=["GET", "PUT"])
    def settings() -> Response:
        if request.method == "GET":
            return jsonify(manager.settings())
        payload = request.get_json(silent=True) or {}
        try:
            updated = manager.update_settings(
                depth_max_mm=payload.get("depth_max_mm"),
                jpeg_quality=payload.get("jpeg_quality"),
            )
        except (TypeError, ValueError) as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify({"ok": True, "settings": updated})

    def mjpeg_stream(stream_name: str) -> Iterator[bytes]:
        sequence = -1
        while True:
            sequence, jpeg = manager.wait_for_frame(stream_name, sequence)
            if jpeg is None:
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: "
                + str(len(jpeg)).encode("ascii")
                + b"\r\n\r\n"
                + jpeg
                + b"\r\n"
            )

    @app.get("/stream/<stream_name>.mjpg")
    def stream(stream_name: str) -> Response:
        if stream_name not in {"rgb", "depth"}:
            return jsonify({"error": "unknown stream"}), 404
        return Response(
            mjpeg_stream(stream_name),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8080")), threaded=True)

"""Isolated loopback service for tomato instance/VLM/RGB-D processing.

The service never starts on import.  Heavy inference backends are selected by
configuration and imported only in the worker process.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
from pathlib import Path
import threading
from typing import Any, Protocol

from script.utils import load_mapping_file
from script.tomato_vision_spool import TomatoVisionSpoolReader


logger = logging.getLogger(__name__)
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


def _mapping(value: object, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    return value


def _non_empty(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _bounded_int(
    value: object,
    *,
    field_name: str,
    lower: int,
    upper: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
        raise ValueError(f"{field_name} must be an integer in [{lower}, {upper}]")
    return value


class VisionWorkerBackend(Protocol):
    backend_id: str

    def process(
        self,
        *,
        bundle_path: Path,
        observation_sequence: int,
        requested_track_id: str | None,
    ) -> Mapping[str, object]: ...

    def health(self) -> Mapping[str, object]: ...

    def close(self) -> None: ...


class ReplayVisionWorkerBackend:
    """Return a frozen worker result from a capture bundle sidecar."""

    backend_id = "replay_v1"

    def process(
        self,
        *,
        bundle_path: Path,
        observation_sequence: int,
        requested_track_id: str | None,
    ) -> Mapping[str, object]:
        del observation_sequence
        bundle = load_mapping_file(bundle_path)
        if bundle.get("schema_version") != 1:
            raise ValueError("capture bundle schema_version must be 1")
        vision = _mapping(bundle.get("tomato_vision"), field_name="capture.tomato_vision")
        if requested_track_id is not None:
            selection = _mapping(
                vision.get("vlm_selection"), field_name="capture.vlm_selection"
            )
            selected_id = selection.get("selected_instance_id")
            rows = vision.get("instances")
            if isinstance(rows, (str, bytes, bytearray, Mapping)):
                raise TypeError("capture.instances must be a sequence")
            try:
                candidates = tuple(rows)  # type: ignore[arg-type]
            except TypeError as exc:
                raise TypeError("capture.instances must be a sequence") from exc
            selected_track = None
            for row in candidates:
                candidate = _mapping(row, field_name="capture instance")
                if candidate.get("instance_id") == selected_id:
                    selected_track = candidate.get("track_id")
                    break
            if selected_track != requested_track_id:
                raise ValueError("capture bundle selection does not match requested_track_id")
        return dict(vision)

    def health(self) -> Mapping[str, object]:
        return {"ready": True, "backend_id": self.backend_id, "simulated": True}

    def close(self) -> None:
        return None


class VisionWorkerServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        *,
        worker_id: str,
        capture_root: Path,
        backend: VisionWorkerBackend,
        max_request_bytes: int,
        max_bundle_bytes: int,
    ) -> None:
        super().__init__(address, VisionWorkerHandler)
        self.worker_id = worker_id
        self.capture_root = capture_root
        self.backend = backend
        self.max_request_bytes = max_request_bytes
        self.max_bundle_bytes = max_bundle_bytes
        self._close_lock = threading.Lock()
        self._resources_closed = False

    def close_resources(self) -> None:
        with self._close_lock:
            if self._resources_closed:
                return
            self._resources_closed = True
        self.backend.close()
        self.server_close()


class VisionWorkerHandler(BaseHTTPRequestHandler):
    server: VisionWorkerServer

    def log_message(self, format: str, *args: object) -> None:
        logger.debug("vision worker request " + format, *args)

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/health":
            self._json_response(404, {"error": "not_found"})
            return
        health = dict(self.server.backend.health())
        self._json_response(
            200,
            {
                "schema_version": 1,
                "worker_id": self.server.worker_id,
                **health,
            },
        )

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/process":
            self._json_response(404, {"error": "not_found"})
            return
        try:
            payload = self._read_request_json()
            result = self._process(payload)
        except (TypeError, ValueError, FileNotFoundError) as exc:
            self._json_response(
                422,
                {"error": "invalid_request", "message": str(exc)[:300]},
            )
            return
        except TimeoutError as exc:
            self._json_response(
                409,
                {"error": "new_capture_timeout", "message": str(exc)[:300]},
            )
            return
        except RuntimeError as exc:
            self._json_response(
                503,
                {"error": "backend_unavailable", "message": str(exc)[:300]},
            )
            return
        except Exception:
            logger.exception("vision worker request failed")
            self._json_response(500, {"error": "internal_error"})
            return
        self._json_response(200, result)

    def _read_request_json(self) -> Mapping[str, Any]:
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip()
        if content_type != "application/json":
            raise ValueError("Content-Type must be application/json")
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise ValueError("Content-Length is required")
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise ValueError("Content-Length must be an integer") from exc
        if not 1 <= length <= self.server.max_request_bytes:
            raise ValueError("request body exceeds the configured limit")
        raw = self.rfile.read(length)
        try:
            return _mapping(json.loads(raw.decode("utf-8")), field_name="request")
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("request body must be valid UTF-8 JSON") from exc

    def _process(self, payload: Mapping[str, Any]) -> dict[str, object]:
        allowed = {
            "schema_version",
            "request_id",
            "observation_sequence",
            "capture_bundle",
            "capture_reference_kind",
            "requested_track_id",
            "minimum_sample_monotonic_ns",
            "new_frame_wait_timeout_s",
        }
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError("unknown request fields: " + ", ".join(sorted(unknown)))
        if payload.get("schema_version") != 1:
            raise ValueError("request schema_version must be 1")
        request_id = _non_empty(payload.get("request_id"), field_name="request_id")
        observation_sequence = _bounded_int(
            payload.get("observation_sequence"),
            field_name="observation_sequence",
            lower=0,
            upper=2_147_483_647,
        )
        requested_track_id = payload.get("requested_track_id")
        if requested_track_id is not None:
            requested_track_id = _non_empty(
                requested_track_id, field_name="requested_track_id"
            )
        reference_kind = payload.get("capture_reference_kind")
        if reference_kind not in {"bundle", "latest"}:
            raise ValueError("capture_reference_kind must be bundle or latest")
        minimum_sample_ns = payload.get("minimum_sample_monotonic_ns")
        if minimum_sample_ns is not None:
            minimum_sample_ns = _bounded_int(
                minimum_sample_ns,
                field_name="minimum_sample_monotonic_ns",
                lower=0,
                upper=9_223_372_036_854_775_807,
            )
        wait_timeout = payload.get("new_frame_wait_timeout_s")
        if (
            isinstance(wait_timeout, bool)
            or not isinstance(wait_timeout, (int, float))
            or not 0.0 <= wait_timeout <= 5.0
        ):
            raise ValueError("new_frame_wait_timeout_s must be in [0, 5]")
        if reference_kind == "bundle" and (
            minimum_sample_ns is not None or float(wait_timeout) != 0.0
        ):
            raise ValueError(
                "minimum sample and new-frame wait are only valid for latest references"
            )
        if minimum_sample_ns is None and float(wait_timeout) != 0.0:
            raise ValueError(
                "new_frame_wait_timeout_s requires minimum_sample_monotonic_ns"
            )
        bundle_ref = _non_empty(payload.get("capture_bundle"), field_name="capture_bundle")
        relative = Path(bundle_ref)
        if relative.is_absolute():
            raise ValueError("capture_bundle must be relative")
        bundle_path = (self.server.capture_root / relative).resolve()
        if not bundle_path.is_relative_to(self.server.capture_root):
            raise ValueError("capture_bundle escapes capture_root")
        if not bundle_path.is_file():
            raise FileNotFoundError(bundle_path)
        if bundle_path.stat().st_size > self.server.max_bundle_bytes:
            raise ValueError("capture bundle sidecar exceeds the configured limit")
        if reference_kind == "latest":
            resolved = TomatoVisionSpoolReader(
                self.server.capture_root,
                latest_reference=relative,
            ).resolve_latest(
                minimum_sample_monotonic_ns=minimum_sample_ns,
                timeout_s=float(wait_timeout),
            )
            bundle_path = resolved.bundle_path
            if bundle_path.stat().st_size > self.server.max_bundle_bytes:
                raise ValueError("resolved capture sidecar exceeds the configured limit")
        vision = self.server.backend.process(
            bundle_path=bundle_path,
            observation_sequence=observation_sequence,
            requested_track_id=requested_track_id,
        )
        return {
            "schema_version": 1,
            "request_id": request_id,
            "worker_id": self.server.worker_id,
            "tomato_vision": dict(vision),
        }

    def _json_response(self, status: int, payload: Mapping[str, object]) -> None:
        raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)


def build_worker_from_config(
    config: Mapping[str, Any],
    *,
    config_dir: str | Path,
    bind_port_override: int | None = None,
) -> VisionWorkerServer:
    allowed = {
        "schema_version",
        "worker_id",
        "host",
        "port",
        "capture_root",
        "max_request_bytes",
        "max_bundle_bytes",
        "backend",
    }
    unknown = set(config) - allowed
    if unknown:
        raise ValueError("unknown worker config fields: " + ", ".join(sorted(unknown)))
    if config.get("schema_version") != 1:
        raise ValueError("worker config schema_version must be 1")
    worker_id = _non_empty(config.get("worker_id"), field_name="worker_id")
    host = _non_empty(config.get("host", "127.0.0.1"), field_name="host")
    if host not in _LOOPBACK_HOSTS:
        raise ValueError("vision worker must bind to a loopback host")
    port = _bounded_int(
        config.get("port", 8765), field_name="port", lower=0, upper=65535
    )
    if bind_port_override is not None:
        port = _bounded_int(
            bind_port_override,
            field_name="bind_port_override",
            lower=0,
            upper=65535,
        )
    max_request_bytes = _bounded_int(
        config.get("max_request_bytes", 64_000),
        field_name="max_request_bytes",
        lower=1024,
        upper=5_000_000,
    )
    max_bundle_bytes = _bounded_int(
        config.get("max_bundle_bytes", 2_000_000),
        field_name="max_bundle_bytes",
        lower=1024,
        upper=20_000_000,
    )
    root = Path(config_dir).expanduser().resolve()
    capture_ref = _non_empty(config.get("capture_root"), field_name="capture_root")
    capture_root = (root / capture_ref).resolve()
    if not capture_root.is_dir():
        raise FileNotFoundError(capture_root)
    backend_config = _mapping(config.get("backend"), field_name="backend")
    backend_type = backend_config.get("type")
    if backend_type == "replay":
        if set(backend_config) != {"type"}:
            raise ValueError("replay backend accepts only the type field")
        backend: VisionWorkerBackend = ReplayVisionWorkerBackend()
    elif backend_type == "ultralytics_vlm":
        from script.tomato_vision_worker_inference import UltralyticsVlmBackend

        options = dict(backend_config)
        options.pop("type", None)
        backend = UltralyticsVlmBackend(
            capture_root=capture_root,
            config_dir=root,
            **options,
        )
    elif backend_type == "ark_vlm_box":
        from script.tomato_vision_worker_ark import ArkVlmBoxDepthBackend

        options = dict(backend_config)
        options.pop("type", None)
        backend = ArkVlmBoxDepthBackend(
            capture_root=capture_root,
            config_dir=root,
            **options,
        )
    else:
        raise ValueError(
            "backend.type must be replay, ultralytics_vlm, or ark_vlm_box"
        )
    try:
        return VisionWorkerServer(
            (host, port),
            worker_id=worker_id,
            capture_root=capture_root,
            backend=backend,
            max_request_bytes=max_request_bytes,
            max_bundle_bytes=max_bundle_bytes,
        )
    except BaseException:
        backend.close()
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve the isolated tomato vision worker")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate configuration/backend construction without serving",
    )
    args = parser.parse_args(argv)
    config_path = args.config.expanduser().resolve()
    server: VisionWorkerServer | None = None
    try:
        server = build_worker_from_config(
            load_mapping_file(config_path),
            config_dir=config_path.parent,
            bind_port_override=0 if args.check else None,
        )
        if args.check:
            logger.info(
                "vision worker config valid worker_id=%s backend=%s",
                server.worker_id,
                server.backend.backend_id,
            )
            return 0
        logger.info(
            "vision worker listening host=%s port=%s backend=%s",
            server.server_address[0],
            server.server_address[1],
            server.backend.backend_id,
        )
        server.serve_forever(poll_interval=0.2)
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception:
        logger.exception("vision worker failed")
        return 1
    finally:
        if server is not None:
            server.close_resources()


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ReplayVisionWorkerBackend",
    "VisionWorkerBackend",
    "VisionWorkerServer",
    "build_worker_from_config",
    "main",
]

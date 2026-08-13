"""Offline mock and read-only Orbbec viewer camera-health adapters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping
from urllib.request import Request, urlopen

from module.robotics import Clock
from proto import CameraHealthSnapshot


def _sdk_layout(root: Path) -> tuple[bool, tuple[str, ...]]:
    required = (
        root / "camera" / "capture.py",
        root / "vendor" / "pyorbbecsdk" / "stubs" / "pyorbbecsdk.pyi",
        root / "vendor" / "pyorbbecsdk" / "install" / "lib" / "libOrbbecSDK.so",
    )
    extension_dir = root / "vendor" / "pyorbbecsdk" / "install" / "lib"
    extension_present = any(extension_dir.glob("pyorbbecsdk.cpython-*.so"))
    available = all(path.is_file() for path in required) and extension_present
    evidence = tuple(f"file:{path}" for path in required if path.is_file())
    return available, evidence


class MockCameraHealthPort:
    """Deterministic RGB-D health source; it never imports or opens the SDK."""

    def __init__(
        self,
        *,
        clock: Clock,
        sdk_root: str | Path | None = None,
        sdk_available: bool = True,
        device_connected: bool = True,
        rgb_live: bool = True,
        depth_live: bool = True,
        calibration_id: str = "orbbec_gemini_pro_mock_calibration_v1",
        device_id: str = "mock-orbbec-gemini-pro",
        stale_by_s: float = 0.0,
    ) -> None:
        for name, value in {
            "sdk_available": sdk_available,
            "device_connected": device_connected,
            "rgb_live": rgb_live,
            "depth_live": depth_live,
        }.items():
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a bool")
        if (
            isinstance(stale_by_s, bool)
            or not isinstance(stale_by_s, (int, float))
            or stale_by_s < 0
        ):
            raise ValueError("stale_by_s must be >= 0")
        self._clock = clock
        self._sdk_root = None if sdk_root is None else Path(sdk_root).expanduser().resolve()
        self._configured_sdk_available = sdk_available
        self._device_connected = device_connected
        self._rgb_live = rgb_live
        self._depth_live = depth_live
        self._calibration_id = calibration_id
        self._device_id = device_id
        self._stale_by_ns = round(float(stale_by_s) * 1_000_000_000)
        self._closed = False

    def snapshot(self) -> CameraHealthSnapshot:
        if self._closed:
            raise RuntimeError("camera health port is closed")
        sdk_available = self._configured_sdk_available
        evidence = ["mock:orbbec_camera_health"]
        details: dict[str, Any] = {"live_probe": False}
        if self._sdk_root is not None:
            layout_available, layout_evidence = _sdk_layout(self._sdk_root)
            sdk_available = sdk_available and layout_available
            evidence.extend(layout_evidence)
            details["sdk_root"] = str(self._sdk_root)
            details["sdk_layout_available"] = layout_available
        now_ns = self._clock.monotonic_ns()
        return CameraHealthSnapshot(
            backend_name="mock_orbbec_gemini_pro",
            sdk_available=sdk_available,
            device_connected=self._device_connected,
            rgb_live=self._rgb_live,
            depth_live=self._depth_live,
            simulated=True,
            sample_monotonic_ns=max(0, now_ns - self._stale_by_ns),
            calibration_id=self._calibration_id,
            device_id=self._device_id,
            evidence_sources=tuple(evidence),
            details=details,
        )

    def close(self) -> None:
        self._closed = True


class OrbbecViewerCameraHealthPort:
    """Read only ``/api/status``; it never constructs an Orbbec Pipeline."""

    def __init__(
        self,
        *,
        clock: Clock,
        sdk_root: str | Path,
        status_url: str = "http://127.0.0.1:8080/api/status",
        calibration_id: str | None = None,
        live_probe: bool = False,
        timeout_s: float = 0.5,
    ) -> None:
        if not isinstance(status_url, str) or not status_url.strip():
            raise ValueError("status_url must be a non-empty string")
        if not isinstance(live_probe, bool):
            raise TypeError("live_probe must be a bool")
        if (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, (int, float))
            or timeout_s <= 0
        ):
            raise ValueError("timeout_s must be > 0")
        self._clock = clock
        self._sdk_root = Path(sdk_root).expanduser().resolve()
        self._status_url = status_url.strip()
        self._calibration_id = calibration_id
        self._live_probe = live_probe
        self._timeout_s = float(timeout_s)
        self._closed = False

    @staticmethod
    def parse_status(
        payload: Mapping[str, Any],
        *,
        clock: Clock,
        sdk_available: bool,
        calibration_id: str | None,
        evidence_sources: tuple[str, ...],
    ) -> CameraHealthSnapshot:
        if not isinstance(payload, Mapping):
            raise TypeError("Orbbec viewer status must be a mapping")
        streams = payload.get("streams", {})
        device = payload.get("device", {})
        if not isinstance(streams, Mapping) or not isinstance(device, Mapping):
            raise TypeError("Orbbec viewer status streams and device must be mappings")
        rgb = streams.get("rgb", {})
        depth = streams.get("depth", {})
        if not isinstance(rgb, Mapping) or not isinstance(depth, Mapping):
            raise TypeError("Orbbec viewer rgb/depth status must be mappings")
        serial = device.get("serial")
        if serial is not None and not isinstance(serial, str):
            serial = str(serial)
        return CameraHealthSnapshot(
            backend_name="orbbec_gemini_pro_viewer",
            sdk_available=sdk_available,
            device_connected=payload.get("state") in {"connected", "partial"},
            rgb_live=rgb.get("live") is True,
            depth_live=depth.get("live") is True,
            simulated=False,
            sample_monotonic_ns=clock.monotonic_ns(),
            calibration_id=calibration_id,
            device_id=serial,
            evidence_sources=evidence_sources,
            details={
                "viewer_state": payload.get("state", "unknown"),
                "rgb_error": rgb.get("error"),
                "depth_error": depth.get("error"),
                "rgb_source": rgb.get("source"),
                "depth_source": depth.get("source"),
                "firmware": device.get("firmware"),
                "sdk_version": device.get("sdk_version"),
            },
        )

    def snapshot(self) -> CameraHealthSnapshot:
        if self._closed:
            raise RuntimeError("camera health port is closed")
        sdk_available, layout_evidence = _sdk_layout(self._sdk_root)
        evidence = (*layout_evidence, f"viewer_status:{self._status_url}")
        if not self._live_probe:
            return CameraHealthSnapshot(
                backend_name="orbbec_gemini_pro_viewer",
                sdk_available=sdk_available,
                device_connected=False,
                rgb_live=False,
                depth_live=False,
                simulated=False,
                sample_monotonic_ns=self._clock.monotonic_ns(),
                calibration_id=self._calibration_id,
                evidence_sources=evidence,
                details={
                    "sdk_root": str(self._sdk_root),
                    "live_probe": False,
                    "reason": "live probe disabled; no device access attempted",
                },
            )
        request = Request(
            self._status_url,
            method="GET",
            headers={"Accept": "application/json"},
        )
        with urlopen(request, timeout=self._timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return self.parse_status(
            payload,
            clock=self._clock,
            sdk_available=sdk_available,
            calibration_id=self._calibration_id,
            evidence_sources=evidence,
        )

    def close(self) -> None:
        self._closed = True


__all__ = ["MockCameraHealthPort", "OrbbecViewerCameraHealthPort"]

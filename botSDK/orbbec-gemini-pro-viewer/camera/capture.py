from __future__ import annotations

import importlib
import os
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DRIVER_DIR = PROJECT_ROOT / "vendor" / "pyorbbecsdk" / "install" / "lib"
ORBBEC_RGB_VENDOR_ID = "2bc5"
ORBBEC_RGB_PRODUCT_ID = "0511"


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def depth_to_colormap(
    depth_mm: np.ndarray,
    min_depth_mm: int = 100,
    max_depth_mm: int = 5000,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert millimetre depth data to a fixed-scale Turbo colour map."""
    if depth_mm.ndim != 2:
        raise ValueError("depth_mm must be a two-dimensional array")
    if min_depth_mm < 0 or max_depth_mm <= min_depth_mm:
        raise ValueError("invalid depth range")

    valid = (depth_mm >= min_depth_mm) & (depth_mm <= max_depth_mm)
    normalized = np.zeros(depth_mm.shape, dtype=np.uint8)
    if np.any(valid):
        scaled = (
            (depth_mm[valid].astype(np.float32) - min_depth_mm)
            * (255.0 / (max_depth_mm - min_depth_mm))
        )
        normalized[valid] = np.clip(scaled, 0, 255).astype(np.uint8)

    colorized = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    colorized[~valid] = 0
    return colorized, valid


def _center_distance(depth_mm: np.ndarray, valid: np.ndarray) -> int | None:
    height, width = depth_mm.shape
    radius = 5
    y0, y1 = max(0, height // 2 - radius), min(height, height // 2 + radius + 1)
    x0, x1 = max(0, width // 2 - radius), min(width, width // 2 + radius + 1)
    patch = depth_mm[y0:y1, x0:x1]
    patch_valid = valid[y0:y1, x0:x1]
    values = patch[patch_valid]
    return int(np.median(values)) if values.size else None


@dataclass
class StreamState:
    name: str
    state: str = "waiting"
    width: int = 0
    height: int = 0
    fps: float = 0.0
    target_fps: int = 30
    frame_count: int = 0
    sequence: int = 0
    jpeg: bytes | None = None
    timestamp: str | None = None
    last_seen_monotonic: float = 0.0
    source: str = ""
    format: str = ""
    error: str | None = None
    center_distance_mm: int | None = None
    sample_times: deque[float] = field(default_factory=lambda: deque(maxlen=45))

    def snapshot(self, now: float) -> dict[str, Any]:
        stale = not self.last_seen_monotonic or now - self.last_seen_monotonic > 2.5
        state = "waiting" if self.state == "streaming" and stale else self.state
        return {
            "state": state,
            "live": state == "streaming" and not stale,
            "width": self.width,
            "height": self.height,
            "fps": round(self.fps, 1),
            "target_fps": self.target_fps,
            "frame_count": self.frame_count,
            "timestamp": self.timestamp,
            "source": self.source,
            "format": self.format,
            "error": self.error,
            "center_distance_mm": self.center_distance_mm,
        }


class CameraManager:
    """Owns one Orbbec depth capture and one Linux UVC RGB capture."""

    def __init__(
        self,
        driver_dir: str | Path | None = None,
        rgb_width: int = 1280,
        rgb_height: int = 720,
        rgb_fps: int = 30,
        depth_width: int = 1280,
        depth_height: int = 800,
        depth_fps: int = 30,
        depth_max_mm: int = 5000,
        jpeg_quality: int = 84,
    ) -> None:
        self.driver_dir = Path(
            driver_dir or os.environ.get("ORBBEC_DRIVER_DIR", DEFAULT_DRIVER_DIR)
        ).resolve()
        self.rgb_width = rgb_width
        self.rgb_height = rgb_height
        self.rgb_fps = rgb_fps
        self.depth_width = depth_width
        self.depth_height = depth_height
        self.depth_fps = depth_fps
        self.depth_min_mm = 100
        self.depth_max_mm = depth_max_mm
        self.jpeg_quality = jpeg_quality

        self._condition = threading.Condition(threading.RLock())
        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []
        self._started = False
        self._device: dict[str, Any] = {
            "name": "Orbbec Gemini Pro",
            "serial": None,
            "firmware": None,
            "pid": "0x0614",
            "vid": "0x2bc5",
            "connection": "USB 2.0",
            "sdk_version": None,
        }
        self._streams = {
            "rgb": StreamState(name="rgb", target_fps=rgb_fps),
            "depth": StreamState(name="depth", target_fps=depth_fps),
        }

    def start(self) -> None:
        with self._condition:
            if self._started:
                return
            self._started = True
            self._stop_event.clear()
            self._threads = [
                threading.Thread(
                    target=self._depth_worker,
                    name="orbbec-depth",
                    daemon=True,
                ),
                threading.Thread(
                    target=self._rgb_worker,
                    name="orbbec-rgb",
                    daemon=True,
                ),
            ]
            for thread in self._threads:
                thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
        for thread in self._threads:
            if thread.is_alive():
                thread.join(timeout=3.0)
        with self._condition:
            self._started = False

    def retry(self) -> None:
        with self._condition:
            for stream in self._streams.values():
                if stream.state != "streaming":
                    stream.state = "waiting"
                    stream.error = None
            self._condition.notify_all()

    def update_settings(
        self,
        *,
        depth_max_mm: int | None = None,
        jpeg_quality: int | None = None,
    ) -> dict[str, int]:
        with self._condition:
            if depth_max_mm is not None:
                if not 1000 <= depth_max_mm <= 10000:
                    raise ValueError("depth_max_mm must be between 1000 and 10000")
                self.depth_max_mm = int(depth_max_mm)
            if jpeg_quality is not None:
                if not 50 <= jpeg_quality <= 95:
                    raise ValueError("jpeg_quality must be between 50 and 95")
                self.jpeg_quality = int(jpeg_quality)
            return {
                "depth_max_mm": self.depth_max_mm,
                "jpeg_quality": self.jpeg_quality,
            }

    def settings(self) -> dict[str, int]:
        with self._condition:
            return {
                "depth_max_mm": self.depth_max_mm,
                "jpeg_quality": self.jpeg_quality,
            }

    def status(self) -> dict[str, Any]:
        now = time.monotonic()
        with self._condition:
            streams = {
                name: stream.snapshot(now) for name, stream in self._streams.items()
            }
            live_count = sum(int(stream["live"]) for stream in streams.values())
            if live_count == 2:
                state = "connected"
            elif live_count == 1:
                state = "partial"
            else:
                state = "disconnected"
            return {
                "state": state,
                "device": dict(self._device),
                "streams": streams,
                "settings": self.settings(),
                "server_time": _now_iso(),
            }

    def wait_for_frame(
        self,
        stream_name: str,
        after_sequence: int,
        timeout: float = 5.0,
    ) -> tuple[int, bytes | None]:
        deadline = time.monotonic() + timeout
        with self._condition:
            stream = self._streams[stream_name]
            while (
                stream.sequence <= after_sequence
                and not self._stop_event.is_set()
                and time.monotonic() < deadline
            ):
                self._condition.wait(timeout=max(0.05, deadline - time.monotonic()))
            return stream.sequence, stream.jpeg

    def _set_stream_state(
        self,
        stream_name: str,
        state: str,
        *,
        error: str | None = None,
        source: str | None = None,
    ) -> None:
        with self._condition:
            stream = self._streams[stream_name]
            stream.state = state
            stream.error = error
            if source is not None:
                stream.source = source
            self._condition.notify_all()

    def _publish(
        self,
        stream_name: str,
        jpeg: bytes,
        *,
        width: int,
        height: int,
        source: str,
        frame_format: str,
        center_distance_mm: int | None = None,
    ) -> None:
        now = time.monotonic()
        with self._condition:
            stream = self._streams[stream_name]
            stream.state = "streaming"
            stream.error = None
            stream.width = width
            stream.height = height
            stream.source = source
            stream.format = frame_format
            stream.center_distance_mm = center_distance_mm
            stream.frame_count += 1
            stream.sequence += 1
            stream.jpeg = jpeg
            stream.timestamp = _now_iso()
            stream.last_seen_monotonic = now
            stream.sample_times.append(now)
            if len(stream.sample_times) > 1:
                elapsed = stream.sample_times[-1] - stream.sample_times[0]
                if elapsed > 0:
                    stream.fps = (len(stream.sample_times) - 1) / elapsed
            self._condition.notify_all()

    def _encode_jpeg(self, image: np.ndarray) -> bytes | None:
        with self._condition:
            quality = self.jpeg_quality
        ok, encoded = cv2.imencode(
            ".jpg",
            image,
            [cv2.IMWRITE_JPEG_QUALITY, quality],
        )
        return encoded.tobytes() if ok else None

    def _import_orbbec_sdk(self):
        if not self.driver_dir.exists():
            raise RuntimeError(
                f"Orbbec driver not built: {self.driver_dir}. Run scripts/install.sh."
            )
        driver_dir = str(self.driver_dir)
        if driver_dir not in sys.path:
            sys.path.insert(0, driver_dir)
        return importlib.import_module("pyorbbecsdk")

    def _select_depth_profile(self, profiles: Any, sdk: Any) -> Any:
        """Prefer the requested full-resolution mode, then use the SDK default."""
        errors: list[str] = []
        for frame_format in (sdk.OBFormat.Y12, sdk.OBFormat.Y11):
            try:
                return profiles.get_video_stream_profile(
                    self.depth_width,
                    self.depth_height,
                    frame_format,
                    self.depth_fps,
                )
            except Exception as exc:
                errors.append(str(exc))

        profile = profiles.get_default_video_stream_profile()
        if profile is None:
            detail = "; ".join(error for error in errors if error)
            raise RuntimeError(f"找不到可用的深度视频模式：{detail}")
        return profile

    def _depth_worker(self) -> None:
        while not self._stop_event.is_set():
            pipeline = None
            try:
                sdk = self._import_orbbec_sdk()
                pipeline = sdk.Pipeline()
                device_info = pipeline.get_device().get_device_info()
                with self._condition:
                    self._device.update(
                        {
                            "name": device_info.get_name(),
                            "serial": device_info.get_serial_number(),
                            "firmware": device_info.get_firmware_version(),
                            "pid": f"0x{device_info.get_pid():04x}",
                            "vid": f"0x{device_info.get_vid():04x}",
                            "connection": device_info.get_connection_type(),
                            "sdk_version": sdk.get_version(),
                        }
                    )

                profiles = pipeline.get_stream_profile_list(
                    sdk.OBSensorType.DEPTH_SENSOR
                )
                profile = self._select_depth_profile(profiles, sdk)
                config = sdk.Config()
                config.enable_stream(profile)
                pipeline.start(config)
                self._set_stream_state(
                    "depth",
                    "waiting",
                    source="Orbbec SDK / PID 0614",
                )

                while not self._stop_event.is_set():
                    frames = pipeline.wait_for_frames(500)
                    if frames is None:
                        continue
                    depth_frame = frames.get_depth_frame()
                    if depth_frame is None:
                        continue

                    width = depth_frame.get_width()
                    height = depth_frame.get_height()
                    expected = width * height
                    raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
                    if raw.size < expected:
                        continue
                    scale = float(depth_frame.get_depth_scale())
                    depth_mm = (
                        raw[:expected]
                        .reshape((height, width))
                        .astype(np.float32)
                        * scale
                    )
                    with self._condition:
                        max_depth = self.depth_max_mm
                    colorized, valid = depth_to_colormap(
                        depth_mm,
                        self.depth_min_mm,
                        max_depth,
                    )
                    jpeg = self._encode_jpeg(colorized)
                    if jpeg is None:
                        continue
                    self._publish(
                        "depth",
                        jpeg,
                        width=width,
                        height=height,
                        source="Orbbec SDK / PID 0614",
                        frame_format="Y16 → Turbo",
                        center_distance_mm=_center_distance(depth_mm, valid),
                    )
            except Exception as exc:
                self._set_stream_state(
                    "depth",
                    "error",
                    error=str(exc),
                    source="Orbbec SDK / PID 0614",
                )
            finally:
                if pipeline is not None:
                    try:
                        pipeline.stop()
                    except Exception:
                        pass
            self._stop_event.wait(1.5)

    @staticmethod
    def _find_rgb_nodes() -> list[Path]:
        candidates: list[tuple[int, Path]] = []
        for video_path in sorted(Path("/sys/class/video4linux").glob("video*")):
            try:
                resolved = video_path.resolve()
                usb_parent = next(
                    parent
                    for parent in [resolved, *resolved.parents]
                    if (parent / "idVendor").exists()
                    and (parent / "idProduct").exists()
                )
                vendor = (usb_parent / "idVendor").read_text().strip().lower()
                product = (usb_parent / "idProduct").read_text().strip().lower()
                if vendor != ORBBEC_RGB_VENDOR_ID or product != ORBBEC_RGB_PRODUCT_ID:
                    continue
                index_path = video_path / "index"
                index = int(index_path.read_text().strip()) if index_path.exists() else 99
                candidates.append((index, Path("/dev") / video_path.name))
            except (OSError, StopIteration, ValueError):
                continue
        return [path for _, path in sorted(candidates)]

    def _open_rgb_capture(self) -> tuple[cv2.VideoCapture, Path]:
        nodes = self._find_rgb_nodes()
        if not nodes:
            raise RuntimeError("未找到 Gemini RGB UVC 节点；请重新插拔相机 USB")

        errors: list[str] = []
        for node in nodes:
            capture = cv2.VideoCapture(str(node), cv2.CAP_V4L2)
            if not capture.isOpened():
                errors.append(f"{node} 无法打开")
                capture.release()
                continue
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            capture.set(
                cv2.CAP_PROP_FOURCC,
                cv2.VideoWriter_fourcc(*"MJPG"),
            )
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.rgb_width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.rgb_height)
            capture.set(cv2.CAP_PROP_FPS, self.rgb_fps)
            return capture, node
        raise RuntimeError("；".join(errors) or "Gemini RGB UVC 节点不可用")

    def _rgb_worker(self) -> None:
        while not self._stop_event.is_set():
            capture: cv2.VideoCapture | None = None
            try:
                self._set_stream_state("rgb", "waiting")
                capture, node = self._open_rgb_capture()
                self._set_stream_state("rgb", "waiting", source=str(node))
                failures = 0
                while not self._stop_event.is_set():
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        failures += 1
                        if failures >= 20:
                            raise RuntimeError(f"{node} 连续读取失败")
                        time.sleep(0.03)
                        continue
                    failures = 0
                    jpeg = self._encode_jpeg(frame)
                    if jpeg is None:
                        continue
                    height, width = frame.shape[:2]
                    self._publish(
                        "rgb",
                        jpeg,
                        width=width,
                        height=height,
                        source=str(node),
                        frame_format="MJPEG → BGR",
                    )
            except Exception as exc:
                self._set_stream_state("rgb", "error", error=str(exc))
            finally:
                if capture is not None:
                    capture.release()
            self._stop_event.wait(1.5)

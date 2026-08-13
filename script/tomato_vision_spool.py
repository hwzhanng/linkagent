"""Atomic filesystem spool for synchronized raw RGB-D captures.

The module owns no camera SDK and performs no alignment. A vendor adapter must
provide an already synchronized RGB PNG, D2C-aligned 16-bit depth PNG, device
timestamps, intrinsics, and a calibrated camera-to-world pose. The spool only
validates and atomically publishes that evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import tempfile
import time
from typing import Mapping
import uuid

from proto import CameraIntrinsics, Pose
from script.tomato_vision_capture import RawRgbdCaptureBundle


_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_LATEST_FIELDS = frozenset(
    {
        "schema_version",
        "generation",
        "capture_id",
        "capture_bundle",
        "sample_monotonic_ns",
        "rgb_sample_monotonic_ns",
        "depth_sample_monotonic_ns",
        "calibration_id",
        "device_id",
        "sidecar_sha256",
        "published_unix_ns",
    }
)


def _non_empty(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _non_negative_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be an integer >= 0")
    return value


def _positive_float(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite number > 0")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{field_name} must be a finite number > 0")
    return result


def _relative_path(value: str | Path, *, field_name: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{field_name} must be a confined relative path")
    return path


def _confined(root: Path, relative: Path, *, field_name: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"{field_name} escapes capture_root")
    return path


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_fsynced(path: Path, payload: bytes) -> None:
    with path.open("xb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


@dataclass(frozen=True, slots=True)
class PngDescriptor:
    width_px: int
    height_px: int
    bit_depth: int
    colour_type: int


def inspect_png(payload: bytes, *, field_name: str) -> PngDescriptor:
    """Read the PNG signature and IHDR without decoding image pixels."""

    if not isinstance(payload, bytes):
        raise TypeError(f"{field_name} must be bytes")
    if len(payload) < 33 or payload[:8] != _PNG_SIGNATURE:
        raise ValueError(f"{field_name} must be a complete PNG")
    if len(payload) < 45 or payload[-12:-8] != b"\x00\x00\x00\x00" or payload[-8:-4] != b"IEND":
        raise ValueError(f"{field_name} must end with an IEND chunk")
    length = struct.unpack(">I", payload[8:12])[0]
    if length != 13 or payload[12:16] != b"IHDR":
        raise ValueError(f"{field_name} must begin with a standard IHDR chunk")
    width, height, bit_depth, colour_type, compression, filtering, interlace = struct.unpack(
        ">IIBBBBB", payload[16:29]
    )
    if width <= 0 or height <= 0:
        raise ValueError(f"{field_name} dimensions must be positive")
    if compression != 0 or filtering != 0 or interlace not in {0, 1}:
        raise ValueError(f"{field_name} uses unsupported PNG header settings")
    return PngDescriptor(width, height, bit_depth, colour_type)


@dataclass(frozen=True, slots=True)
class RgbdSpoolFrame:
    capture_id: str
    rgb_png: bytes
    aligned_depth_png: bytes
    depth_scale_m_per_unit: float
    intrinsics: CameraIntrinsics
    camera_pose_in_world: Pose
    rgb_sample_monotonic_ns: int
    depth_sample_monotonic_ns: int
    device_id: str
    evidence_sources: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "capture_id", _non_empty(self.capture_id, field_name="capture_id"))
        if not isinstance(self.rgb_png, bytes) or not isinstance(self.aligned_depth_png, bytes):
            raise TypeError("rgb_png and aligned_depth_png must be bytes")
        if not isinstance(self.intrinsics, CameraIntrinsics):
            raise TypeError("intrinsics must be CameraIntrinsics")
        if not isinstance(self.camera_pose_in_world, Pose):
            raise TypeError("camera_pose_in_world must be Pose")
        if self.camera_pose_in_world.frame_id != "world":
            raise ValueError("camera_pose_in_world.frame_id must be 'world'")
        object.__setattr__(
            self,
            "depth_scale_m_per_unit",
            _positive_float(
                self.depth_scale_m_per_unit, field_name="depth_scale_m_per_unit"
            ),
        )
        for name in ("rgb_sample_monotonic_ns", "depth_sample_monotonic_ns"):
            object.__setattr__(
                self, name, _non_negative_int(getattr(self, name), field_name=name)
            )
        object.__setattr__(self, "device_id", _non_empty(self.device_id, field_name="device_id"))
        if isinstance(self.evidence_sources, (str, bytes, bytearray, Mapping)):
            raise TypeError("evidence_sources must be a sequence of strings")
        sources = tuple(
            _non_empty(item, field_name="evidence_sources item")
            for item in self.evidence_sources
        )
        if not sources:
            raise ValueError("evidence_sources must not be empty")
        object.__setattr__(self, "evidence_sources", sources)

        rgb = inspect_png(self.rgb_png, field_name="rgb_png")
        depth = inspect_png(self.aligned_depth_png, field_name="aligned_depth_png")
        expected = (self.intrinsics.width_px, self.intrinsics.height_px)
        if (rgb.width_px, rgb.height_px) != expected:
            raise ValueError("RGB PNG dimensions do not match camera intrinsics")
        if (depth.width_px, depth.height_px) != expected:
            raise ValueError("aligned depth PNG dimensions do not match camera intrinsics")
        if rgb.bit_depth != 8 or rgb.colour_type not in {2, 6}:
            raise ValueError("RGB PNG must be 8-bit RGB or RGBA")
        if depth.bit_depth != 16 or depth.colour_type != 0:
            raise ValueError("aligned depth PNG must be 16-bit grayscale sensor units")

    @property
    def sample_monotonic_ns(self) -> int:
        return max(self.rgb_sample_monotonic_ns, self.depth_sample_monotonic_ns)

    @property
    def skew_s(self) -> float:
        return abs(
            self.rgb_sample_monotonic_ns - self.depth_sample_monotonic_ns
        ) / 1.0e9

    def sidecar_payload(self) -> dict[str, object]:
        intrinsics = self.intrinsics
        return {
            "schema_version": 1,
            "capture_id": self.capture_id,
            "rgb_path": "rgb.png",
            "aligned_depth_path": "aligned-depth.png",
            "depth_scale_m_per_unit": self.depth_scale_m_per_unit,
            "camera": {
                "intrinsics": {
                    "width_px": intrinsics.width_px,
                    "height_px": intrinsics.height_px,
                    "fx_px": intrinsics.fx_px,
                    "fy_px": intrinsics.fy_px,
                    "cx_px": intrinsics.cx_px,
                    "cy_px": intrinsics.cy_px,
                    "calibration_id": intrinsics.calibration_id,
                },
                "camera_pose_in_world": self.camera_pose_in_world.as_dict(),
                "rgb_sample_monotonic_ns": self.rgb_sample_monotonic_ns,
                "depth_sample_monotonic_ns": self.depth_sample_monotonic_ns,
                "depth_aligned_to_rgb": True,
                "device_id": self.device_id,
                "evidence_sources": list(self.evidence_sources),
            },
        }


@dataclass(frozen=True, slots=True)
class LatestCapturePointer:
    generation: int
    capture_id: str
    capture_bundle: str
    sample_monotonic_ns: int
    rgb_sample_monotonic_ns: int
    depth_sample_monotonic_ns: int
    calibration_id: str
    device_id: str
    sidecar_sha256: str
    published_unix_ns: int

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "LatestCapturePointer":
        if set(payload) != _LATEST_FIELDS:
            raise ValueError("latest capture pointer fields do not match schema v1")
        if payload.get("schema_version") != 1:
            raise ValueError("latest capture pointer schema_version must be 1")
        generation = _non_negative_int(payload.get("generation"), field_name="generation")
        if generation < 1:
            raise ValueError("latest capture pointer generation must be >= 1")
        digest = _non_empty(payload.get("sidecar_sha256"), field_name="sidecar_sha256")
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("sidecar_sha256 must be a lowercase SHA-256 digest")
        return cls(
            generation=generation,
            capture_id=_non_empty(payload.get("capture_id"), field_name="capture_id"),
            capture_bundle=_non_empty(
                payload.get("capture_bundle"), field_name="capture_bundle"
            ),
            sample_monotonic_ns=_non_negative_int(
                payload.get("sample_monotonic_ns"), field_name="sample_monotonic_ns"
            ),
            rgb_sample_monotonic_ns=_non_negative_int(
                payload.get("rgb_sample_monotonic_ns"),
                field_name="rgb_sample_monotonic_ns",
            ),
            depth_sample_monotonic_ns=_non_negative_int(
                payload.get("depth_sample_monotonic_ns"),
                field_name="depth_sample_monotonic_ns",
            ),
            calibration_id=_non_empty(
                payload.get("calibration_id"), field_name="calibration_id"
            ),
            device_id=_non_empty(payload.get("device_id"), field_name="device_id"),
            sidecar_sha256=digest,
            published_unix_ns=_non_negative_int(
                payload.get("published_unix_ns"), field_name="published_unix_ns"
            ),
        )

    def as_dict(self) -> dict[str, object]:
        return {"schema_version": 1, **{name: getattr(self, name) for name in self.__dataclass_fields__}}


@dataclass(frozen=True, slots=True)
class SpoolPublication:
    pointer: LatestCapturePointer
    bundle_path: Path
    rgb_sha256: str
    aligned_depth_sha256: str


@dataclass(frozen=True, slots=True)
class ResolvedSpoolCapture:
    pointer: LatestCapturePointer
    bundle_path: Path
    bundle: RawRgbdCaptureBundle


class TomatoVisionSpoolReader:
    def __init__(
        self,
        capture_root: str | Path,
        *,
        latest_reference: str | Path = "latest.json",
        captures_reference: str | Path = "captures",
        max_pointer_bytes: int = 64_000,
    ) -> None:
        self.root = Path(capture_root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)
        self.latest_relative = _relative_path(
            latest_reference, field_name="latest_reference"
        )
        self.latest_path = _confined(
            self.root, self.latest_relative, field_name="latest_reference"
        )
        self.captures_relative = _relative_path(
            captures_reference, field_name="captures_reference"
        )
        self.captures_path = _confined(
            self.root, self.captures_relative, field_name="captures_reference"
        )
        if not self.captures_path.is_dir():
            raise FileNotFoundError(self.captures_path)
        if isinstance(max_pointer_bytes, bool) or not isinstance(max_pointer_bytes, int) or max_pointer_bytes < 1024:
            raise ValueError("max_pointer_bytes must be an integer >= 1024")
        self.max_pointer_bytes = max_pointer_bytes

    def read_pointer(self) -> LatestCapturePointer:
        if not self.latest_path.is_file():
            raise FileNotFoundError(self.latest_path)
        raw = self.latest_path.read_bytes()
        if len(raw) > self.max_pointer_bytes:
            raise ValueError("latest capture pointer exceeds the configured limit")
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("latest capture pointer must be UTF-8 JSON") from exc
        if not isinstance(payload, Mapping):
            raise TypeError("latest capture pointer must be a mapping")
        return LatestCapturePointer.from_mapping(payload)

    def resolve_pointer(self, pointer: LatestCapturePointer) -> ResolvedSpoolCapture:
        relative = _relative_path(pointer.capture_bundle, field_name="capture_bundle")
        path = _confined(self.root, relative, field_name="capture_bundle")
        if not path.is_file():
            raise FileNotFoundError(path)
        if _sha256_file(path) != pointer.sidecar_sha256:
            raise ValueError("latest capture sidecar SHA-256 mismatch")
        bundle = RawRgbdCaptureBundle.load(path, capture_root=self.root)
        sample_ns = max(
            bundle.rgb_sample_monotonic_ns, bundle.depth_sample_monotonic_ns
        )
        if bundle.capture_id != pointer.capture_id:
            raise ValueError("latest pointer capture_id does not match sidecar")
        if sample_ns != pointer.sample_monotonic_ns:
            raise ValueError("latest pointer sample timestamp does not match sidecar")
        if bundle.rgb_sample_monotonic_ns != pointer.rgb_sample_monotonic_ns or bundle.depth_sample_monotonic_ns != pointer.depth_sample_monotonic_ns:
            raise ValueError("latest pointer stream timestamps do not match sidecar")
        if bundle.intrinsics.calibration_id != pointer.calibration_id:
            raise ValueError("latest pointer calibration_id does not match sidecar")
        if bundle.device_id != pointer.device_id:
            raise ValueError("latest pointer device_id does not match sidecar")
        return ResolvedSpoolCapture(pointer=pointer, bundle_path=path, bundle=bundle)

    def resolve_latest(
        self,
        *,
        minimum_sample_monotonic_ns: int | None = None,
        timeout_s: float = 0.0,
        poll_interval_s: float = 0.02,
    ) -> ResolvedSpoolCapture:
        if minimum_sample_monotonic_ns is not None:
            minimum_sample_monotonic_ns = _non_negative_int(
                minimum_sample_monotonic_ns,
                field_name="minimum_sample_monotonic_ns",
            )
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or timeout_s < 0:
            raise ValueError("timeout_s must be >= 0")
        if isinstance(poll_interval_s, bool) or not isinstance(poll_interval_s, (int, float)) or not 0.001 <= poll_interval_s <= 0.25:
            raise ValueError("poll_interval_s must be in [0.001, 0.25]")
        deadline = time.monotonic() + float(timeout_s)
        while True:
            pointer = self.read_pointer()
            if (
                minimum_sample_monotonic_ns is None
                or pointer.sample_monotonic_ns > minimum_sample_monotonic_ns
            ):
                return self.resolve_pointer(pointer)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("no newer RGB-D capture was published before timeout")
            time.sleep(min(float(poll_interval_s), remaining))

    def resolve_bundle(self, reference: str | Path) -> RawRgbdCaptureBundle:
        relative = _relative_path(reference, field_name="capture_bundle")
        path = _confined(self.root, relative, field_name="capture_bundle")
        return RawRgbdCaptureBundle.load(path, capture_root=self.root)

    def capture_paths(self) -> tuple[Path, ...]:
        return tuple(
            path / "capture.json"
            for path in sorted(self.captures_path.iterdir(), key=lambda item: item.name)
            if path.is_dir()
            and not path.is_symlink()
            and not path.name.startswith(".")
            and (path / "capture.json").is_file()
        )


class TomatoVisionSpoolWriter:
    def __init__(
        self,
        capture_root: str | Path,
        *,
        captures_reference: str | Path = "captures",
        latest_reference: str | Path = "latest.json",
        max_rgb_depth_skew_s: float = 0.035,
        max_image_bytes: int = 128_000_000,
        reserve_free_bytes: int = 0,
    ) -> None:
        self.root = Path(capture_root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.captures_relative = _relative_path(
            captures_reference, field_name="captures_reference"
        )
        self.captures_path = _confined(
            self.root, self.captures_relative, field_name="captures_reference"
        )
        self.captures_path.mkdir(parents=True, exist_ok=True)
        if not self.captures_path.resolve().is_relative_to(self.root):
            raise ValueError("captures directory escapes capture_root")
        self.latest_relative = _relative_path(
            latest_reference, field_name="latest_reference"
        )
        self.latest_path = _confined(
            self.root, self.latest_relative, field_name="latest_reference"
        )
        if not 0.0 < _positive_float(
            max_rgb_depth_skew_s, field_name="max_rgb_depth_skew_s"
        ) <= 1.0:
            raise ValueError("max_rgb_depth_skew_s must be <= 1.0")
        self.max_rgb_depth_skew_s = float(max_rgb_depth_skew_s)
        self.max_image_bytes = _non_negative_int(
            max_image_bytes, field_name="max_image_bytes"
        )
        if self.max_image_bytes < 1024:
            raise ValueError("max_image_bytes must be >= 1024")
        self.reserve_free_bytes = _non_negative_int(
            reserve_free_bytes, field_name="reserve_free_bytes"
        )
        self.lock_path = self.root / ".writer.lock"

    def _current_pointer(self) -> LatestCapturePointer | None:
        if not self.latest_path.exists():
            return None
        reader = TomatoVisionSpoolReader(
            self.root, latest_reference=self.latest_relative
        )
        pointer = reader.read_pointer()
        # Do not advance past a corrupt latest pointer.  An operator must first
        # audit/recover the spool so evidence cannot be silently skipped.
        reader.resolve_pointer(pointer)
        return pointer

    def publish(self, frame: RgbdSpoolFrame) -> SpoolPublication:
        if not isinstance(frame, RgbdSpoolFrame):
            raise TypeError("frame must be RgbdSpoolFrame")
        if frame.skew_s > self.max_rgb_depth_skew_s:
            raise ValueError("RGB/depth timestamp skew exceeds the configured limit")
        for name, payload in {
            "rgb_png": frame.rgb_png,
            "aligned_depth_png": frame.aligned_depth_png,
        }.items():
            if len(payload) > self.max_image_bytes:
                raise ValueError(f"{name} exceeds max_image_bytes")
        required_bytes = len(frame.rgb_png) + len(frame.aligned_depth_png) + 128_000
        if shutil.disk_usage(self.root).free < required_bytes + self.reserve_free_bytes:
            raise OSError("capture spool has insufficient free space")

        with self.lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            staging: Path | None = None
            committed = False
            try:
                current = self._current_pointer()
                if current is not None and frame.sample_monotonic_ns <= current.sample_monotonic_ns:
                    raise ValueError(
                        "capture sample timestamp must advance beyond the published latest frame"
                    )
                key = f"{frame.sample_monotonic_ns:020d}-{uuid.uuid4().hex}"
                final_directory = self.captures_path / key
                staging = Path(
                    tempfile.mkdtemp(prefix=".staging-", dir=self.captures_path)
                )
                _write_fsynced(staging / "rgb.png", frame.rgb_png)
                _write_fsynced(staging / "aligned-depth.png", frame.aligned_depth_png)
                sidecar_bytes = json.dumps(
                    frame.sidecar_payload(),
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                _write_fsynced(staging / "capture.json", sidecar_bytes)
                RawRgbdCaptureBundle.load(
                    staging / "capture.json", capture_root=self.root
                )
                _fsync_directory(staging)
                os.replace(staging, final_directory)
                committed = True
                _fsync_directory(self.captures_path)

                relative_bundle = (
                    self.captures_relative / key / "capture.json"
                ).as_posix()
                pointer = LatestCapturePointer(
                    generation=1 if current is None else current.generation + 1,
                    capture_id=frame.capture_id,
                    capture_bundle=relative_bundle,
                    sample_monotonic_ns=frame.sample_monotonic_ns,
                    rgb_sample_monotonic_ns=frame.rgb_sample_monotonic_ns,
                    depth_sample_monotonic_ns=frame.depth_sample_monotonic_ns,
                    calibration_id=frame.intrinsics.calibration_id,
                    device_id=frame.device_id,
                    sidecar_sha256=_sha256_bytes(sidecar_bytes),
                    published_unix_ns=time.time_ns(),
                )
                _atomic_write(
                    self.latest_path,
                    json.dumps(
                        pointer.as_dict(), sort_keys=True, separators=(",", ":")
                    ).encode("utf-8"),
                )
                return SpoolPublication(
                    pointer=pointer,
                    bundle_path=final_directory / "capture.json",
                    rgb_sha256=_sha256_bytes(frame.rgb_png),
                    aligned_depth_sha256=_sha256_bytes(frame.aligned_depth_png),
                )
            finally:
                if not committed and staging is not None and staging.exists():
                    shutil.rmtree(staging)
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def recover_staging(
        self, *, older_than_s: float = 60.0, dry_run: bool = True
    ) -> tuple[Path, ...]:
        if isinstance(older_than_s, bool) or not isinstance(older_than_s, (int, float)) or older_than_s < 0:
            raise ValueError("older_than_s must be >= 0")
        if not isinstance(dry_run, bool):
            raise TypeError("dry_run must be a bool")
        cutoff = time.time() - float(older_than_s)
        candidates = tuple(sorted(
            path
            for path in self.captures_path.iterdir()
            if path.name.startswith(".staging-")
            and path.is_dir()
            and not path.is_symlink()
            and path.stat().st_mtime <= cutoff
        ))
        if not dry_run:
            with self.lock_path.open("a+b") as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                try:
                    for path in candidates:
                        if path.parent == self.captures_path and path.name.startswith(".staging-"):
                            shutil.rmtree(path)
                    _fsync_directory(self.captures_path)
                finally:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        return candidates

    def prune(self, *, keep_last: int, dry_run: bool = True) -> tuple[Path, ...]:
        if isinstance(keep_last, bool) or not isinstance(keep_last, int) or keep_last < 1:
            raise ValueError("keep_last must be an integer >= 1")
        if not isinstance(dry_run, bool):
            raise TypeError("dry_run must be a bool")
        with self.lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                current = self._current_pointer()
                protected = None
                if current is not None:
                    protected = _confined(
                        self.root,
                        _relative_path(
                            current.capture_bundle, field_name="capture_bundle"
                        ),
                        field_name="capture_bundle",
                    ).parent
                captures = sorted(
                    (
                        path
                        for path in self.captures_path.iterdir()
                        if path.is_dir()
                        and not path.is_symlink()
                        and not path.name.startswith(".")
                        and (path / "capture.json").is_file()
                    ),
                    key=lambda path: path.name,
                    reverse=True,
                )
                keep = set(captures[:keep_last])
                if protected is not None:
                    keep.add(protected)
                removable = tuple(path for path in captures if path not in keep)
                if not dry_run:
                    for path in removable:
                        if path.parent == self.captures_path:
                            shutil.rmtree(path)
                    _fsync_directory(self.captures_path)
                return removable
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


__all__ = [
    "LatestCapturePointer",
    "PngDescriptor",
    "ResolvedSpoolCapture",
    "RgbdSpoolFrame",
    "SpoolPublication",
    "TomatoVisionSpoolReader",
    "TomatoVisionSpoolWriter",
    "inspect_png",
]

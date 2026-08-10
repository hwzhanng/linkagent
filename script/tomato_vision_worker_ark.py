"""Ark multimodal tomato boxes plus aligned-depth pseudo-mask worker backend.

This is an explicit temporary replacement for YOLO-seg.  Ark supplies semantic
2D boxes and image points; aligned depth creates a conservative box-local mask.
The backend never treats VLM output as metric geometry or motion authority.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from io import BytesIO
import hashlib
import json
import math
import os
from pathlib import Path
import threading
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from script.tomato_vision_capture import RawRgbdCaptureBundle


_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


def _non_empty(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _mapping(value: object, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    return value


def _unit_interval(value: object, *, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"{field_name} must be finite and in [0, 1]")
    return float(value)


def _positive(value: object, *, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{field_name} must be finite and > 0")
    return float(value)


def _positive_int(value: object, *, field_name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field_name} must be an integer >= {minimum}")
    return value


def _point(
    value: object,
    *,
    field_name: str,
    width: int,
    height: int,
) -> tuple[int, int]:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise ValueError(f"{field_name} must contain integer x/y")
    try:
        x, y = tuple(value)  # type: ignore[arg-type,misc]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must contain integer x/y") from exc
    if any(isinstance(item, bool) or not isinstance(item, int) for item in (x, y)):
        raise ValueError(f"{field_name} must contain integer x/y")
    if not 0 <= x < width or not 0 <= y < height:
        raise ValueError(f"{field_name} is outside the RGB image")
    return x, y


def _bbox(
    value: object,
    *,
    field_name: str,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise ValueError(f"{field_name} must contain x1/y1/x2/y2")
    try:
        items = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(f"{field_name} must contain x1/y1/x2/y2") from exc
    if len(items) != 4:
        raise ValueError(f"{field_name} must contain x1/y1/x2/y2")
    prepared: list[float] = []
    for item in items:
        if (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
        ):
            raise ValueError(f"{field_name} values must be finite numbers")
        prepared.append(float(item))
    x1, y1, x2, y2 = prepared
    if not (0.0 <= x1 < x2 <= width and 0.0 <= y1 < y2 <= height):
        raise ValueError(f"{field_name} must be ordered and inside the RGB image")
    if (x2 - x1) * (y2 - y1) < 16.0:
        raise ValueError(f"{field_name} is too small")
    return x1, y1, x2, y2


def _coordinates_1000(
    value: object,
    *,
    field_name: str,
    length: int,
) -> tuple[int, ...]:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise ValueError(f"{field_name} must contain {length} normalized integers")
    try:
        items = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(
            f"{field_name} must contain {length} normalized integers"
        ) from exc
    if len(items) != length or any(
        isinstance(item, bool) or not isinstance(item, int) or not 0 <= item <= 1000
        for item in items
    ):
        raise ValueError(f"{field_name} values must be integers in [0, 1000]")
    return items


def _scaled_box_1000(
    value: object,
    *,
    field_name: str,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = _coordinates_1000(
        value, field_name=field_name, length=4
    )
    return _bbox(
        (x1 * width / 1000.0, y1 * height / 1000.0, x2 * width / 1000.0, y2 * height / 1000.0),
        field_name=field_name,
        width=width,
        height=height,
    )


def _scaled_point_1000(
    value: object,
    *,
    field_name: str,
    width: int,
    height: int,
) -> tuple[int, int]:
    x, y = _coordinates_1000(value, field_name=field_name, length=2)
    return _point(
        (
            min(width - 1, round(x * (width - 1) / 1000.0)),
            min(height - 1, round(y * (height - 1) / 1000.0)),
        ),
        field_name=field_name,
        width=width,
        height=height,
    )


def _bbox_iou(left: Sequence[float], right: Sequence[float]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return 0.0 if union <= 0.0 else intersection / union


class ArkResponsesTomatoBoxDetector:
    """Strict Ark Responses client for semantic tomato boxes and image points."""

    def __init__(
        self,
        *,
        base_url: str = "https://ark.cn-beijing.volces.com/api/coding/v3",
        model_id: str = "doubao-seed-2.0-pro",
        api_key_env: str = "ARK_API_KEY",
        timeout_s: float = 60.0,
        max_response_bytes: int = 256_000,
        max_output_tokens: int = 4096,
        thinking_type: str = "disabled",
    ) -> None:
        parsed = urlparse(base_url)
        if (
            parsed.scheme not in {"http", "https"}
            or parsed.hostname is None
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("ark_vlm.base_url must be credential-free HTTP(S)")
        if parsed.scheme == "http" and parsed.hostname not in _LOOPBACK_HOSTS:
            raise ValueError("non-loopback Ark endpoints must use HTTPS")
        self.base_url = base_url.rstrip("/")
        self.model_id = _non_empty(model_id, field_name="ark_vlm.model_id")
        self.api_key_env = _non_empty(
            api_key_env, field_name="ark_vlm.api_key_env"
        )
        self.timeout_s = _positive(timeout_s, field_name="ark_vlm.timeout_s")
        self.max_response_bytes = _positive_int(
            max_response_bytes,
            field_name="ark_vlm.max_response_bytes",
            minimum=1024,
        )
        if self.max_response_bytes > 2_000_000:
            raise ValueError("ark_vlm.max_response_bytes must be <= 2000000")
        self.max_output_tokens = _positive_int(
            max_output_tokens,
            field_name="ark_vlm.max_output_tokens",
            minimum=512,
        )
        if self.max_output_tokens > 8192:
            raise ValueError("ark_vlm.max_output_tokens must be <= 8192")
        if thinking_type not in {"disabled", "enabled", "auto"}:
            raise ValueError("ark_vlm.thinking_type must be disabled, enabled, or auto")
        self.thinking_type = thinking_type
        self.last_response_id: str | None = None
        self.last_resolved_model: str | None = None

    @property
    def configured(self) -> bool:
        return bool(os.environ.get(self.api_key_env))

    def detect(
        self,
        *,
        rgb_png: bytes,
        width: int,
        height: int,
        requested_track_id: str | None,
    ) -> list[dict[str, object]]:
        secret = os.environ.get(self.api_key_env)
        if not secret:
            raise RuntimeError(
                f"configured Ark credential environment variable "
                f"{self.api_key_env!r} is missing"
            )
        prompt = (
            "Detect every visible tomato fruit in this RGB image. Return all coordinates "
            "in a normalized 0-to-1000 image coordinate system, with x measured left to "
            "right and y top to bottom. Return xyxy boxes, a point near each fruit center, "
            "and the visible stem attachment point; both points must be inside the box. "
            "Estimate ripeness and occlusion semantically. Do not infer depth, "
            "metric pose, reachability, grasp success, or robot motion. Exclude leaves, "
            "signs, red non-fruit objects, and boxes smaller than 4x4 pixels."
        )
        if requested_track_id is not None:
            prompt += (
                " A previously pinned track must be recovered by local tracking after "
                "detection; do not invent or return a track ID."
            )
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "boxes": {
                    "type": "array",
                    "maxItems": 20,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "bbox_xyxy_1000": {
                                "type": "array",
                                "items": {
                                    "type": "integer",
                                    "minimum": 0,
                                    "maximum": 1000,
                                },
                                "minItems": 4,
                                "maxItems": 4,
                            },
                            "confidence": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                            },
                            "ripeness_score": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                            },
                            "occlusion_ratio": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                            },
                            "center_point_xy_1000": {
                                "type": "array",
                                "items": {
                                    "type": "integer",
                                    "minimum": 0,
                                    "maximum": 1000,
                                },
                                "minItems": 2,
                                "maxItems": 2,
                            },
                            "stem_point_xy_1000": {
                                "type": "array",
                                "items": {
                                    "type": "integer",
                                    "minimum": 0,
                                    "maximum": 1000,
                                },
                                "minItems": 2,
                                "maxItems": 2,
                            },
                            "reason": {"type": "string", "maxLength": 512},
                        },
                        "required": [
                            "bbox_xyxy_1000",
                            "confidence",
                            "ripeness_score",
                            "occlusion_ratio",
                            "center_point_xy_1000",
                            "stem_point_xy_1000",
                            "reason",
                        ],
                    },
                }
            },
            "required": ["boxes"],
        }
        payload = {
            "model": self.model_id,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {
                            "type": "input_image",
                            "image_url": "data:image/png;base64,"
                            + base64.b64encode(rgb_png).decode("ascii"),
                        },
                    ],
                }
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "tomato_boxes",
                    "strict": True,
                    "schema": schema,
                }
            },
            "thinking": {"type": self.thinking_type},
            "max_output_tokens": self.max_output_tokens,
        }
        request = Request(
            f"{self.base_url}/responses",
            data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {secret}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "realagent-ark-vision/0.1",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_s) as response:
                raw = response.read(self.max_response_bytes + 1)
        except HTTPError as exc:
            raise RuntimeError(
                f"Ark multimodal endpoint rejected request with HTTP {exc.code}"
            ) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise RuntimeError("Ark multimodal endpoint is unavailable") from exc
        if len(raw) > self.max_response_bytes:
            raise RuntimeError("Ark multimodal response exceeds the configured limit")
        try:
            response = _mapping(
                json.loads(raw.decode("utf-8")), field_name="Ark response"
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("Ark multimodal endpoint returned invalid JSON") from exc
        if response.get("status") != "completed":
            status = str(response.get("status"))[:80]
            details = str(response.get("incomplete_details"))[:160]
            raise RuntimeError(
                f"Ark multimodal response did not complete: status={status}; "
                f"details={details}"
            )
        response_id = response.get("id")
        resolved_model = response.get("model")
        self.last_response_id = response_id if isinstance(response_id, str) else None
        self.last_resolved_model = (
            resolved_model if isinstance(resolved_model, str) else None
        )
        texts: list[str] = []
        output = response.get("output")
        if isinstance(output, (str, bytes, bytearray)) or not isinstance(
            output, Sequence
        ):
            raise RuntimeError("Ark multimodal response output is invalid")
        for item in output:
            if not isinstance(item, Mapping) or item.get("type") != "message":
                continue
            content = item.get("content", ())
            if isinstance(content, (str, bytes, bytearray)) or not isinstance(
                content, Sequence
            ):
                continue
            for part in content:
                if (
                    isinstance(part, Mapping)
                    and part.get("type") == "output_text"
                    and isinstance(part.get("text"), str)
                ):
                    texts.append(part["text"])
        if len(texts) != 1:
            raise RuntimeError(
                "Ark multimodal response must contain exactly one structured output"
            )
        try:
            structured = _mapping(
                json.loads(texts[0]), field_name="Ark structured tomato boxes"
            )
        except json.JSONDecodeError as exc:
            raise RuntimeError("Ark structured output is not valid JSON") from exc
        if set(structured) != {"boxes"}:
            raise ValueError("Ark structured output fields do not match tomato_boxes")
        boxes = structured["boxes"]
        if isinstance(boxes, (str, bytes, bytearray, Mapping)) or not isinstance(
            boxes, Sequence
        ):
            raise TypeError("Ark tomato boxes must be a sequence")
        if len(boxes) > 20:
            raise ValueError("Ark returned more than 20 tomato boxes")
        detections: list[dict[str, object]] = []
        allowed = {
            "bbox_xyxy_1000",
            "confidence",
            "ripeness_score",
            "occlusion_ratio",
            "center_point_xy_1000",
            "stem_point_xy_1000",
            "reason",
        }
        for index, raw_box in enumerate(boxes):
            item = _mapping(raw_box, field_name=f"Ark boxes[{index}]")
            if set(item) != allowed:
                raise ValueError(f"Ark boxes[{index}] fields do not match schema")
            reason = _non_empty(item["reason"], field_name=f"Ark boxes[{index}].reason")
            if len(reason) > 512:
                raise ValueError("Ark tomato reason exceeds 512 characters")
            prepared_box = _scaled_box_1000(
                item["bbox_xyxy_1000"],
                field_name=f"Ark boxes[{index}].bbox_xyxy_1000",
                width=width,
                height=height,
            )
            center = _scaled_point_1000(
                item["center_point_xy_1000"],
                field_name=f"Ark boxes[{index}].center_point_xy_1000",
                width=width,
                height=height,
            )
            stem = _scaled_point_1000(
                item["stem_point_xy_1000"],
                field_name=f"Ark boxes[{index}].stem_point_xy_1000",
                width=width,
                height=height,
            )
            x1, y1, x2, y2 = prepared_box
            if not (x1 <= center[0] < x2 and y1 <= center[1] < y2):
                raise ValueError("Ark center_point_xy must be inside its tomato box")
            if not (x1 <= stem[0] < x2 and y1 <= stem[1] < y2):
                raise ValueError("Ark stem_point_xy must be inside its tomato box")
            detections.append(
                {
                    "bbox": prepared_box,
                    "confidence": _unit_interval(
                        item["confidence"], field_name="Ark confidence"
                    ),
                    "ripeness": _unit_interval(
                        item["ripeness_score"], field_name="Ark ripeness_score"
                    ),
                    "occlusion": _unit_interval(
                        item["occlusion_ratio"], field_name="Ark occlusion_ratio"
                    ),
                    "center_point": center,
                    "stem_point": stem,
                    "reason": reason,
                    "label": "tomato",
                }
            )
        return detections


class ArkVlmBoxDepthBackend:
    """Temporary VLM-box detector with local aligned-depth pseudo masks."""

    backend_id = "ark_vlm_box_aligned_depth_pseudomask_v1"

    def __init__(
        self,
        *,
        capture_root: str | Path,
        config_dir: str | Path,
        ark_vlm: Mapping[str, object],
        minimum_confidence: float = 0.45,
        max_instances: int = 20,
        max_depth_samples: int = 512,
        min_depth_m: float = 0.10,
        max_depth_m: float = 3.0,
        depth_band_m: float = 0.08,
        track_iou_threshold: float = 0.25,
        track_iou_ambiguity_margin: float = 0.05,
        max_image_pixels: int = 8_847_360,
        max_image_file_bytes: int = 128_000_000,
    ) -> None:
        del config_dir
        self._capture_root = Path(capture_root).expanduser().resolve()
        if not self._capture_root.is_dir():
            raise FileNotFoundError(self._capture_root)
        self._minimum_confidence = _unit_interval(
            minimum_confidence, field_name="backend.minimum_confidence"
        )
        self._max_instances = _positive_int(
            max_instances, field_name="backend.max_instances"
        )
        self._max_depth_samples = _positive_int(
            max_depth_samples, field_name="backend.max_depth_samples", minimum=6
        )
        self._min_depth_m = _positive(min_depth_m, field_name="backend.min_depth_m")
        self._max_depth_m = _positive(max_depth_m, field_name="backend.max_depth_m")
        if self._min_depth_m >= self._max_depth_m:
            raise ValueError("backend depth limits must be ordered")
        self._depth_band_m = _positive(
            depth_band_m, field_name="backend.depth_band_m"
        )
        self._track_iou = _unit_interval(
            track_iou_threshold, field_name="backend.track_iou_threshold"
        )
        self._track_iou_ambiguity_margin = _unit_interval(
            track_iou_ambiguity_margin,
            field_name="backend.track_iou_ambiguity_margin",
        )
        self._max_image_pixels = _positive_int(
            max_image_pixels, field_name="backend.max_image_pixels"
        )
        self._max_image_file_bytes = _positive_int(
            max_image_file_bytes, field_name="backend.max_image_file_bytes"
        )
        vlm_config = dict(_mapping(ark_vlm, field_name="backend.ark_vlm"))
        vlm_type = vlm_config.pop("type", None)
        if vlm_type != "ark_responses":
            raise ValueError("backend.ark_vlm.type must be ark_responses")
        self._detector = ArkResponsesTomatoBoxDetector(**vlm_config)  # type: ignore[arg-type]
        try:
            import numpy as np  # type: ignore[import-not-found]
            from PIL import Image  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "ark_vlm_box backend requires numpy and Pillow"
            ) from exc
        self._np = np
        self._Image = Image
        self._lock = threading.Lock()
        self._next_track_id = 1
        self._previous_tracks: dict[str, tuple[float, float, float, float]] = {}
        self._closed = False

    def _read_images(
        self, capture: RawRgbdCaptureBundle
    ) -> tuple[Any, Any, bytes, str, str]:
        expected_pixels = capture.intrinsics.width_px * capture.intrinsics.height_px
        if expected_pixels > self._max_image_pixels:
            raise ValueError("capture dimensions exceed the configured pixel limit")
        for path in (capture.rgb_path, capture.aligned_depth_path):
            if path.stat().st_size > self._max_image_file_bytes:
                raise ValueError("capture image exceeds the configured file-size limit")
        rgb_bytes = capture.rgb_path.read_bytes()
        depth_bytes = capture.aligned_depth_path.read_bytes()
        try:
            with self._Image.open(BytesIO(rgb_bytes)) as source:
                rgb = source.convert("RGB")
            with self._Image.open(BytesIO(depth_bytes)) as source:
                depth_array = self._np.asarray(source).copy()
        except OSError as exc:
            raise ValueError("capture bundle contains an unreadable RGB/depth image") from exc
        if rgb.size != (capture.intrinsics.width_px, capture.intrinsics.height_px):
            raise ValueError("RGB dimensions do not match camera intrinsics")
        if depth_array.ndim != 2 or depth_array.shape != (
            capture.intrinsics.height_px,
            capture.intrinsics.width_px,
        ):
            raise ValueError("aligned depth dimensions do not match camera intrinsics")
        depth_m = depth_array.astype(self._np.float64) * capture.depth_scale_m_per_unit
        png = BytesIO()
        rgb.save(png, format="PNG")
        return (
            rgb,
            depth_m,
            png.getvalue(),
            hashlib.sha256(rgb_bytes).hexdigest(),
            hashlib.sha256(depth_bytes).hexdigest(),
        )

    def _nearest_depth(
        self, depth_m: Any, point: tuple[int, int]
    ) -> tuple[int, int, float]:
        x, y = point
        height, width = depth_m.shape
        for radius in range(0, 9):
            candidates: list[tuple[int, int, float]] = []
            for v in range(max(0, y - radius), min(height, y + radius + 1)):
                for u in range(max(0, x - radius), min(width, x + radius + 1)):
                    depth = float(depth_m[v, u])
                    if self._min_depth_m <= depth <= self._max_depth_m:
                        candidates.append((u, v, depth))
            if candidates:
                return min(
                    candidates,
                    key=lambda item: (
                        (item[0] - x) ** 2 + (item[1] - y) ** 2,
                        item[2],
                    ),
                )
        raise ValueError("no valid aligned depth near Ark image point")

    def _depth_mask(self, depth_m: Any, detection: dict[str, object]) -> Any:
        x1, y1, x2, y2 = detection["bbox"]
        center_u, center_v, center_depth = self._nearest_depth(
            depth_m, detection["center_point"]  # type: ignore[arg-type]
        )
        height, width = depth_m.shape
        yy, xx = self._np.ogrid[:height, :width]
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        radius_x = max(1.0, (x2 - x1) / 2.0)
        radius_y = max(1.0, (y2 - y1) / 2.0)
        in_box = (xx >= x1) & (xx < x2) & (yy >= y1) & (yy < y2)
        in_ellipse = (
            ((xx - center_x) / radius_x) ** 2
            + ((yy - center_y) / radius_y) ** 2
            <= 1.0
        )
        valid_depth = (
            (depth_m >= self._min_depth_m)
            & (depth_m <= self._max_depth_m)
            & (self._np.abs(depth_m - center_depth) <= self._depth_band_m)
        )
        mask = in_box & in_ellipse & valid_depth
        mask[center_v, center_u] = True
        detection["center_point"] = (center_u, center_v)
        if int(mask.sum()) < 6:
            raise ValueError("Ark box has too little aligned-depth support")
        return mask

    def _assign_tracks(self, detections: list[dict[str, object]]) -> None:
        available = dict(self._previous_tracks)
        current: dict[str, tuple[float, float, float, float]] = {}
        for detection in detections:
            bbox = detection["bbox"]
            ranked = sorted(
                (
                    (_bbox_iou(bbox, previous_bbox), track_id)
                    for track_id, previous_bbox in available.items()
                ),
                reverse=True,
            )
            ambiguous = (
                len(ranked) > 1
                and ranked[0][0] >= self._track_iou
                and ranked[1][0] >= self._track_iou
                and ranked[0][0] - ranked[1][0]
                < self._track_iou_ambiguity_margin
            )
            if ranked and ranked[0][0] >= self._track_iou and not ambiguous:
                track_id = ranked[0][1]
                available.pop(track_id)
            else:
                track_id = f"tomato-{self._next_track_id:06d}"
                self._next_track_id += 1
            detection["track_id"] = track_id
            current[track_id] = bbox  # type: ignore[assignment]
        self._previous_tracks = current

    @staticmethod
    def _deproject(
        capture: RawRgbdCaptureBundle, u: int, v: int, depth_m: float
    ) -> list[float]:
        intrinsics = capture.intrinsics
        return [
            (u - intrinsics.cx_px) * depth_m / intrinsics.fx_px,
            (v - intrinsics.cy_px) * depth_m / intrinsics.fy_px,
            depth_m,
        ]

    def _instance_payload(
        self,
        *,
        capture: RawRgbdCaptureBundle,
        detection: Mapping[str, object],
        instance_id: str,
        depth_m: Any,
        selected: bool,
    ) -> dict[str, object]:
        mask = detection["mask"]
        coordinates = self._np.argwhere(mask)
        count = int(len(coordinates))
        indexes = (
            self._np.linspace(
                0, count - 1, num=self._max_depth_samples, dtype=int
            )
            if count > self._max_depth_samples
            else self._np.arange(count)
        )
        sampled = coordinates[indexes]
        sampled_vu = [(int(v), int(u)) for v, u in sampled]
        center_u, center_v = detection["center_point"]  # type: ignore[misc]
        center_vu = (int(center_v), int(center_u))
        if center_vu not in sampled_vu:
            if len(sampled_vu) >= self._max_depth_samples:
                sampled_vu[0] = center_vu
            else:
                sampled_vu.append(center_vu)
        mask_pixels = [(u, v) for v, u in sampled_vu]
        depth_samples = [
            [u, v, float(depth_m[v, u])]
            for v, u in sampled_vu
        ]
        stem_camera = None
        if selected:
            stem_u, stem_v, stem_depth = self._nearest_depth(
                depth_m, detection["stem_point"]  # type: ignore[arg-type]
            )
            stem_camera = self._deproject(
                capture, stem_u, stem_v, stem_depth
            )
        return {
            "instance_id": instance_id,
            "track_id": detection["track_id"],
            "bbox_xyxy_px": list(detection["bbox"]),
            "mask_pixels_xy": [list(point) for point in mask_pixels],
            "detector_confidence": detection["confidence"],
            "ripeness_score": detection["ripeness"],
            "occlusion_ratio": detection["occlusion"],
            "depth_samples_uvm": depth_samples,
            "stem_point_camera_m": stem_camera,
            "mask_area_px": count,
            "valid_depth_pixel_count": count,
        }

    def process(
        self,
        *,
        bundle_path: Path,
        observation_sequence: int,
        requested_track_id: str | None,
    ) -> Mapping[str, object]:
        del observation_sequence
        with self._lock:
            if self._closed:
                raise RuntimeError("Ark vision inference backend is closed")
            capture = RawRgbdCaptureBundle.load(
                bundle_path, capture_root=self._capture_root
            )
            _, depth_m, rgb_png, rgb_sha256, depth_sha256 = self._read_images(
                capture
            )
            detections = [
                item
                for item in self._detector.detect(
                    rgb_png=rgb_png,
                    width=capture.intrinsics.width_px,
                    height=capture.intrinsics.height_px,
                    requested_track_id=requested_track_id,
                )
                if float(item["confidence"]) >= self._minimum_confidence
            ][: self._max_instances]
            if not detections:
                raise ValueError("Ark VLM returned no tomato box above confidence gate")
            supported: list[dict[str, object]] = []
            for detection in detections:
                try:
                    detection["mask"] = self._depth_mask(depth_m, detection)
                except ValueError:
                    continue
                supported.append(detection)
            if not supported:
                raise ValueError("no Ark tomato box had enough aligned-depth support")
            self._assign_tracks(supported)
            forced = next(
                (
                    index
                    for index, item in enumerate(supported)
                    if item["track_id"] == requested_track_id
                ),
                None,
            )
            if requested_track_id is not None and forced is None:
                raise ValueError("requested_track_id is not visible in the Ark frame")
            selected_index = (
                forced
                if forced is not None
                else max(
                    range(len(supported)),
                    key=lambda index: (
                        float(supported[index]["confidence"])
                        * float(supported[index]["ripeness"])
                        * (1.0 - float(supported[index]["occlusion"])),
                        -index,
                    ),
                )
            )
            selected = supported[selected_index]
            instances = [
                self._instance_payload(
                    capture=capture,
                    detection=detection,
                    instance_id=f"ark-box-depth-{index + 1}",
                    depth_m=depth_m,
                    selected=index == selected_index,
                )
                for index, detection in enumerate(supported)
            ]
            camera_payload = capture.camera_payload()
            response_evidence = []
            if self._detector.last_response_id is not None:
                response_evidence.append(
                    f"ark-response-id:{self._detector.last_response_id}"
                )
            if self._detector.last_resolved_model is not None:
                response_evidence.append(
                    f"ark-resolved-model:{self._detector.last_resolved_model}"
                )
            camera_payload["evidence_sources"] = [
                *capture.evidence_sources,
                f"capture:{capture.capture_id}",
                f"rgb-sha256:{rgb_sha256}",
                f"aligned-depth-sha256:{depth_sha256}",
                f"ark-vlm-model:{self._detector.model_id}",
                *response_evidence,
                "instance-authority:ark-vlm-box+aligned-depth-pseudomask",
            ]
            return {
                "camera": camera_payload,
                "instances": instances,
                "vlm_selection": {
                    "selected_instance_id": f"ark-box-depth-{selected_index + 1}",
                    "confidence": selected["confidence"],
                    "reason": selected["reason"],
                    "point_xy": list(selected["center_point"]),
                    "selector_model": self._detector.model_id,
                },
            }

    def health(self) -> Mapping[str, object]:
        return {
            "ready": not self._closed and self._detector.configured,
            "backend_id": self.backend_id,
            "simulated": False,
            "model_id": self._detector.model_id,
            "pixel_authority": "ark_vlm_box+aligned_depth_pseudomask",
        }

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._previous_tracks.clear()


__all__ = ["ArkResponsesTomatoBoxDetector", "ArkVlmBoxDepthBackend"]

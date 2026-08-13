"""Optional Ultralytics YOLO-seg plus structured VLM worker backend.

All heavy dependencies are imported after configuration and local model-path
validation.  Supplying a model alias is intentionally unsupported because
Ultralytics may otherwise download weights implicitly.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from io import BytesIO
import hashlib
import json
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


def _unit_interval(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a number in [0, 1]")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{field_name} must be in [0, 1]")
    return result


def _positive_int(value: object, *, field_name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field_name} must be an integer >= {minimum}")
    return value


def _mapping(value: object, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    return value


def _point(value: object, *, field_name: str, width: int, height: int) -> tuple[int, int]:
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class HttpJsonVlmSelector:
    """Call an explicitly configured structured VLM gateway."""

    def __init__(
        self,
        *,
        endpoint: str,
        model_id: str,
        api_key_env: str | None = None,
        timeout_s: float = 15.0,
        max_response_bytes: int = 64_000,
    ) -> None:
        parsed = urlparse(endpoint)
        if (
            parsed.scheme not in {"http", "https"}
            or parsed.hostname is None
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
        ):
            raise ValueError("vlm.endpoint must be credential-free HTTP(S)")
        if parsed.scheme == "http" and parsed.hostname not in _LOOPBACK_HOSTS:
            raise ValueError("non-loopback VLM endpoints must use HTTPS")
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or timeout_s <= 0:
            raise ValueError("vlm.timeout_s must be > 0")
        if not 1024 <= _positive_int(
            max_response_bytes, field_name="vlm.max_response_bytes"
        ) <= 2_000_000:
            raise ValueError("vlm.max_response_bytes must be <= 2000000")
        self.endpoint = endpoint
        self.model_id = _non_empty(model_id, field_name="vlm.model_id")
        self.timeout_s = float(timeout_s)
        self.max_response_bytes = max_response_bytes
        self.api_key_env = None
        if api_key_env is not None:
            self.api_key_env = _non_empty(api_key_env, field_name="vlm.api_key_env")

    def select(
        self,
        *,
        overlay_png: bytes,
        candidates: Sequence[Mapping[str, object]],
        requested_track_id: str | None,
        width: int,
        height: int,
    ) -> dict[str, object]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key_env is not None:
            secret = os.environ.get(self.api_key_env)
            if not secret:
                raise RuntimeError(
                    f"configured VLM credential environment variable {self.api_key_env!r} is missing"
                )
            headers["Authorization"] = f"Bearer {secret}"
        prompt = (
            "Choose exactly one numbered tomato instance. Prefer a ripe, clearly visible "
            "fruit and avoid severe occlusion. Return only the structured JSON contract. "
            "point_xy must be inside the chosen fruit mask; stem_point_xy marks its visible "
            "stem attachment. Do not infer metric depth, pose, or robot reachability."
        )
        if requested_track_id is not None:
            prompt += f" Keep the already pinned track_id {requested_track_id!r}."
        body = json.dumps(
            {
                "schema_version": 1,
                "model": self.model_id,
                "prompt": prompt,
                "requested_track_id": requested_track_id,
                "candidates": list(candidates),
                "image_png_base64": base64.b64encode(overlay_png).decode("ascii"),
            },
            separators=(",", ":"),
        ).encode("utf-8")
        request = Request(self.endpoint, data=body, headers=headers, method="POST")
        try:
            with urlopen(request, timeout=self.timeout_s) as response:
                raw = response.read(self.max_response_bytes + 1)
        except HTTPError as exc:
            raise RuntimeError(f"VLM gateway rejected request with HTTP {exc.code}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise RuntimeError("VLM gateway is unavailable") from exc
        if len(raw) > self.max_response_bytes:
            raise RuntimeError("VLM gateway response exceeds the configured limit")
        try:
            payload = _mapping(json.loads(raw.decode("utf-8")), field_name="VLM response")
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("VLM gateway returned invalid JSON") from exc
        allowed = {
            "schema_version",
            "selected_instance_id",
            "confidence",
            "reason",
            "point_xy",
            "stem_point_xy",
            "ripeness_score",
            "occlusion_ratio",
        }
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError("unknown VLM response fields: " + ", ".join(sorted(unknown)))
        if payload.get("schema_version") != 1:
            raise ValueError("VLM response schema_version must be 1")
        candidate_ids = {item["instance_id"] for item in candidates}
        selected = _non_empty(
            payload.get("selected_instance_id"), field_name="selected_instance_id"
        )
        if selected not in candidate_ids:
            raise ValueError("VLM selected an instance that YOLO-seg did not produce")
        reason = _non_empty(payload.get("reason"), field_name="reason")
        if len(reason) > 512:
            raise ValueError("VLM response reason must not exceed 512 characters")
        return {
            "selected_instance_id": selected,
            "confidence": _unit_interval(payload.get("confidence"), field_name="confidence"),
            "reason": reason,
            "point_xy": _point(
                payload.get("point_xy"), field_name="point_xy", width=width, height=height
            ),
            "stem_point_xy": _point(
                payload.get("stem_point_xy"),
                field_name="stem_point_xy",
                width=width,
                height=height,
            ),
            "ripeness_score": _unit_interval(
                payload.get("ripeness_score"), field_name="ripeness_score"
            ),
            "occlusion_ratio": _unit_interval(
                payload.get("occlusion_ratio"), field_name="occlusion_ratio"
            ),
        }

    @property
    def configured(self) -> bool:
        return self.api_key_env is None or bool(os.environ.get(self.api_key_env))


class UltralyticsVlmBackend:
    """Real worker backend with YOLO mask authority and RGB-D metric authority."""

    backend_id = "ultralytics_yolo_seg_structured_vlm_rgbd_v1"

    def __init__(
        self,
        *,
        capture_root: str | Path,
        config_dir: str | Path,
        model_path: str,
        vlm: Mapping[str, object],
        confidence: float = 0.50,
        tomato_class_names: Sequence[str] = ("tomato",),
        max_instances: int = 20,
        max_depth_samples: int = 512,
        min_depth_m: float = 0.10,
        max_depth_m: float = 3.0,
        track_iou_threshold: float = 0.30,
        track_iou_ambiguity_margin: float = 0.05,
        max_image_pixels: int = 8_847_360,
        max_image_file_bytes: int = 128_000_000,
    ) -> None:
        self._capture_root = Path(capture_root).expanduser().resolve()
        if not self._capture_root.is_dir():
            raise FileNotFoundError(self._capture_root)
        root = Path(config_dir).expanduser().resolve()
        reference = Path(_non_empty(model_path, field_name="backend.model_path"))
        self._model_path = (root / reference).resolve() if not reference.is_absolute() else reference.resolve()
        if not self._model_path.is_file():
            raise FileNotFoundError(
                f"local YOLO segmentation weights are required: {self._model_path}"
            )
        self._confidence = _unit_interval(confidence, field_name="backend.confidence")
        if self._confidence <= 0.0:
            raise ValueError("backend.confidence must be > 0")
        if isinstance(tomato_class_names, (str, bytes, bytearray, Mapping)):
            raise TypeError("backend.tomato_class_names must be a sequence")
        self._class_names = frozenset(
            _non_empty(item, field_name="backend.tomato_class_names item").casefold()
            for item in tomato_class_names
        )
        if not self._class_names:
            raise ValueError("backend.tomato_class_names must not be empty")
        self._max_instances = _positive_int(max_instances, field_name="backend.max_instances")
        self._max_depth_samples = _positive_int(
            max_depth_samples, field_name="backend.max_depth_samples", minimum=6
        )
        if not 0.0 < min_depth_m < max_depth_m:
            raise ValueError("backend depth limits must be positive and ordered")
        self._min_depth_m = float(min_depth_m)
        self._max_depth_m = float(max_depth_m)
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
        vlm_config = dict(_mapping(vlm, field_name="backend.vlm"))
        vlm_type = vlm_config.pop("type", None)
        if vlm_type != "http_json":
            raise ValueError("backend.vlm.type must be http_json")
        self._selector = HttpJsonVlmSelector(**vlm_config)  # type: ignore[arg-type]

        try:
            import numpy as np  # type: ignore[import-not-found]
            from PIL import Image, ImageDraw  # type: ignore[import-not-found]
            from ultralytics import YOLO  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics_vlm backend requires ultralytics, numpy, and Pillow"
            ) from exc
        self._np = np
        self._Image = Image
        self._ImageDraw = ImageDraw
        model_sha256_before_load = _sha256(self._model_path)
        try:
            self._model = YOLO(str(self._model_path))
        except Exception as exc:
            raise RuntimeError("failed to load local YOLO segmentation weights") from exc
        self._model_sha256 = _sha256(self._model_path)
        if self._model_sha256 != model_sha256_before_load:
            raise RuntimeError("YOLO segmentation weights changed while being loaded")
        self._lock = threading.Lock()
        self._next_track_id = 1
        self._previous_tracks: dict[str, tuple[float, float, float, float]] = {}
        self._closed = False

    def _read_images(self, capture: RawRgbdCaptureBundle) -> tuple[Any, Any, str, str]:
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
        if depth_array.dtype.kind not in {"u", "i"}:
            raise ValueError("aligned depth image must contain integer sensor units")
        return (
            rgb,
            depth_array.astype(self._np.float64) * capture.depth_scale_m_per_unit,
            hashlib.sha256(rgb_bytes).hexdigest(),
            hashlib.sha256(depth_bytes).hexdigest(),
        )

    def _detect(self, rgb: Any) -> list[dict[str, object]]:
        try:
            results = self._model.predict(
                source=self._np.asarray(rgb),
                conf=self._confidence,
                verbose=False,
            )
        except Exception as exc:
            raise RuntimeError("YOLO segmentation inference failed") from exc
        if len(results) != 1:
            raise RuntimeError("YOLO segmentation must return exactly one image result")
        result = results[0]
        if result.boxes is None or result.masks is None:
            raise ValueError("YOLO model returned no instance masks")
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        scores = result.boxes.conf.detach().cpu().numpy()
        class_ids = result.boxes.cls.detach().cpu().numpy()
        masks = result.masks.data.detach().cpu().numpy()
        names = result.names
        detections: list[dict[str, object]] = []
        for bbox, score, class_id, raw_mask in zip(boxes, scores, class_ids, masks):
            label = str(names[int(class_id)])
            if label.casefold() not in self._class_names:
                continue
            mask_image = self._Image.fromarray((raw_mask > 0.5).astype("uint8") * 255)
            if mask_image.size != rgb.size:
                mask_image = mask_image.resize(rgb.size, resample=self._Image.Resampling.NEAREST)
            mask = self._np.asarray(mask_image) > 0
            if int(mask.sum()) == 0:
                continue
            detections.append(
                {
                    "bbox": tuple(float(item) for item in bbox),
                    "confidence": float(score),
                    "label": label,
                    "mask": mask,
                }
            )
        detections.sort(key=lambda item: float(item["confidence"]), reverse=True)
        if not detections:
            raise ValueError("YOLO-seg returned no configured tomato class instances")
        return detections[: self._max_instances]

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

    def _overlay(self, rgb: Any, detections: Sequence[Mapping[str, object]]) -> bytes:
        pixels = self._np.asarray(rgb).copy().astype(self._np.float32)
        colours = (
            (255, 64, 64),
            (64, 220, 96),
            (64, 128, 255),
            (255, 190, 48),
            (190, 80, 255),
            (48, 220, 220),
        )
        for index, detection in enumerate(detections):
            mask = detection["mask"]
            colour = self._np.asarray(colours[index % len(colours)], dtype=self._np.float32)
            pixels[mask] = pixels[mask] * 0.58 + colour * 0.42
        overlay = self._Image.fromarray(
            self._np.clip(pixels, 0, 255).astype("uint8"), mode="RGB"
        )
        draw = self._ImageDraw.Draw(overlay)
        for index, detection in enumerate(detections, start=1):
            x1, y1, x2, y2 = detection["bbox"]
            draw.rectangle((x1, y1, x2, y2), outline=(255, 40, 40), width=4)
            draw.text((x1 + 4, y1 + 4), str(index), fill=(255, 255, 0), stroke_width=2, stroke_fill=(0, 0, 0))
        buffer = BytesIO()
        overlay.save(buffer, format="PNG")
        return buffer.getvalue()

    def _nearest_depth(self, depth_m: Any, point: tuple[int, int]) -> tuple[int, int, float]:
        x, y = point
        height, width = depth_m.shape
        for radius in range(0, 6):
            candidates: list[tuple[float, int, int]] = []
            for v in range(max(0, y - radius), min(height, y + radius + 1)):
                for u in range(max(0, x - radius), min(width, x + radius + 1)):
                    depth = float(depth_m[v, u])
                    if self._min_depth_m <= depth <= self._max_depth_m:
                        candidates.append((depth, u, v))
            if candidates:
                center_depth = float(depth_m[y, x])
                if self._min_depth_m <= center_depth <= self._max_depth_m:
                    depth, u, v = min(
                        candidates,
                        key=lambda item: (
                            abs(item[0] - center_depth),
                            (item[1] - x) ** 2 + (item[2] - y) ** 2,
                        ),
                    )
                else:
                    depth, u, v = min(
                        candidates,
                        key=lambda item: (
                            (item[1] - x) ** 2 + (item[2] - y) ** 2,
                            item[0],
                        ),
                    )
                return u, v, depth
        raise ValueError("no valid aligned depth near the VLM stem point")

    @staticmethod
    def _deproject(capture: RawRgbdCaptureBundle, u: int, v: int, depth_m: float) -> list[float]:
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
        selected_point: tuple[int, int] | None,
        stem_point: tuple[int, int] | None,
        ripeness: float,
        occlusion: float,
    ) -> dict[str, object]:
        mask = detection["mask"]
        mask_coordinates = self._np.argwhere(mask)
        mask_area = int(len(mask_coordinates))
        valid_dense = mask & (depth_m >= self._min_depth_m) & (depth_m <= self._max_depth_m)
        valid_coordinates = self._np.argwhere(valid_dense)
        valid_count = int(len(valid_coordinates))
        if mask_area > self._max_depth_samples:
            mask_indexes = self._np.linspace(
                0, mask_area - 1, num=self._max_depth_samples, dtype=int
            )
            sampled_mask = mask_coordinates[mask_indexes]
        else:
            sampled_mask = mask_coordinates
        if valid_count > self._max_depth_samples:
            indexes = self._np.linspace(
                0, valid_count - 1, num=self._max_depth_samples, dtype=int
            )
            sampled = valid_coordinates[indexes]
        else:
            sampled = valid_coordinates
        mask_pixels = {(int(u), int(v)) for v, u in sampled_mask}
        mask_pixels.update((int(u), int(v)) for v, u in sampled)
        if selected_point is not None:
            mask_pixels.add(selected_point)
        depth_samples = [
            [int(u), int(v), float(depth_m[int(v), int(u)])]
            for v, u in sampled
        ]
        x1, y1, x2, y2 = detection["bbox"]
        bbox_area = max(1.0, (x2 - x1) * (y2 - y1))
        if stem_point is None:
            stem_camera = None
        else:
            stem_u, stem_v, stem_depth = self._nearest_depth(depth_m, stem_point)
            stem_camera = self._deproject(capture, stem_u, stem_v, stem_depth)
        return {
            "instance_id": instance_id,
            "track_id": detection["track_id"],
            "bbox_xyxy_px": list(detection["bbox"]),
            "mask_pixels_xy": [list(point) for point in sorted(mask_pixels)],
            "detector_confidence": detection["confidence"],
            "ripeness_score": ripeness,
            "occlusion_ratio": occlusion if stem_point is not None else max(0.0, min(1.0, 1.0 - mask_area / bbox_area)),
            "depth_samples_uvm": depth_samples,
            "stem_point_camera_m": stem_camera,
            "mask_area_px": mask_area,
            "valid_depth_pixel_count": valid_count,
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
                raise RuntimeError("vision inference backend is closed")
            capture = RawRgbdCaptureBundle.load(
                bundle_path, capture_root=self._capture_root
            )
            rgb, depth_m, rgb_sha256, depth_sha256 = self._read_images(capture)
            detections = self._detect(rgb)
            self._assign_tracks(detections)
            forced_instance = None
            candidates_for_vlm: list[dict[str, object]] = []
            for index, detection in enumerate(detections, start=1):
                instance_id = f"som-{index}"
                if detection["track_id"] == requested_track_id:
                    forced_instance = instance_id
                candidates_for_vlm.append(
                    {
                        "number": index,
                        "instance_id": instance_id,
                        "track_id": detection["track_id"],
                        "label": detection["label"],
                        "detector_confidence": detection["confidence"],
                        "bbox_xyxy_px": list(detection["bbox"]),
                    }
                )
            if requested_track_id is not None and forced_instance is None:
                raise ValueError("requested_track_id is not visible in the current YOLO frame")
            selection = self._selector.select(
                overlay_png=self._overlay(rgb, detections),
                candidates=candidates_for_vlm,
                requested_track_id=requested_track_id,
                width=capture.intrinsics.width_px,
                height=capture.intrinsics.height_px,
            )
            selected_id = selection["selected_instance_id"]
            if forced_instance is not None and selected_id != forced_instance:
                raise ValueError("VLM changed the pinned target identity")
            selected_index = int(str(selected_id).split("-", 1)[1]) - 1
            selected_mask = detections[selected_index]["mask"]
            point_xy = selection["point_xy"]
            containing = sum(bool(item["mask"][point_xy[1], point_xy[0]]) for item in detections)
            if containing != 1 or not bool(selected_mask[point_xy[1], point_xy[0]]):
                raise ValueError("VLM point must belong to exactly one selected YOLO mask")
            instances = []
            for index, detection in enumerate(detections, start=1):
                instance_id = f"som-{index}"
                selected = instance_id == selected_id
                instances.append(
                    self._instance_payload(
                        capture=capture,
                        detection=detection,
                        instance_id=instance_id,
                        depth_m=depth_m,
                        selected_point=point_xy if selected else None,
                        stem_point=selection["stem_point_xy"] if selected else None,
                        ripeness=float(selection["ripeness_score"]) if selected else 0.0,
                        occlusion=float(selection["occlusion_ratio"]) if selected else 0.0,
                    )
                )
            camera_payload = capture.camera_payload()
            camera_payload["evidence_sources"] = [
                *capture.evidence_sources,
                f"capture:{capture.capture_id}",
                f"rgb-sha256:{rgb_sha256}",
                f"aligned-depth-sha256:{depth_sha256}",
                f"yolo-model-sha256:{self._model_sha256}",
            ]
            return {
                "camera": camera_payload,
                "instances": instances,
                "vlm_selection": {
                    "selected_instance_id": selected_id,
                    "confidence": selection["confidence"],
                    "reason": selection["reason"],
                    "point_xy": list(point_xy),
                    "selector_model": self._selector.model_id,
                },
            }

    def health(self) -> Mapping[str, object]:
        return {
            "ready": not self._closed and self._selector.configured,
            "backend_id": self.backend_id,
            "simulated": False,
            "model_path": str(self._model_path),
            "model_sha256": self._model_sha256,
        }

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._previous_tracks.clear()


__all__ = ["HttpJsonVlmSelector", "UltralyticsVlmBackend"]

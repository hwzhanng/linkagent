"""SDK-neutral computer-vision result DTOs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ._base_proto import _finite_float, _finite_float_tuple, _freeze_mapping


@dataclass(frozen=True, slots=True)
class VisionInstance:
    """One labelled image instance with an optional pixel-space bounding box.

    ``bbox_xyxy`` follows ``(x_min, y_min, x_max, y_max)`` in image pixels.
    ``mask`` may be any backend-neutral array-like payload; concrete perception
    adapters must convert vendor result objects before assigning it.
    """

    label: str
    score: float
    bbox_xyxy: tuple[float, float, float, float] | None = None
    mask: object | None = None
    instance_id: str | int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("label must be a non-empty string")
        object.__setattr__(self, "label", self.label.strip())

        score = _finite_float(self.score, field_name="score")
        if not 0.0 <= score <= 1.0:
            raise ValueError("score must be between 0.0 and 1.0")
        object.__setattr__(self, "score", score)

        if self.bbox_xyxy is not None:
            bbox = _finite_float_tuple(
                self.bbox_xyxy,
                field_name="bbox_xyxy",
                length=4,
            )
            if bbox[2] < bbox[0] or bbox[3] < bbox[1]:
                raise ValueError(
                    "bbox_xyxy must satisfy x_max >= x_min and y_max >= y_min"
                )
            object.__setattr__(self, "bbox_xyxy", bbox)

        if self.instance_id is not None and not isinstance(
            self.instance_id,
            (str, int),
        ):
            raise TypeError("instance_id must be a string or integer when set")
        if isinstance(self.instance_id, bool):
            raise TypeError("instance_id must not be a bool")
        object.__setattr__(
            self,
            "metadata",
            _freeze_mapping(self.metadata, field_name="metadata"),
        )

    def as_dict(self, *, include_mask: bool = False) -> dict[str, Any]:
        """Return a small serializable mapping, excluding mask data by default."""

        result: dict[str, Any] = {
            "label": self.label,
            "score": self.score,
            "bbox_xyxy": self.bbox_xyxy,
            "instance_id": self.instance_id,
            "has_mask": self.mask is not None,
            "metadata": dict(self.metadata),
        }
        if include_mask:
            result["mask"] = self.mask
        return result


@dataclass(frozen=True, slots=True)
class VisionResult:
    """Perception output tied to an optional observation sequence."""

    model: str
    image_size: tuple[int, int] | None = None
    instances: tuple[VisionInstance, ...] = ()
    elapsed_s: float = 0.0
    observation_sequence: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be a non-empty string")
        object.__setattr__(self, "model", self.model.strip())

        if self.image_size is not None:
            if (
                isinstance(self.image_size, (str, bytes, bytearray))
                or len(self.image_size) != 2
            ):
                raise ValueError("image_size must be a (width, height) pair")
            width, height = self.image_size
            if any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in (width, height)
            ):
                raise ValueError("image_size values must be integers > 0")
            object.__setattr__(self, "image_size", (width, height))

        instances = tuple(self.instances)
        if not all(isinstance(item, VisionInstance) for item in instances):
            raise TypeError("instances must contain only VisionInstance values")
        object.__setattr__(self, "instances", instances)

        elapsed_s = _finite_float(self.elapsed_s, field_name="elapsed_s")
        if elapsed_s < 0.0:
            raise ValueError("elapsed_s must be >= 0")
        object.__setattr__(self, "elapsed_s", elapsed_s)

        if self.observation_sequence is not None and (
            isinstance(self.observation_sequence, bool)
            or not isinstance(self.observation_sequence, int)
            or self.observation_sequence < 0
        ):
            raise ValueError("observation_sequence must be an integer >= 0 when set")
        object.__setattr__(
            self,
            "metadata",
            _freeze_mapping(self.metadata, field_name="metadata"),
        )

    def as_dict(self, *, include_masks: bool = False) -> dict[str, Any]:
        """Return a mapping suitable for a compact Agent tool result."""

        return {
            "model": self.model,
            "image_size": self.image_size,
            "instances": [
                instance.as_dict(include_mask=include_masks)
                for instance in self.instances
            ],
            "elapsed_s": self.elapsed_s,
            "observation_sequence": self.observation_sequence,
            "metadata": dict(self.metadata),
        }

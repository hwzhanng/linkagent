"""Metadata-driven perception implementation for deterministic tests."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from proto import Observation, VisionInstance, VisionResult

from ._base import BasePerceptionTool


_INSTANCE_FIELDS = {
    "label",
    "score",
    "bbox_xyxy",
    "mask",
    "instance_id",
    "metadata",
}


def _coerce_instance(value: object, *, index: int) -> VisionInstance:
    if isinstance(value, VisionInstance):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(
            f"detections[{index}] must be a VisionInstance or a mapping"
        )
    unknown = set(value) - _INSTANCE_FIELDS
    if unknown:
        names = ", ".join(sorted(str(name) for name in unknown))
        raise TypeError(f"unsupported detections[{index}] field(s): {names}")
    try:
        return VisionInstance(**dict(value))
    except (TypeError, ValueError) as exc:
        raise type(exc)(f"invalid detections[{index}]: {exc}") from exc


class MockPerceptionTool(BasePerceptionTool):
    """Read ``detections`` from an Observation's metadata.

    Call options are ``labels``, ``min_score`` (or ``threshold``), and
    ``max_instances``.  Filtering is deterministic and preserves source order.
    """

    def __init__(
        self,
        *,
        name: str = "mock_perception",
        metadata_key: str = "detections",
        model: str = "mock",
        fail: bool = False,
        failure_message: str = "simulated perception failure",
    ) -> None:
        super().__init__(name)
        if not isinstance(metadata_key, str) or not metadata_key.strip():
            raise ValueError("metadata_key must be a non-empty string")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        if not isinstance(fail, bool):
            raise TypeError("fail must be a bool")
        if not isinstance(failure_message, str) or not failure_message:
            raise ValueError("failure_message must be a non-empty string")
        self._metadata_key = metadata_key.strip()
        self._model = model.strip()
        self._fail = fail
        self._failure_message = failure_message

    def _run(
        self,
        observation: Observation,
        options: dict[str, Any],
    ) -> VisionResult:
        if self._fail:
            raise RuntimeError(self._failure_message)

        labels = options.pop("labels", None)
        min_score = options.pop("min_score", None)
        threshold = options.pop("threshold", None)
        max_instances = options.pop("max_instances", None)
        self._reject_options(options)

        label_filter = self._prepare_labels(labels)
        if min_score is not None and threshold is not None:
            raise TypeError("use only one of min_score and threshold")
        score_threshold = self._prepare_threshold(
            min_score if min_score is not None else threshold
        )
        max_count = self._prepare_max_instances(max_instances)

        raw_detections = observation.metadata.get(self._metadata_key, ())
        if isinstance(raw_detections, VisionResult):
            source_instances = raw_detections.instances
            image_size = raw_detections.image_size
            result_metadata = raw_detections.metadata
        else:
            if isinstance(raw_detections, (str, bytes, bytearray, Mapping)):
                raise TypeError(
                    f"Observation.metadata[{self._metadata_key!r}] must be an "
                    "iterable of detections"
                )
            try:
                source_instances = tuple(
                    _coerce_instance(item, index=index)
                    for index, item in enumerate(raw_detections)
                )
            except TypeError as exc:
                if "detections[" in str(exc):
                    raise
                raise TypeError(
                    f"Observation.metadata[{self._metadata_key!r}] must be an "
                    "iterable of detections"
                ) from exc
            image_size = observation.metadata.get("image_size")
            result_metadata = {}

        instances = tuple(
            instance
            for instance in source_instances
            if (label_filter is None or instance.label in label_filter)
            and instance.score >= score_threshold
        )
        if max_count is not None:
            instances = instances[:max_count]

        return VisionResult(
            model=self._model,
            image_size=image_size,
            instances=instances,
            observation_sequence=observation.sequence,
            metadata=result_metadata,
        )

    @staticmethod
    def _prepare_labels(value: object) -> frozenset[str] | None:
        if value is None:
            return None
        if isinstance(value, str):
            labels = (value,)
        elif isinstance(value, Iterable) and not isinstance(
            value,
            (bytes, bytearray, Mapping),
        ):
            labels = tuple(value)
        else:
            raise TypeError("labels must be a string, an iterable of strings, or None")
        if any(not isinstance(label, str) or not label.strip() for label in labels):
            raise ValueError("labels must contain non-empty strings")
        return frozenset(label.strip() for label in labels)

    @staticmethod
    def _prepare_threshold(value: object) -> float:
        if value is None:
            return 0.0
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("min_score/threshold must be a real number")
        threshold = float(value)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("min_score/threshold must be between 0.0 and 1.0")
        return threshold

    @staticmethod
    def _prepare_max_instances(value: object) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("max_instances must be an integer >= 0 or None")
        return value

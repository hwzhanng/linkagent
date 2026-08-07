"""Metadata-driven grasp planner used without robotics SDKs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from proto import GraspCandidate, Observation, Pose

from ._base import BaseManipulation


_CANDIDATE_FIELDS = {
    "pregrasp_tcp_pose",
    "grasp_tcp_pose",
    "score",
    "model_score",
    "forward_alignment",
    "metadata",
}
_SIMPLE_CANDIDATE_FIELDS = {
    "position_m",
    "quaternion_wxyz",
    "frame_id",
    "pregrasp_position_m",
    "pregrasp_quaternion_wxyz",
    "score",
    "model_score",
    "forward_alignment",
    "metadata",
}
_POSE_FIELDS = {"position_m", "quaternion_wxyz", "frame_id"}


def _coerce_pose(value: object, *, field_name: str) -> Pose:
    if isinstance(value, Pose):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a Pose or a mapping")
    unknown = set(value) - _POSE_FIELDS
    if unknown:
        names = ", ".join(sorted(str(name) for name in unknown))
        raise TypeError(f"unsupported {field_name} field(s): {names}")
    try:
        return Pose(**dict(value))
    except (TypeError, ValueError) as exc:
        raise type(exc)(f"invalid {field_name}: {exc}") from exc


def _coerce_candidate(value: object, *, index: int) -> GraspCandidate:
    if isinstance(value, GraspCandidate):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(
            f"grasp_candidates[{index}] must be a GraspCandidate or a mapping"
        )

    fields = set(value)
    if "grasp_tcp_pose" in value or "pregrasp_tcp_pose" in value:
        unknown = fields - _CANDIDATE_FIELDS
        if unknown:
            names = ", ".join(sorted(str(name) for name in unknown))
            raise TypeError(
                f"unsupported grasp_candidates[{index}] field(s): {names}"
            )
        prepared = dict(value)
        if "pregrasp_tcp_pose" not in prepared or "grasp_tcp_pose" not in prepared:
            raise ValueError(
                f"grasp_candidates[{index}] requires both pregrasp_tcp_pose "
                "and grasp_tcp_pose"
            )
        prepared["pregrasp_tcp_pose"] = _coerce_pose(
            prepared["pregrasp_tcp_pose"],
            field_name=f"grasp_candidates[{index}].pregrasp_tcp_pose",
        )
        prepared["grasp_tcp_pose"] = _coerce_pose(
            prepared["grasp_tcp_pose"],
            field_name=f"grasp_candidates[{index}].grasp_tcp_pose",
        )
    else:
        unknown = fields - _SIMPLE_CANDIDATE_FIELDS
        if unknown:
            names = ", ".join(sorted(str(name) for name in unknown))
            raise TypeError(
                f"unsupported grasp_candidates[{index}] field(s): {names}"
            )
        prepared = dict(value)
        missing = {
            field
            for field in ("position_m", "quaternion_wxyz", "frame_id", "score")
            if field not in prepared
        }
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(
                f"grasp_candidates[{index}] is missing required field(s): {names}"
            )
        grasp_pose = Pose(
            position_m=prepared.pop("position_m"),
            quaternion_wxyz=prepared.pop("quaternion_wxyz"),
            frame_id=prepared["frame_id"],
        )
        pregrasp_pose = Pose(
            position_m=prepared.pop(
                "pregrasp_position_m",
                grasp_pose.position_m,
            ),
            quaternion_wxyz=prepared.pop(
                "pregrasp_quaternion_wxyz",
                grasp_pose.quaternion_wxyz,
            ),
            frame_id=prepared.pop("frame_id"),
        )
        prepared["pregrasp_tcp_pose"] = pregrasp_pose
        prepared["grasp_tcp_pose"] = grasp_pose

    try:
        return GraspCandidate(**prepared)
    except (TypeError, ValueError) as exc:
        raise type(exc)(f"invalid grasp_candidates[{index}]: {exc}") from exc


class MockManipulation(BaseManipulation):
    """Read and score-rank ``grasp_candidates`` from Observation metadata."""

    def __init__(
        self,
        *,
        name: str = "mock_manipulation",
        metadata_key: str = "grasp_candidates",
        fail: bool = False,
        failure_message: str = "simulated manipulation failure",
    ) -> None:
        super().__init__(name)
        if not isinstance(metadata_key, str) or not metadata_key.strip():
            raise ValueError("metadata_key must be a non-empty string")
        if not isinstance(fail, bool):
            raise TypeError("fail must be a bool")
        if not isinstance(failure_message, str) or not failure_message:
            raise ValueError("failure_message must be a non-empty string")
        self._metadata_key = metadata_key.strip()
        self._fail = fail
        self._failure_message = failure_message

    def _rank_grasp_candidates(
        self,
        observation: Observation,
        options: dict[str, Any],
    ) -> tuple[GraspCandidate, ...]:
        if self._fail:
            raise RuntimeError(self._failure_message)
        min_score = options.pop("min_score", None)
        max_candidates = options.pop("max_candidates", None)
        self._reject_options(options)
        score_threshold = self._prepare_min_score(min_score)
        max_count = self._prepare_max_candidates(max_candidates)

        raw_candidates = observation.metadata.get(self._metadata_key, ())
        if isinstance(raw_candidates, (str, bytes, bytearray, Mapping)):
            raise TypeError(
                f"Observation.metadata[{self._metadata_key!r}] must be an "
                "iterable of grasp candidates"
            )
        try:
            candidates = tuple(
                _coerce_candidate(item, index=index)
                for index, item in enumerate(raw_candidates)
            )
        except TypeError as exc:
            if "grasp_candidates[" in str(exc):
                raise
            raise TypeError(
                f"Observation.metadata[{self._metadata_key!r}] must be an "
                "iterable of grasp candidates"
            ) from exc

        selected = tuple(
            sorted(
                (
                    candidate
                    for candidate in candidates
                    if candidate.score >= score_threshold
                ),
                key=lambda candidate: candidate.score,
                reverse=True,
            )
        )
        if max_count is not None:
            selected = selected[:max_count]
        return selected

    @staticmethod
    def _prepare_min_score(value: object) -> float:
        if value is None:
            return float("-inf")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("min_score must be a real number or None")
        score = float(value)
        if score != score or score in {float("inf"), float("-inf")}:
            raise ValueError("min_score must be finite")
        return score

    @staticmethod
    def _prepare_max_candidates(value: object) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("max_candidates must be an integer >= 0 or None")
        return value

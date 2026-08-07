"""Neutral pose and grasp-candidate DTOs.

Lengths and quaternion ordering are intentionally explicit in field names:
positions are metres and rotations use ``(w, x, y, z)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping

from ._base_proto import _finite_float, _finite_float_tuple, _freeze_mapping


@dataclass(frozen=True, slots=True)
class Pose:
    """A rigid pose in ``frame_id``, using metres and quaternion ``wxyz``."""

    position_m: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]
    frame_id: str

    def __post_init__(self) -> None:
        position_m = _finite_float_tuple(
            self.position_m,
            field_name="position_m",
            length=3,
        )
        quaternion_wxyz = _finite_float_tuple(
            self.quaternion_wxyz,
            field_name="quaternion_wxyz",
            length=4,
        )
        quaternion_norm = math.sqrt(sum(value * value for value in quaternion_wxyz))
        if not math.isclose(quaternion_norm, 1.0, rel_tol=1e-5, abs_tol=1e-6):
            raise ValueError("quaternion_wxyz must be normalized")
        if not isinstance(self.frame_id, str) or not self.frame_id.strip():
            raise ValueError("frame_id must be a non-empty string")
        object.__setattr__(self, "position_m", position_m)
        object.__setattr__(self, "quaternion_wxyz", quaternion_wxyz)
        object.__setattr__(self, "frame_id", self.frame_id.strip())

    def as_dict(self) -> dict[str, object]:
        """Return a serializable pose mapping with explicit units/order."""

        return {
            "position_m": self.position_m,
            "quaternion_wxyz": self.quaternion_wxyz,
            "frame_id": self.frame_id,
        }


@dataclass(frozen=True, slots=True)
class GraspCandidate:
    """One ranked TCP grasp and its pre-grasp pose.

    Both poses state their coordinate frame, use metre positions, and use
    ``quaternion_wxyz``.  Larger ``score`` values rank ahead of smaller values.
    """

    pregrasp_tcp_pose: Pose
    grasp_tcp_pose: Pose
    score: float
    model_score: float | None = None
    forward_alignment: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.pregrasp_tcp_pose, Pose):
            raise TypeError("pregrasp_tcp_pose must be a Pose")
        if not isinstance(self.grasp_tcp_pose, Pose):
            raise TypeError("grasp_tcp_pose must be a Pose")
        if self.pregrasp_tcp_pose.frame_id != self.grasp_tcp_pose.frame_id:
            raise ValueError("pregrasp and grasp poses must use the same frame_id")
        object.__setattr__(self, "score", _finite_float(self.score, field_name="score"))
        if self.model_score is not None:
            object.__setattr__(
                self,
                "model_score",
                _finite_float(self.model_score, field_name="model_score"),
            )
        if self.forward_alignment is not None:
            forward_alignment = _finite_float(
                self.forward_alignment,
                field_name="forward_alignment",
            )
            if not -1.0 <= forward_alignment <= 1.0:
                raise ValueError("forward_alignment must be between -1.0 and 1.0")
            object.__setattr__(self, "forward_alignment", forward_alignment)
        object.__setattr__(
            self,
            "metadata",
            _freeze_mapping(self.metadata, field_name="metadata"),
        )

    @property
    def position_m(self) -> tuple[float, float, float]:
        """Grasp TCP position in metres (convenience alias)."""

        return self.grasp_tcp_pose.position_m

    @property
    def quaternion_wxyz(self) -> tuple[float, float, float, float]:
        """Grasp TCP orientation in explicit ``wxyz`` order."""

        return self.grasp_tcp_pose.quaternion_wxyz

    @property
    def frame_id(self) -> str:
        """Coordinate frame shared by the pre-grasp and grasp poses."""

        return self.grasp_tcp_pose.frame_id

    def as_dict(self) -> dict[str, object]:
        """Return a serializable, unit-explicit candidate mapping."""

        return {
            "pregrasp_tcp_pose": self.pregrasp_tcp_pose.as_dict(),
            "grasp_tcp_pose": self.grasp_tcp_pose.as_dict(),
            "score": self.score,
            "model_score": self.model_score,
            "forward_alignment": self.forward_alignment,
            "metadata": dict(self.metadata),
        }

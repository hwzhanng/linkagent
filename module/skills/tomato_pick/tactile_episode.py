"""Small, replayable JSONL contract for G20 tactile episodes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Iterable

from proto import ArmSide, HandState, Pose, TactileFrame

from .tactile_matrix import TactileMatrixFeatureFrame


TACTILE_EPISODE_SCHEMA = "linkagent.g20_tactile_episode.v1"


def _non_empty(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _finite_tuple(value: Iterable[float], *, field_name: str, length: int) -> tuple[float, ...]:
    values = tuple(float(item) for item in value)
    if len(values) != length or not all(math.isfinite(item) for item in values):
        raise ValueError(f"{field_name} must contain {length} finite values")
    return values


@dataclass(frozen=True, slots=True)
class TactileEpisodeMetadata:
    episode_id: str
    side: ArmSide
    created_utc: str
    object_description: str
    calibration_id: str
    source_feature_extractor_id: str
    schema: str = TACTILE_EPISODE_SCHEMA

    def __post_init__(self) -> None:
        for field_name in (
            "episode_id",
            "created_utc",
            "object_description",
            "calibration_id",
            "source_feature_extractor_id",
        ):
            object.__setattr__(self, field_name, _non_empty(getattr(self, field_name), field_name=field_name))
        if not isinstance(self.side, ArmSide):
            object.__setattr__(self, "side", ArmSide(self.side))
        if self.schema != TACTILE_EPISODE_SCHEMA:
            raise ValueError(f"schema must be {TACTILE_EPISODE_SCHEMA}")


@dataclass(frozen=True, slots=True)
class TactileEpisodeSample:
    sample_monotonic_ns: int
    source_sequence: int
    phase: str
    hand_joint_position_rad: tuple[float, ...]
    hand_joint_velocity_rad_s: tuple[float, ...]
    tactile_matrix_raw: tuple[tuple[tuple[int, ...], ...], ...]
    tactile_matrix_valid: tuple[bool, ...]
    tactile_matrix_age_s: tuple[float | None, ...]
    object_pose_in_palm: dict[str, object] | None = None
    reference_force_n: tuple[float, ...] | None = None
    matrix_features_50d: tuple[float, ...] | None = None
    matrix_feature_extractor_id: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.sample_monotonic_ns, bool) or self.sample_monotonic_ns < 0:
            raise ValueError("sample_monotonic_ns must be an integer >= 0")
        if isinstance(self.source_sequence, bool) or self.source_sequence < 0:
            raise ValueError("source_sequence must be an integer >= 0")
        object.__setattr__(self, "phase", _non_empty(self.phase, field_name="phase"))
        object.__setattr__(self, "hand_joint_position_rad", _finite_tuple(
            self.hand_joint_position_rad, field_name="hand_joint_position_rad", length=16
        ))
        object.__setattr__(self, "hand_joint_velocity_rad_s", _finite_tuple(
            self.hand_joint_velocity_rad_s, field_name="hand_joint_velocity_rad_s", length=16
        ))
        matrices = tuple(tuple(tuple(row) for row in finger) for finger in self.tactile_matrix_raw)
        if len(matrices) != 5 or any(len(finger) != 12 or any(len(row) != 6 for row in finger) for finger in matrices):
            raise ValueError("tactile_matrix_raw must have shape 5x12x6")
        if any(isinstance(item, bool) or not isinstance(item, int) or not 0 <= item <= 255
               for finger in matrices for row in finger for item in row):
            raise ValueError("tactile_matrix_raw values must be uint8 integers")
        object.__setattr__(self, "tactile_matrix_raw", matrices)
        valid = tuple(self.tactile_matrix_valid)
        if len(valid) != 5 or any(not isinstance(item, bool) for item in valid):
            raise ValueError("tactile_matrix_valid must contain five bools")
        object.__setattr__(self, "tactile_matrix_valid", valid)
        ages = tuple(self.tactile_matrix_age_s)
        if len(ages) != 5 or any(item is not None and (not math.isfinite(item) or item < 0.0) for item in ages):
            raise ValueError("tactile_matrix_age_s must contain five non-negative values or nulls")
        object.__setattr__(self, "tactile_matrix_age_s", ages)
        if self.reference_force_n is not None:
            force = _finite_tuple(self.reference_force_n, field_name="reference_force_n", length=5)
            if any(item < 0.0 for item in force):
                raise ValueError("reference_force_n values must be non-negative")
            object.__setattr__(self, "reference_force_n", force)
        if self.matrix_features_50d is not None:
            object.__setattr__(self, "matrix_features_50d", _finite_tuple(
                self.matrix_features_50d, field_name="matrix_features_50d", length=50
            ))
            object.__setattr__(self, "matrix_feature_extractor_id", _non_empty(
                self.matrix_feature_extractor_id, field_name="matrix_feature_extractor_id"
            ))
        elif self.matrix_feature_extractor_id is not None:
            raise ValueError("matrix_feature_extractor_id requires matrix_features_50d")

    @classmethod
    def capture(
        cls,
        *,
        hand: HandState,
        tactile: TactileFrame,
        now_ns: int,
        phase: str,
        object_pose_in_palm: Pose | None = None,
        reference_force_n: tuple[float, ...] | None = None,
        matrix_features: TactileMatrixFeatureFrame | None = None,
    ) -> "TactileEpisodeSample":
        if hand.side is not tactile.side:
            raise ValueError("hand and tactile frame must use the same side")
        if isinstance(now_ns, bool) or not isinstance(now_ns, int) or now_ns < 0:
            raise ValueError("now_ns must be an integer >= 0")
        matrices = tuple(
            finger.matrix_weight_raw
            if finger.matrix_weight_raw is not None
            else tuple(tuple(0 for _ in range(6)) for _ in range(12))
            for finger in tactile.fingers
        )
        ages = tuple(
            (now_ns - finger.matrix_sample_monotonic_ns) / 1.0e9
            if finger.matrix_sample_monotonic_ns is not None and now_ns >= finger.matrix_sample_monotonic_ns
            else None
            for finger in tactile.fingers
        )
        if matrix_features is not None and matrix_features.source_sequence != tactile.source_sequence:
            raise ValueError("matrix features must come from the same tactile source sequence")
        return cls(
            sample_monotonic_ns=now_ns,
            source_sequence=tactile.source_sequence,
            phase=phase,
            hand_joint_position_rad=hand.joint_positions_rad,
            hand_joint_velocity_rad_s=hand.joint_velocities_rad_s,
            tactile_matrix_raw=matrices,
            tactile_matrix_valid=tuple(finger.matrix_valid for finger in tactile.fingers),
            tactile_matrix_age_s=ages,
            object_pose_in_palm=object_pose_in_palm.as_dict() if object_pose_in_palm is not None else None,
            reference_force_n=reference_force_n,
            matrix_features_50d=matrix_features.values if matrix_features is not None else None,
            matrix_feature_extractor_id=(
                matrix_features.feature_extractor_id if matrix_features is not None else None
            ),
        )


def write_tactile_episode_jsonl(
    path: str | Path,
    metadata: TactileEpisodeMetadata,
    samples: Iterable[TactileEpisodeSample],
) -> Path:
    """Write one metadata line followed by ordered sample lines."""

    target = Path(path)
    records = tuple(samples)
    if not records:
        raise ValueError("an episode requires at least one sample")
    if any(current.sample_monotonic_ns <= previous.sample_monotonic_ns
           for previous, current in zip(records, records[1:])):
        raise ValueError("episode sample times must be strictly increasing")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as stream:
        metadata_payload = asdict(metadata)
        metadata_payload["side"] = metadata.side.value
        stream.write(json.dumps({"type": "metadata", **metadata_payload}, ensure_ascii=False) + "\n")
        for sample in records:
            stream.write(json.dumps({"type": "sample", **asdict(sample)}, ensure_ascii=False) + "\n")
    return target


def read_tactile_episode_jsonl(
    path: str | Path,
) -> tuple[TactileEpisodeMetadata, tuple[TactileEpisodeSample, ...]]:
    """Read the same small contract for deterministic offline replay."""

    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError("tactile episode is empty")
    header = json.loads(lines[0])
    if header.pop("type", None) != "metadata":
        raise ValueError("first JSONL record must be metadata")
    metadata = TactileEpisodeMetadata(**header)
    samples: list[TactileEpisodeSample] = []
    for line in lines[1:]:
        payload = json.loads(line)
        if payload.pop("type", None) != "sample":
            raise ValueError("non-header JSONL records must be samples")
        samples.append(TactileEpisodeSample(**payload))
    if not samples:
        raise ValueError("tactile episode contains no samples")
    return metadata, tuple(samples)


__all__ = [
    "TACTILE_EPISODE_SCHEMA",
    "TactileEpisodeMetadata",
    "TactileEpisodeSample",
    "read_tactile_episode_jsonl",
    "write_tactile_episode_jsonl",
]

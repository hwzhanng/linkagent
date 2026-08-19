"""Strict in-memory contract for future left-hand tactile calibration episodes."""

from __future__ import annotations

from typing import Any, Mapping

import torch


TACTILE_DATASET_SCHEMA = "left_g20_tactile_episode_v1"
PHASE_NAMES = (
    "no_contact",
    "approach",
    "gentle_contact",
    "stable_hold",
    "incipient_slip",
    "release",
)

FIELD_WIDTHS = {
    "joint_position_rad": (16,),
    "joint_velocity_rad_s": (16,),
    "tactile_raw": (5, 12, 6),
    "tactile_valid": (5,),
    "sample_age_s": (5,),
    "object_pose_in_palm_xyzw": (7,),
    "object_pose_valid": (),
    "reference_force_n": (6,),
    "reference_force_valid": (6,),
    "phase_code": (),
}


def validate_metadata(metadata: Mapping[str, Any]) -> None:
    required = {
        "schema",
        "episode_id",
        "hand",
        "timestamp_source",
        "tactile_unit",
        "joint_position_unit",
        "joint_velocity_unit",
        "reference_force_unit",
        "object_description",
        "operator_supervised",
    }
    if set(metadata) != required:
        raise ValueError(
            f"metadata mismatch: missing={sorted(required - set(metadata))}, "
            f"extra={sorted(set(metadata) - required)}"
        )
    if metadata["schema"] != TACTILE_DATASET_SCHEMA or metadata["hand"] != "left_g20":
        raise ValueError("wrong tactile schema or hand")
    if metadata["joint_position_unit"] != "rad" or metadata["joint_velocity_unit"] != "rad/s":
        raise ValueError("joint units must be SI radians")
    if metadata["reference_force_unit"] != "N":
        raise ValueError("reference force unit must be N")
    if not isinstance(metadata["operator_supervised"], bool):
        raise ValueError("operator_supervised must be boolean")
    for name in ("episode_id", "timestamp_source", "tactile_unit", "object_description"):
        if not isinstance(metadata[name], str) or not metadata[name].strip():
            raise ValueError(f"metadata {name} must be a nonempty string")


def validate_episode(fields: Mapping[str, torch.Tensor], metadata: Mapping[str, Any]) -> None:
    """Fail closed on shape, dtype, units, timestamps, and validity-mask drift."""

    validate_metadata(metadata)
    expected = {"timestamp_s", *FIELD_WIDTHS}
    if set(fields) != expected:
        raise ValueError(
            f"episode fields mismatch: missing={sorted(expected - set(fields))}, "
            f"extra={sorted(set(fields) - expected)}"
        )
    timestamp = fields["timestamp_s"]
    if not isinstance(timestamp, torch.Tensor) or timestamp.ndim != 1 or timestamp.numel() < 2:
        raise ValueError("timestamp_s must be a one-dimensional tensor with at least two samples")
    count = timestamp.numel()
    if not torch.isfinite(timestamp).all() or not bool((timestamp[1:] > timestamp[:-1]).all()):
        raise ValueError("timestamps must be finite and strictly increasing")
    for name, width in FIELD_WIDTHS.items():
        value = fields[name]
        if not isinstance(value, torch.Tensor) or value.shape != (count, *width):
            raise ValueError(f"{name} must have shape {(count, *width)}")
        if name not in ("tactile_valid", "object_pose_valid", "reference_force_valid"):
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} contains NaN or Inf")
    for name in ("tactile_valid", "object_pose_valid", "reference_force_valid"):
        if fields[name].dtype != torch.bool:
            raise ValueError(f"{name} must be boolean")
    if fields["phase_code"].dtype not in (torch.int32, torch.int64):
        raise ValueError("phase_code must be an integer tensor")
    if bool(((fields["phase_code"] < 0) | (fields["phase_code"] >= len(PHASE_NAMES))).any()):
        raise ValueError("phase_code is outside the frozen vocabulary")
    if bool((fields["tactile_raw"] < 0.0).any()):
        raise ValueError("raw tactile values must be nonnegative")
    if bool((fields["sample_age_s"] < 0.0).any()):
        raise ValueError("sample age must be nonnegative")
    if bool((fields["reference_force_n"] < 0.0).any()):
        raise ValueError("reference force must be nonnegative")
    pose_valid = fields["object_pose_valid"]
    if pose_valid.any():
        quaternion = fields["object_pose_in_palm_xyzw"][pose_valid, 3:]
        norm = torch.linalg.vector_norm(quaternion, dim=1)
        if bool(((norm - 1.0).abs() > 0.02).any()):
            raise ValueError("valid object quaternions must have norm within 2% of one")


def summarize_episode(
    fields: Mapping[str, torch.Tensor], metadata: Mapping[str, Any]
) -> dict[str, Any]:
    validate_episode(fields, metadata)
    timestamp = fields["timestamp_s"]
    dt = timestamp[1:] - timestamp[:-1]
    phase_counts = {
        name: int((fields["phase_code"] == index).sum())
        for index, name in enumerate(PHASE_NAMES)
    }
    return {
        "schema": TACTILE_DATASET_SCHEMA,
        "episode_id": metadata["episode_id"],
        "samples": timestamp.numel(),
        "duration_s": float(timestamp[-1] - timestamp[0]),
        "median_sample_rate_hz": float(1.0 / dt.median()),
        "maximum_inter_sample_gap_s": float(dt.max()),
        "tactile_valid_fraction_by_finger": fields["tactile_valid"].float().mean(dim=0).tolist(),
        "reference_force_valid_fraction": float(fields["reference_force_valid"].float().mean()),
        "object_pose_valid_fraction": float(fields["object_pose_valid"].float().mean()),
        "phase_counts": phase_counts,
        "hardware_release_allowed": False,
    }


__all__ = [
    "FIELD_WIDTHS",
    "PHASE_NAMES",
    "TACTILE_DATASET_SCHEMA",
    "summarize_episode",
    "validate_episode",
    "validate_metadata",
]

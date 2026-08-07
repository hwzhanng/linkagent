"""Kinematics recovered from the Linkbot A7_V2 controller.

The vendor ``lbot_api_server`` uses the URDF chain from ``base_link`` to
``L8_Link``/``R8_Link``, KDL for FK, and TRAC-IK (Distance mode) for IK.  This
module reproduces the observable conventions on x86_64 without loading the
AArch64 executable:

* joints are radians;
* positions are metres;
* orientations are roll/pitch/yaw radians;
* FK applies ``raw * base_offset * flange_offset``;
* IK removes those offsets in the reverse order.

The numerical IK fallback uses SciPy's bounded least-squares solver.  It is
not represented as vendor source code, but is checked against the live
controller's FK service and preserves the same seed and limit semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import warnings

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


JOINT_COUNT = 7


@dataclass(frozen=True, slots=True)
class Pose:
    position: tuple[float, float, float]
    rpy: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class IKResult:
    joints: tuple[float, ...]
    success: bool
    position_error: float
    rotation_error: float
    iterations: int


@dataclass(frozen=True, slots=True)
class JointGeometry:
    origin: tuple[float, float, float]
    axis: tuple[float, float, float]


# Exact values from the Pi's installed
# lkls73_i1_dual_arm_description.urdf.  Joint 8 is fixed and is included as
# the final origin with a zero axis.
_GEOMETRY = {
    "left": (
        JointGeometry((0.0, 0.164, 0.0), (0.0, 1.0, 0.0)),
        JointGeometry((0.0, 0.058, 0.0), (1.0, 0.0, 0.0)),
        JointGeometry((-2.5018e-5, 0.0, -0.123), (0.0, 0.0, -1.0)),
        JointGeometry((2.4267e-5, 0.0, -0.1621), (0.0, 1.0, 0.0)),
        JointGeometry((0.0, 0.0, -0.124), (0.0, 0.0, -1.0)),
        JointGeometry((0.0, 0.0, -0.0881), (0.0, 1.0, 0.0)),
        JointGeometry((-2.6563e-5, 0.0, -0.0655), (1.0, 0.0, 0.0)),
        JointGeometry((2.5732e-5, 0.0, -0.0455), (0.0, 0.0, 0.0)),
    ),
    "right": (
        JointGeometry((3.32e-5, -0.164, 0.0), (0.0, -1.0, 0.0)),
        JointGeometry((-3.32e-5, -0.058001, 0.0), (1.0, 0.0, 0.0)),
        JointGeometry((2.5018e-5, 0.0, -0.123), (0.0, 0.0, -1.0)),
        JointGeometry((-2.5308e-5, 0.0, -0.1621), (0.0, 1.0, 0.0)),
        JointGeometry((0.0, 0.0, -0.124), (0.0, 0.0, -1.0)),
        JointGeometry((0.0, 0.0, -0.0881), (0.0, 1.0, 0.0)),
        JointGeometry((2.5631e-5, 0.0, -0.0655), (1.0, 0.0, 0.0)),
        JointGeometry((-2.5732e-5, 0.0, -0.0455), (0.0, 0.0, 0.0)),
    ),
}


def _translation(xyz: tuple[float, float, float]) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, 3] = xyz
    return transform


def _rotation(axis: tuple[float, float, float], angle: float) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    vector = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm:
        transform[:3, :3] = Rotation.from_rotvec(vector / norm * angle).as_matrix()
    return transform


def pose_matrix(position: list[float] | tuple[float, ...], rpy: list[float] | tuple[float, ...]) -> np.ndarray:
    if len(position) != 3 or len(rpy) != 3:
        raise ValueError("position and rpy must each contain three values")
    values = [float(value) for value in (*position, *rpy)]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("pose values must be finite")
    roll, pitch, yaw = values[3:]
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = Rotation.from_euler("ZYX", [yaw, pitch, roll]).as_matrix()
    transform[:3, 3] = values[:3]
    return transform


def offset_matrix(offset: list[float] | tuple[float, ...]) -> np.ndarray:
    """Convert vendor offset order ``r,p,y,x,y,z`` to a transform."""
    if len(offset) != 6:
        raise ValueError("offset must contain roll,pitch,yaw,x,y,z")
    roll, pitch, yaw, x, y, z = (float(value) for value in offset)
    return pose_matrix((x, y, z), (roll, pitch, yaw))


def matrix_pose(transform: np.ndarray) -> Pose:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        yaw, pitch, roll = Rotation.from_matrix(transform[:3, :3]).as_euler("ZYX")
    return Pose(
        tuple(float(value) for value in transform[:3, 3]),
        (float(roll), float(pitch), float(yaw)),
    )


class ArmKinematics:
    def __init__(
        self,
        arm: str,
        *,
        config_path: str | Path | None = None,
        base_offset: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        flange_offset: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    ) -> None:
        if arm not in _GEOMETRY:
            raise ValueError("arm must be left or right")
        self.arm = arm
        self.geometry = _GEOMETRY[arm]
        path = Path(config_path) if config_path else Path(__file__).with_name(f"a7_v2_{arm}.json")
        with path.open(encoding="utf-8") as stream:
            config = json.load(stream)
        limits = config["joint_limits"]["current"]
        self.lower = np.asarray(limits["lower"], dtype=float)
        self.upper = np.asarray(limits["upper"], dtype=float)
        self.base_offset = offset_matrix(base_offset)
        self.flange_offset = offset_matrix(flange_offset)

    @property
    def joint_limits(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        return tuple(self.lower), tuple(self.upper)

    def validate_joints(self, joints: list[float] | tuple[float, ...], *, enforce_limits: bool = False) -> np.ndarray:
        if len(joints) != JOINT_COUNT:
            raise ValueError("joints must contain exactly seven radians")
        values = np.asarray(joints, dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("joint values must be finite")
        if enforce_limits and (np.any(values < self.lower) or np.any(values > self.upper)):
            raise ValueError("joint target exceeds the configured software limits")
        return values

    def raw_forward_matrix(self, joints: list[float] | tuple[float, ...]) -> np.ndarray:
        values = self.validate_joints(joints)
        transform = np.eye(4, dtype=float)
        for index, item in enumerate(self.geometry):
            transform = transform @ _translation(item.origin)
            if index < JOINT_COUNT:
                transform = transform @ _rotation(item.axis, float(values[index]))
        return transform

    def forward_matrix(self, joints: list[float] | tuple[float, ...]) -> np.ndarray:
        return self.raw_forward_matrix(joints) @ self.base_offset @ self.flange_offset

    def forward(self, joints: list[float] | tuple[float, ...]) -> Pose:
        return matrix_pose(self.forward_matrix(joints))

    def inverse(
        self,
        position: list[float] | tuple[float, ...],
        rpy: list[float] | tuple[float, ...],
        seed: list[float] | tuple[float, ...],
        *,
        max_nfev: int = 350,
        position_tolerance: float = 2.0e-5,
        rotation_tolerance: float = 2.0e-4,
    ) -> IKResult:
        seed_values = np.clip(self.validate_joints(seed), self.lower, self.upper)
        requested = pose_matrix(position, rpy)
        # Exact inverse of the vendor FK's post-multiplication order.
        target = requested @ np.linalg.inv(self.flange_offset) @ np.linalg.inv(self.base_offset)
        target_rotation = Rotation.from_matrix(target[:3, :3])

        def residual(values: np.ndarray) -> np.ndarray:
            current = self.raw_forward_matrix(values)
            position_error = current[:3, 3] - target[:3, 3]
            current_rotation = Rotation.from_matrix(current[:3, :3])
            rotation_error = (current_rotation.inv() * target_rotation).as_rotvec()
            # A very small seed term selects the nearby redundant-arm solution,
            # corresponding to TRAC-IK Distance mode, without compromising pose.
            seed_error = (values - seed_values) * 1.0e-7
            return np.concatenate((position_error, rotation_error, seed_error))

        starts = [
            seed_values,
            np.clip((seed_values + (self.lower + self.upper) * 0.5) * 0.5, self.lower, self.upper),
            (self.lower + self.upper) * 0.5,
        ]
        best = None
        total_evaluations = 0
        for start in starts:
            result = least_squares(
                residual,
                start,
                bounds=(self.lower, self.upper),
                xtol=1.0e-11,
                ftol=1.0e-11,
                gtol=1.0e-11,
                max_nfev=max_nfev,
            )
            total_evaluations += int(result.nfev)
            transform = self.raw_forward_matrix(result.x)
            p_error = float(np.linalg.norm(transform[:3, 3] - target[:3, 3]))
            r_error = float(
                np.linalg.norm(
                    (Rotation.from_matrix(transform[:3, :3]).inv() * target_rotation).as_rotvec()
                )
            )
            candidate = (p_error + r_error, result.x, p_error, r_error)
            if best is None or candidate[0] < best[0]:
                best = candidate
            if p_error <= position_tolerance and r_error <= rotation_tolerance:
                break

        assert best is not None
        _, joints, p_error, r_error = best
        success = p_error <= position_tolerance and r_error <= rotation_tolerance
        return IKResult(
            tuple(float(value) for value in joints),
            success,
            p_error,
            r_error,
            total_evaluations,
        )

"""Trajectory generation compatible with the recovered Linkbot API server."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .kinematics import ArmKinematics, Pose


DEFAULT_SPEED = 0.5
DEFAULT_ACCELERATION = 0.5
OFFICIAL_SAMPLE_PERIOD = 0.01  # vendor node constructs rclcpp::Rate(100.0)
MOVEL_EQUIVALENT_RADIUS = 0.01  # recovered KDL::Path_Line argument, metres/radian


class TrajectoryError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class JointSample:
    time: float
    positions: tuple[float, ...]
    velocities: tuple[float, ...]
    accelerations: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class _TrapProfile:
    start: float
    end: float
    max_velocity: float
    max_acceleration: float
    acceleration_time: float
    cruise_time: float
    duration: float

    @classmethod
    def create(cls, start: float, end: float, speed: float, acceleration: float) -> "_TrapProfile":
        speed = speed if speed > 0.0 else DEFAULT_SPEED
        acceleration = acceleration if acceleration > 0.0 else DEFAULT_ACCELERATION
        distance = abs(end - start)
        if distance == 0.0:
            return cls(start, end, speed, acceleration, 0.0, 0.0, 0.0)
        if distance < speed * speed / acceleration:
            acceleration_time = math.sqrt(distance / acceleration)
            cruise_time = 0.0
        else:
            acceleration_time = speed / acceleration
            cruise_time = (distance - speed * speed / acceleration) / speed
        return cls(
            start,
            end,
            speed,
            acceleration,
            acceleration_time,
            cruise_time,
            2.0 * acceleration_time + cruise_time,
        )

    def evaluate(self, time_value: float) -> tuple[float, float, float]:
        distance = abs(self.end - self.start)
        if distance == 0.0 or time_value >= self.duration:
            return self.end, 0.0, 0.0
        direction = 1.0 if self.end > self.start else -1.0
        time_value = max(0.0, time_value)
        peak_velocity = self.max_acceleration * self.acceleration_time
        if time_value <= self.acceleration_time:
            travelled = 0.5 * self.max_acceleration * time_value * time_value
            return (
                self.start + direction * travelled,
                direction * self.max_acceleration * time_value,
                direction * self.max_acceleration,
            )
        cruise_end = self.acceleration_time + self.cruise_time
        acceleration_distance = 0.5 * self.max_acceleration * self.acceleration_time**2
        if time_value <= cruise_end:
            travelled = acceleration_distance + peak_velocity * (time_value - self.acceleration_time)
            return self.start + direction * travelled, direction * peak_velocity, 0.0
        remaining = self.duration - time_value
        travelled = distance - 0.5 * self.max_acceleration * remaining * remaining
        return (
            self.start + direction * travelled,
            direction * self.max_acceleration * remaining,
            -direction * self.max_acceleration,
        )


def _sample_times(duration: float, period: float) -> list[float]:
    if period <= 0.0 or not math.isfinite(period):
        raise ValueError("sample period must be finite and positive")
    if duration <= 0.0:
        return [0.0]
    count = int(math.floor(duration / period))
    result = [index * period for index in range(count + 1)]
    if duration - result[-1] > 1.0e-12:
        result.append(duration)
    else:
        result[-1] = duration
    return result


def sample_movej(
    start: Iterable[float],
    target: Iterable[float],
    speed: float,
    acceleration: float,
    *,
    period: float = OFFICIAL_SAMPLE_PERIOD,
) -> list[JointSample]:
    """Generate the vendor's independent per-joint trapezoids at 100 Hz.

    Shorter joints reach their goals first and hold while the longest joint
    finishes.  An exact final target is appended when the duration does not
    land on a sample boundary; this removes the vendor implementation's
    possible final sub-sample truncation.
    """
    start_values = tuple(float(value) for value in start)
    target_values = tuple(float(value) for value in target)
    if len(start_values) != 7 or len(target_values) != 7:
        raise ValueError("MoveJ requires seven start and target joints")
    if not all(math.isfinite(value) for value in (*start_values, *target_values, speed, acceleration)):
        raise ValueError("trajectory inputs must be finite")
    profiles = [
        _TrapProfile.create(begin, end, speed, acceleration)
        for begin, end in zip(start_values, target_values, strict=True)
    ]
    duration = max(profile.duration for profile in profiles)
    samples: list[JointSample] = []
    for time_value in _sample_times(duration, period):
        evaluated = [profile.evaluate(time_value) for profile in profiles]
        samples.append(
            JointSample(
                time_value,
                tuple(value[0] for value in evaluated),
                tuple(value[1] for value in evaluated),
                tuple(value[2] for value in evaluated),
            )
        )
    return samples


def sample_smooth_movej(
    start: Iterable[float],
    target: Iterable[float],
    duration: float,
    *,
    period: float = OFFICIAL_SAMPLE_PERIOD,
) -> list[JointSample]:
    """Quintic zero-velocity/zero-acceleration joint trajectory.

    This is provided for recorded/VLA trajectories that need lower jerk than
    the official trapezoid.  It is deliberately separate from ``MoveJ`` so
    compatibility calls retain vendor timing semantics.
    """
    begin = np.asarray(tuple(start), dtype=float)
    end = np.asarray(tuple(target), dtype=float)
    if begin.shape != (7,) or end.shape != (7,):
        raise ValueError("smooth MoveJ requires seven start and target joints")
    if duration <= 0.0 or not math.isfinite(duration):
        raise ValueError("duration must be finite and positive")
    delta = end - begin
    result: list[JointSample] = []
    for time_value in _sample_times(duration, period):
        u = min(1.0, time_value / duration)
        blend = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
        blend_velocity = (30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4) / duration
        blend_acceleration = (60.0 * u - 180.0 * u**2 + 120.0 * u**3) / duration**2
        result.append(
            JointSample(
                time_value,
                tuple(begin + delta * blend),
                tuple(delta * blend_velocity),
                tuple(delta * blend_acceleration),
            )
        )
    return result


def sample_movel(
    kinematics: ArmKinematics,
    start_joints: Iterable[float],
    target_pose: Pose,
    speed: float,
    acceleration: float,
    *,
    period: float = OFFICIAL_SAMPLE_PERIOD,
) -> list[JointSample]:
    """Reproduce KDL Path_Line + Trap sampling and solve IK sequentially."""
    seed = tuple(float(value) for value in start_joints)
    start_pose = kinematics.forward(seed)
    start_position = np.asarray(start_pose.position)
    end_position = np.asarray(target_pose.position)
    start_rotation = Rotation.from_euler(
        "ZYX", [start_pose.rpy[2], start_pose.rpy[1], start_pose.rpy[0]]
    )
    end_rotation = Rotation.from_euler(
        "ZYX", [target_pose.rpy[2], target_pose.rpy[1], target_pose.rpy[0]]
    )
    angular_distance = float(np.linalg.norm((start_rotation.inv() * end_rotation).as_rotvec()))
    linear_distance = float(np.linalg.norm(end_position - start_position))
    # KDL Path_Line uses the larger normalized motion component.
    path_length = max(linear_distance, MOVEL_EQUIVALENT_RADIUS * angular_distance)
    if path_length <= 1.0e-12:
        return [JointSample(0.0, seed, (0.0,) * 7, (0.0,) * 7)]
    profile = _TrapProfile.create(0.0, path_length, speed, acceleration)
    key_rotations = Rotation.concatenate([start_rotation, end_rotation])
    slerp = Slerp([0.0, 1.0], key_rotations)
    result: list[JointSample] = []
    previous = np.asarray(seed, dtype=float)
    previous_velocity = np.zeros(7, dtype=float)
    previous_time = 0.0
    for time_value in _sample_times(profile.duration, period):
        path_position, _, _ = profile.evaluate(time_value)
        fraction = min(1.0, max(0.0, path_position / path_length))
        position = start_position + (end_position - start_position) * fraction
        rotation = slerp([fraction])[0]
        yaw, pitch, roll = rotation.as_euler("ZYX")
        solution = kinematics.inverse(position, (roll, pitch, yaw), previous)
        if not solution.success:
            raise TrajectoryError(
                f"IK failed at t={time_value:.3f}s: "
                f"position_error={solution.position_error:.3g}, "
                f"rotation_error={solution.rotation_error:.3g}"
            )
        current = np.asarray(solution.joints)
        delta_time = time_value - previous_time
        if delta_time > 0.0:
            velocity = (current - previous) / delta_time
            acceleration_values = (velocity - previous_velocity) / delta_time
        else:
            velocity = np.zeros(7, dtype=float)
            acceleration_values = np.zeros(7, dtype=float)
        result.append(
            JointSample(
                time_value,
                tuple(current),
                tuple(velocity),
                tuple(acceleration_values),
            )
        )
        previous = current
        previous_velocity = velocity
        previous_time = time_value
    return result

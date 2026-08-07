"""Small dependency-free quaternion helpers for grasp-frame residual control."""

from __future__ import annotations

import math

from proto import Pose


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def quaternion_conjugate(q: tuple[float, ...]) -> tuple[float, float, float, float]:
    return (q[0], -q[1], -q[2], -q[3])


def quaternion_multiply(
    left: tuple[float, ...],
    right: tuple[float, ...],
) -> tuple[float, float, float, float]:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    result = (
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    )
    norm = math.sqrt(sum(value * value for value in result))
    if norm <= 1e-12:
        raise ValueError("quaternion multiplication produced zero norm")
    return tuple(value / norm for value in result)  # type: ignore[return-value]


def rotate_vector(
    quaternion_wxyz: tuple[float, ...],
    vector_xyz: tuple[float, ...],
) -> tuple[float, float, float]:
    """Rotate a vector without normalizing away its magnitude."""

    w, x, y, z = quaternion_wxyz
    vx, vy, vz = vector_xyz
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def rotation_vector_to_quaternion(
    rotation_vector: tuple[float, ...],
) -> tuple[float, float, float, float]:
    """Convert an exponential-coordinate rotation vector to a unit quaternion."""

    if len(rotation_vector) != 3:
        raise ValueError("rotation_vector must contain xyz")
    angle = math.sqrt(sum(value * value for value in rotation_vector))
    if angle < 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    scale = math.sin(0.5 * angle) / angle
    return (
        math.cos(0.5 * angle),
        rotation_vector[0] * scale,
        rotation_vector[1] * scale,
        rotation_vector[2] * scale,
    )


def quaternion_to_rotvec(q: tuple[float, ...]) -> tuple[float, float, float]:
    w, x, y, z = q
    if w < 0.0:
        w, x, y, z = -w, -x, -y, -z
    w = clamp(w, -1.0, 1.0)
    angle = 2.0 * math.acos(w)
    sine = math.sqrt(max(0.0, 1.0 - w * w))
    if sine < 1e-8:
        return (2.0 * x, 2.0 * y, 2.0 * z)
    scale = angle / sine
    return (x * scale, y * scale, z * scale)


def pose_error_in_grasp_frame(
    *,
    reference: Pose,
    current: Pose,
    grasp_frame: Pose,
) -> tuple[float, ...]:
    if len({reference.frame_id, current.frame_id, grasp_frame.frame_id}) != 1:
        raise ValueError("reference, current, and grasp_frame must share frame_id")
    world_delta = tuple(
        current_value - reference_value
        for current_value, reference_value in zip(
            current.position_m,
            reference.position_m,
        )
    )
    local_delta = rotate_vector(
        quaternion_conjugate(grasp_frame.quaternion_wxyz),
        world_delta,
    )
    # Spatial rotation error: current * reference^-1, then rotate the vector
    # into the frozen grasp axes.  Isaac packs the same convention.
    orientation_delta = quaternion_multiply(
        current.quaternion_wxyz,
        quaternion_conjugate(reference.quaternion_wxyz),
    )
    local_rotation = rotate_vector(
        quaternion_conjugate(grasp_frame.quaternion_wxyz),
        quaternion_to_rotvec(orientation_delta),
    )
    return (*local_delta, *local_rotation)


def twist_in_grasp_frame(
    twist: tuple[float, ...],
    *,
    grasp_frame: Pose,
) -> tuple[float, ...]:
    """Rotate a spatial EE twist from ``Pose.frame_id`` into grasp axes."""

    if len(twist) != 6:
        raise ValueError("twist must contain linear xyz and angular xyz")
    inverse = quaternion_conjugate(grasp_frame.quaternion_wxyz)
    return (
        *rotate_vector(inverse, tuple(twist[:3])),
        *rotate_vector(inverse, tuple(twist[3:])),
    )


def apply_grasp_frame_delta(
    *,
    current: Pose,
    grasp_frame: Pose,
    delta_xyz: tuple[float, ...],
    delta_rotation_vector: tuple[float, ...],
) -> Pose:
    if current.frame_id != grasp_frame.frame_id:
        raise ValueError("current and grasp_frame must share frame_id")
    world_delta = rotate_vector(grasp_frame.quaternion_wxyz, delta_xyz)
    position = tuple(
        current_value + delta
        for current_value, delta in zip(current.position_m, world_delta)
    )
    world_rotation_vector = rotate_vector(
        grasp_frame.quaternion_wxyz,
        delta_rotation_vector,
    )
    orientation_delta = rotation_vector_to_quaternion(world_rotation_vector)
    orientation = quaternion_multiply(orientation_delta, current.quaternion_wxyz)
    return Pose(position, orientation, current.frame_id)


def clamp_pose_to_plan_workspace(
    *,
    pose: Pose,
    contact_pose: Pose,
    grasp_frame: Pose,
    limits_m: tuple[float, ...],
    rotation_limits_rad: tuple[float, ...],
) -> Pose:
    world_delta = tuple(
        value - origin for value, origin in zip(pose.position_m, contact_pose.position_m)
    )
    local_delta = rotate_vector(
        quaternion_conjugate(grasp_frame.quaternion_wxyz),
        world_delta,
    )
    clamped_local = tuple(
        clamp(value, -limit, limit)
        for value, limit in zip(local_delta, limits_m)
    )
    clamped_world = rotate_vector(grasp_frame.quaternion_wxyz, clamped_local)
    position = tuple(
        origin + delta for origin, delta in zip(contact_pose.position_m, clamped_world)
    )
    rotation_error = pose_error_in_grasp_frame(
        reference=contact_pose,
        current=pose,
        grasp_frame=grasp_frame,
    )[3:6]
    clamped_rotation_g = tuple(
        clamp(value, -limit, limit)
        for value, limit in zip(rotation_error, rotation_limits_rad)
    )
    clamped_rotation_w = rotate_vector(
        grasp_frame.quaternion_wxyz,
        clamped_rotation_g,
    )
    orientation = quaternion_multiply(
        rotation_vector_to_quaternion(clamped_rotation_w),
        contact_pose.quaternion_wxyz,
    )
    return Pose(position, orientation, pose.frame_id)


__all__ = [
    "apply_grasp_frame_delta",
    "clamp",
    "clamp_pose_to_plan_workspace",
    "pose_error_in_grasp_frame",
    "quaternion_conjugate",
    "quaternion_multiply",
    "rotation_vector_to_quaternion",
    "rotate_vector",
    "twist_in_grasp_frame",
]

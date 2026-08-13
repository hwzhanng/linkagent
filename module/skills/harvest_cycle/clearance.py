"""Read-only pregrasp swept-volume clearance check over a calibrated point sample."""

from __future__ import annotations

import math

from module.robotics import RoboticsBundle
from proto import PregraspClearanceEvidence, PregraspClearanceRequest

from module.skills.core import (
    AtomicSkillBase,
    AtomicSkillDescriptor,
    AtomicSkillFailureCode,
    AtomicSkillResult,
    AtomicSkillStatus,
    SkillTraceEventKind,
    SkillTraceRecorder,
)

from .ports import ClearanceScenePort


Point3 = tuple[float, float, float]


def _point_segment_distance(point: Point3, start: Point3, end: Point3) -> float:
    delta = tuple(end[i] - start[i] for i in range(3))
    length_sq = sum(value * value for value in delta)
    if length_sq <= 1.0e-16:
        return math.dist(point, start)
    ratio = sum((point[i] - start[i]) * delta[i] for i in range(3)) / length_sq
    ratio = min(1.0, max(0.0, ratio))
    nearest = tuple(start[i] + ratio * delta[i] for i in range(3))
    return math.dist(point, nearest)


def _rotate_world_to_target(point: Point3, center: Point3, quaternion: tuple[float, ...]) -> Point3:
    """Rotate a world displacement by the inverse target quaternion."""

    w, x, y, z = quaternion
    vx, vy, vz = (point[i] - center[i] for i in range(3))
    # R(q)^T * v, expanded to avoid a geometry dependency.
    return (
        (1 - 2 * (y * y + z * z)) * vx + 2 * (x * y + w * z) * vy + 2 * (x * z - w * y) * vz,
        2 * (x * y - w * z) * vx + (1 - 2 * (x * x + z * z)) * vy + 2 * (y * z + w * x) * vz,
        2 * (x * z + w * y) * vx + 2 * (y * z - w * x) * vy + (1 - 2 * (x * x + y * y)) * vz,
    )


class PregraspClearanceCheckSkill(AtomicSkillBase):
    descriptor = AtomicSkillDescriptor(
        skill_id="pregrasp_clearance_check",
        version="1.0.0",
        description="Check the planned hand corridor against a calibrated scene point sample.",
        effectful=False,
        required_resources=("arm_state", "arm_ik", "clearance_scene"),
        hardware_approved=False,
    )

    def __init__(
        self,
        *,
        robotics: RoboticsBundle,
        scene_port: ClearanceScenePort,
        max_scene_age_s: float = 0.5,
        min_coverage_ratio: float = 0.80,
        min_scene_points: int = 3,
        transit_clearance_m: float = 0.08,
        approach_clearance_m: float = 0.06,
        finger_clearance_m: float = 0.04,
        target_exclusion_margin_m: float = 0.012,
    ) -> None:
        super().__init__(clock=robotics.clock)
        if not isinstance(robotics, RoboticsBundle):
            raise TypeError("robotics must be a RoboticsBundle")
        if not isinstance(scene_port, ClearanceScenePort):
            raise TypeError("scene_port must implement ClearanceScenePort")
        for name, value in {
            "max_scene_age_s": max_scene_age_s,
            "transit_clearance_m": transit_clearance_m,
            "approach_clearance_m": approach_clearance_m,
            "finger_clearance_m": finger_clearance_m,
            "target_exclusion_margin_m": target_exclusion_margin_m,
        }.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"{name} must be > 0")
        if isinstance(min_coverage_ratio, bool) or not isinstance(min_coverage_ratio, (int, float)) or not 0 <= min_coverage_ratio <= 1:
            raise ValueError("min_coverage_ratio must be in [0, 1]")
        if isinstance(min_scene_points, bool) or not isinstance(min_scene_points, int) or min_scene_points < 1:
            raise ValueError("min_scene_points must be an integer >= 1")
        self._robotics = robotics
        self._scene_port = scene_port
        self._max_scene_age_s = float(max_scene_age_s)
        self._min_coverage_ratio = float(min_coverage_ratio)
        self._min_scene_points = min_scene_points
        self._radii = (
            ("transit", float(transit_clearance_m)),
            ("approach", float(approach_clearance_m)),
            ("finger_corridor", float(finger_clearance_m)),
        )
        self._target_margin = float(target_exclusion_margin_m)

    def run(self, request: object) -> AtomicSkillResult:
        if not isinstance(request, PregraspClearanceRequest):
            raise TypeError("request must be a PregraspClearanceRequest")
        cached = self._cached(request.execution_id)
        if cached is not None:
            return cached
        trace = self._recorder(request.execution_id)
        trace.record("clearance", SkillTraceEventKind.STARTED, input_summary={
            "target_id": request.target.target_id,
            "grasp_plan_id": request.plan.grasp_plan_id,
            "observation_sequence": request.observation.sequence,
        })
        now_ns = self._clock.monotonic_ns()
        if now_ns >= request.plan.expires_monotonic_ns:
            return self._failed(trace, AtomicSkillFailureCode.TARGET_STALE, "grasp plan expired before clearance check")
        arm = self._robotics.arm.get_state(request.plan.arm_side)
        if not arm.enabled or arm.ee_pose.frame_id != "world":
            return self._failed(trace, AtomicSkillFailureCode.CLEARANCE_CONTEXT_MISSING, "valid world-frame arm state is unavailable")
        ik = self._robotics.arm.ik(request.plan.arm_side, request.plan.pregrasp_pose, arm.joint_positions_rad)
        if not ik.success:
            return self._failed(trace, AtomicSkillFailureCode.CLEARANCE_IK_UNREACHABLE, "pregrasp pose is not IK reachable")
        try:
            scene = self._scene_port.capture(request)
        except (RuntimeError, ValueError, TypeError) as exc:
            return self._failed(trace, AtomicSkillFailureCode.CLEARANCE_CONTEXT_MISSING, str(exc))
        scene_age_s = max(0.0, (self._clock.monotonic_ns() - scene.sample_monotonic_ns) / 1.0e9)
        if scene_age_s > self._max_scene_age_s:
            return self._failed(trace, AtomicSkillFailureCode.CLEARANCE_SCENE_STALE, "clearance scene is stale")
        if len(scene.points_world_m) < self._min_scene_points or scene.coverage_ratio < self._min_coverage_ratio:
            return self._failed(trace, AtomicSkillFailureCode.CLEARANCE_CONTEXT_MISSING, "clearance scene has insufficient point coverage")

        target = request.target
        axes = tuple(axis + self._target_margin for axis in target.semi_axes_m)
        remaining: list[Point3] = []
        ignored = 0
        for point in scene.points_world_m:
            local = _rotate_world_to_target(point, target.center_pose.position_m, target.center_pose.quaternion_wxyz)
            if sum((local[i] / axes[i]) ** 2 for i in range(3)) <= 1.0:
                ignored += 1
            else:
                remaining.append(point)

        positions = (
            arm.ee_pose.position_m,
            request.plan.pregrasp_pose.position_m,
            request.plan.contact_pose.position_m,
            target.center_pose.position_m,
        )
        nearest_distance = math.inf
        nearest_margin = math.inf
        nearest_point: Point3 | None = None
        nearest_segment: str | None = None
        required = 0.0
        for (segment, radius), start, end in zip(self._radii, positions, positions[1:]):
            for point in remaining:
                distance = _point_segment_distance(point, start, end)
                margin = distance - radius
                if margin < nearest_margin:
                    nearest_margin = margin
                    nearest_distance = distance
                    nearest_point = point
                    nearest_segment = segment
                    required = radius

        collision_free = nearest_point is None or nearest_margin >= 0.0
        minimum = nearest_distance if nearest_point is not None else math.inf
        evidence = PregraspClearanceEvidence(
            target_id=target.target_id,
            grasp_plan_id=request.plan.grasp_plan_id,
            scene_id=scene.scene_id,
            sample_monotonic_ns=scene.sample_monotonic_ns,
            point_count=len(scene.points_world_m),
            coverage_ratio=scene.coverage_ratio,
            checked_segments=tuple(name for name, _ in self._radii),
            clearance_required_m=required,
            minimum_clearance_m=minimum,
            ignored_target_point_count=ignored,
            ik_reachable=True,
            collision_free=collision_free,
            simulated=scene.simulated,
            evidence_sources=scene.evidence_sources,
            nearest_obstacle_point_world_m=nearest_point,
            blocked_segment=None if collision_free else nearest_segment,
        )
        if not collision_free:
            return self._failed(
                trace,
                AtomicSkillFailureCode.CLEARANCE_BLOCKED,
                f"planned {nearest_segment} corridor is blocked",
                output={"clearance_evidence": evidence.as_dict(), "motion_authorized": False},
            )
        trace.record("clearance", SkillTraceEventKind.SUCCEEDED, output_summary=evidence.as_dict(), evidence_refs=scene.evidence_sources)
        return self._result(
            trace,
            status=AtomicSkillStatus.SUCCEEDED,
            message="pregrasp corridor is clear",
            output={"clearance_evidence": evidence.as_dict(), "motion_authorized": True},
        )

    def _failed(
        self,
        trace: SkillTraceRecorder,
        code: AtomicSkillFailureCode,
        message: str,
        *,
        output: dict[str, object] | None = None,
    ) -> AtomicSkillResult:
        trace.record("clearance", SkillTraceEventKind.FAILED, message=message)
        return self._result(
            trace,
            status=AtomicSkillStatus.RETRYABLE_FAILURE,
            failure_code=code,
            message=message,
            output={} if output is None else output,
        )

    def close(self) -> None:
        self._scene_port.close()
        super().close()


__all__ = ["PregraspClearanceCheckSkill"]

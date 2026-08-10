"""Motion admission backed by live Isaac state and RGB-D clearance evidence."""

from __future__ import annotations

from proto import ArmState, CollisionStatus, SafetyAssessment, TomatoGraspPlan

from .robotics import IsaacHarvestRoboticsBundle


class IsaacMotionSafetyPort:
    """Bind reachability and workspace admission to a passed scene clearance."""

    def __init__(
        self,
        bundle: IsaacHarvestRoboticsBundle,
        *,
        workspace_xyzxyz_m: tuple[float, ...] = (0.0, -0.70, 1.45, 0.90, 0.45, 2.30),
    ) -> None:
        if len(tuple(workspace_xyzxyz_m)) != 6:
            raise ValueError("workspace_xyzxyz_m must contain six values")
        self._bundle = bundle
        self._workspace = tuple(float(value) for value in workspace_xyzxyz_m)
        self._clear_plans: set[str] = set()

    def record_clearance(self, grasp_plan_id: str) -> None:
        if not isinstance(grasp_plan_id, str) or not grasp_plan_id.strip():
            raise ValueError("grasp_plan_id must be non-empty")
        self._clear_plans.add(grasp_plan_id.strip())

    def assess_pregrasp(
        self, plan: TomatoGraspPlan, arm_state: ArmState
    ) -> SafetyAssessment:
        return self._assess(plan, arm_state, stage="pregrasp")

    def assess_skill_envelope(
        self, plan: TomatoGraspPlan, arm_state: ArmState
    ) -> SafetyAssessment:
        return self._assess(plan, arm_state, stage="ppo-envelope")

    def _inside(self, position: tuple[float, ...]) -> bool:
        return all(
            self._workspace[index] <= position[index] <= self._workspace[index + 3]
            for index in range(3)
        )

    def _assess(
        self,
        plan: TomatoGraspPlan,
        arm_state: ArmState,
        *,
        stage: str,
    ) -> SafetyAssessment:
        if not isinstance(plan, TomatoGraspPlan) or not isinstance(arm_state, ArmState):
            raise TypeError("plan and arm_state must use frozen DTOs")
        poses = (
            (plan.pregrasp_pose,)
            if stage == "pregrasp"
            else (plan.pregrasp_pose, plan.contact_pose, plan.retreat_pose)
        )
        workspace_ok = all(self._inside(pose.position_m) for pose in poses)
        ik_results = tuple(
            self._bundle.arm.ik(
                plan.arm_side, pose, arm_state.joint_positions_rad
            )
            for pose in poses
        )
        reachable = all(item.success for item in ik_results)
        clearance_ok = plan.grasp_plan_id in self._clear_plans
        approved = workspace_ok and reachable and clearance_ok and arm_state.enabled
        reasons: list[str] = []
        if not workspace_ok:
            reasons.append("pose outside Isaac acceptance workspace")
        if not reachable:
            reasons.append("Isaac DLS reachability admission failed")
        if not clearance_ok:
            reasons.append("rendered RGB-D clearance evidence missing")
        if not arm_state.enabled:
            reasons.append("Isaac controller disabled")
        now_ns = self._bundle.clock.monotonic_ns()
        return SafetyAssessment(
            assessment_id=(
                f"isaac-safety-{stage}-{plan.grasp_plan_id}-"
                f"{arm_state.stamp.source_sequence}"
            ),
            grasp_plan_id=plan.grasp_plan_id,
            observation_sequence=plan.source_observation_sequence,
            arm_state_sequence=arm_state.stamp.source_sequence,
            reachable=reachable and arm_state.enabled,
            joint_limits_ok=reachable,
            path_sampled_ok=clearance_ok and workspace_ok,
            collision_status=(
                CollisionStatus.SIMULATED_SAFE
                if approved
                else CollisionStatus.UNSAFE
            ),
            allowed_workspace_xyzxyz_m=self._workspace,
            expires_monotonic_ns=max(1, now_ns + 5_000_000_000),
            reason=None if approved else "; ".join(reasons),
        )


__all__ = ["IsaacMotionSafetyPort"]

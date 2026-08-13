"""Validate frozen PPO and post-pick atoms in one real Isaac physics scene."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher


def _write_report(destination: Path, report: dict[str, object]) -> None:
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/simulation/harvest_cycle_acceptance/ppo-runner/acceptance.json"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--physics-cpu", action="store_true")
    parser.add_argument(
        "--full-cycle",
        action="store_true",
        help="Continue through retreat, transport, placement, verification, and reset.",
    )
    parser.add_argument(
        "--inject-drop-before-retreat",
        action="store_true",
        help=(
            "Physically open the hand after a verified pick and require zero-motion "
            "drop detection plus dropped-payload localization."
        ),
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _basis_quaternion(matrix) -> tuple[float, float, float, float]:
    m00, m01, m02 = (float(matrix[0, index]) for index in range(3))
    m10, m11, m12 = (float(matrix[1, index]) for index in range(3))
    m20, m21, m22 = (float(matrix[2, index]) for index in range(3))
    trace = m00 + m11 + m22
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        q = (0.25 * scale, (m21 - m12) / scale, (m02 - m20) / scale, (m10 - m01) / scale)
    elif m00 > m11 and m00 > m22:
        scale = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        q = ((m21 - m12) / scale, 0.25 * scale, (m01 + m10) / scale, (m02 + m20) / scale)
    elif m11 > m22:
        scale = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        q = ((m02 - m20) / scale, (m01 + m10) / scale, 0.25 * scale, (m12 + m21) / scale)
    else:
        scale = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        q = ((m10 - m01) / scale, (m02 + m20) / scale, (m12 + m21) / scale, 0.25 * scale)
    norm = math.sqrt(sum(value * value for value in q))
    return tuple(value / norm for value in q)  # type: ignore[return-value]


def main() -> int:
    args = _parser().parse_args()
    launcher = AppLauncher(args)
    app = launcher.app
    try:
        import gymnasium as gym
        import torch

        from module.robotics import HandSynergyCommand
        from module.skills.harvest_cycle import (
            LocalizeDroppedPayloadSkill,
            MoveToPickReadySkill,
            PayloadPlaceSkill,
            PayloadRetreatSkill,
            ResetForNextTargetSkill,
            SafeRetractSkill,
            TransportPayloadSkill,
        )
        from module.skills.tomato_pick import (
            TomatoPickSkillContext,
            TomatoPickSkillRunner,
            WorldThenHandTargetStatePort,
            build_policy,
        )
        from proto import (
            ArmSide,
            DroppedPayloadLocalizationRequest,
            HandSynergy,
            HarvestPayload,
            MoveToPickReadyRequest,
            PayloadPlaceRequest,
            PayloadRetreatRequest,
            PayloadTransportRequest,
            PlacementReceipt,
            Pose,
            ResetForNextTargetRequest,
            SafeRetractRequest,
            TomatoDetectionResult,
            TomatoPickSkillRequest,
            TomatoTarget,
        )
        from simulation.harvest_cycle import (
            HarvestCycleIsaacEnvCfg,
            PHYSICS_TASK_ID,
        )
        from simulation.harvest_cycle.perception import (
            IsaacDroppedPayloadObservationPort,
            IsaacPlacementVerifierPort,
        )
        from simulation.harvest_cycle.planning import (
            IsaacTrainingDistributionGraspPlanner,
            plan_container_for_retained_payload,
        )
        from simulation.harvest_cycle.robotics import IsaacHarvestRoboticsBundle
        from simulation.harvest_cycle.safety import IsaacMotionSafetyPort
        from simulation.harvest_cycle.verification import IsaacPickEvidenceSource
        from task.agri_pick.verification import IndependentPickVerifier, PickVerifierThresholds

        repo_root = Path(__file__).resolve().parents[1]
        cfg = HarvestCycleIsaacEnvCfg()
        cfg.scene.num_envs = 1
        cfg.seed = args.seed
        cfg.action_delay_steps = (0, 0)
        cfg.target_position_noise_m = 0.0
        cfg.fruit_axis_relative_noise = 0.0
        cfg.tactile_force_gain = (1.0, 1.0)
        cfg.tactile_force_bias_n = (0.0, 0.0)
        if args.physics_cpu:
            cfg.sim.device = "cpu"
        report: dict[str, object] = {
            "schema_version": 1,
            "passed": False,
            "backend": "isaac_physx_cpu" if args.physics_cpu else "isaac_cuda_physx",
            "full_cycle_requested": args.full_cycle,
            "drop_injection_requested": args.inject_drop_before_retreat,
            "seed": args.seed,
            "policy_residual_translation_limits_m": tuple(
                float(value) for value in cfg.max_residual_translation_m
            ),
            "policy_residual_rotation_limits_rad": tuple(
                float(value) for value in cfg.max_residual_rotation_rad
            ),
            "stages": {},
        }
        env = None
        bundle = None
        try:
            env = gym.make(PHYSICS_TASK_ID, cfg=cfg)
            env.reset(seed=args.seed)
            base = env.unwrapped
            bundle = IsaacHarvestRoboticsBundle(base, position_tolerance_m=0.004)
            basis = base._grasp_basis_w[0]
            fruit = base._tomato.data.root_pos_w.torch[0]
            axes = base._measured_fruit_axes_m[0]
            target = TomatoTarget(
                target_id="isaac-physics-tomato-000001",
                center_pose=Pose(
                    tuple(float(value) for value in fruit),
                    _basis_quaternion(basis),
                    "world",
                ),
                semi_axes_m=tuple(float(value) for value in axes),
                ripeness_score=1.0,
                pick_confidence=1.0,
                occlusion=0.0,
                stem_direction_xyz=tuple(float(value) for value in basis[:, 2]),
                observation_sequence=1,
                sample_monotonic_ns=bundle.clock.monotonic_ns(),
            )
            detection = TomatoDetectionResult(
                targets=(target,),
                observation_sequence=1,
                model="isaac-calibrated-geometry-fixture-v1",
                sample_monotonic_ns=target.sample_monotonic_ns,
            )
            planner = IsaacTrainingDistributionGraspPlanner(bundle)
            plan = planner.plan(
                target,
                bundle.arm.get_state(ArmSide.RIGHT),
                bundle.hand.get_state(ArmSide.RIGHT),
            )
            safety = IsaacMotionSafetyPort(bundle)
            safety.record_clearance(plan.grasp_plan_id)
            safe_retract = SafeRetractSkill(robotics=bundle)
            retract_result = safe_retract.run(
                SafeRetractRequest(
                    execution_id="isaac-pre-pick-safe-retract",
                    arm_side=ArmSide.RIGHT,
                    safe_pose=plan.retreat_pose,
                    reason="exercise bounded recovery before PPO admission",
                    max_duration_s=5.0,
                )
            )
            report["stages"]["safe_retract"] = retract_result.as_dict()  # type: ignore[index]
            if not retract_result.success:
                raise RuntimeError(
                    f"Isaac safe retract failed: {retract_result.message}"
                )
            ready = MoveToPickReadySkill(
                robotics=bundle,
                safety=safety,
                execution_mode="simulation",
                max_target_age_s=5.0,
            )
            ready_result = ready.run(
                MoveToPickReadyRequest(
                    "isaac-ppo-runner-ready", target, plan, 2.0
                )
            )
            report["stages"]["move_to_pick_ready"] = ready_result.as_dict()  # type: ignore[index]
            if not ready_result.success:
                raise RuntimeError(
                    f"Isaac PPO ready admission failed: {ready_result.message}"
                )
            policy = build_policy(
                {
                    "type": "profile",
                    "path": repo_root
                    / "config/skills/tomato_pick_clean_v5_model1350_best.json",
                }
            )
            report["policy_id"] = policy.manifest.policy_id
            report["policy_checkpoint_sha256"] = policy.manifest.checkpoint_sha256
            verifier = IndependentPickVerifier(
                IsaacPickEvidenceSource(bundle),
                PickVerifierThresholds(
                    profile_id="isaac-clean-v5-physics-v1",
                    tactile_calibration_id="isaac_g20_contact_v1",
                    stable_hold_s=0.50,
                    minimum_contact_fingers=2,
                    minimum_normal_force_n=0.60,
                    maximum_normal_force_n=15.0,
                    maximum_slip_ratio=0.75,
                ),
            )
            runner = TomatoPickSkillRunner(policy, maximum_normal_force_n=15.0)
            pick_result = runner.run(
                TomatoPickSkillRequest(
                    execution_id="isaac-ppo-runner-pick",
                    target_id=target.target_id,
                    grasp_plan_id=plan.grasp_plan_id,
                    arm_side=ArmSide.RIGHT,
                    policy_id=policy.manifest.policy_id,
                    max_duration_s=8.0,
                ),
                TomatoPickSkillContext(
                    target=target,
                    plan=plan,
                    safety=ready.assessment_for(plan.grasp_plan_id),
                    perception=detection,
                    arm=bundle.arm,
                    hand=bundle.hand,
                    tactile=bundle.tactile,
                    leases=bundle.leases,
                    verifier=verifier,
                    clock=bundle.clock,
                    target_state=WorldThenHandTargetStatePort(),
                ),
            )
            report["stages"]["tomato_pick"] = pick_result.as_dict()  # type: ignore[index]
            state = bundle.payload_state()
            post_pick: dict[str, object] = {}
            post_pick_passed = True
            if args.inject_drop_before_retreat and pick_result.success:
                payload = HarvestPayload(
                    payload_id=f"payload-{target.target_id}",
                    target_id=target.target_id,
                    pick_execution_id=pick_result.execution_id,
                    arm_side=ArmSide.RIGHT,
                    detached=True,
                    retained=True,
                    acquired_monotonic_ns=bundle.clock.monotonic_ns(),
                )
                arm_before_drop_recovery = bundle.arm.get_state(
                    ArmSide.RIGHT
                ).ee_pose
                lease = bundle.leases.acquire(
                    owner_id="isaac-inject-physical-drop",
                    side=ArmSide.RIGHT,
                    ttl_s=3.0,
                )
                try:
                    open_handle = bundle.hand.command_synergy(
                        lease,
                        HandSynergyCommand(
                            command_id="isaac-inject-physical-drop-open",
                            side=ArmSide.RIGHT,
                            target=HandSynergy(0.0, 0.0, 0.0),
                            timeout_s=2.0,
                        ),
                    )
                finally:
                    bundle.leases.release(lease)
                bundle.advance(0.60)
                physically_dropped = bundle.payload_state()
                retreat_result = PayloadRetreatSkill(
                    robotics=bundle,
                    min_contact_fingers=2,
                    min_normal_force_n=0.60,
                ).run(
                    PayloadRetreatRequest(
                        execution_id="isaac-injected-drop-retreat",
                        payload=payload,
                        retreat_pose=plan.retreat_pose,
                        max_duration_s=5.0,
                    )
                )
                arm_after_drop_recovery = bundle.arm.get_state(
                    ArmSide.RIGHT
                ).ee_pose
                zero_motion_error_m = math.dist(
                    arm_before_drop_recovery.position_m,
                    arm_after_drop_recovery.position_m,
                )
                localize_result = LocalizeDroppedPayloadSkill(
                    observation_port=IsaacDroppedPayloadObservationPort(
                        bundle=bundle
                    ),
                    clock=bundle.clock,
                    recovery_workspace_xyzxyz_m=(
                        0.0,
                        -0.70,
                        0.0,
                        0.90,
                        0.45,
                        1.90,
                    ),
                    minimum_confidence=0.60,
                ).run(
                    DroppedPayloadLocalizationRequest(
                        execution_id="isaac-localize-injected-drop",
                        payload=payload,
                        container=plan_container_for_retained_payload(bundle),
                        max_observation_age_s=1.0,
                    )
                )
                expected_drop_abort = (
                    not retreat_result.success
                    and retreat_result.failure_code is not None
                    and retreat_result.failure_code.value == "payload_dropped"
                )
                post_pick = {
                    "physical_drop_command_state": open_handle.state.value,
                    "physics_payload_state_after_drop": physically_dropped,
                    "retreat": retreat_result.as_dict(),
                    "localize_dropped_payload": localize_result.as_dict(),
                    "zero_motion_error_m": zero_motion_error_m,
                }
                post_pick_passed = (
                    open_handle.state.value == "succeeded"
                    and bool(physically_dropped["dropped"])
                    and expected_drop_abort
                    and zero_motion_error_m <= 0.004
                    and localize_result.success
                    and localize_result.output.get("motion_authorized") is False
                    and localize_result.output.get(
                        "frozen_hanging_pick_ppo_reuse_allowed"
                    )
                    is False
                )
            elif args.full_cycle and pick_result.success:
                payload = HarvestPayload(
                    payload_id=f"payload-{target.target_id}",
                    target_id=target.target_id,
                    pick_execution_id=pick_result.execution_id,
                    arm_side=ArmSide.RIGHT,
                    detached=True,
                    retained=True,
                    acquired_monotonic_ns=bundle.clock.monotonic_ns(),
                )
                retreat = PayloadRetreatSkill(
                    robotics=bundle,
                    min_contact_fingers=2,
                    min_normal_force_n=0.60,
                )
                retreat_result = retreat.run(
                    PayloadRetreatRequest(
                        execution_id="isaac-payload-retreat",
                        payload=payload,
                        retreat_pose=plan.retreat_pose,
                        max_duration_s=5.0,
                    )
                )
                post_pick["retreat"] = retreat_result.as_dict()
                post_pick_passed = retreat_result.success

                if retreat_result.success:
                    container = plan_container_for_retained_payload(bundle)
                    transport = TransportPayloadSkill(
                        robotics=bundle,
                        vertical_clearance_m=0.06,
                        min_contact_fingers=2,
                        min_normal_force_n=0.60,
                    )
                    transport_result = transport.run(
                        PayloadTransportRequest(
                            execution_id="isaac-payload-transport",
                            payload=payload,
                            container=container,
                            max_duration_s=12.0,
                        )
                    )
                    post_pick["container"] = {
                        "container_id": container.container_id,
                        "placement_pose": container.placement_pose.as_dict(),
                    }
                    post_pick["transport"] = transport_result.as_dict()
                    post_pick_passed = transport_result.success

                    if transport_result.success:
                        preplace = transport.preplace_pose_for(
                            transport_result.execution_id
                        )
                        place = PayloadPlaceSkill(
                            robotics=bundle,
                            min_contact_fingers=2,
                            min_normal_force_n=0.60,
                        )
                        place_result = place.run(
                            PayloadPlaceRequest(
                                execution_id="isaac-payload-place",
                                payload=payload,
                                container=container,
                                preplace_pose=preplace,
                                max_duration_s=10.0,
                            )
                        )
                        post_pick["place"] = place_result.as_dict()
                        post_pick_passed = place_result.success

                        if place_result.success:
                            output = place_result.output
                            receipt = PlacementReceipt(
                                placement_id=str(output["placement_id"]),
                                payload_id=payload.payload_id,
                                container_id=container.container_id,
                                release_confirmed=bool(output["release_confirmed"]),
                                hand_empty=bool(output["hand_empty"]),
                                dropped=bool(output["dropped"]),
                                evidence_sources=tuple(output["evidence_sources"]),
                            )
                            placement = IsaacPlacementVerifierPort(
                                bundle=bundle
                            ).verify(receipt, payload, container)
                            post_pick["placement_verification"] = {
                                "placement_id": placement.placement_id,
                                "container_id": placement.container_id,
                                "placed_in_container": placement.placed_in_container,
                                "release_confirmed": placement.release_confirmed,
                                "hand_empty": placement.hand_empty,
                                "dropped": placement.dropped,
                                "evidence_sources": placement.evidence_sources,
                                "successful": placement.successful,
                            }
                            reset = ResetForNextTargetSkill(robotics=bundle)
                            reset_result = reset.run(
                                ResetForNextTargetRequest(
                                    execution_id="isaac-reset-for-next-target",
                                    completed_cycle_id="isaac-harvest-cycle-000001",
                                    arm_side=ArmSide.RIGHT,
                                    reset_pose=bundle.reference_pose,
                                    max_duration_s=12.0,
                                )
                            )
                            post_pick["reset"] = reset_result.as_dict()
                            post_pick_passed = (
                                placement.successful and reset_result.success
                            )

            requested_chain_passed = (
                post_pick_passed
                if args.full_cycle or args.inject_drop_before_retreat
                else True
            )
            report.update({
                "passed": (
                    retract_result.success
                    and ready_result.success
                    and pick_result.success
                    and bool(state["detached"])
                    and bool(state["retained"])
                    and requested_chain_passed
                ),
                "safe_retract": retract_result.as_dict(),
                "ready": ready_result.as_dict(),
                "pick": pick_result.as_dict(),
                "physics_payload_state": state,
                "post_pick": post_pick,
                "simulated_substeps": bundle.sequence,
            })
            _write_report(args.output, report)
            print(
                "ISAAC_TOMATO_PICK_RUNNER_ACCEPTANCE="
                + json.dumps(
                    {
                        "passed": report["passed"],
                        "status": pick_result.status.value,
                        "failure_code": None if pick_result.failure_code is None else pick_result.failure_code.value,
                        "physics_payload_state": state,
                        "simulated_substeps": bundle.sequence,
                    },
                    sort_keys=True,
                )
            )
            return 0 if report["passed"] else 2
        except Exception as exc:
            report["passed"] = False
            report["error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
            if bundle is not None:
                report["simulated_substeps"] = bundle.sequence
                report["physics_payload_state"] = bundle.payload_state()
            _write_report(args.output, report)
            print(
                "ISAAC_TOMATO_PICK_RUNNER_ACCEPTANCE="
                + json.dumps(
                    {
                        "passed": False,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                    sort_keys=True,
                )
            )
            return 2
        finally:
            if bundle is not None:
                bundle.close()
            if env is not None:
                env.close()
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())

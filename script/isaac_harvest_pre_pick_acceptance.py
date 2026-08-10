"""Run pre-pick atomic skills against one live Isaac/PhysX harvest scene."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from enum import Enum
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher

from script.isaac_harvest_perception_acceptance import _load_local_credential


def _plain(value):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/simulation/harvest_cycle_acceptance/pre-pick"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-pick",
        action="store_true",
        help="Continue from the admitted ready pose through frozen PPO picking.",
    )
    parser.add_argument(
        "--full-cycle",
        action="store_true",
        help=(
            "Continue the Ark/RGB-D target through frozen PPO, retained retreat, "
            "transport, placement verification, and reset."
        ),
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _ark_backend(repo_root: Path, spool_root: Path):
    from script.tomato_vision_worker_ark import ArkVlmBoxDepthBackend

    return ArkVlmBoxDepthBackend(
        capture_root=spool_root,
        config_dir=repo_root / "config",
        ark_vlm={
            "type": "ark_responses",
            "base_url": "https://ark.cn-beijing.volces.com/api/coding/v3",
            "model_id": "doubao-seed-2.0-pro",
            "api_key_env": "ARK_API_KEY",
            "timeout_s": 60.0,
            "max_response_bytes": 256_000,
            "max_output_tokens": 4096,
            "thinking_type": "disabled",
        },
        minimum_confidence=0.35,
        depth_band_m=0.08,
    )


def _observation(sequence: int, bundle, payload):
    from proto import Observation

    return Observation(
        sequence=sequence,
        timestamp_s=bundle.clock.monotonic_ns() / 1.0e9,
        rgb=f"isaac:rtx-rgb:{sequence}",
        depth_m=f"isaac:aligned-depth:{sequence}",
        metadata={"tomato_vision": payload},
    )


def _detect(observation, bundle):
    from module.skills.tomato_vision import ReplayTomatoVisionPerceptionPort

    port = ReplayTomatoVisionPerceptionPort(
        clock=bundle.clock,
        expected_calibration_id="isaac_top_rgbd_v1",
        expected_device_id="isaac-top-rgbd-01",
        minimum_detector_confidence=0.35,
        minimum_selection_confidence=0.35,
        max_fit_residual_m=0.020,
        silhouette_axis_floor_ratio=0.86,
        visible_surface_center_correction_ratio=0.82,
        pull_axis_prior_world=(0.0, 0.0, -1.0),
        max_pull_axis_deviation_deg=3.0,
    )
    try:
        return port.detect(observation)
    finally:
        port.close()


def _camera_pose(payload) -> tuple[float, ...]:
    pose = payload["camera"]["camera_pose_in_world"]
    return tuple(float(value) for value in (*pose["position_m"], *pose["quaternion_wxyz"]))


def main() -> int:
    args = _parser().parse_args()
    launcher = AppLauncher(args)
    app = launcher.app
    try:
        import gymnasium as gym
        import torch

        from module.skills.harvest_cycle import (
            ActiveViewRepositionSkill,
            MoveToPickReadySkill,
            PayloadPlaceSkill,
            PayloadRetreatSkill,
            PregraspClearanceCheckSkill,
            ResetForNextTargetSkill,
            SelectHarvestTargetSkill,
            TargetReobserveSkill,
            TransportPayloadSkill,
        )
        from module.skills.system_self_check import SystemSelfCheckSkill
        from module.skills.tomato_pick import (
            TomatoPickSkillContext,
            TomatoPickSkillRunner,
            WorldThenHandTargetStatePort,
            build_policy,
        )
        from proto import (
            ActiveViewRepositionRequest,
            ArmSide,
            HarvestContainer,
            HarvestPayload,
            MoveToPickReadyRequest,
            PayloadPlaceRequest,
            PayloadRetreatRequest,
            PayloadTransportRequest,
            PlacementReceipt,
            Pose,
            PregraspClearanceRequest,
            ResetForNextTargetRequest,
            SelectHarvestTargetRequest,
            SystemSelfCheckRequest,
            TargetReobserveRequest,
            TomatoPickSkillRequest,
        )
        from script.tomato_vision_worker_ark import ArkVlmBoxDepthBackend
        from simulation.harvest_cycle import HarvestCycleIsaacEnvCfg
        from simulation.harvest_cycle.perception import (
            IsaacCameraHealthPort,
            IsaacClearanceScenePort,
            IsaacEmergencyStopHealthPort,
            IsaacPlacementVerifierPort,
            IsaacTargetObservationPort,
            IsaacVisionWorkerHealthPort,
            publish_isaac_rgbd,
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
        _load_local_credential(repo_root / ".env.local", "ARK_API_KEY")
        output_root = args.output_root.expanduser().resolve()
        spool_root = output_root / "spool"
        spool_root.mkdir(parents=True, exist_ok=True)

        cfg = HarvestCycleIsaacEnvCfg()
        cfg.scene.num_envs = 1
        cfg.seed = args.seed
        env = None
        backend: ArkVlmBoxDepthBackend | None = None
        bundle = None
        self_check = None
        clearance_skill = None
        reobserve_skill = None
        results = {}
        try:
            env = gym.make("LinkAgent-HarvestCycle-Acceptance-v1", cfg=cfg)
            env.reset(seed=args.seed)
            base = env.unwrapped
            zeros = torch.zeros((1, 11), device=base.device)
            for _ in range(4):
                env.step(zeros)
            bundle = IsaacHarvestRoboticsBundle(base, position_tolerance_m=0.004)
            backend = _ark_backend(repo_root, spool_root)
            policy = build_policy(
                {
                    "type": "profile",
                    "path": repo_root
                    / "config/skills/tomato_pick_clean_v5_model1350_best.json",
                }
            )
            container = HarvestContainer(
                container_id="isaac-harvest-bin-01",
                placement_pose=Pose(
                    base.container_placement_position_w, (1.0, 0.0, 0.0, 0.0), "world"
                ),
                available=True,
                capacity_remaining=20,
            )
            self_check = SystemSelfCheckSkill(
                robotics=bundle,
                camera_health=IsaacCameraHealthPort(bundle),
                vision_worker_health=IsaacVisionWorkerHealthPort(
                    bundle, ready=lambda: bool(backend and backend.health()["ready"]), backend_id=backend.backend_id
                ),
                emergency_stop_health=IsaacEmergencyStopHealthPort(bundle),
                policy_manifest=policy.manifest,
                container=container,
                execution_mode="simulation",
                expected_camera_calibration_id="isaac_top_rgbd_v1",
                expected_camera_device_id="isaac-top-rgbd-01",
                expected_tactile_calibration_id="isaac_g20_contact_v1",
            )
            results["system_self_check"] = self_check.run(
                SystemSelfCheckRequest("isaac-prepick-self-check", ArmSide.RIGHT)
            )

            first_publication = publish_isaac_rgbd(
                bundle, spool_root, capture_id=f"isaac-seed-{args.seed}-prepick-view-1"
            )
            first_payload = backend.process(
                bundle_path=first_publication.bundle_path,
                observation_sequence=1,
                requested_track_id=None,
            )
            first_observation = _observation(1, bundle, first_payload)
            first_detection = _detect(first_observation, bundle)
            first_target = first_detection.targets[0]
            fruit_truth = tuple(float(value) for value in base._tomato.data.root_pos_w.torch[0])

            selection_skill = SelectHarvestTargetSkill(
                clock=bundle.clock,
                reachability_profile_id="isaac-right-harvest-workspace-v1",
                reachability_arm_side=ArmSide.RIGHT,
                workspace_min_m=(0.0, -0.70, 1.45),
                workspace_max_m=(0.90, 0.45, 2.30),
                min_pick_confidence=0.35,
                max_occlusion_ratio=0.70,
            )
            results["select_harvest_target"] = selection_skill.run(
                SelectHarvestTargetRequest(
                    "isaac-prepick-select", first_detection, ArmSide.RIGHT, 5.0
                )
            )

            reference_pose = bundle.arm.get_state(ArmSide.RIGHT).ee_pose
            active_view = ActiveViewRepositionSkill(
                robotics=bundle,
                execution_mode="simulation",
                view_profile_id="fixed-top-camera-arm-parallax-v1",
                view_arm_side=ArmSide.RIGHT,
                view_offsets_world_m=((0.0, 0.030, 0.0),),
                workspace_min_m=(0.0, -0.70, 1.45),
                workspace_max_m=(0.90, 0.45, 2.30),
                position_tolerance_m=0.006,
                orientation_tolerance_rad=0.05,
            )
            results["active_view_reposition"] = active_view.run(
                ActiveViewRepositionRequest(
                    "isaac-prepick-active-view",
                    first_target,
                    ArmSide.RIGHT,
                    reference_pose,
                    0,
                    2.0,
                )
            )

            second_publication = publish_isaac_rgbd(
                bundle, spool_root, capture_id=f"isaac-seed-{args.seed}-prepick-view-2"
            )
            second_payload = backend.process(
                bundle_path=second_publication.bundle_path,
                observation_sequence=2,
                requested_track_id=first_target.target_id,
            )
            second_observation = _observation(2, bundle, second_payload)
            second_detection = _detect(second_observation, bundle)
            second_target = second_detection.targets[0]
            target_port = IsaacTargetObservationPort(
                observe_target=lambda requested_id, sequence: second_target
            )
            reobserve_skill = TargetReobserveSkill(
                observation_port=target_port,
                clock=bundle.clock,
                max_occlusion_ratio=0.70,
                require_depth=True,
                require_new_sample=True,
            )
            results["target_reobserve"] = reobserve_skill.run(
                TargetReobserveRequest(
                    execution_id="isaac-prepick-reobserve",
                    target_id=first_target.target_id,
                    observation=second_observation,
                    max_observation_age_s=5.0,
                    prior_sample_monotonic_ns=first_target.sample_monotonic_ns,
                    require_new_sample=True,
                )
            )

            planner = IsaacTrainingDistributionGraspPlanner(bundle, plan_ttl_s=30.0)
            plan = planner.plan(
                second_target,
                bundle.arm.get_state(ArmSide.RIGHT),
                bundle.hand.get_state(ArmSide.RIGHT),
            )
            clearance_skill = PregraspClearanceCheckSkill(
                robotics=bundle,
                scene_port=IsaacClearanceScenePort(bundle=bundle, pixel_stride=8),
                min_coverage_ratio=0.45,
                transit_clearance_m=0.025,
                approach_clearance_m=0.020,
                finger_clearance_m=0.012,
            )
            results["pregrasp_clearance_check"] = clearance_skill.run(
                PregraspClearanceRequest(
                    "isaac-prepick-clearance", second_target, plan, second_observation, 2.0
                )
            )
            safety = IsaacMotionSafetyPort(bundle)
            if results["pregrasp_clearance_check"].success:
                safety.record_clearance(plan.grasp_plan_id)
            ready_skill = MoveToPickReadySkill(
                robotics=bundle,
                safety=safety,
                execution_mode="simulation",
                max_target_age_s=5.0,
            )
            results["move_to_pick_ready"] = ready_skill.run(
                MoveToPickReadyRequest(
                    "isaac-prepick-ready", second_target, plan, 3.0
                )
            )
            pick_requested = args.include_pick or args.full_cycle
            if pick_requested and results["move_to_pick_ready"].success:
                verifier = IndependentPickVerifier(
                    IsaacPickEvidenceSource(bundle),
                    PickVerifierThresholds(
                        profile_id="isaac-clean-v5-physics-v1",
                        tactile_calibration_id="isaac_g20_contact_v1",
                        stable_hold_s=0.50,
                        minimum_contact_fingers=2,
                        minimum_normal_force_n=0.60,
                        # Match the frozen Clean-v5 environment termination
                        # threshold used to train and evaluate model 1350.
                        maximum_normal_force_n=15.0,
                        maximum_slip_ratio=0.75,
                    ),
                )
                runner = TomatoPickSkillRunner(policy, maximum_normal_force_n=15.0)
                results["execute_tomato_pick_ppo"] = runner.run(
                    TomatoPickSkillRequest(
                        execution_id="isaac-prepick-ppo",
                        target_id=second_target.target_id,
                        grasp_plan_id=plan.grasp_plan_id,
                        arm_side=ArmSide.RIGHT,
                        policy_id=policy.manifest.policy_id,
                        max_duration_s=8.0,
                    ),
                    TomatoPickSkillContext(
                        target=second_target,
                        plan=plan,
                        safety=ready_skill.assessment_for(plan.grasp_plan_id),
                        perception=second_detection,
                        arm=bundle.arm,
                        hand=bundle.hand,
                        tactile=bundle.tactile,
                        leases=bundle.leases,
                        verifier=verifier,
                        clock=bundle.clock,
                        target_state=WorldThenHandTargetStatePort(),
                    ),
                )

            pick_payload_state = bundle.payload_state()
            placement_verification = None
            if (
                args.full_cycle
                and "execute_tomato_pick_ppo" in results
                and results["execute_tomato_pick_ppo"].success
                and bool(pick_payload_state["detached"])
                and bool(pick_payload_state["retained"])
            ):
                payload = HarvestPayload(
                    payload_id=f"payload-{second_target.target_id}",
                    target_id=second_target.target_id,
                    pick_execution_id=results["execute_tomato_pick_ppo"].execution_id,
                    arm_side=ArmSide.RIGHT,
                    detached=True,
                    retained=True,
                    acquired_monotonic_ns=bundle.clock.monotonic_ns(),
                )
                results["payload_retreat"] = PayloadRetreatSkill(
                    robotics=bundle,
                    min_contact_fingers=2,
                    min_normal_force_n=0.60,
                ).run(
                    PayloadRetreatRequest(
                        execution_id="isaac-full-cycle-retreat",
                        payload=payload,
                        retreat_pose=plan.retreat_pose,
                        max_duration_s=5.0,
                    )
                )
                if results["payload_retreat"].success:
                    container = plan_container_for_retained_payload(bundle)
                    transport_skill = TransportPayloadSkill(
                        robotics=bundle,
                        vertical_clearance_m=0.06,
                        min_contact_fingers=2,
                        min_normal_force_n=0.60,
                    )
                    results["transport_payload"] = transport_skill.run(
                        PayloadTransportRequest(
                            execution_id="isaac-full-cycle-transport",
                            payload=payload,
                            container=container,
                            max_duration_s=12.0,
                        )
                    )
                    if results["transport_payload"].success:
                        preplace = transport_skill.preplace_pose_for(
                            results["transport_payload"].execution_id
                        )
                        results["payload_place"] = PayloadPlaceSkill(
                            robotics=bundle,
                            min_contact_fingers=2,
                            min_normal_force_n=0.60,
                        ).run(
                            PayloadPlaceRequest(
                                execution_id="isaac-full-cycle-place",
                                payload=payload,
                                container=container,
                                preplace_pose=preplace,
                                max_duration_s=10.0,
                            )
                        )
                        if results["payload_place"].success:
                            place_output = results["payload_place"].output
                            receipt = PlacementReceipt(
                                placement_id=str(place_output["placement_id"]),
                                payload_id=payload.payload_id,
                                container_id=container.container_id,
                                release_confirmed=bool(
                                    place_output["release_confirmed"]
                                ),
                                hand_empty=bool(place_output["hand_empty"]),
                                dropped=bool(place_output["dropped"]),
                                evidence_sources=tuple(
                                    place_output["evidence_sources"]
                                ),
                            )
                            placement_verification = IsaacPlacementVerifierPort(
                                bundle=bundle
                            ).verify(receipt, payload, container)
                            results["reset_for_next_target"] = (
                                ResetForNextTargetSkill(robotics=bundle).run(
                                    ResetForNextTargetRequest(
                                        execution_id="isaac-full-cycle-reset",
                                        completed_cycle_id="isaac-full-cycle-000001",
                                        arm_side=ArmSide.RIGHT,
                                        reset_pose=bundle.reference_pose,
                                        max_duration_s=12.0,
                                    )
                                )
                            )

            second_truth_error = math.dist(second_target.center_pose.position_m, fruit_truth)
            camera_pose_drift = max(
                abs(a - b)
                for a, b in zip(
                    _camera_pose(first_payload), _camera_pose(second_payload), strict=True
                )
            )
            requested_pick_passed = (
                not pick_requested
                or (
                    "execute_tomato_pick_ppo" in results
                    and results["execute_tomato_pick_ppo"].success
                    and bool(pick_payload_state["detached"])
                    and bool(pick_payload_state["retained"])
                )
            )
            requested_full_cycle_passed = (
                not args.full_cycle
                or (
                    placement_verification is not None
                    and placement_verification.successful
                    and "reset_for_next_target" in results
                    and results["reset_for_next_target"].success
                )
            )
            passed = (
                all(result.success for result in results.values())
                and requested_pick_passed
                and requested_full_cycle_passed
                and first_target.target_id == second_target.target_id
                and second_target.sample_monotonic_ns > first_target.sample_monotonic_ns
                and first_publication.rgb_sha256 != second_publication.rgb_sha256
                and camera_pose_drift <= 1.0e-6
                and second_truth_error <= 0.055
            )
            report = {
                "schema_version": 1,
                "passed": passed,
                "backend": "isaac_lab_cuda_physx",
                "pick_requested": pick_requested,
                "full_cycle_requested": args.full_cycle,
                "camera_installation": "fixed_top_center_forward_down",
                "camera_pose_max_drift": camera_pose_drift,
                "first_rgb_sha256": first_publication.rgb_sha256,
                "second_rgb_sha256": second_publication.rgb_sha256,
                "image_changed_after_arm_motion": first_publication.rgb_sha256 != second_publication.rgb_sha256,
                "track_id": second_target.target_id,
                "track_preserved": first_target.target_id == second_target.target_id,
                "first_metric_center_error_m": math.dist(first_target.center_pose.position_m, fruit_truth),
                "second_metric_center_error_m": second_truth_error,
                "ark_response_ids": [
                    source.split(":", 1)[1]
                    for payload in (first_payload, second_payload)
                    for source in payload["camera"]["evidence_sources"]
                    if source.startswith("ark-response-id:")
                ],
                "plan": plan.summary(),
                "pick_payload_state": pick_payload_state,
                "final_physics_payload_state": bundle.payload_state(),
                "placement_verification": (
                    None
                    if placement_verification is None
                    else {
                        "placement_id": placement_verification.placement_id,
                        "container_id": placement_verification.container_id,
                        "placed_in_container": placement_verification.placed_in_container,
                        "release_confirmed": placement_verification.release_confirmed,
                        "hand_empty": placement_verification.hand_empty,
                        "dropped": placement_verification.dropped,
                        "evidence_sources": placement_verification.evidence_sources,
                        "successful": placement_verification.successful,
                    }
                ),
                "skills": {name: result.as_dict() for name, result in results.items()},
            }
            output_root.mkdir(parents=True, exist_ok=True)
            (output_root / "acceptance.json").write_text(
                json.dumps(_plain(report), indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            concise = {
                "passed": passed,
                "skills": {name: result.status.value for name, result in results.items()},
                "track_preserved": report["track_preserved"],
                "camera_pose_max_drift": camera_pose_drift,
                "second_metric_center_error_m": second_truth_error,
            }
            print("ISAAC_HARVEST_PRE_PICK_ACCEPTANCE=" + json.dumps(concise, sort_keys=True))
            return 0 if passed else 2
        except Exception as exc:
            failure_report = {
                "schema_version": 1,
                "passed": False,
                "backend": "isaac_lab_cuda_physx",
                "camera_installation": "fixed_top_center_forward_down",
                "pick_requested": args.include_pick or args.full_cycle,
                "full_cycle_requested": args.full_cycle,
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
                "skills": {
                    name: result.as_dict() for name, result in results.items()
                },
            }
            if bundle is not None:
                failure_report["simulated_substeps"] = bundle.sequence
                failure_report["physics_payload_state"] = bundle.payload_state()
            output_root.mkdir(parents=True, exist_ok=True)
            (output_root / "acceptance.json").write_text(
                json.dumps(_plain(failure_report), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(
                "ISAAC_HARVEST_PRE_PICK_FAILURE="
                + json.dumps(
                    {"type": type(exc).__name__, "message": str(exc)},
                    sort_keys=True,
                ),
                flush=True,
            )
            return 2
        finally:
            if reobserve_skill is not None:
                reobserve_skill.close()
            if clearance_skill is not None:
                clearance_skill.close()
            if self_check is not None:
                self_check.close()
            if backend is not None:
                backend.close()
            if bundle is not None:
                bundle.close()
            if env is not None:
                env.close()
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())

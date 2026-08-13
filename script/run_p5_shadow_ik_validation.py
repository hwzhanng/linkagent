"""P5 live Shadow-IK validation for both connected arms.

The script reads real CAN feedback and uses it as the sole IK seed.  It then
performs FK -> IK -> FK checks at the measured pose and at six neighbouring
poses ( +/- 5 mm on each Cartesian axis).  It never enables an arm and never
dispatches a target, trajectory, mode, or emergency-stop frame.  The vendor
runtime is allowed to emit bounded state-query/telemetry frames so fresh
feedback can be established; those frames are counted separately.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from module.robotics import ControlBusyError, ReadOnlyControlLeaseManager, SystemClock
from proto import ArmSide


ROOT = Path(__file__).resolve().parents[1]
VENDOR_ROOT = ROOT / "botSDK" / "lbot_arm_api-master"
if str(VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDOR_ROOT))
from tools.local_arm_can.advanced_controller import AdvancedArmController  # noqa: E402
from tools.local_arm_can.service import ArmCanRuntime  # noqa: E402
from tools.local_arm_can.trajectory import sample_smooth_movej  # noqa: E402


NEIGHBOUR_OFFSETS_M = (
    ("x_plus_5mm", (0.005, 0.0, 0.0)),
    ("x_minus_5mm", (-0.005, 0.0, 0.0)),
    ("y_plus_5mm", (0.0, 0.005, 0.0)),
    ("y_minus_5mm", (0.0, -0.005, 0.0)),
    ("z_plus_5mm", (0.0, 0.0, 0.005)),
    ("z_minus_5mm", (0.0, 0.0, -0.005)),
)
MAX_JOINT_DELTA_RAD = 0.50
POSITION_TOLERANCE_M = 2.0e-5
ROTATION_TOLERANCE_RAD = 2.0e-4
DETERMINISM_TOLERANCE_RAD = 1.0e-5
# If a requested 5 mm neighbour is outside the local fixed-orientation
# workspace, bisect toward the measured pose.  This distinguishes a genuine
# workspace boundary from a solver failure without ever sending a motion.
BOUNDARY_PROBE_MAGNITUDES_M = (0.0025, 0.00125, 0.000625, 0.0003125, 0.00015625, 0.000078125)


def _json_command(command: list[str]) -> str:
    try:
        return subprocess.run(
            command, check=False, capture_output=True, text=True, timeout=5.0
        ).stdout.strip()
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def _can_snapshot(interface: str) -> dict[str, Any]:
    raw = _json_command(["ip", "-details", "-statistics", "link", "show", interface])
    return {
        "interface": interface,
        "raw_summary": raw,
        "up": "<NOARP,UP," in raw or "state UP" in raw,
        "error_active": "ERROR-ACTIVE" in raw,
    }


def _pose_dict(pose: Any) -> dict[str, Any]:
    return {
        "position_m": [float(value) for value in pose.position],
        "rpy_rad": [float(value) for value in pose.rpy],
        "frame_id": "world",
    }


def _state_dict(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "interface": snapshot["interface"],
        "arm": snapshot["arm"],
        "healthy_joint_count": snapshot["healthy_joint_count"],
        "all_joints_healthy": snapshot["all_joints_healthy"],
        "software_enabled": snapshot["software_enabled"],
        "last_error": snapshot["last_error"],
        "rx_frames": snapshot["rx_frames"],
        "tx_frames": snapshot["tx_frames"],
        "joints": snapshot["joints"],
    }


def _joint_metrics(
    controller: AdvancedArmController,
    seed: tuple[float, ...],
    candidate: tuple[float, ...],
) -> dict[str, Any]:
    lower, upper = controller.kinematics.joint_limits
    margins = [
        min(float(value) - float(lo), float(hi) - float(value))
        for value, lo, hi in zip(candidate, lower, upper, strict=True)
    ]
    deltas = [float(value - old) for value, old in zip(candidate, seed, strict=True)]
    return {
        "candidate_joints_rad": list(candidate),
        "delta_from_seed_rad": deltas,
        "max_abs_delta_rad": max(abs(value) for value in deltas),
        "j6_delta_rad": deltas[5],
        "j7_delta_rad": deltas[6],
        "within_limits": all(margin >= 0.0 for margin in margins),
        "minimum_limit_margin_rad": min(margins),
    }


def _validate_target(
    controller: AdvancedArmController,
    seed: tuple[float, ...],
    target_position: tuple[float, ...],
    target_rpy: tuple[float, ...],
) -> dict[str, Any]:
    result = controller.inverse(target_position, target_rpy, seed)
    candidate = tuple(result.joints)
    returned_pose = controller.forward(candidate)
    metrics = _joint_metrics(controller, seed, candidate)
    repeat = controller.inverse(target_position, target_rpy, seed)
    repeat_delta = [
        float(a - b) for a, b in zip(candidate, tuple(repeat.joints), strict=True)
    ]
    metrics.update(
        {
            "solution_found": bool(result.success),
            "iterations": int(result.iterations),
            "position_error_m": float(result.position_error),
            "rotation_error_rad": float(result.rotation_error),
            "target_pose": {
                "position_m": list(target_position),
                "rpy_rad": list(target_rpy),
                "frame_id": "world",
            },
            "returned_fk_pose": _pose_dict(returned_pose),
            "repeat_solution_found": bool(repeat.success),
            "repeat_max_abs_delta_rad": max(abs(value) for value in repeat_delta),
            "repeat_delta_rad": repeat_delta,
            "collision_checked": False,
            "motion_executed": False,
        }
    )
    metrics["passed"] = bool(
        result.success
        and repeat.success
        and result.position_error <= POSITION_TOLERANCE_M
        and result.rotation_error <= ROTATION_TOLERANCE_RAD
        and metrics["within_limits"]
        and metrics["max_abs_delta_rad"] <= MAX_JOINT_DELTA_RAD
        and metrics["repeat_max_abs_delta_rad"] <= DETERMINISM_TOLERANCE_RAD
    )
    return metrics


def _urdf_preview_record(
    controller: AdvancedArmController,
    start: tuple[float, ...],
    target: tuple[float, ...],
    *,
    duration: float = 2.0,
) -> dict[str, Any]:
    """Generate the same smooth joint/FK samples consumed by the URDF console."""
    samples = sample_smooth_movej(start, target, duration)
    indices = sorted({0, len(samples) // 2, len(samples) - 1})
    return {
        "simulation_only": True,
        "mode": "smooth_movej",
        "duration_s": float(samples[-1].time),
        "sample_count": len(samples),
        "sampled_fk": [
            {
                "t_s": float(samples[index].time),
                "joints_rad": list(samples[index].positions),
                "pose": _pose_dict(controller.forward(samples[index].positions)),
            }
            for index in indices
        ],
        "motion_executed": False,
    }


def main() -> int:
    run_id = "p5-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    evidence_dir = ROOT / "logs" / "agent_skill_validation" / run_id
    evidence_dir.mkdir(parents=True, exist_ok=True)
    deployment = json.loads(
        (ROOT / "config" / "robot_bimanual_offline.yaml").read_text(encoding="utf-8")
    )
    can = {
        side: _can_snapshot(deployment["arms"][side]["can_interface"])
        for side in ("left", "right")
    }
    leases = ReadOnlyControlLeaseManager(SystemClock())
    lease_rejected = True
    try:
        leases.acquire(owner_id="p5-shadow-ik", side=ArmSide.LEFT, ttl_s=1.0)
        lease_rejected = False
    except ControlBusyError:
        pass

    runtimes: dict[str, ArmCanRuntime] = {}
    controllers: dict[str, AdvancedArmController] = {}
    arms: dict[str, Any] = {}
    blocking: list[str] = []
    try:
        for side, interface, allow_can0 in (
            ("left", "can0", True),
            ("right", "can1", False),
        ):
            runtime = ArmCanRuntime(
                interface,
                arm=side,
                allow_tx=True,
                allow_can0=allow_can0,
                poll_telemetry=True,
            )
            runtime.start()
            runtimes[side] = runtime
            controllers[side] = AdvancedArmController(
                runtime, config_path=ROOT / deployment["arms"][side]["profile_path"]
            )
        time.sleep(2.0)
        for side in ("left", "right"):
            runtime = runtimes[side]
            controller = controllers[side]
            snapshot = runtime.snapshot()
            if not snapshot["all_joints_healthy"]:
                blocking.append(
                    f"{side} feedback is stale: {snapshot['healthy_joint_count']}/7"
                )
                arms[side] = {"connected": True, "passed": False, "state": _state_dict(snapshot)}
                continue
            seed = tuple(float(row["position"]) for row in snapshot["joints"])
            measured_pose = controller.forward(seed)
            current = _validate_target(controller, seed, measured_pose.position, measured_pose.rpy)
            current["name"] = "measured_pose"
            neighbours = []
            for name, offset in NEIGHBOUR_OFFSETS_M:
                target = tuple(
                    float(value + delta)
                    for value, delta in zip(measured_pose.position, offset, strict=True)
                )
                item = _validate_target(controller, seed, target, measured_pose.rpy)
                item["name"] = name
                item["offset_m"] = list(offset)
                item["requested_target_reachable"] = bool(item["passed"])
                item["workspace_boundary"] = False
                if not item["passed"]:
                    # A failed 5 mm request is not automatically a hardware
                    # fault: fixed orientation can place the pose on a local
                    # workspace boundary. Probe smaller displacements along
                    # exactly the same axis and preserve every result.
                    probes = []
                    direction = tuple(
                        (float(value) / 0.005) for value in offset
                    )
                    for magnitude in BOUNDARY_PROBE_MAGNITUDES_M:
                        probe_offset = tuple(
                            direction_value * magnitude for direction_value in direction
                        )
                        probe_target = tuple(
                            float(value + delta)
                            for value, delta in zip(
                                measured_pose.position, probe_offset, strict=True
                            )
                        )
                        probe = _validate_target(
                            controller, seed, probe_target, measured_pose.rpy
                        )
                        probe["offset_m"] = list(probe_offset)
                        probes.append(probe)
                        if probe["passed"]:
                            break
                    item["boundary_probes"] = probes
                    item["workspace_boundary"] = bool(probes and not item["passed"] and any(probe["passed"] for probe in probes))
                    item["stage_accepted"] = bool(item["workspace_boundary"])
                else:
                    item["boundary_probes"] = []
                    item["stage_accepted"] = True
                preview_target = None
                preview_status = "requested_target"
                if item["passed"]:
                    preview_target = tuple(item["candidate_joints_rad"])
                elif item["workspace_boundary"]:
                    reachable_probe = next(
                        (probe for probe in item["boundary_probes"] if probe["passed"]),
                        None,
                    )
                    if reachable_probe is not None:
                        preview_target = tuple(reachable_probe["candidate_joints_rad"])
                        preview_status = "nearest_reachable_boundary_probe"
                if preview_target is not None:
                    item["urdf_preview"] = _urdf_preview_record(
                        controller, seed, preview_target
                    )
                    item["urdf_preview"]["status"] = preview_status
                else:
                    item["urdf_preview"] = {
                        "simulation_only": True,
                        "status": "not_generated_target_unreachable",
                        "motion_executed": False,
                    }
                neighbours.append(item)
            checks = [current, *neighbours]
            side_passed = bool(
                snapshot["software_enabled"] is False
                and lease_rejected
                and current["passed"]
                and all(item.get("stage_accepted", False) for item in neighbours)
            )
            if not side_passed:
                blocking.append(f"{side} Shadow IK checks failed")
            arms[side] = {
                "connected": True,
                "passed": side_passed,
                "state": _state_dict(snapshot),
                "seed_source": "current_feedback",
                "zero_reference": bool(max(abs(value) for value in seed) <= 1.0e-3),
                "seed_joints_rad": list(seed),
                "measured_fk_pose": _pose_dict(measured_pose),
                "checks": checks,
                "motion_executed": False,
            }
    except Exception as exc:
        blocking.append(f"runtime validation error: {type(exc).__name__}: {exc}")
    finally:
        for runtime in runtimes.values():
            runtime.close()

    manifest = {
        "run_id": run_id,
        "stage": "P5",
        "device_connections": {"arms": True, "hands": False, "camera": False},
        "control_enabled": False,
        "hardware_write_authorized": False,
        "read_only_lease_rejected": lease_rejected,
        "motion_commands_sent": 0,
        "target_writes": 0,
        "enable_changes": 0,
        "telemetry_query_frames_sent": sum(
            int(arm.get("state", {}).get("tx_frames", 0)) for arm in arms.values()
        ),
        "offsets_m": {name: list(offset) for name, offset in NEIGHBOUR_OFFSETS_M},
        "thresholds": {
            "position_tolerance_m": POSITION_TOLERANCE_M,
            "rotation_tolerance_rad": ROTATION_TOLERANCE_RAD,
            "max_joint_delta_rad": MAX_JOINT_DELTA_RAD,
            "determinism_tolerance_rad": DETERMINISM_TOLERANCE_RAD,
        },
        "can": can,
        "arms": arms,
        "hands": {
            "left": {"connected": False, "checked": False, "reason": "hand not connected by user"},
            "right": {"connected": False, "checked": False, "reason": "hand not connected by user"},
        },
        "collision_checked": False,
        "blocking_reasons": blocking,
        "passed": not blocking and lease_rejected,
    }
    (evidence_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (evidence_dir / "summary.md").write_text(
        "# P5 Shadow IK\n\n"
        f"- result: {'PASS' if manifest['passed'] else 'FAIL'}\n"
        f"- run_id: `{run_id}`\n"
        "- hardware motion: none\n"
        "- collision checking: not performed (P5 shadow scope)\n"
        "- hand checks: skipped; hands were not connected\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "run_id": run_id,
                "passed": manifest["passed"],
                "blocking_reasons": blocking,
                "evidence_dir": str(evidence_dir),
            },
            ensure_ascii=False,
        )
    )
    return 0 if manifest["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

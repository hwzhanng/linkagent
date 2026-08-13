"""Execute exactly one authorized, low-speed right-arm P6 trajectory.

This is the first P6 write-capable script.  It has no retry loop and never
executes a second target.  A trajectory failure leaves the runtime enabled at
the last accepted point, matching the project's drop-prevention rule.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
VENDOR_ROOT = ROOT / "botSDK" / "lbot_arm_api-master"
if str(VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDOR_ROOT))
from tools.local_arm_can.advanced_controller import AdvancedArmController  # noqa: E402
from tools.local_arm_can.service import ArmCanRuntime  # noqa: E402
from tools.local_arm_can.trajectory import sample_smooth_movej  # noqa: E402


SIDE = "right"
INTERFACE = "can1"
CONFIRMATION = "RIGHT_ARM_AREA_CLEAR"
TARGET_NAME = "z_plus_5mm"
DURATION_S = 5.0
MAX_VELOCITY_RAD_S = 0.12
MAX_SEED_DRIFT_RAD = 0.02
MAX_FINAL_ERROR_RAD = 0.03


def _pose_dict(pose: Any) -> dict[str, Any]:
    return {
        "position_m": list(pose.position),
        "rpy_rad": list(pose.rpy),
        "frame_id": "world",
    }


def _snapshot_summary(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "interface": snapshot["interface"],
        "arm": snapshot["arm"],
        "healthy_joint_count": snapshot["healthy_joint_count"],
        "all_joints_healthy": snapshot["all_joints_healthy"],
        "software_enabled": snapshot["software_enabled"],
        "last_error": snapshot["last_error"],
        "rx_frames": snapshot["rx_frames"],
        "tx_frames": snapshot["tx_frames"],
        "joint_positions_rad": [float(row["position"]) for row in snapshot["joints"]],
        "joints": snapshot["joints"],
    }


def _build_target(controller: AdvancedArmController, seed: tuple[float, ...]) -> dict[str, Any]:
    measured = controller.forward(seed)
    target_position = (
        measured.position[0],
        measured.position[1],
        measured.position[2] + 0.005,
    )
    ik = controller.inverse(target_position, measured.rpy, seed)
    candidate = tuple(ik.joints)
    deltas = [float(a - b) for a, b in zip(candidate, seed, strict=True)]
    samples = sample_smooth_movej(seed, candidate, DURATION_S)
    max_velocity = max(
        abs(float(value)) for sample in samples for value in sample.velocities
    )
    lower, upper = controller.kinematics.joint_limits
    margins = [
        min(float(value) - float(lo), float(hi) - float(value))
        for value, lo, hi in zip(candidate, lower, upper, strict=True)
    ]
    return {
        "target": TARGET_NAME,
        "seed_joints_rad": list(seed),
        "measured_fk_pose": _pose_dict(measured),
        "target_position_m": list(target_position),
        "target_rpy_rad": list(measured.rpy),
        "ik_solution_found": bool(ik.success),
        "target_joints_rad": list(candidate),
        "ik_position_error_m": float(ik.position_error),
        "ik_rotation_error_rad": float(ik.rotation_error),
        "ik_iterations": int(ik.iterations),
        "delta_from_seed_rad": deltas,
        "max_joint_delta_rad": max(abs(value) for value in deltas),
        "j6_delta_rad": deltas[5],
        "j7_delta_rad": deltas[6],
        "within_limits": all(value >= 0.0 for value in margins),
        "minimum_limit_margin_rad": min(margins),
        "duration_s": DURATION_S,
        "sample_count": len(samples),
        "max_velocity_rad_s": max_velocity,
        "velocity_limit_rad_s": MAX_VELOCITY_RAD_S,
        "within_velocity_limit": max_velocity <= MAX_VELOCITY_RAD_S,
        "motion_executed": False,
    }


def main() -> int:
    run_id = "p6-right-execute-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    evidence_dir = ROOT / "logs" / "agent_skill_validation" / run_id
    evidence_dir.mkdir(parents=True, exist_ok=True)
    runtime = ArmCanRuntime(
        INTERFACE,
        arm=SIDE,
        allow_tx=True,
        allow_can0=False,
        poll_telemetry=True,
    )
    controller = AdvancedArmController(
        runtime,
        config_path=ROOT / "botSDK/lbot_arm_api-master/tools/local_arm_can/a7_v2_right.json",
    )
    started = False
    enabled_by_this_run = False
    trajectory_samples_sent: list[dict[str, Any]] = []
    before: dict[str, Any] | None = None
    enabled_snapshot: dict[str, Any] | None = None
    candidate: dict[str, Any] | None = None
    after: dict[str, Any] | None = None
    error: str | None = None
    trajectory_succeeded = False
    try:
        runtime.start()
        started = True
        time.sleep(2.0)
        initial = runtime.snapshot()
        before = _snapshot_summary(initial)
        if not initial["all_joints_healthy"]:
            raise RuntimeError(
                f"right feedback is stale: {initial['healthy_joint_count']}/7"
            )
        if initial["software_enabled"]:
            raise RuntimeError("right arm was already enabled; refusing to take ownership")

        # The user authorization is consumed exactly once here.  No command
        # target has been sent yet.
        runtime.set_enabled(True, CONFIRMATION)
        enabled_by_this_run = True
        time.sleep(0.2)
        enabled_state = runtime.snapshot()
        enabled_snapshot = _snapshot_summary(enabled_state)
        if not enabled_state["software_enabled"] or not enabled_state["all_joints_healthy"]:
            raise RuntimeError("right arm did not remain enabled with fresh 7/7 feedback")

        pre_enable_seed = tuple(before["joint_positions_rad"])
        enabled_seed = tuple(enabled_snapshot["joint_positions_rad"])
        seed_drift = max(
            abs(float(a) - float(b))
            for a, b in zip(enabled_seed, pre_enable_seed, strict=True)
        )
        if seed_drift > MAX_SEED_DRIFT_RAD:
            raise RuntimeError(
                f"feedback changed {seed_drift:.6f} rad during enable; no trajectory sent"
            )
        candidate = _build_target(controller, enabled_seed)
        if not candidate["ik_solution_found"]:
            raise RuntimeError("z+5mm IK did not converge")
        if not candidate["within_limits"]:
            raise RuntimeError("IK target exceeds software limits")
        if not candidate["within_velocity_limit"]:
            raise RuntimeError("candidate trajectory exceeds 0.12 rad/s")

        # Exactly one blocking smooth trajectory.  There is deliberately no
        # retry, fallback target, or automatic reset.
        controller.smooth_movej(
            candidate["target_joints_rad"],
            DURATION_S,
            block=True,
            confirmation=CONFIRMATION,
            on_sample=lambda sample: trajectory_samples_sent.append(
                {"t_s": float(sample.time), "joints_rad": list(sample.positions)}
            ),
        )
        trajectory_succeeded = True
        time.sleep(0.3)
        after_snapshot = runtime.snapshot()
        after = _snapshot_summary(after_snapshot)
        final_target = tuple(candidate["target_joints_rad"])
        final_joints = tuple(after["joint_positions_rad"])
        final_error = max(
            abs(float(a) - float(b))
            for a, b in zip(final_joints, final_target, strict=True)
        )
        if final_error > MAX_FINAL_ERROR_RAD:
            error = f"trajectory completed but final feedback error is {final_error:.6f} rad"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        # If no trajectory has been dispatched, return to disabled state.  If
        # a trajectory failed after dispatch, retain enablement at the last
        # accepted point and let the operator acknowledge/reset.
        if enabled_by_this_run and not trajectory_samples_sent:
            try:
                runtime.set_enabled(False, None)
            except Exception as disable_error:
                error += f"; safe disable failed: {type(disable_error).__name__}: {disable_error}"
    finally:
        if started:
            try:
                if after is None:
                    after = _snapshot_summary(runtime.snapshot())
            except Exception as snapshot_error:
                error = error or f"post-failure snapshot error: {type(snapshot_error).__name__}"
        runtime.close()

    manifest = {
        "run_id": run_id,
        "stage": "P6",
        "side": SIDE,
        "target": TARGET_NAME,
        "authorization": {
            "operator_confirmed": True,
            "confirmation_string": CONFIRMATION,
            "single_use": True,
        },
        "hardware_connected": True,
        "hands_connected": False,
        "motion_commands_sent": len(trajectory_samples_sent),
        "trajectory_samples_dispatched": len(trajectory_samples_sent),
        "enable_changes": 1 if enabled_by_this_run else 0,
        "target_writes": len(trajectory_samples_sent),
        "trajectory_succeeded": trajectory_succeeded,
        "before": before,
        "enabled_state": enabled_snapshot,
        "candidate": candidate,
        "after": after,
        "error": error,
        "motion_executed": bool(trajectory_samples_sent),
        "passed": bool(trajectory_succeeded and error is None and after is not None),
        "note": "One authorized right-arm trajectory only; no retry or automatic reset.",
    }
    (evidence_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "run_id": run_id,
                "passed": manifest["passed"],
                "trajectory_succeeded": trajectory_succeeded,
                "trajectory_samples_dispatched": len(trajectory_samples_sent),
                "error": error,
                "evidence_dir": str(evidence_dir),
            },
            ensure_ascii=False,
        )
    )
    return 0 if manifest["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

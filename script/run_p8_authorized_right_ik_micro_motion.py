"""Execute one explicitly authorized, low-speed 1 mm right-arm IK motion.

This is an R3 commissioning action, separate from the real-API read-only
checks.  It consumes one physical confirmation, dispatches one smooth target,
verifies feedback, and disables the right arm after success.  A failure after
trajectory dispatch leaves the arm enabled at the last accepted point.
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
OFFSET_M = 0.001
DURATION_S = 4.0
MAX_VELOCITY_RAD_S = 0.08
MAX_FINAL_ERROR_RAD = 0.03


def _summary(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "interface": snapshot["interface"],
        "arm": snapshot["arm"],
        "healthy_joint_count": snapshot["healthy_joint_count"],
        "all_joints_healthy": snapshot["all_joints_healthy"],
        "software_enabled": snapshot["software_enabled"],
        "last_error": snapshot["last_error"],
        "joint_positions_rad": [float(row["position"]) for row in snapshot["joints"]],
    }


def _pose(pose: Any) -> dict[str, Any]:
    return {"position_m": list(pose.position), "rpy_rad": list(pose.rpy), "frame_id": "world"}


def main() -> int:
    run_id = "p8-right-ik-micro-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    evidence_dir = ROOT / "logs" / "agent_skill_validation" / run_id
    evidence_dir.mkdir(parents=True, exist_ok=True)
    runtime = ArmCanRuntime(INTERFACE, arm=SIDE, allow_tx=True, allow_can0=False, poll_telemetry=True)
    controller = AdvancedArmController(
        runtime,
        config_path=ROOT / "botSDK/lbot_arm_api-master/tools/local_arm_can/a7_v2_right.json",
    )
    before: dict[str, Any] | None = None
    candidate: dict[str, Any] | None = None
    after_motion: dict[str, Any] | None = None
    after_disable: dict[str, Any] | None = None
    samples_sent: list[dict[str, Any]] = []
    enabled_by_run = False
    trajectory_succeeded = False
    error: str | None = None
    try:
        runtime.start()
        time.sleep(1.5)
        initial = runtime.snapshot()
        before = _summary(initial)
        if not initial["all_joints_healthy"]:
            raise RuntimeError(f"right feedback is stale: {initial['healthy_joint_count']}/7")
        if initial["software_enabled"]:
            raise RuntimeError("right arm is already enabled; refusing to take ownership")
        seed = tuple(before["joint_positions_rad"])
        measured = controller.forward(seed)
        target_position = (measured.position[0], measured.position[1], measured.position[2] + OFFSET_M)
        ik = controller.inverse(target_position, measured.rpy, seed)
        target = tuple(ik.joints)
        planned = sample_smooth_movej(seed, target, DURATION_S)
        max_velocity = max(abs(float(v)) for sample in planned for v in sample.velocities)
        lower, upper = controller.kinematics.joint_limits
        margins = [min(float(v) - float(lo), float(hi) - float(v)) for v, lo, hi in zip(target, lower, upper, strict=True)]
        candidate = {
            "offset_m": OFFSET_M,
            "seed_joints_rad": list(seed),
            "measured_fk_pose": _pose(measured),
            "target_position_m": list(target_position),
            "target_rpy_rad": list(measured.rpy),
            "solution_found": bool(ik.success),
            "target_joints_rad": list(target),
            "position_error_m": float(ik.position_error),
            "rotation_error_rad": float(ik.rotation_error),
            "within_limits": all(value >= 0.0 for value in margins),
            "minimum_limit_margin_rad": min(margins),
            "duration_s": DURATION_S,
            "sample_count": len(planned),
            "max_velocity_rad_s": max_velocity,
            "velocity_limit_rad_s": MAX_VELOCITY_RAD_S,
            "within_velocity_limit": max_velocity <= MAX_VELOCITY_RAD_S,
            "motion_executed": False,
        }
        if not candidate["solution_found"] or not candidate["within_limits"] or not candidate["within_velocity_limit"]:
            raise RuntimeError("IK micro-motion preflight rejected the candidate")

        runtime.set_enabled(True, CONFIRMATION)
        enabled_by_run = True
        time.sleep(0.2)
        enabled = runtime.snapshot()
        if not enabled["software_enabled"] or not enabled["all_joints_healthy"]:
            raise RuntimeError("right arm did not remain enabled with fresh feedback")
        controller.smooth_movej(
            target,
            DURATION_S,
            block=True,
            confirmation=CONFIRMATION,
            on_sample=lambda sample: samples_sent.append(
                {"t_s": float(sample.time), "joints_rad": list(sample.positions)}
            ),
        )
        trajectory_succeeded = True
        time.sleep(0.3)
        after_motion = _summary(runtime.snapshot())
        final_error = max(abs(float(a) - float(b)) for a, b in zip(after_motion["joint_positions_rad"], target, strict=True))
        candidate["final_max_joint_error_rad"] = final_error
        if final_error > MAX_FINAL_ERROR_RAD:
            raise RuntimeError(f"final feedback error {final_error:.6f} rad exceeds limit")
        # Success path: leave the arm safe and disabled after the visual demo.
        runtime.set_enabled(False, None)
        time.sleep(0.2)
        after_disable = _summary(runtime.snapshot())
        if after_disable["software_enabled"] is not False:
            raise RuntimeError("right arm did not reach disabled state after success")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        # Before any target was dispatched, disable is safe.  After dispatch,
        # preserve enablement so a failed trajectory cannot drop the arm.
        if enabled_by_run and not samples_sent:
            try:
                runtime.set_enabled(False, None)
                after_disable = _summary(runtime.snapshot())
            except Exception as disable_error:
                error += f"; safe disable failed: {type(disable_error).__name__}: {disable_error}"
    finally:
        runtime.close()

    manifest = {
        "run_id": run_id,
        "stage": "P8-R3",
        "side": SIDE,
        "target": "right_ee_z_plus_1mm",
        "authorization": {"operator_confirmed": True, "confirmation_string": CONFIRMATION, "single_use": True},
        "hardware_connected": True,
        "hands_connected": False,
        "hands_skipped": True,
        "motion_commands_sent": len(samples_sent),
        "target_writes": len(samples_sent),
        "enable_changes": 2 if enabled_by_run and after_disable is not None else (1 if enabled_by_run else 0),
        "trajectory_samples_dispatched": len(samples_sent),
        "trajectory_succeeded": trajectory_succeeded,
        "motion_executed": bool(samples_sent),
        "before": before,
        "candidate": candidate,
        "after_motion": after_motion,
        "after_disable": after_disable,
        "error": error,
        "passed": bool(trajectory_succeeded and error is None and after_disable is not None),
        "note": "One authorized right-arm 1 mm IK micro-motion; no retry or second target; success path disables arm.",
    }
    (evidence_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"run_id": run_id, "passed": manifest["passed"], "trajectory_succeeded": trajectory_succeeded, "samples": len(samples_sent), "after_disabled": bool(after_disable and after_disable.get("software_enabled") is False), "evidence_dir": str(evidence_dir)}, ensure_ascii=False))
    return 0 if manifest["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

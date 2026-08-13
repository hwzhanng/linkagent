"""Generate P6 single-arm low-speed candidate trajectories without executing them."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

from module.runtime import RuntimeContext
from module.tools.env import MockEnv
from task.api_validation import HardwareReadOnlyValidationTask


ROOT = Path(__file__).resolve().parents[1]
VENDOR_ROOT = ROOT / "botSDK" / "lbot_arm_api-master"
if str(VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDOR_ROOT))
from tools.local_arm_can.trajectory import sample_smooth_movej  # noqa: E402


P6_DURATION_S = 5.0
P6_MAX_VELOCITY_RAD_S = 0.12
P6_TARGET = "z_plus_5mm"


def _arm_candidate(
    state: dict[str, Any], ik: dict[str, Any], *, side: str
) -> dict[str, Any]:
    seed = tuple(float(value) for value in ik["seed_joint_positions_rad"])
    target = tuple(float(value) for value in ik["joint_positions_rad"])
    samples = sample_smooth_movej(seed, target, P6_DURATION_S)
    max_velocity = max(
        abs(float(velocity))
        for sample in samples
        for velocity in sample.velocities
    )
    max_acceleration = max(
        abs(float(acceleration))
        for sample in samples
        for acceleration in sample.accelerations
    )
    deltas = [float(a - b) for a, b in zip(target, seed, strict=True)]
    return {
        "side": side,
        "target": P6_TARGET,
        "seed_source": "current_feedback_at_preflight",
        "seed_joints_rad": list(seed),
        "target_joints_rad": list(target),
        "duration_s": P6_DURATION_S,
        "sample_count": len(samples),
        "max_velocity_rad_s": max_velocity,
        "max_acceleration_rad_s2": max_acceleration,
        "max_joint_delta_rad": max(abs(value) for value in deltas),
        "j6_delta_rad": deltas[5],
        "j7_delta_rad": deltas[6],
        "trajectory_start_matches_seed": all(
            abs(float(a) - float(b)) <= 1.0e-12
            for a, b in zip(samples[0].positions, seed, strict=True)
        ),
        "trajectory_end_matches_target": all(
            abs(float(a) - float(b)) <= 1.0e-12
            for a, b in zip(samples[-1].positions, target, strict=True)
        ),
        "software_enabled": state["software_enabled"],
        "feedback_healthy": state["all_joints_healthy"],
        "ik_solution_found": ik["solution_found"],
        "ik_position_error_m": ik["position_error_m"],
        "ik_rotation_error_rad": ik["rotation_error_rad"],
        "within_p6_velocity_limit": max_velocity <= P6_MAX_VELOCITY_RAD_S,
        "motion_executed": False,
        "candidate_only": True,
        "physical_area_clear_confirmed": False,
        "physical_estop_confirmed": False,
        "urdf_preview_confirmed": False,
        "single_use_authorization_token": None,
        "ready_for_real_execution": False,
        "sampled_endpoints": {
            "start": list(samples[0].positions),
            "middle": list(samples[len(samples) // 2].positions),
            "end": list(samples[-1].positions),
        },
    }


def main() -> int:
    run_id = "p6-preflight-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    evidence_dir = ROOT / "logs" / "agent_skill_validation" / run_id
    evidence_dir.mkdir(parents=True, exist_ok=True)
    env = MockEnv(observations=[{"sequence": 0, "timestamp_s": 0.0}])
    observation = env.reset()
    context = RuntimeContext(env)
    context.update_observation(observation)
    task = HardwareReadOnlyValidationTask(
        required_tools=("get_live_arm_state", "compute_live_shadow_ik")
    )
    candidates: dict[str, Any] = {}
    blocking: list[str] = []
    try:
        tools = {tool.name: tool for tool in task.build_tools(context)}
        for side in ("left", "right"):
            state = tools["get_live_arm_state"].invoke({"side": side})
            ik = tools["compute_live_shadow_ik"].invoke(
                {"side": side, "target": P6_TARGET}
            )
            candidate = _arm_candidate(state, ik, side=side)
            candidates[side] = candidate
            if not candidate["feedback_healthy"]:
                blocking.append(f"{side} feedback is not 7/7 fresh")
            if candidate["software_enabled"] is not False:
                blocking.append(f"{side} software is enabled; preflight refuses execution")
            if not candidate["ik_solution_found"]:
                blocking.append(f"{side} {P6_TARGET} IK did not converge")
            if not candidate["within_p6_velocity_limit"]:
                blocking.append(f"{side} trajectory exceeds P6 velocity limit")
            if not candidate["trajectory_start_matches_seed"] or not candidate["trajectory_end_matches_target"]:
                blocking.append(f"{side} trajectory endpoints are not exact")
    except Exception as exc:
        blocking.append(f"preflight error: {type(exc).__name__}: {exc}")
    finally:
        task.close()
        env.close()

    manifest = {
        "run_id": run_id,
        "stage": "P6-preflight",
        "candidate_target": P6_TARGET,
        "duration_s": P6_DURATION_S,
        "max_velocity_limit_rad_s": P6_MAX_VELOCITY_RAD_S,
        "hardware_connected": True,
        "hands_connected": False,
        "control_enabled": False,
        "hardware_write_authorized": False,
        "motion_commands_sent": 0,
        "target_writes": 0,
        "enable_changes": 0,
        "physical_confirmation_required": True,
        "candidates": candidates,
        "blocking_reasons": blocking,
        "passed": not blocking,
        "ready_for_real_execution": False,
        "note": "Candidate trajectories only; P6 requires fresh operator confirmations and a single-use token.",
    }
    (evidence_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"run_id": run_id, "passed": manifest["passed"], "blocking_reasons": blocking, "evidence_dir": str(evidence_dir)}, ensure_ascii=False))
    return 0 if manifest["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

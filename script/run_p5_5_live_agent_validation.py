"""Run the P5.5 real Ark API -> Agent -> read-only hardware Skill checks."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

from module.agent import TopAgent, build_model
from module.agent.models._http import ProviderModelError
from module.runtime import RuntimeContext
from module.tools.env import MockEnv
from script.utils import load_mapping_file
from task.api_validation import HardwareReadOnlyValidationTask


ROOT = Path(__file__).resolve().parents[1]


def _tool_message(result: Mapping[str, Any]) -> Mapping[str, Any] | None:
    messages = result.get("messages", [])
    if not isinstance(messages, list):
        return None
    return next(
        (
            message
            for message in messages
            if isinstance(message, Mapping) and message.get("role") == "tool"
        ),
        None,
    )


def _baseline() -> Mapping[str, Any] | None:
    candidates = sorted(
        (ROOT / "logs" / "agent_skill_validation").glob("p5-*/manifest.json")
    )
    if not candidates:
        return None
    try:
        payload = json.loads(candidates[-1].read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, Mapping) else None


def _max_joint_delta_to_baseline(
    side: str, content: Mapping[str, Any], baseline: Mapping[str, Any] | None
) -> float | None:
    if baseline is None:
        return None
    arms = baseline.get("arms", {})
    if not isinstance(arms, Mapping):
        return None
    arm = arms.get(side, {})
    if not isinstance(arm, Mapping):
        return None
    seed = arm.get("seed_joints_rad")
    current = content.get("joint_positions_rad")
    if not isinstance(seed, list) or not isinstance(current, list) or len(seed) != len(current):
        return None
    try:
        return max(abs(float(a) - float(b)) for a, b in zip(seed, current, strict=True))
    except (TypeError, ValueError):
        return None


def main() -> int:
    if not os.environ.get("ARK_API_KEY"):
        print("ERROR: ARK_API_KEY is not set in this process", file=sys.stderr)
        return 2
    model_profiles = load_mapping_file(ROOT / "config" / "_model.yaml")
    model_config = model_profiles["ark_openai_doubao"]
    model = build_model(model_config)
    baseline = _baseline()
    run_id = "p5.5-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    evidence_dir = ROOT / "logs" / "agent_skill_validation" / run_id
    evidence_dir.mkdir(parents=True, exist_ok=True)
    cases = [
        ("get_live_arm_state", {"side": "left"}),
        ("get_live_arm_state", {"side": "right"}),
        ("compute_live_shadow_ik", {"side": "left", "target": "z_plus_5mm"}),
        ("compute_live_shadow_ik", {"side": "right", "target": "z_plus_5mm"}),
        ("get_live_camera_preview", {}),
    ]
    records: list[dict[str, Any]] = []
    provider_error: str | None = None
    for index, (expected_tool, expected_arguments) in enumerate(cases, start=1):
        task = HardwareReadOnlyValidationTask(required_tools=(expected_tool,))
        env = MockEnv(observations=[{"sequence": 0, "timestamp_s": 0.0}])
        context = RuntimeContext(env)
        observation = env.reset()
        context.update_observation(observation)
        try:
            if expected_tool == "get_live_arm_state":
                hint = f"side={expected_arguments['side']}"
            elif expected_tool == "compute_live_shadow_ik":
                hint = (
                    f"side={expected_arguments['side']}, target={expected_arguments['target']}"
                )
            else:
                hint = "no arguments"
            prompt = (
                f"调用 {expected_tool} 一次，参数必须是 {hint}。不要调用其他工具。"
                "这是 P5.5 只读验证；调用后用中文简短报告真实返回，明确没有执行运动。"
            )
            agent = TopAgent(
                model=model,
                tools=task.build_tools(context),
                system_prompt=task.build_prompt(),
                max_iterations=3,
                max_tool_calls=1,
                max_tool_calls_per_turn=1,
            )
            try:
                result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
            except ProviderModelError as exc:
                # Stop after the first provider failure.  Do not retry a bad
                # credential five times and do not include any credential in
                # evidence.
                provider_error = f"{type(exc).__name__}: {exc}"
                records.append(
                    {
                        "case": index,
                        "expected_tool": expected_tool,
                        "expected_arguments": dict(expected_arguments),
                        "status": "provider_error",
                        "success": False,
                        "error": provider_error,
                    }
                )
                break
            message = _tool_message(result)
            content = message.get("content") if message else None
            success = bool(
                message
                and message.get("name") == expected_tool
                and message.get("status") == "success"
                and isinstance(content, Mapping)
                and content.get("hardware_accessed") is True
                and content.get("motion_executed") is False
                and content.get("write_control_authorized") is False
            )
            if expected_tool == "get_live_arm_state":
                success = success and content.get("arm") == expected_arguments["side"]  # type: ignore[union-attr]
                success = success and content.get("all_joints_healthy") is True  # type: ignore[union-attr]
                success = success and content.get("software_enabled") is False  # type: ignore[union-attr]
            elif expected_tool == "compute_live_shadow_ik":
                success = success and content.get("side") == expected_arguments["side"]  # type: ignore[union-attr]
                success = success and content.get("target") == expected_arguments["target"]  # type: ignore[union-attr]
                success = success and content.get("solution_found") is True  # type: ignore[union-attr]
                success = success and float(content.get("position_error_m", 1.0)) <= 2.0e-5  # type: ignore[union-attr]
            else:
                success = success and content.get("rgb_depth_live") is True  # type: ignore[union-attr]
            record: dict[str, Any] = {
                "case": index,
                "expected_tool": expected_tool,
                "expected_arguments": dict(expected_arguments),
                "tool_call_id": message.get("tool_call_id") if message else None,
                "status": message.get("status") if message else "missing",
                "execution_id": content.get("execution_id") if isinstance(content, Mapping) else None,
                "success": success,
                "model_output": result.get("output"),
            }
            if isinstance(content, Mapping):
                record["hardware_accessed"] = content.get("hardware_accessed")
                record["motion_executed"] = content.get("motion_executed")
                record["write_control_authorized"] = content.get("write_control_authorized")
                if expected_tool == "get_live_arm_state":
                    record["side"] = content.get("arm")
                    record["healthy_joint_count"] = content.get("healthy_joint_count")
                    record["software_enabled"] = content.get("software_enabled")
                    record["max_joint_delta_to_p5_baseline"] = _max_joint_delta_to_baseline(
                        str(content.get("arm")), content, baseline
                    )
                if expected_tool == "compute_live_shadow_ik":
                    record["solution_found"] = content.get("solution_found")
                    record["position_error_m"] = content.get("position_error_m")
                    record["j6_delta_rad"] = content.get("j6_delta_rad")
                    record["j7_delta_rad"] = content.get("j7_delta_rad")
                if expected_tool == "get_live_camera_preview":
                    device = content.get("device", {})
                    record["camera_serial"] = device.get("serial") if isinstance(device, Mapping) else None
                    record["rgb_depth_live"] = content.get("rgb_depth_live")
            records.append(record)
        finally:
            task.close()
            env.close()

    passed = len(records) == len(cases) and all(record["success"] for record in records)
    manifest = {
        "run_id": run_id,
        "stage": "P5.5",
        "provider": "ark_openai_responses",
        "model": model_config.get("model"),
        "base_url": model_config.get("base_url"),
        "api_key_recorded": False,
        "hardware_connected": True,
        "hands_connected": False,
        "control_enabled": False,
        "hardware_write_authorized": False,
        "motion_commands_sent": 0,
        "target_writes": 0,
        "enable_changes": 0,
        "p5_baseline": str(baseline.get("run_id")) if baseline else None,
        "provider_error": provider_error,
        "records": records,
        "passed": passed,
    }
    (evidence_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {"run_id": run_id, "passed": passed, "cases": len(records), "evidence_dir": str(evidence_dir)},
            ensure_ascii=False,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

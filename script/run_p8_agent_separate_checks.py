"""Run three independent *real API Agent* checks for the current safe scope.

Each case creates a fresh read-only Task and TopAgent, and the only expected
tool call is made by the Ark model.  This is intentionally not a direct local
Skill invocation and it does not access the hand.
"""

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
CASES = (
    ("get_live_arm_state", {"side": "right"}),
    ("compute_live_shadow_ik", {"side": "right", "target": "z_plus_5mm"}),
    ("get_live_camera_preview", {}),
)


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


def _validate(name: str, message: Mapping[str, Any] | None) -> tuple[bool, dict[str, Any]]:
    content = message.get("content") if message else None
    record: dict[str, Any] = {
        "expected_tool": name,
        "tool_call_id": message.get("tool_call_id") if message else None,
        "status": message.get("status") if message else "missing",
        "execution_id": content.get("execution_id") if isinstance(content, Mapping) else None,
        "success": False,
    }
    if not isinstance(content, Mapping):
        return False, record
    common = (
        message is not None
        and message.get("name") == name
        and message.get("status") == "success"
        and content.get("hardware_accessed") is True
        and content.get("motion_executed") is False
        and content.get("write_control_authorized") is False
    )
    record.update(
        {
            "hardware_accessed": content.get("hardware_accessed"),
            "motion_executed": content.get("motion_executed"),
            "write_control_authorized": content.get("write_control_authorized"),
        }
    )
    if name == "get_live_arm_state":
        record.update(
            {
                "arm": content.get("arm"),
                "healthy_joint_count": content.get("healthy_joint_count"),
                "all_joints_healthy": content.get("all_joints_healthy"),
                "software_enabled": content.get("software_enabled"),
            }
        )
        ok = common and content.get("arm") == "right" and content.get("healthy_joint_count") == 7 and content.get("all_joints_healthy") is True and content.get("software_enabled") is False
    elif name == "compute_live_shadow_ik":
        record.update(
            {
                "side": content.get("side"),
                "target": content.get("target"),
                "solution_found": content.get("solution_found"),
                "position_error_m": content.get("position_error_m"),
                "rotation_error_rad": content.get("rotation_error_rad"),
                "within_limits": content.get("within_limits"),
            }
        )
        try:
            position_ok = float(content.get("position_error_m", 1.0)) <= 2.0e-5
        except (TypeError, ValueError):
            position_ok = False
        ok = common and content.get("side") == "right" and content.get("target") == "z_plus_5mm" and content.get("solution_found") is True and content.get("within_limits") is True and position_ok
    else:
        device = content.get("device", {})
        record.update(
            {
                "camera_serial": device.get("serial") if isinstance(device, Mapping) else None,
                "rgb_depth_live": content.get("rgb_depth_live"),
                "viewer_url": content.get("viewer_url"),
                "rgb_stream_url": content.get("rgb_stream_url"),
                "depth_stream_url": content.get("depth_stream_url"),
            }
        )
        ok = common and content.get("rgb_depth_live") is True
    record["success"] = bool(ok)
    return bool(ok), record


def main() -> int:
    if not os.environ.get("ARK_API_KEY"):
        print("ERROR: ARK_API_KEY is not set in this process", file=sys.stderr)
        return 2
    profiles = load_mapping_file(ROOT / "config" / "_model.yaml")
    model_config = profiles["ark_openai_doubao"]
    model = build_model(model_config)
    run_id = "p8-separate-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    evidence_dir = ROOT / "logs" / "agent_skill_validation" / run_id
    evidence_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    provider_errors: list[str] = []

    for tool_name, arguments in CASES:
        task = HardwareReadOnlyValidationTask(required_tools=(tool_name,))
        env = MockEnv(observations=[{"sequence": 0, "timestamp_s": 0.0}])
        context = RuntimeContext(env)
        context.update_observation(env.reset())
        try:
            hint = ", ".join(f"{key}={value}" for key, value in arguments.items()) or "无参数"
            prompt = (
                f"只调用一次工具 {tool_name}，参数为 {hint}，不要调用其他工具。"
                "这是 P8 真实硬件只读验证；调用后用中文报告真实返回，并明确"
                "hardware_accessed=true、motion_executed=false、write_control_authorized=false。"
            )
            result: Mapping[str, Any]
            try:
                agent = TopAgent(
                    model=model,
                    tools=task.build_tools(context),
                    system_prompt=task.build_prompt() + " 当前仅验证右臂，灵巧手和触觉跳过。",
                    max_iterations=3,
                    max_tool_calls=1,
                    max_tool_calls_per_turn=1,
                )
                result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
            except ProviderModelError as exc:
                error = f"{type(exc).__name__}: {exc}"
                provider_errors.append(error)
                records.append({"expected_tool": tool_name, "success": False, "provider_error": error})
                continue
            message = _tool_message(result)
            ok, record = _validate(tool_name, message)
            record["model_output"] = result.get("output")
            records.append(record)
        finally:
            task.close()
            env.close()

    passed = len(records) == len(CASES) and all(record.get("success") is True for record in records)
    manifest = {
        "run_id": run_id,
        "stage": "P8",
        "objective": "three real Ark Agent calls: right-arm state, Shadow IK, RGB-D preview",
        "provider": "ark_openai_responses",
        "model": model_config.get("model"),
        "base_url": model_config.get("base_url"),
        "api_key_recorded": False,
        "agent_invocations": len(CASES),
        "hands_connected": False,
        "hands_skipped": True,
        "hardware_write_authorized": False,
        "motion_commands_sent": 0,
        "target_writes": 0,
        "enable_changes": 0,
        "provider_errors": provider_errors,
        "records": records,
        "passed": passed,
    }
    (evidence_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"run_id": run_id, "passed": passed, "agent_invocations": len(CASES), "evidence_dir": str(evidence_dir)}, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Run one real Ark Agent invocation for right-arm state, Shadow IK and RGB-D.

The task is intentionally read-only and does not expose the hand, tactile,
enable, reset, trajectory, or emergency-stop controls.
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
REQUIRED = (
    "get_live_arm_state",
    "compute_live_shadow_ik",
    "get_live_camera_preview",
)


def _tool_messages(result: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    messages = result.get("messages", [])
    if not isinstance(messages, list):
        return []
    return [
        message
        for message in messages
        if isinstance(message, Mapping) and message.get("role") == "tool"
    ]


def _record(message: Mapping[str, Any]) -> dict[str, Any]:
    content = message.get("content")
    name = message.get("name")
    result: dict[str, Any] = {
        "name": name,
        "tool_call_id": message.get("tool_call_id"),
        "status": message.get("status"),
        "execution_id": content.get("execution_id") if isinstance(content, Mapping) else None,
        "success": False,
    }
    if not isinstance(content, Mapping):
        return result
    result.update(
        {
            "hardware_accessed": content.get("hardware_accessed"),
            "motion_executed": content.get("motion_executed"),
            "write_control_authorized": content.get("write_control_authorized"),
        }
    )
    common_ok = (
        message.get("status") == "success"
        and content.get("hardware_accessed") is True
        and content.get("motion_executed") is False
        and content.get("write_control_authorized") is False
    )
    if name == "get_live_arm_state":
        result.update(
            {
                "arm": content.get("arm"),
                "healthy_joint_count": content.get("healthy_joint_count"),
                "all_joints_healthy": content.get("all_joints_healthy"),
                "software_enabled": content.get("software_enabled"),
            }
        )
        result["success"] = bool(
            common_ok
            and content.get("arm") == "right"
            and content.get("healthy_joint_count") == 7
            and content.get("all_joints_healthy") is True
            and content.get("software_enabled") is False
        )
    elif name == "compute_live_shadow_ik":
        result.update(
            {
                "side": content.get("side"),
                "target": content.get("target"),
                "solution_found": content.get("solution_found"),
                "position_error_m": content.get("position_error_m"),
                "rotation_error_rad": content.get("rotation_error_rad"),
                "within_limits": content.get("within_limits"),
            }
        )
        result["success"] = bool(
            common_ok
            and content.get("side") == "right"
            and content.get("target") == "z_plus_5mm"
            and content.get("solution_found") is True
            and content.get("within_limits") is True
            and float(content.get("position_error_m", 1.0)) <= 2.0e-5
        )
    elif name == "get_live_camera_preview":
        device = content.get("device", {})
        result.update(
            {
                "camera_serial": device.get("serial") if isinstance(device, Mapping) else None,
                "rgb_depth_live": content.get("rgb_depth_live"),
                "viewer_url": content.get("viewer_url"),
                "rgb_stream_url": content.get("rgb_stream_url"),
                "depth_stream_url": content.get("depth_stream_url"),
            }
        )
        result["success"] = bool(common_ok and content.get("rgb_depth_live") is True)
    return result


def main() -> int:
    if not os.environ.get("ARK_API_KEY"):
        print("ERROR: ARK_API_KEY is not set in this process", file=sys.stderr)
        return 2
    model_profiles = load_mapping_file(ROOT / "config" / "_model.yaml")
    model_config = model_profiles["ark_openai_doubao"]
    model = build_model(model_config)
    run_id = "p8-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    evidence_dir = ROOT / "logs" / "agent_skill_validation" / run_id
    evidence_dir.mkdir(parents=True, exist_ok=True)

    task = HardwareReadOnlyValidationTask(required_tools=REQUIRED)
    env = MockEnv(observations=[{"sequence": 0, "timestamp_s": 0.0}])
    context = RuntimeContext(env)
    context.update_observation(env.reset())
    prompt = (
        "执行 P8 真实硬件只读组合验证。在本次 Agent 调用中各调用一次且仅调用以下三个工具："
        "get_live_arm_state(side=right)、compute_live_shadow_ik(side=right,target=z_plus_5mm)、"
        "get_live_camera_preview()。禁止调用任何其他工具，禁止访问灵巧手和触觉，禁止使能、"
        "复位、轨迹、目标写入、急停或任何运动。最后用中文汇总七关节状态、Shadow IK/FK"
        "误差和 RGB/Depth live 状态，并明确 hardware_accessed=true、motion_executed=false、"
        "write_control_authorized=false。"
    )
    result: Mapping[str, Any] = {"messages": [], "output": None}
    provider_error: str | None = None
    try:
        agent = TopAgent(
            model=model,
            tools=task.build_tools(context),
            system_prompt=task.build_prompt() + " 当前阶段为 P8，灵巧手和触觉明确跳过。",
            max_iterations=6,
            max_tool_calls=3,
            # Some Ark Responses turns return an empty output when asked to
            # emit three parallel calls.  Keep one call per model turn while
            # retaining one TopAgent invocation and the same three-tool
            # whitelist; this preserves the orchestration proof and gives the
            # provider a normal sequential tool-call exchange.
            max_tool_calls_per_turn=1,
        )
        try:
            result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
        except ProviderModelError as exc:
            provider_error = f"{type(exc).__name__}: {exc}"
    finally:
        task.close()
        env.close()

    records = [_record(message) for message in _tool_messages(result)]
    names = [record.get("name") for record in records]
    passed = (
        provider_error is None
        and len(records) == len(REQUIRED)
        and set(names) == set(REQUIRED)
        and len(names) == len(set(names))
        and all(record.get("success") is True for record in records)
    )
    manifest = {
        "run_id": run_id,
        "stage": "P8",
        "objective": "one Agent invocation reads right arm, computes read-only Shadow IK, and reads RGB-D",
        "provider": "ark_openai_responses",
        "model": model_config.get("model"),
        "base_url": model_config.get("base_url"),
        "api_key_recorded": False,
        "right_arm_selected": True,
        "right_arm_expected_software_enabled": False,
        "hands_connected": False,
        "hands_skipped": True,
        "control_enabled": False,
        "hardware_write_authorized": False,
        "motion_commands_sent": 0,
        "target_writes": 0,
        "enable_changes": 0,
        "agent_invocations": 1,
        "required_tools": list(REQUIRED),
        "provider_error": provider_error,
        "tool_records": records,
        "model_output": result.get("output"),
        "passed": passed,
    }
    (evidence_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "run_id": run_id,
                "passed": passed,
                "agent_invocations": 1,
                "tool_calls": len(records),
                "evidence_dir": str(evidence_dir),
            },
            ensure_ascii=False,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

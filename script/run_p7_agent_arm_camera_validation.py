"""Run one real Ark Agent invocation that reads the right arm and RGB-D camera.

This stage is deliberately read-only.  The hand is not connected and is not
exposed as a tool.  No IK, enable, target, trajectory, reset, or emergency-stop
operation is available to the Agent.
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


def _tool_messages(result: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    messages = result.get("messages", [])
    if not isinstance(messages, list):
        return []
    return [
        message
        for message in messages
        if isinstance(message, Mapping) and message.get("role") == "tool"
    ]


def _safe_tool_record(message: Mapping[str, Any]) -> dict[str, Any]:
    content = message.get("content")
    record: dict[str, Any] = {
        "name": message.get("name"),
        "tool_call_id": message.get("tool_call_id"),
        "status": message.get("status"),
        "execution_id": content.get("execution_id") if isinstance(content, Mapping) else None,
        "success": False,
    }
    if not isinstance(content, Mapping):
        return record
    record.update(
        {
            "hardware_accessed": content.get("hardware_accessed"),
            "motion_executed": content.get("motion_executed"),
            "write_control_authorized": content.get("write_control_authorized"),
        }
    )
    if message.get("name") == "get_live_arm_state":
        record.update(
            {
                "arm": content.get("arm"),
                "healthy_joint_count": content.get("healthy_joint_count"),
                "all_joints_healthy": content.get("all_joints_healthy"),
                "software_enabled": content.get("software_enabled"),
                "interface": content.get("interface"),
            }
        )
        record["success"] = bool(
            message.get("status") == "success"
            and content.get("hardware_accessed") is True
            and content.get("motion_executed") is False
            and content.get("write_control_authorized") is False
            and content.get("arm") == "right"
            and content.get("healthy_joint_count") == 7
            and content.get("all_joints_healthy") is True
            and content.get("software_enabled") is False
        )
    elif message.get("name") == "get_live_camera_preview":
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
        record["success"] = bool(
            message.get("status") == "success"
            and content.get("hardware_accessed") is True
            and content.get("motion_executed") is False
            and content.get("write_control_authorized") is False
            and content.get("rgb_depth_live") is True
        )
    return record


def main() -> int:
    if not os.environ.get("ARK_API_KEY"):
        print("ERROR: ARK_API_KEY is not set in this process", file=sys.stderr)
        return 2

    model_profiles = load_mapping_file(ROOT / "config" / "_model.yaml")
    model_config = model_profiles["ark_openai_doubao"]
    model = build_model(model_config)
    run_id = "p7-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    evidence_dir = ROOT / "logs" / "agent_skill_validation" / run_id
    evidence_dir.mkdir(parents=True, exist_ok=True)

    # One Task and one TopAgent invocation.  Only the right arm state and camera
    # tools are exposed; compute IK and all hand tools are intentionally absent.
    task = HardwareReadOnlyValidationTask(
        required_tools=("get_live_arm_state", "get_live_camera_preview")
    )
    env = MockEnv(observations=[{"sequence": 0, "timestamp_s": 0.0}])
    context = RuntimeContext(env)
    observation = env.reset()
    context.update_observation(observation)
    prompt = (
        "执行 P7 臂与相机组合只读验证，必须在本次调用中各调用一次且仅调用以下两个工具："
        "get_live_arm_state，参数 side=right；get_live_camera_preview，无参数。"
        "不要调用 IK、手部、使能、复位、目标写入、轨迹、急停或任何其他工具。"
        "调用完成后用中文报告右臂七关节健康数、软件使能状态、RGB/Depth 是否 live，"
        "并明确 hardware_accessed=true、motion_executed=false、write_control_authorized=false。"
    )
    provider_error: str | None = None
    result: Mapping[str, Any] = {"messages": [], "output": None}
    try:
        agent = TopAgent(
            model=model,
            tools=task.build_tools(context),
            system_prompt=task.build_prompt()
            + " 本阶段仅验证右臂和相机，灵巧手已跳过。",
            max_iterations=5,
            max_tool_calls=2,
            max_tool_calls_per_turn=2,
        )
        try:
            result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
        except ProviderModelError as exc:
            provider_error = f"{type(exc).__name__}: {exc}"
    finally:
        task.close()
        env.close()

    tool_records = [_safe_tool_record(message) for message in _tool_messages(result)]
    names = [record.get("name") for record in tool_records]
    expected = {"get_live_arm_state", "get_live_camera_preview"}
    passed = (
        provider_error is None
        and len(tool_records) == 2
        and set(names) == expected
        and len(names) == len(set(names))
        and all(record.get("success") is True for record in tool_records)
    )
    manifest = {
        "run_id": run_id,
        "stage": "P7",
        "objective": "one Agent invocation reads right arm state and RGB-D camera",
        "provider": "ark_openai_responses",
        "model": model_config.get("model"),
        "base_url": model_config.get("base_url"),
        "api_key_recorded": False,
        "hardware_connected": True,
        "right_arm_selected": True,
        "right_arm_disable_authorized": True,
        "hands_connected": False,
        "hands_skipped": True,
        "control_enabled": False,
        "hardware_write_authorized": False,
        "motion_commands_sent": 0,
        "target_writes": 0,
        "enable_changes": 0,
        "provider_error": provider_error,
        "agent_invocations": 1,
        "required_tools": ["get_live_arm_state", "get_live_camera_preview"],
        "tool_records": tool_records,
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
                "tool_calls": len(tool_records),
                "evidence_dir": str(evidence_dir),
            },
            ensure_ascii=False,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

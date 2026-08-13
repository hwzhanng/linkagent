"""Run P3 with the real model API and deterministic fake/read-only backends."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping

from module.agent import TopAgent, build_model
from module.runtime import RuntimeContext
from module.tools.env import MockEnv
from script.utils import load_mapping_file
from task.api_validation import HardwareCapabilityValidationTask


ROOT = Path(__file__).resolve().parents[1]


def _tool_message(result: Mapping[str, Any]) -> Mapping[str, Any] | None:
    messages = result.get("messages", [])
    if not isinstance(messages, list):
        return None
    return next(
        (m for m in messages if isinstance(m, Mapping) and m.get("role") == "tool"),
        None,
    )


def main() -> int:
    if not __import__("os").environ.get("ARK_API_KEY"):
        print("ERROR: ARK_API_KEY is not set in this process", file=sys.stderr)
        return 2
    profiles = load_mapping_file(ROOT / "config" / "_model.yaml")
    model_config = profiles["ark_openai_doubao"]
    model = build_model(model_config)
    env = MockEnv(observations=[{"sequence": 0, "timestamp_s": 0.0}])
    context = RuntimeContext(env)
    context.update_observation(env.reset())
    run_id = "p3-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    evidence_dir = ROOT / "logs" / "agent_skill_validation" / run_id
    evidence_dir.mkdir(parents=True, exist_ok=True)
    cases = (
        ("get_validation_capabilities", "读取当前 P3 只读能力和门禁状态"),
        ("probe_inverse_kinematics", "对 world 坐标 [0.0, 0.0, 0.70] 计算 IK，不得运动"),
        ("probe_camera_preview", "读取 wrist 相机 replay 第 0 帧，不得打开真实相机"),
    )
    records: list[dict[str, Any]] = []
    try:
        for index, (tool_name, instruction) in enumerate(cases, start=1):
            task = HardwareCapabilityValidationTask(
                stage="p3", required_tools=(tool_name,)
            )
            try:
                agent = TopAgent(
                    model=model,
                    tools=task.build_tools(context),
                    system_prompt=(
                        "这是 P3 Fake Gateway/replay 验证。只能调用白名单中的一个工具。"
                        "所有结果都是 simulated；禁止声称真实机械臂或相机已经动作。"
                    ),
                    max_iterations=3,
                    max_tool_calls=1,
                    max_tool_calls_per_turn=1,
                )
                result = agent.invoke(
                    {"messages": [{"role": "user", "content": instruction}]}
                )
                message = _tool_message(result)
                content = message.get("content") if message else None
                success = bool(
                    message
                    and message.get("name") == tool_name
                    and message.get("status") == "success"
                    and isinstance(content, Mapping)
                    and content.get("simulated") is True
                    and content.get("hardware_accessed") is False
                    and content.get("motion_executed") is False
                )
                if tool_name == "probe_inverse_kinematics":
                    success = success and content.get("max_position_residual_m") == 0.0 and content.get("solution_found") is True  # type: ignore[union-attr]
                if tool_name == "probe_camera_preview":
                    success = success and content.get("rgb_live") is True and content.get("depth_live") is True  # type: ignore[union-attr]
                records.append(
                    {
                        "case": index,
                        "tool": tool_name,
                        "tool_call_id": message.get("tool_call_id") if message else None,
                        "execution_id": content.get("execution_id") if isinstance(content, Mapping) else None,
                        "success": success,
                        "content": content,
                        "model_output": result.get("output"),
                    }
                )
            finally:
                task.close()
    finally:
        env.close()
    passed = len(records) == 3 and all(record["success"] for record in records)
    manifest = {
        "run_id": run_id,
        "stage": "P3",
        "provider": "ark_openai_responses",
        "model": model_config.get("model"),
        "base_url": model_config.get("base_url"),
        "backend": "fake_gateway_shadow_and_rgbd_replay",
        "hardware_connected": False,
        "control_enabled": False,
        "api_key_recorded": False,
        "passed": passed,
        "records": records,
    }
    (evidence_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"run_id": run_id, "passed": passed, "cases": len(records), "evidence_dir": str(evidence_dir)}, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

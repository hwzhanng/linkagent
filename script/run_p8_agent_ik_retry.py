"""Retry only the real Ark Agent -> live Shadow IK call, read-only."""

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
        (m for m in messages if isinstance(m, Mapping) and m.get("role") == "tool"),
        None,
    )


def main() -> int:
    if not os.environ.get("ARK_API_KEY"):
        print("ERROR: ARK_API_KEY is not set in this process", file=sys.stderr)
        return 2
    profiles = load_mapping_file(ROOT / "config" / "_model.yaml")
    model_config = profiles["ark_openai_doubao"]
    model = build_model(model_config)
    run_id = "p8-ik-retry-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    evidence_dir = ROOT / "logs" / "agent_skill_validation" / run_id
    evidence_dir.mkdir(parents=True, exist_ok=True)
    attempts: list[dict[str, Any]] = []
    for attempt in range(1, 4):
        task = HardwareReadOnlyValidationTask(required_tools=("compute_live_shadow_ik",))
        env = MockEnv(observations=[{"sequence": 0, "timestamp_s": 0.0}])
        context = RuntimeContext(env)
        context.update_observation(env.reset())
        try:
            prompt = (
                "只调用一次 compute_live_shadow_ik 工具，参数必须是 side=right、target=z_plus_5mm。"
                "这是实时机械臂只读 Shadow IK 验证；禁止调用其他工具，禁止使能、复位、目标写入、"
                "轨迹或任何运动。调用后用中文报告 solution_found、FK/IK 位置误差和限位状态，并明确"
                "hardware_accessed=true、motion_executed=false、write_control_authorized=false。"
            )
            record: dict[str, Any] = {"attempt": attempt, "success": False}
            try:
                agent = TopAgent(
                    model=model,
                    tools=task.build_tools(context),
                    system_prompt=task.build_prompt() + " 当前仅验证右臂 Shadow IK，灵巧手和触觉跳过。",
                    max_iterations=3,
                    max_tool_calls=1,
                    max_tool_calls_per_turn=1,
                )
                result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
            except ProviderModelError as exc:
                record["provider_error"] = f"{type(exc).__name__}: {exc}"
                attempts.append(record)
                continue
            message = _tool_message(result)
            content = message.get("content") if message else None
            record.update(
                {
                    "tool_call_id": message.get("tool_call_id") if message else None,
                    "status": message.get("status") if message else "missing",
                    "execution_id": content.get("execution_id") if isinstance(content, Mapping) else None,
                    "model_output": result.get("output"),
                }
            )
            if isinstance(content, Mapping):
                record.update(
                    {
                        "hardware_accessed": content.get("hardware_accessed"),
                        "motion_executed": content.get("motion_executed"),
                        "write_control_authorized": content.get("write_control_authorized"),
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
                record["success"] = bool(
                    message
                    and message.get("name") == "compute_live_shadow_ik"
                    and message.get("status") == "success"
                    and content.get("hardware_accessed") is True
                    and content.get("motion_executed") is False
                    and content.get("write_control_authorized") is False
                    and content.get("side") == "right"
                    and content.get("target") == "z_plus_5mm"
                    and content.get("solution_found") is True
                    and content.get("within_limits") is True
                    and position_ok
                )
            attempts.append(record)
            if record["success"]:
                break
        finally:
            task.close()
            env.close()
    passed = any(record.get("success") is True for record in attempts)
    manifest = {
        "run_id": run_id,
        "stage": "P8",
        "objective": "real Ark Agent invokes right-arm live Shadow IK",
        "provider": "ark_openai_responses",
        "model": model_config.get("model"),
        "base_url": model_config.get("base_url"),
        "api_key_recorded": False,
        "hands_connected": False,
        "hands_skipped": True,
        "hardware_write_authorized": False,
        "motion_commands_sent": 0,
        "target_writes": 0,
        "enable_changes": 0,
        "attempts": attempts,
        "passed": passed,
    }
    (evidence_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"run_id": run_id, "passed": passed, "attempts": len(attempts), "evidence_dir": str(evidence_dir)}, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

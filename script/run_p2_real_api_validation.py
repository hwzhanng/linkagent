"""Run the P2 real-API, read-only capability validation.

The API key is read only from ``ARK_API_KEY`` by the provider factory.  This
script never writes credentials to evidence files and never constructs a robot
or camera backend.
"""

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
        (
            message
            for message in messages
            if isinstance(message, Mapping) and message.get("role") == "tool"
        ),
        None,
    )


def main() -> int:
    if not __import__("os").environ.get("ARK_API_KEY"):
        print("ERROR: ARK_API_KEY is not set in this process", file=sys.stderr)
        return 2

    model_profiles = load_mapping_file(ROOT / "config" / "_model.yaml")
    model_config = model_profiles["ark_openai_doubao"]
    model = build_model(model_config)
    env = MockEnv(observations=[{"sequence": 0, "timestamp_s": 0.0}])
    context = RuntimeContext(env)
    context.update_observation(env.reset())
    run_id = "p2-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    evidence_dir = ROOT / "logs" / "agent_skill_validation" / run_id
    evidence_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    cases = [
        ("get_validation_capabilities", None) for _ in range(5)
    ] + [
        ("probe_nonce", f"p2-nonce-{index:02d}") for index in range(5)
    ]

    try:
        for index, (expected_tool, nonce) in enumerate(cases, start=1):
            task = HardwareCapabilityValidationTask(
                stage="p2",
                required_tools=(expected_tool,),
                expected_nonce=nonce,
            )
            try:
                argument_hint = (
                    f" with nonce exactly {nonce!r}" if nonce is not None else ""
                )
                prompt = (
                    f"调用 {expected_tool} 一次{argument_hint}。"
                    "不要调用任何其他工具。工具成功后用一句中文确认，"
                    "不要声称访问了真实硬件。"
                )
                agent = TopAgent(
                    model=model,
                    tools=task.build_tools(context),
                    system_prompt=(
                        "这是 P2 真实 API 只读验证。只能使用当前白名单中的一个工具；"
                        "禁止机械臂、灵巧手、CAN、相机或网络设备控制。"
                    ),
                    max_iterations=3,
                    max_tool_calls=1,
                    max_tool_calls_per_turn=1,
                )
                result = agent.invoke(
                    {"messages": [{"role": "user", "content": prompt}]}
                )
                message = _tool_message(result)
                content = message.get("content") if message else None
                success = bool(
                    message
                    and message.get("name") == expected_tool
                    and message.get("status") == "success"
                    and isinstance(content, Mapping)
                    and content.get("hardware_accessed") is False
                    and content.get("motion_executed") is False
                )
                if expected_tool == "probe_nonce":
                    success = success and content.get("nonce") == nonce  # type: ignore[union-attr]
                records.append(
                    {
                        "case": index,
                        "expected_tool": expected_tool,
                        "tool_call_id": message.get("tool_call_id") if message else None,
                        "status": message.get("status") if message else "missing",
                        "execution_id": content.get("execution_id") if isinstance(content, Mapping) else None,
                        "handler_calls": task.state.nonce_calls + task.state.capability_calls,
                        "success": success,
                        "model_output": result.get("output"),
                    }
                )
            finally:
                task.close()
    finally:
        env.close()

    passed = len(records) == 10 and all(record["success"] for record in records)
    manifest = {
        "run_id": run_id,
        "stage": "P2",
        "provider": "ark_openai_responses",
        "model": model_config.get("model"),
        "base_url": model_config.get("base_url"),
        "hardware_connected": False,
        "control_enabled": False,
        "api_key_recorded": False,
        "independent_tool_workflows": len(records),
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

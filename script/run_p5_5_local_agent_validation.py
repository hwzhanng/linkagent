"""Exercise the P5.5 Agent tool boundary with a local scripted model.

This is not a substitute for the real Ark API.  It proves that the project's
TopAgent dispatches the new live, read-only tools and that their real hardware
results are returned without motion.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from module.agent import TopAgent
from module.agent.models.mock import MockChatModel
from module.runtime import RuntimeContext
from module.tools.env import MockEnv
from task.api_validation import HardwareReadOnlyValidationTask


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    run_id = "p5.5-local-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    evidence_dir = ROOT / "logs" / "agent_skill_validation" / run_id
    evidence_dir.mkdir(parents=True, exist_ok=True)
    env = MockEnv(observations=[{"sequence": 0, "timestamp_s": 0.0}])
    observation = env.reset()
    context = RuntimeContext(env)
    context.update_observation(observation)
    task = HardwareReadOnlyValidationTask()
    model = MockChatModel(
        tool_sequence=[
            {"name": "get_live_arm_state", "arguments": {"side": "left"}},
            {
                "name": "compute_live_shadow_ik",
                "arguments": {"side": "left", "target": "z_plus_5mm"},
            },
            {"name": "get_live_camera_preview", "arguments": {}},
        ],
        final_output="P5.5 local Agent read-only validation complete",
    )
    try:
        result = TopAgent(
            model=model,
            tools=task.build_tools(context),
            system_prompt=task.build_prompt(),
            max_iterations=5,
            max_tool_calls=3,
            max_tool_calls_per_turn=1,
        ).invoke(task.build_input(observation))
        messages = [
            message
            for message in result.get("messages", [])
            if isinstance(message, dict) and message.get("role") == "tool"
        ]
        records: list[dict[str, Any]] = []
        for message in messages:
            content = message.get("content", {})
            records.append(
                {
                    "name": message.get("name"),
                    "status": message.get("status"),
                    "execution_id": content.get("execution_id") if isinstance(content, dict) else None,
                    "hardware_accessed": content.get("hardware_accessed") if isinstance(content, dict) else None,
                    "motion_executed": content.get("motion_executed") if isinstance(content, dict) else None,
                    "write_control_authorized": content.get("write_control_authorized") if isinstance(content, dict) else None,
                }
            )
        passed = bool(
            task.is_complete(context, result)
            and len(records) == 3
            and all(
                record["status"] == "success"
                and record["hardware_accessed"] is True
                and record["motion_executed"] is False
                and record["write_control_authorized"] is False
                for record in records
            )
        )
        manifest = {
            "run_id": run_id,
            "stage": "P5.5",
            "api_mode": "local_scripted_agent",
            "hardware_connected": True,
            "hands_connected": False,
            "control_enabled": False,
            "hardware_write_authorized": False,
            "motion_commands_sent": 0,
            "records": records,
            "agent_output": result.get("output"),
            "passed": passed,
        }
    finally:
        task.close()
        env.close()
    (evidence_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"run_id": run_id, "passed": manifest["passed"], "evidence_dir": str(evidence_dir)}, ensure_ascii=False))
    return 0 if manifest["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

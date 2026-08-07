"""Offline vertical-slice Task that locates one configured target."""

from __future__ import annotations

from typing import Any, Mapping

from module.agent import AgentTool, RecoverableToolError
from module.runtime import RuntimeContext
from proto import Observation
from task._base_task import BaseTask


class LocateTargetTask(BaseTask):
    """Locate a target through one Task-owned perception capability."""

    def __init__(self, *, target: str) -> None:
        if not isinstance(target, str) or not target.strip():
            raise ValueError("target must be a non-empty string")
        self._target = target.strip()
        super().__init__()

    def build_prompt(self) -> str:
        return (
            f"Locate the configured target {self._target!r} with locate_target. "
            "Only report completion after that tool returns a non-empty success."
        )

    def build_tools(self, context: RuntimeContext) -> list[AgentTool]:
        detector = self.get_perception_tool("target_detector")

        def locate_target() -> Mapping[str, Any]:
            """Locate the configured target in the latest camera observation."""

            observation = context.require_observation()
            result = detector.run(observation, labels=[self._target])
            if not result.instances:
                raise RecoverableToolError(f"no {self._target!r} was found")
            return result.as_dict()

        return [
            AgentTool(
                name="locate_target",
                description=(
                    "Locate the configured target in the latest observation; "
                    "success requires at least one matching instance."
                ),
                handler=locate_target,
            )
        ]

    def build_input(self, observation: Observation) -> Mapping[str, Any]:
        return {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "A fresh observation is available at sequence "
                        f"{observation.sequence}. Complete the task using only "
                        "the allowed tools."
                    ),
                }
            ]
        }

    def is_complete(
        self,
        context: RuntimeContext,
        agent_result: Mapping[str, Any],
    ) -> bool:
        del context
        messages = agent_result.get("messages", [])
        if not isinstance(messages, list):
            return False
        return any(
            isinstance(message, Mapping)
            and message.get("role") == "tool"
            and message.get("name") == "locate_target"
            and message.get("status") == "success"
            for message in messages
        )

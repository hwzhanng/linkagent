"""Shared payload-state reader for mock, simulation, and future hardware ports."""

from __future__ import annotations

from collections.abc import Mapping

from module.robotics import RoboticsBundle


def read_payload_state(robotics: RoboticsBundle) -> dict[str, object]:
    context = robotics.verifier_context()
    source = context.get("payload_state")
    if callable(source):
        value = source()
        if not isinstance(value, Mapping):
            raise TypeError("payload_state verifier must return a mapping")
        return dict(value)
    world = context.get("mock_world")
    return {
        "detached": bool(getattr(world, "detached", False)),
        "retained": bool(getattr(world, "grasped", False)),
        "dropped": bool(getattr(world, "dropped", False)),
    }


__all__ = ["read_payload_state"]

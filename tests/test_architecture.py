"""Static guardrails for the dependency and offline-execution boundaries."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess
import sys
import unittest

from module.agent import MODEL_REGISTRY
from module.tools.env import ENV_REGISTRY
from module.tools.manipulation import MANIPULATION_REGISTRY
from module.tools.perception import PERCEPTION_REGISTRY
from module.robotics import (
    ARM_PORT_REGISTRY,
    HAND_PORT_REGISTRY,
    ROBOTICS_REGISTRY,
    TACTILE_PORT_REGISTRY,
)
from module.skills.tomato_pick import SKILL_REGISTRY
from module.skills import ATOMIC_SKILL_REGISTRY
from module.skills.harvest_cycle import (
    CLEARANCE_SCENE_REGISTRY,
    DROPPED_PAYLOAD_OBSERVATION_REGISTRY,
    PLACEMENT_VERIFIER_REGISTRY,
    TARGET_OBSERVATION_REGISTRY,
)
from module.skills.system_self_check import (
    CAMERA_HEALTH_REGISTRY,
    EMERGENCY_STOP_HEALTH_REGISTRY,
    VISION_WORKER_HEALTH_REGISTRY,
)
from task import TASK_REGISTRY
from task.agri_pick.factory import (
    PERCEPTION_REGISTRY as TOMATO_PERCEPTION_REGISTRY,
    PLANNER_REGISTRY,
    SAFETY_REGISTRY,
    VERIFIER_REGISTRY,
)


ROOT = Path(__file__).resolve().parents[1]


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


class ArchitectureBoundaryTests(unittest.TestCase):
    def test_runtime_does_not_import_concrete_tasks_or_adapters(self) -> None:
        imports = imported_modules(ROOT / "module/runtime.py")
        forbidden_prefixes = (
            "task.locate_target",
            "module.tools.env.mock",
            "module.tools.perception",
            "module.tools.manipulation",
        )
        self.assertFalse(
            any(name.startswith(forbidden_prefixes) for name in imports),
            imports,
        )

    def test_runtime_import_does_not_transitively_load_concrete_implementations(
        self,
    ) -> None:
        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import json, sys; import module.runtime; "
                    "print(json.dumps(sorted(sys.modules)))"
                ),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        loaded_modules = set(json.loads(probe.stdout))
        forbidden = {
            "module.agent.agent",
            "module.agent.models.factory",
            "module.agent.models.mock",
            "module.tools.env.factory",
            "module.tools.env.mock",
            "module.tools.manipulation.factory",
            "module.tools.perception.factory",
            "task",
            "task.factory",
            "task.locate_target",
        }
        self.assertTrue(forbidden.isdisjoint(loaded_modules), loaded_modules & forbidden)

    def test_agent_layer_does_not_import_runtime_task_or_environment(self) -> None:
        imports: set[str] = set()
        for path in (ROOT / "module/agent").rglob("*.py"):
            imports.update(imported_modules(path))
        forbidden_prefixes = ("module.runtime", "module.tools", "task", "proto")
        self.assertFalse(
            any(name.startswith(forbidden_prefixes) for name in imports),
            imports,
        )

    def test_capability_layers_do_not_import_agent_runtime_or_task(self) -> None:
        imports: set[str] = set()
        for path in (ROOT / "module/tools").rglob("*.py"):
            imports.update(imported_modules(path))
        forbidden_prefixes = ("module.agent", "module.runtime", "task")
        self.assertFalse(
            any(name.startswith(forbidden_prefixes) for name in imports),
            imports,
        )

    def test_core_has_no_third_party_sdk_or_heavy_runtime_imports(self) -> None:
        imports: set[str] = set()
        # Isaac/PPO entry points under ``script`` intentionally import Torch and
        # Isaac Lab inside the isolated simulation runtime.  The import boundary
        # guarded here is the reusable Agent/DTO/Task core.
        for root_name in ("module", "proto", "task"):
            for path in (ROOT / root_name).rglob("*.py"):
                imports.update(imported_modules(path))
        forbidden_roots = {
            "httpx",
            "langchain",
            "numpy",
            "openai",
            "requests",
            "robosuite",
            "torch",
            "ultralytics",
        }
        self.assertTrue(forbidden_roots.isdisjoint(imports), imports)

    def test_library_modules_do_not_configure_handlers_or_print(self) -> None:
        violations: list[str] = []
        for root_name in ("module", "proto", "task"):
            for path in (ROOT / root_name).rglob("*.py"):
                source = path.read_text(encoding="utf-8")
                if "basicConfig(" in source or "print(" in source:
                    violations.append(str(path.relative_to(ROOT)))
        self.assertEqual([], violations)

    def test_registries_are_explicit_and_only_expose_reviewed_implementations(self) -> None:
        self.assertEqual(
            {"mock", "ark_anthropic_messages", "ark_openai_responses"},
            set(MODEL_REGISTRY),
        )
        self.assertEqual({"mock"}, set(ENV_REGISTRY))
        self.assertEqual({"mock"}, set(PERCEPTION_REGISTRY))
        self.assertEqual({"mock"}, set(MANIPULATION_REGISTRY))
        self.assertEqual(
            {
                "locate_target",
                "agri_pick",
                "harvest_cycle",
                "hardware_capability_validation",
                "hardware_read_only_validation",
            },
            set(TASK_REGISTRY),
        )
        self.assertEqual(
            {"mock", "fake_gateway", "linkbot_g20_shadow"},
            set(ROBOTICS_REGISTRY),
        )
        self.assertEqual({"mock"}, set(ARM_PORT_REGISTRY))
        self.assertEqual({"mock"}, set(HAND_PORT_REGISTRY))
        self.assertEqual({"mock"}, set(TACTILE_PORT_REGISTRY))
        self.assertEqual(
            {"mock", "profile", "torch_export", "torchscript"},
            set(SKILL_REGISTRY),
        )
        self.assertEqual(
            {
                "system_self_check",
                "active_view_reposition",
                "select_harvest_target",
                "move_to_pick_ready",
                "localize_dropped_payload",
                "pregrasp_clearance_check",
                "reset_for_next_target",
                "target_reobserve",
                "transport_payload",
                "safe_retract",
                "payload_retreat",
                "payload_place",
            },
            set(ATOMIC_SKILL_REGISTRY),
        )
        self.assertEqual({"mock", "orbbec_viewer"}, set(CAMERA_HEALTH_REGISTRY))
        self.assertEqual({"mock"}, set(EMERGENCY_STOP_HEALTH_REGISTRY))
        self.assertEqual(
            {"mock", "loopback"}, set(VISION_WORKER_HEALTH_REGISTRY)
        )
        self.assertEqual(
            {"mock", "tomato_vision_replay", "tomato_vision_worker"},
            set(TARGET_OBSERVATION_REGISTRY),
        )
        self.assertEqual({"mock"}, set(PLACEMENT_VERIFIER_REGISTRY))
        self.assertEqual(
            {"mock", "observation_pointcloud"}, set(CLEARANCE_SCENE_REGISTRY)
        )
        self.assertEqual({"mock"}, set(DROPPED_PAYLOAD_OBSERVATION_REGISTRY))
        self.assertEqual(
            {"mock", "tomato_vision_replay", "tomato_vision_worker"},
            set(TOMATO_PERCEPTION_REGISTRY),
        )
        self.assertEqual({"mock", "training_distribution"}, set(PLANNER_REGISTRY))
        self.assertEqual({"mock"}, set(SAFETY_REGISTRY))
        self.assertEqual({"mock", "independent"}, set(VERIFIER_REGISTRY))

    def test_shipped_config_contains_no_credential_fields(self) -> None:
        secret_names = {
            "api_key",
            "apikey",
            "authorization",
            "password",
            "secret",
            "token",
        }

        def visit(value: object, *, path: str) -> list[str]:
            findings: list[str] = []
            if isinstance(value, dict):
                for key, child in value.items():
                    normalized = str(key).lower().replace("-", "_")
                    if normalized in secret_names:
                        findings.append(f"{path}.{key}")
                    findings.extend(visit(child, path=f"{path}.{key}"))
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    findings.extend(visit(child, path=f"{path}[{index}]"))
            return findings

        findings: list[str] = []
        for path in (ROOT / "config").glob("*.yaml"):
            value = json.loads(path.read_text(encoding="utf-8"))
            findings.extend(visit(value, path=path.name))
        self.assertEqual([], findings)


if __name__ == "__main__":
    unittest.main()

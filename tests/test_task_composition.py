"""Offline tests for Task ownership, configuration, and composition.

Only standard-library test helpers are used here.  The suite deliberately runs
the shipped vertical slice without importing a provider SDK or contacting an
external service.
"""

from __future__ import annotations

from contextlib import redirect_stderr
import importlib.util
import io
import json
import logging
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from module.agent import MODEL_REGISTRY, MockChatModel, TopAgent
from module.runtime import AgentRuntime, CompletionReason, RuntimeContext, RuntimeState
from module.tools.env import ENV_REGISTRY, MockEnv
from module.tools.manipulation import (
    MANIPULATION_REGISTRY,
    MockManipulation,
)
from module.tools.perception import PERCEPTION_REGISTRY, MockPerceptionTool
from proto import Observation
from script.run import build_runtime, build_runtime_from_file, configure_logging, main
from script.utils import load_mapping_file
from task import TASK_REGISTRY, BaseTask, build_task
from task.locate_target import LocateTargetTask


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.yaml"


def _locate_task_config(
    *,
    perception_profile: str = "offline",
    manipulation_type: str | None = None,
) -> dict[str, object]:
    """Return one fresh Task config suitable for focused tests."""

    config: dict[str, object] = {
        "type": "locate_target",
        "target": "bottle",
        "perception_tools": [
            {
                "name": "target_detector",
                "profile": perception_profile,
            }
        ],
    }
    if manipulation_type is not None:
        config["manipulation_tools"] = [
            {
                "name": "grasp_planner",
                "type": manipulation_type,
            }
        ]
    return config


class ConfigurationTests(unittest.TestCase):
    def test_json_compatible_yaml_loads_and_non_mapping_root_is_rejected(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            mapping_path = directory / "mapping.yaml"
            mapping_path.write_text(
                json.dumps({"agent": {"model": "offline"}, "max_steps": 2}),
                encoding="utf-8",
            )
            sequence_path = directory / "sequence.yaml"
            sequence_path.write_text('["not", "a", "mapping"]', encoding="utf-8")

            self.assertEqual(
                load_mapping_file(mapping_path),
                {"agent": {"model": "offline"}, "max_steps": 2},
            )
            with self.assertRaisesRegex(TypeError, "root.*must be a mapping"):
                load_mapping_file(sequence_path)

    def test_conventional_yaml_is_optional_and_missing_dependency_is_explicit(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            yaml_path = Path(temporary_directory) / "ordinary.yaml"
            yaml_path.write_text(
                "agent:\n  model: offline\nruntime:\n  max_steps: 2\n",
                encoding="utf-8",
            )

            if importlib.util.find_spec("yaml") is None:
                with self.assertRaisesRegex(
                    ValueError,
                    "not JSON-compatible YAML.*PyYAML.*not installed",
                ):
                    load_mapping_file(yaml_path)
            else:
                self.assertEqual(
                    load_mapping_file(yaml_path),
                    {
                        "agent": {"model": "offline"},
                        "runtime": {"max_steps": 2},
                    },
                )

            # Exercise the no-optional-dependency branch even on developer
            # machines that happen to have PyYAML installed.
            with patch.dict(sys.modules, {"yaml": None}):
                with self.assertRaisesRegex(
                    ValueError,
                    "not JSON-compatible YAML.*PyYAML.*not installed",
                ):
                    load_mapping_file(yaml_path)


class TaskFactoryAndFlowTests(unittest.TestCase):
    def test_task_factory_reports_unknown_type_and_profile(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "unknown task type.*missing.*locate_target",
        ):
            build_task({"type": "missing"})

        with self.assertRaisesRegex(
            ValueError,
            "unknown perception profile.*missing",
        ):
            build_task(
                _locate_task_config(perception_profile="missing"),
                perception_profiles={"known": {"type": "mock"}},
            )

        sneaky_config = _locate_task_config(perception_profile="known")
        sneaky_config["perception_tools"][0]["type"] = "unreviewed-adapter"
        with self.assertRaisesRegex(ValueError, "cannot override profile type"):
            build_task(
                sneaky_config,
                perception_profiles={"known": {"type": "mock"}},
            )

    def test_locate_target_exact_allowlist_retries_empty_frame_then_completes(
        self,
    ) -> None:
        env = MockEnv(
            observations=(
                Observation(sequence=0, metadata={"detections": []}),
                Observation(
                    sequence=1,
                    metadata={
                        "detections": [
                            {
                                "label": "bottle",
                                "score": 0.95,
                                "bbox_xyxy": [1, 2, 30, 80],
                            }
                        ]
                    },
                ),
            )
        )
        context = RuntimeContext(env)
        task = build_task(
            _locate_task_config(),
            perception_profiles={"offline": {"type": "mock"}},
        )
        self.assertIsInstance(task, BaseTask)
        self.assertIsInstance(task, LocateTargetTask)

        tools = task.build_tools(context)
        self.assertEqual(tuple(tool.name for tool in tools), ("locate_target",))
        self.assertEqual(len(tools), 1)

        agent = TopAgent(
            model=MockChatModel(tool_sequence=("locate_target",)),
            tools=tools,
            system_prompt=task.build_prompt(),
            max_iterations=3,
        )
        self.assertEqual(tuple(tool.name for tool in agent.tools), ("locate_target",))
        runtime = AgentRuntime(
            context=context,
            task=task,
            agent=agent,
            max_steps=2,
        )

        # The first tool call raises for an empty result and is represented as
        # a failed tool message.  Only the fresh second frame can complete.
        with self.assertLogs("module.agent.agent", level=logging.WARNING) as captured:
            report = runtime.run(seed=7)

        self.assertTrue(any("no 'bottle' was found" in line for line in captured.output))
        self.assertTrue(report.completed)
        self.assertIs(report.reason, CompletionReason.TASK_COMPLETE)
        self.assertEqual(report.control_steps, 2)
        self.assertEqual(report.last_observation_sequence, 1)
        self.assertEqual(runtime.state, RuntimeState.CLOSED)
        self.assertTrue(env.closed)


class TaskResourceOwnershipTests(unittest.TestCase):
    def test_task_close_releases_perception_and_manipulation_once(self) -> None:
        created_perceptions: list[_CountingPerception] = []
        created_manipulations: list[_CountingManipulation] = []

        def build_counting_perception() -> _CountingPerception:
            backend = _CountingPerception()
            created_perceptions.append(backend)
            return backend

        def build_counting_manipulation() -> _CountingManipulation:
            backend = _CountingManipulation()
            created_manipulations.append(backend)
            return backend

        original_perception_registry = dict(PERCEPTION_REGISTRY)
        original_manipulation_registry = dict(MANIPULATION_REGISTRY)
        with (
            patch.dict(
                PERCEPTION_REGISTRY,
                {"counting-test": build_counting_perception},
            ),
            patch.dict(
                MANIPULATION_REGISTRY,
                {"counting-test": build_counting_manipulation},
            ),
        ):
            task = build_task(
                _locate_task_config(manipulation_type="counting-test"),
                perception_profiles={
                    "offline": {"type": "counting-test"},
                },
            )
            perception = task.get_perception_tool("target_detector")
            manipulation = task.get_manipulation_tool("grasp_planner")

            task.close()
            task.close()

        self.assertEqual(PERCEPTION_REGISTRY, original_perception_registry)
        self.assertEqual(MANIPULATION_REGISTRY, original_manipulation_registry)
        self.assertIs(perception, created_perceptions[0])
        self.assertIs(manipulation, created_manipulations[0])
        self.assertTrue(perception.closed)
        self.assertTrue(manipulation.closed)
        self.assertEqual(perception.close_calls, 1)
        self.assertEqual(manipulation.close_calls, 1)

    def test_task_does_not_repeat_successful_or_failed_close_attempts(self) -> None:
        perception = _FailingClosePerception()
        manipulation = _CountingManipulation()

        with (
            patch.dict(
                PERCEPTION_REGISTRY,
                {"failing-close-test": lambda: perception},
            ),
            patch.dict(
                MANIPULATION_REGISTRY,
                {"counting-close-test": lambda: manipulation},
            ),
        ):
            task = build_task(
                _locate_task_config(manipulation_type="counting-close-test"),
                perception_profiles={
                    "offline": {"type": "failing-close-test"},
                },
            )

            with self.assertLogs("task._base_task", level=logging.ERROR):
                with self.assertRaisesRegex(RuntimeError, "perception close failed"):
                    task.close()
            task.close()

        self.assertEqual(perception.close_calls, 1)
        self.assertEqual(manipulation.close_calls, 1)


class CompositionAndScriptTests(unittest.TestCase):
    def test_default_configuration_builds_and_runs_to_completion(self) -> None:
        runtime = build_runtime_from_file(DEFAULT_CONFIG)

        report = runtime.run(seed=11)

        self.assertTrue(report.completed)
        self.assertIs(report.reason, CompletionReason.TASK_COMPLETE)
        self.assertEqual(report.control_steps, 1)
        self.assertEqual(runtime.state, RuntimeState.CLOSED)
        self.assertTrue(runtime.context.env.closed)

    def test_cli_succeeds_and_writes_a_run_log(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            log_directory = Path(temporary_directory) / "logs"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "script.run",
                    "--config",
                    str(DEFAULT_CONFIG),
                    "--seed",
                    "19",
                    "--log-dir",
                    str(log_directory),
                    "--no-color",
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )

            self.assertEqual(
                completed.returncode,
                0,
                msg=f"stdout={completed.stdout!r}\nstderr={completed.stderr!r}",
            )
            self.assertIn("task completed in 1 control step(s)", completed.stderr)
            log_paths = list(log_directory.glob("run_*.log"))
            self.assertEqual(len(log_paths), 1)
            log_text = log_paths[0].read_text(encoding="utf-8")
            self.assertIn("runtime stopped completed=True", log_text)
            self.assertIn("task completed in 1 control step(s)", log_text)

    def test_logging_setup_failure_returns_one_without_mutating_handlers(self) -> None:
        root_logger = logging.getLogger()
        original_handlers = tuple(root_logger.handlers)
        original_level = root_logger.level
        stderr = io.StringIO()

        with TemporaryDirectory() as temporary_directory:
            invalid_log_directory = Path(temporary_directory) / "already-a-file"
            invalid_log_directory.write_text("not a directory", encoding="utf-8")
            with redirect_stderr(stderr):
                status = main(
                    (
                        "--config",
                        str(DEFAULT_CONFIG),
                        "--log-dir",
                        str(invalid_log_directory),
                        "--no-color",
                    )
                )

        self.assertEqual(status, 1)
        self.assertIn("logging setup failed", stderr.getvalue())
        self.assertEqual(tuple(root_logger.handlers), original_handlers)
        self.assertEqual(root_logger.level, original_level)

    def test_invalid_config_fails_before_env_creation_and_restores_registry(
        self,
    ) -> None:
        config = load_mapping_file(DEFAULT_CONFIG)
        config["env"] = {"type": "tracking-env-test"}
        config["agent"] = {
            **config["agent"],
            "model": "missing-profile",
        }
        created_envs: list[MockEnv] = []

        def build_tracking_env() -> MockEnv:
            env = MockEnv()
            created_envs.append(env)
            return env

        original_registry = dict(ENV_REGISTRY)
        with patch.dict(ENV_REGISTRY, {"tracking-env-test": build_tracking_env}):
            with self.assertRaisesRegex(
                ValueError,
                "unknown model profile.*missing-profile",
            ):
                build_runtime(config, config_dir=DEFAULT_CONFIG.parent)
            self.assertEqual(created_envs, [])

        self.assertEqual(ENV_REGISTRY, original_registry)
        self.assertNotIn("tracking-env-test", ENV_REGISTRY)

    def test_model_construction_failure_occurs_before_environment_creation(self) -> None:
        created_envs: list[MockEnv] = []

        def build_tracking_env() -> MockEnv:
            env = MockEnv()
            created_envs.append(env)
            return env

        def build_failing_model() -> MockChatModel:
            raise RuntimeError("model construction failed")

        with TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            models_file = directory / "models.yaml"
            models_file.write_text(
                json.dumps(
                    {"failing": {"type": "failing-model-composition-test"}}
                ),
                encoding="utf-8",
            )
            config = load_mapping_file(DEFAULT_CONFIG)
            config["models_file"] = models_file.name
            config["perception_file"] = str(
                PROJECT_ROOT / "config" / "_perception.yaml"
            )
            config["agent"] = {**config["agent"], "model": "failing"}
            config["env"] = {"type": "tracking-env-composition-test"}

            with (
                patch.dict(
                    ENV_REGISTRY,
                    {"tracking-env-composition-test": build_tracking_env},
                ),
                patch.dict(
                    MODEL_REGISTRY,
                    {"failing-model-composition-test": build_failing_model},
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "model construction failed"):
                    build_runtime(config, config_dir=directory)

        self.assertEqual(created_envs, [])

    def test_failure_after_resource_creation_closes_environment(self) -> None:
        created_envs: list[MockEnv] = []

        def build_tracking_env() -> MockEnv:
            env = MockEnv()
            created_envs.append(env)
            return env

        config = load_mapping_file(DEFAULT_CONFIG)
        config["env"] = {"type": "tracking-env-post-build-test"}
        config["task"] = {"type": "failing-tools-composition-test"}

        with (
            patch.dict(
                ENV_REGISTRY,
                {"tracking-env-post-build-test": build_tracking_env},
            ),
            patch.dict(
                TASK_REGISTRY,
                {"failing-tools-composition-test": _FailingBuildToolsTask},
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "tool construction failed"):
                build_runtime(config, config_dir=DEFAULT_CONFIG.parent)

        self.assertEqual(len(created_envs), 1)
        self.assertTrue(created_envs[0].closed)

    def test_root_and_agent_typos_are_rejected_before_resource_creation(self) -> None:
        created_envs: list[MockEnv] = []

        def build_tracking_env() -> MockEnv:
            env = MockEnv()
            created_envs.append(env)
            return env

        with patch.dict(
            ENV_REGISTRY,
            {"tracking-env-validation-test": build_tracking_env},
        ):
            root_typo = load_mapping_file(DEFAULT_CONFIG)
            root_typo["env"] = {"type": "tracking-env-validation-test"}
            root_typo["unexpected_root_key"] = True
            with self.assertRaisesRegex(ValueError, "unsupported root config keys"):
                build_runtime(root_typo, config_dir=DEFAULT_CONFIG.parent)

            agent_typo = load_mapping_file(DEFAULT_CONFIG)
            agent_typo["env"] = {"type": "tracking-env-validation-test"}
            agent_typo["agent"] = {
                **agent_typo["agent"],
                "max_iteration": 99,
            }
            with self.assertRaisesRegex(ValueError, "unsupported agent config keys"):
                build_runtime(agent_typo, config_dir=DEFAULT_CONFIG.parent)

        self.assertEqual(created_envs, [])

    def test_log_handlers_redact_common_credentials(self) -> None:
        root_logger = logging.getLogger()
        original_handlers = list(root_logger.handlers)
        original_level = root_logger.level
        configured_handlers: list[logging.Handler] = []
        stderr = io.StringIO()

        try:
            with TemporaryDirectory() as temporary_directory, redirect_stderr(stderr):
                log_path = configure_logging(
                    temporary_directory,
                    verbose=True,
                    color=False,
                )
                logging.getLogger("tests.credentials").warning(
                    "api_key=%s token:%s password = %s secret=%s safe=%s",
                    "key-value-123",
                    "token-value-456",
                    "password-value-789",
                    "secret-value-012",
                    "visible",
                )
                configured_handlers = list(root_logger.handlers)
                for handler in configured_handlers:
                    handler.flush()
                log_text = log_path.read_text(encoding="utf-8")
        finally:
            root_logger.handlers.clear()
            for handler in configured_handlers:
                handler.close()
            root_logger.handlers.extend(original_handlers)
            root_logger.setLevel(original_level)

        for credential in (
            "key-value-123",
            "token-value-456",
            "password-value-789",
            "secret-value-012",
        ):
            self.assertNotIn(credential, log_text)
            self.assertNotIn(credential, stderr.getvalue())
        self.assertEqual(log_text.count("<redacted>"), 4)
        self.assertIn("safe=visible", log_text)


class _FailingBuildToolsTask(BaseTask):
    """Task fake that fails only after the composition root owns resources."""

    def __init__(self) -> None:
        super().__init__()

    def build_prompt(self) -> str:
        return "unused"

    def build_tools(self, context: RuntimeContext) -> list[object]:
        del context
        raise RuntimeError("tool construction failed")

    def build_input(self, observation: Observation) -> dict[str, object]:
        del observation
        return {"messages": []}

    def is_complete(
        self,
        context: RuntimeContext,
        agent_result: dict[str, object],
    ) -> bool:
        del context, agent_result
        return False


class _CountingPerception(MockPerceptionTool):
    """Mock perception that exposes how many physical closes occurred."""

    def __init__(self) -> None:
        self.close_calls = 0
        super().__init__(name="counting_perception")

    def _close(self) -> None:
        self.close_calls += 1


class _CountingManipulation(MockManipulation):
    """Mock manipulation that exposes how many physical closes occurred."""

    def __init__(self) -> None:
        self.close_calls = 0
        super().__init__(name="counting_manipulation")

    def _close(self) -> None:
        self.close_calls += 1


class _FailingClosePerception(_CountingPerception):
    def _close(self) -> None:
        self.close_calls += 1
        raise RuntimeError("perception close failed after partial cleanup")


if __name__ == "__main__":
    unittest.main()

"""Offline tests for the bounded agent loop and runtime lifecycle.

The fakes in this module deliberately use only the public, SDK-neutral
contracts.  They make observation refreshes and cleanup ordering observable
without talking to a model provider, simulator, or robot backend.
"""

from __future__ import annotations

import json
import unittest
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from module.agent import (
    AgentTool,
    AgentToolSpec,
    BaseChatModel,
    ModelTurn,
    RecoverableToolError,
    ToolCall,
    ToolCallLimitError,
    TopAgent,
)
from module.runtime import (
    AgentRuntime,
    CompletionReason,
    RuntimeContext,
    RuntimeReport,
    RuntimeState,
)
from module.tools.env import BaseEnv
from proto import Action, Observation, Result
from task import BaseTask


class RecordingChatModel(BaseChatModel):
    """Replay model turns and retain detached transcript snapshots."""

    def __init__(self, turns: Sequence[ModelTurn]) -> None:
        self._turns = tuple(turns)
        self._cursor = 0
        self.reset_count = 0
        self.transcripts: list[tuple[dict[str, Any], ...]] = []
        self.exposed_tool_names: list[tuple[str, ...]] = []

    def reset(self) -> None:
        self.reset_count += 1
        self._cursor = 0

    def invoke(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[AgentToolSpec],
    ) -> ModelTurn:
        self.transcripts.append(tuple(dict(message) for message in messages))
        self.exposed_tool_names.append(tuple(tool.name for tool in tools))
        if self._cursor >= len(self._turns):
            raise AssertionError("scripted model was invoked too many times")
        turn = self._turns[self._cursor]
        self._cursor += 1
        return turn


class TopAgentTests(unittest.TestCase):
    def test_normal_multi_turn_tool_workflow_builds_one_transcript(self) -> None:
        handler_calls: list[tuple[str, Any]] = []

        def double(value: int) -> int:
            handler_calls.append(("double", value))
            return value * 2

        def label(value: str) -> dict[str, str]:
            handler_calls.append(("label", value))
            return {"label": value.upper()}

        tools = (
            AgentTool("double", "Double an integer.", double),
            AgentTool("label", "Normalize a label.", label),
        )
        model = RecordingChatModel(
            (
                ModelTurn(
                    content="calculating",
                    tool_calls=(
                        ToolCall("double", {"value": 3}, id="explicit-call"),
                    ),
                ),
                ModelTurn(
                    content={"phase": 2},
                    tool_calls=(ToolCall("label", {"value": "cup"}),),
                ),
                ModelTurn(content="finished", stop_reason="done"),
            )
        )
        agent = TopAgent(model, tools, system_prompt="Use only task tools.")

        result = agent.invoke(
            {"messages": [{"role": "user", "content": "Run the workflow."}]}
        )

        self.assertEqual(handler_calls, [("double", 3), ("label", "cup")])
        self.assertEqual(result["iterations"], 3)
        self.assertEqual(result["stop_reason"], "done")
        self.assertEqual(result["output"], "finished")
        self.assertEqual(
            [message["role"] for message in result["messages"]],
            ["system", "user", "assistant", "tool", "assistant", "tool", "assistant"],
        )
        first_tool_message = result["messages"][3]
        second_tool_message = result["messages"][5]
        self.assertEqual(first_tool_message["tool_call_id"], "explicit-call")
        self.assertEqual(first_tool_message["content"], 6)
        self.assertEqual(second_tool_message["tool_call_id"], "tool-call-2-1")
        self.assertEqual(second_tool_message["content"], {"label": "CUP"})
        self.assertEqual([len(items) for items in model.transcripts], [2, 4, 6])
        self.assertEqual(model.exposed_tool_names, [("double", "label")] * 3)
        self.assertTrue(all(not hasattr(spec, "handler") for spec in agent.tool_specs))

    def test_binary_values_are_summarized_in_inputs_tool_results_and_output(self) -> None:
        def inspect_frame() -> dict[str, Any]:
            return {
                "mask": bytearray(b"12345"),
                "thumbnail": memoryview(b"abc"),
            }

        model = RecordingChatModel(
            (
                ModelTurn(tool_calls=(ToolCall("inspect_frame"),)),
                ModelTurn(content=b"final-bytes"),
            )
        )
        agent = TopAgent(
            model,
            (AgentTool("inspect_frame", "Inspect the current frame.", inspect_frame),),
        )

        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": {"rgb": b"\x00\x01\x02\x03"},
                    }
                ]
            }
        )

        self.assertEqual(
            result["messages"][0]["content"]["rgb"],
            {"type": "binary", "size_bytes": 4},
        )
        self.assertEqual(
            result["messages"][2]["content"],
            {
                "mask": {"type": "binary", "size_bytes": 5},
                "thumbnail": {"type": "binary", "size_bytes": 3},
            },
        )
        self.assertEqual(
            result["output"],
            {"type": "binary", "size_bytes": 11},
        )
        json.dumps(result)

    def test_unknown_tool_becomes_error_without_executing_allowed_tool(self) -> None:
        known_call_count = 0

        def known_tool() -> str:
            nonlocal known_call_count
            known_call_count += 1
            return "should not run"

        model = RecordingChatModel(
            (
                ModelTurn(tool_calls=(ToolCall("not_allowed", id="missing-1"),)),
                ModelTurn(content="recovered"),
            )
        )
        agent = TopAgent(
            model,
            (AgentTool("known_tool", "An allowed operation.", known_tool),),
        )

        with self.assertLogs("module.agent.agent", level="WARNING"):
            result = agent.invoke(
                {"messages": [{"role": "user", "content": "Try a tool."}]}
            )

        self.assertEqual(known_call_count, 0)
        error_message = result["messages"][2]
        self.assertEqual(error_message["role"], "tool")
        self.assertEqual(error_message["tool_call_id"], "missing-1")
        self.assertEqual(error_message["name"], "not_allowed")
        self.assertEqual(error_message["status"], "error")
        self.assertEqual(error_message["content"]["error"]["type"], "unknown_tool")
        self.assertEqual(result["output"], "recovered")

    def test_tool_exception_becomes_error_and_model_can_recover(self) -> None:
        class ToolFailure(RecoverableToolError):
            pass

        def fail_tool() -> None:
            raise ToolFailure("gripper rejected command")

        model = RecordingChatModel(
            (
                ModelTurn(tool_calls=(ToolCall("fail_tool", id="failure-1"),)),
                ModelTurn(content="use a different action", stop_reason="recovered"),
            )
        )
        agent = TopAgent(
            model,
            (AgentTool("fail_tool", "Exercise the failure boundary.", fail_tool),),
        )

        with self.assertLogs("module.agent.agent", level="WARNING"):
            result = agent.invoke(
                {"messages": [{"role": "user", "content": "Continue safely."}]}
            )

        tool_message = result["messages"][2]
        self.assertEqual(tool_message["status"], "error")
        self.assertEqual(tool_message["content"]["error"]["type"], "ToolFailure")
        self.assertEqual(
            tool_message["content"]["error"]["message"],
            "gripper rejected command",
        )
        self.assertEqual(result["stop_reason"], "recovered")
        self.assertEqual(result["iterations"], 2)

    def test_unexpected_tool_exception_propagates_without_model_recovery(self) -> None:
        fatal_error = PermissionError("backend permission denied")

        def fail_fatally() -> None:
            raise fatal_error

        model = RecordingChatModel(
            (
                ModelTurn(tool_calls=(ToolCall("fail_fatally"),)),
                ModelTurn(content="must not continue"),
            )
        )
        agent = TopAgent(
            model,
            (
                AgentTool(
                    "fail_fatally",
                    "Raise an unexpected backend error.",
                    fail_fatally,
                ),
            ),
        )

        with self.assertRaises(PermissionError) as caught:
            agent.invoke({"messages": [{"role": "user", "content": "run"}]})

        self.assertIs(caught.exception, fatal_error)
        self.assertEqual(len(model.transcripts), 1)

    def test_invalid_tool_arguments_are_recoverable_without_calling_handler(self) -> None:
        call_count = 0

        def typed_tool(value: int) -> int:
            nonlocal call_count
            call_count += 1
            return value

        model = RecordingChatModel(
            (
                ModelTurn(tool_calls=(ToolCall("typed_tool", {"wrong": 1}),)),
                ModelTurn(content="correct later"),
            )
        )
        agent = TopAgent(
            model,
            (AgentTool("typed_tool", "Accept one value.", typed_tool),),
        )

        with self.assertLogs("module.agent.agent", level="WARNING"):
            result = agent.invoke(
                {"messages": [{"role": "user", "content": "run"}]}
            )

        self.assertEqual(call_count, 0)
        self.assertEqual(result["messages"][2]["status"], "error")
        self.assertEqual(
            result["messages"][2]["content"]["error"]["type"],
            "RecoverableToolError",
        )

    def test_tool_call_budgets_reject_a_turn_before_any_handler_runs(self) -> None:
        call_count = 0

        def act() -> None:
            nonlocal call_count
            call_count += 1

        model = RecordingChatModel(
            (
                ModelTurn(
                    tool_calls=(ToolCall("act"), ToolCall("act"), ToolCall("act"))
                ),
            )
        )
        agent = TopAgent(
            model,
            (AgentTool("act", "Perform one bounded action.", act),),
            max_tool_calls=2,
            max_tool_calls_per_turn=2,
        )

        with self.assertRaises(ToolCallLimitError):
            agent.invoke({"messages": [{"role": "user", "content": "run"}]})
        self.assertEqual(call_count, 0)

    def test_total_tool_call_budget_is_enforced_across_model_turns(self) -> None:
        call_count = 0

        def act() -> None:
            nonlocal call_count
            call_count += 1

        model = RecordingChatModel(
            (
                ModelTurn(tool_calls=(ToolCall("act"),)),
                ModelTurn(tool_calls=(ToolCall("act"), ToolCall("act"))),
            )
        )
        agent = TopAgent(
            model,
            (AgentTool("act", "Perform one bounded action.", act),),
            max_tool_calls=2,
            max_tool_calls_per_turn=2,
        )

        with self.assertRaises(ToolCallLimitError):
            agent.invoke({"messages": [{"role": "user", "content": "run"}]})
        self.assertEqual(call_count, 1)

    def test_model_turn_has_a_hard_tool_call_count_limit(self) -> None:
        with self.assertRaisesRegex(ValueError, "too many tool calls"):
            ModelTurn(tool_calls=tuple(ToolCall("act") for _ in range(65)))

    def test_repeated_invoke_resets_model_and_has_no_cross_step_transcript(self) -> None:
        tool_inputs: list[str] = []

        def remember_for_this_invoke(marker: str) -> str:
            tool_inputs.append(marker)
            return marker

        model = RecordingChatModel(
            (
                ModelTurn(
                    tool_calls=(
                        ToolCall("remember", {"marker": "script-marker"}),
                    )
                ),
                ModelTurn(content="fresh completion"),
            )
        )
        agent = TopAgent(
            model,
            (AgentTool("remember", "Record one marker.", remember_for_this_invoke),),
        )

        first = agent.invoke(
            {"messages": [{"role": "user", "content": "outer-step-one"}]}
        )
        second = agent.invoke(
            {"messages": [{"role": "user", "content": "outer-step-two"}]}
        )

        self.assertEqual(model.reset_count, 2)
        self.assertEqual(tool_inputs, ["script-marker", "script-marker"])
        self.assertEqual(first["iterations"], 2)
        self.assertEqual(second["iterations"], 2)
        self.assertEqual([len(items) for items in model.transcripts], [1, 3, 1, 3])
        self.assertEqual(model.transcripts[0][0]["content"], "outer-step-one")
        self.assertEqual(model.transcripts[2][0]["content"], "outer-step-two")
        self.assertNotIn("outer-step-one", json.dumps(second))

    def test_duplicate_tool_names_are_rejected_before_any_model_call(self) -> None:
        model = RecordingChatModel((ModelTurn(content="unused"),))
        first = AgentTool("duplicate", "First.", lambda: None)
        second = AgentTool("duplicate", "Second.", lambda: None)

        with self.assertRaisesRegex(ValueError, "duplicate tool name 'duplicate'"):
            TopAgent(model, (first, second))

        self.assertEqual(model.reset_count, 0)
        self.assertEqual(model.transcripts, [])

    def test_max_iterations_stops_pending_tool_workflow_at_exact_bound(self) -> None:
        call_count = 0

        def keep_working() -> int:
            nonlocal call_count
            call_count += 1
            return call_count

        model = RecordingChatModel(
            (
                ModelTurn(
                    content="iteration one",
                    tool_calls=(ToolCall("keep_working"),),
                ),
                ModelTurn(
                    content="iteration two",
                    tool_calls=(ToolCall("keep_working"),),
                ),
                ModelTurn(content="must not be reached"),
            )
        )
        agent = TopAgent(
            model,
            (AgentTool("keep_working", "Continue the workflow.", keep_working),),
            max_iterations=2,
        )

        with self.assertLogs("module.agent.agent", level="WARNING"):
            result = agent.invoke(
                {"messages": [{"role": "user", "content": "Keep going."}]}
            )

        self.assertEqual(call_count, 2)
        self.assertEqual(len(model.transcripts), 2)
        self.assertEqual(result["iterations"], 2)
        self.assertEqual(result["stop_reason"], "max_iterations")
        self.assertEqual(result["output"], "iteration two")
        self.assertEqual(result["messages"][-1]["role"], "tool")


class RecordingEnv(BaseEnv):
    """Environment fake with deterministic observation sequences and events."""

    def __init__(
        self,
        *,
        reset_sequence: int = 10,
        action_results: Sequence[Result] = (),
        events: list[str] | None = None,
        close_error: BaseException | None = None,
    ) -> None:
        self.reset_sequence = reset_sequence
        self._next_sequence = reset_sequence + 1
        self._action_results = list(action_results)
        self.events = events if events is not None else []
        self.close_error = close_error
        self.reset_seeds: list[int | None] = []
        self.get_obs_count = 0
        self.actions: list[Action] = []
        self.close_count = 0

    def reset(self, seed: int | None = None) -> Observation:
        self.events.append("env.reset")
        self.reset_seeds.append(seed)
        return Observation(sequence=self.reset_sequence)

    def get_obs(self) -> Observation:
        self.events.append("env.get_obs")
        self.get_obs_count += 1
        observation = Observation(sequence=self._next_sequence)
        self._next_sequence += 1
        return observation

    def take_action(self, action: Action) -> Result:
        self.events.append("env.take_action")
        self.actions.append(action)
        if self._action_results:
            return self._action_results.pop(0)
        return Result(success=True, message="accepted")

    def close(self) -> None:
        self.events.append("env.close")
        self.close_count += 1
        if self.close_error is not None:
            raise self.close_error


class RecordingTask(BaseTask):
    """Default-flow task fake with scripted completion decisions."""

    def __init__(
        self,
        completions: Sequence[bool],
        *,
        events: list[str] | None = None,
        close_error: BaseException | None = None,
    ) -> None:
        self._completions = list(completions)
        self.events = events if events is not None else []
        self.close_error = close_error
        self.input_sequences: list[int] = []
        self.completion_results: list[Mapping[str, Any]] = []
        self.close_count = 0
        super().__init__()

    def build_prompt(self) -> str:
        return "Test task."

    def build_tools(self, context: RuntimeContext) -> list[AgentTool]:
        del context
        return []

    def build_input(self, observation: Observation) -> Mapping[str, Any]:
        self.events.append("task.build_input")
        self.input_sequences.append(observation.sequence)
        return {
            "messages": [
                {
                    "role": "user",
                    "content": f"observation:{observation.sequence}",
                }
            ]
        }

    def is_complete(
        self,
        context: RuntimeContext,
        agent_result: Mapping[str, Any],
    ) -> bool:
        del context
        self.events.append("task.is_complete")
        self.completion_results.append(agent_result)
        if not self._completions:
            raise AssertionError("completion script was exhausted")
        return self._completions.pop(0)

    def close(self) -> None:
        self.events.append("task.close")
        self.close_count += 1
        if self.close_error is not None:
            raise self.close_error
        super().close()


class RecordingAgent:
    """Structurally implements AgentInvoker with an optional callback."""

    def __init__(
        self,
        *,
        events: list[str] | None = None,
        callback: Callable[[int, Mapping[str, Any]], Mapping[str, Any]] | None = None,
        error: BaseException | None = None,
    ) -> None:
        self.events = events if events is not None else []
        self.callback = callback
        self.error = error
        self.inputs: list[Mapping[str, Any]] = []
        self.results: list[Mapping[str, Any]] = []

    def invoke(self, agent_input: Mapping[str, Any]) -> dict[str, Any]:
        self.events.append("agent.invoke")
        self.inputs.append(agent_input)
        if self.error is not None:
            raise self.error
        call_number = len(self.inputs)
        if self.callback is None:
            result: Mapping[str, Any] = {"call": call_number}
        else:
            result = self.callback(call_number, agent_input)
        self.results.append(result)
        return dict(result)


def build_runtime(
    env: RecordingEnv,
    task: RecordingTask,
    agent: RecordingAgent,
    *,
    max_steps: int = 3,
) -> AgentRuntime:
    return AgentRuntime(
        context=RuntimeContext(env),
        task=task,
        agent=agent,
        max_steps=max_steps,
    )


class AgentRuntimeTests(unittest.TestCase):
    def test_runtime_report_rejects_malformed_boundary_fields(self) -> None:
        valid = {
            "completed": True,
            "reason": CompletionReason.TASK_COMPLETE,
            "control_steps": 1,
            "last_observation_sequence": 0,
        }
        invalid_fields = (
            ("completed", 1, TypeError),
            ("reason", "garbage", TypeError),
            ("control_steps", True, TypeError),
            ("control_steps", -1, ValueError),
            ("last_observation_sequence", True, TypeError),
            ("last_observation_sequence", -1, ValueError),
            ("last_agent_result", [], TypeError),
        )

        for field_name, value, error_type in invalid_fields:
            with self.subTest(field=field_name, value=value):
                options = {**valid, field_name: value}
                with self.assertRaises(error_type):
                    RuntimeReport(**options)

    def test_invalid_task_is_rejected_before_runtime_acquires_resources(self) -> None:
        env = RecordingEnv()
        context = RuntimeContext(env)

        with self.assertRaisesRegex(TypeError, "task must implement RuntimeTask"):
            AgentRuntime(
                context=context,
                task=object(),  # type: ignore[arg-type]
                agent=RecordingAgent(),
            )

        self.assertEqual(env.reset_seeds, [])
        self.assertEqual(env.close_count, 0)

    def test_ready_runtime_can_be_closed_without_reset_and_only_once(self) -> None:
        events: list[str] = []
        env = RecordingEnv(events=events)
        task = RecordingTask((True,), events=events)
        runtime = build_runtime(env, task, RecordingAgent())

        runtime.close()
        runtime.close()

        self.assertEqual(events, ["task.close", "env.close"])
        self.assertEqual(env.reset_seeds, [])
        self.assertEqual(task.close_count, 1)
        self.assertEqual(env.close_count, 1)
        self.assertIs(runtime.state, RuntimeState.CLOSED)
        with self.assertRaisesRegex(RuntimeError, "can only be run once"):
            runtime.run()

    def test_reset_observation_is_the_first_frame_given_to_task_and_agent(self) -> None:
        events: list[str] = []
        env = RecordingEnv(reset_sequence=41, events=events)
        task = RecordingTask((True,), events=events)
        agent = RecordingAgent(events=events)
        runtime = build_runtime(env, task, agent)

        report = runtime.run(seed=73)

        self.assertEqual(env.reset_seeds, [73])
        self.assertEqual(task.input_sequences, [41])
        self.assertEqual(
            agent.inputs[0]["messages"][0]["content"],
            "observation:41",
        )
        self.assertEqual(report.last_observation_sequence, 41)
        self.assertEqual(events[:3], ["env.reset", "task.build_input", "agent.invoke"])

    def test_first_step_success_does_not_fetch_an_extra_observation(self) -> None:
        env = RecordingEnv(reset_sequence=8)
        task = RecordingTask((True,))
        agent = RecordingAgent()
        runtime = build_runtime(env, task, agent, max_steps=4)

        report = runtime.run()

        self.assertTrue(report.completed)
        self.assertIs(report.reason, CompletionReason.TASK_COMPLETE)
        self.assertEqual(report.control_steps, 1)
        self.assertEqual(env.get_obs_count, 0)
        self.assertEqual(runtime.context.observation_version, 1)

    def test_incomplete_attempt_refreshes_before_next_outer_step(self) -> None:
        env = RecordingEnv(reset_sequence=20)
        task = RecordingTask((False, True))
        agent = RecordingAgent()
        runtime = build_runtime(env, task, agent, max_steps=3)

        report = runtime.run()

        self.assertEqual(task.input_sequences, [20, 21])
        self.assertEqual(env.get_obs_count, 1)
        self.assertEqual(runtime.context.observation_version, 2)
        self.assertTrue(report.completed)
        self.assertEqual(report.control_steps, 2)
        self.assertEqual(report.last_observation_sequence, 21)

    def test_failed_action_still_refreshes_for_the_next_outer_step(self) -> None:
        rejected = Result(success=False, message="motion rejected")
        env = RecordingEnv(reset_sequence=30, action_results=(rejected,))
        task = RecordingTask((False, True))
        step_results: list[Result] = []
        runtime: AgentRuntime

        def invoke(call_number: int, agent_input: Mapping[str, Any]) -> Mapping[str, Any]:
            del agent_input
            if call_number == 1:
                step_results.append(runtime.step(Action("move")))
            return {"attempt": call_number}

        agent = RecordingAgent(callback=invoke)
        runtime = build_runtime(env, task, agent, max_steps=3)

        report = runtime.run()

        self.assertEqual(step_results, [rejected])
        self.assertEqual(task.input_sequences, [30, 31])
        self.assertEqual(env.get_obs_count, 1)
        self.assertEqual(report.last_observation_sequence, 31)

    def test_step_refreshes_immediately_and_outer_loop_does_not_duplicate_it(self) -> None:
        events: list[str] = []
        env = RecordingEnv(reset_sequence=50, events=events)
        task = RecordingTask((False, True), events=events)
        observed_inside_agent: list[int] = []
        runtime: AgentRuntime

        def invoke(call_number: int, agent_input: Mapping[str, Any]) -> Mapping[str, Any]:
            del agent_input
            if call_number == 1:
                result = runtime.step(Action("move", {"distance_m": 0.1}))
                self.assertTrue(result.success)
                observed_inside_agent.append(
                    runtime.context.require_observation().sequence
                )
            return {"attempt": call_number}

        agent = RecordingAgent(events=events, callback=invoke)
        runtime = build_runtime(env, task, agent, max_steps=3)

        report = runtime.run()

        self.assertEqual(observed_inside_agent, [51])
        self.assertEqual(task.input_sequences, [50, 51])
        self.assertEqual(env.get_obs_count, 1)
        self.assertEqual(len(env.actions), 1)
        self.assertEqual(report.control_steps, 2)
        self.assertEqual(
            events,
            [
                "env.reset",
                "task.build_input",
                "agent.invoke",
                "env.take_action",
                "env.get_obs",
                "task.is_complete",
                "task.build_input",
                "agent.invoke",
                "task.is_complete",
                "task.close",
                "env.close",
            ],
        )

    def test_step_limit_returns_an_incomplete_report_with_latest_state(self) -> None:
        env = RecordingEnv(reset_sequence=60)
        task = RecordingTask((False, False))
        agent = RecordingAgent()
        runtime = build_runtime(env, task, agent, max_steps=2)

        report = runtime.run()

        self.assertFalse(report.completed)
        self.assertIs(report.reason, CompletionReason.STEP_LIMIT)
        self.assertEqual(report.control_steps, 2)
        self.assertEqual(report.last_observation_sequence, 61)
        self.assertEqual(report.last_agent_result, {"call": 2})
        self.assertEqual(task.input_sequences, [60, 61])
        self.assertEqual(env.get_obs_count, 1)

    def test_agent_exception_still_closes_task_then_environment(self) -> None:
        events: list[str] = []
        primary_error = RuntimeError("agent exploded")
        env = RecordingEnv(events=events)
        task = RecordingTask((True,), events=events)
        agent = RecordingAgent(events=events, error=primary_error)
        runtime = build_runtime(env, task, agent)

        with self.assertLogs("module.runtime", level="ERROR"):
            with self.assertRaises(RuntimeError) as caught:
                runtime.run()

        self.assertIs(caught.exception, primary_error)
        self.assertEqual(events[-2:], ["task.close", "env.close"])
        self.assertEqual(task.close_count, 1)
        self.assertEqual(env.close_count, 1)
        self.assertIs(runtime.state, RuntimeState.CLOSED)

    def test_task_close_failure_does_not_prevent_environment_close(self) -> None:
        events: list[str] = []
        close_error = RuntimeError("task close failed")
        env = RecordingEnv(events=events)
        task = RecordingTask((True,), events=events, close_error=close_error)
        agent = RecordingAgent(events=events)
        runtime = build_runtime(env, task, agent)

        with self.assertLogs("module.runtime", level="ERROR"):
            with self.assertRaises(RuntimeError) as caught:
                runtime.run()

        self.assertIs(caught.exception, close_error)
        self.assertEqual(events[-2:], ["task.close", "env.close"])
        self.assertEqual(env.close_count, 1)
        self.assertIs(runtime.state, RuntimeState.CLOSED)

    def test_primary_and_cleanup_failures_are_aggregated_in_occurrence_order(self) -> None:
        events: list[str] = []
        primary_error = RuntimeError("agent failed")
        task_close_error = RuntimeError("task cleanup failed")
        env_close_error = RuntimeError("env cleanup failed")
        env = RecordingEnv(events=events, close_error=env_close_error)
        task = RecordingTask(
            (True,),
            events=events,
            close_error=task_close_error,
        )
        agent = RecordingAgent(events=events, error=primary_error)
        runtime = build_runtime(env, task, agent)

        with self.assertLogs("module.runtime", level="ERROR"):
            with self.assertRaises(BaseExceptionGroup) as caught:
                runtime.run()

        group = caught.exception
        self.assertEqual(group.args[0], "runtime execution and cleanup failed")
        self.assertEqual(
            group.exceptions,
            (primary_error, task_close_error, env_close_error),
        )
        self.assertEqual(events[-2:], ["task.close", "env.close"])
        self.assertIs(runtime.state, RuntimeState.CLOSED)

    def test_runtime_is_single_use_and_second_run_does_not_reclose_resources(self) -> None:
        env = RecordingEnv()
        task = RecordingTask((True,))
        agent = RecordingAgent()
        runtime = build_runtime(env, task, agent)

        runtime.run()
        with self.assertRaisesRegex(RuntimeError, "can only be run once"):
            runtime.run()

        self.assertIs(runtime.state, RuntimeState.CLOSED)
        self.assertEqual(env.reset_seeds, [None])
        self.assertEqual(task.close_count, 1)
        self.assertEqual(env.close_count, 1)
        self.assertEqual(len(agent.inputs), 1)


if __name__ == "__main__":
    unittest.main()

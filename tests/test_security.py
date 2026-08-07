"""Regression tests for bounded messages and complete log redaction."""

from __future__ import annotations

from contextlib import redirect_stderr
import io
import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from module.agent import AgentTool, MockChatModel, RecoverableToolError, TopAgent
from script.run import configure_logging


class AgentMessageRedactionTests(unittest.TestCase):
    def test_sensitive_mapping_fields_and_bearer_text_are_redacted(self) -> None:
        agent = TopAgent(
            model=MockChatModel(
                final_output={
                    "provider_api_key": "FINAL-KEY",
                    "authorization": "Bearer FINAL-BEARER",
                    "path": Path("api_key=PATH-SECRET"),
                }
            ),
            system_prompt="Authorization: Bearer SYSTEM-BEARER",
        )

        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": {
                            "openai_api_key": "INPUT-KEY",
                            "access_token": "INPUT-TOKEN",
                            "ssh_private_key": "PRIVATE-KEY",
                            "http_authorization": "Basic AUTH-VALUE",
                            "quoted": (
                                "api_key='MY SECRET VALUE'; "
                                '{\"client_secret\": \"alpha beta\"}; '
                                r'{"password": "abc\" SECOND HALF"}; '
                                'private_key="LINE1\nLINE2"'
                            ),
                            "safe": "visible",
                        },
                    }
                ]
            }
        )
        serialized = json.dumps(result, sort_keys=True)

        for secret in (
            "FINAL-KEY",
            "FINAL-BEARER",
            "SYSTEM-BEARER",
            "INPUT-KEY",
            "INPUT-TOKEN",
            "PRIVATE-KEY",
            "AUTH-VALUE",
            "MY SECRET VALUE",
            "alpha beta",
            "SECOND HALF",
            "LINE1",
            "LINE2",
            "PATH-SECRET",
        ):
            self.assertNotIn(secret, serialized)
        self.assertIn("<redacted>", serialized)
        self.assertIn("visible", serialized)

    def test_non_string_mapping_key_never_invokes_caller_stringification(self) -> None:
        class SecretKey:
            string_calls = 0

            def __str__(self) -> str:
                type(self).string_calls += 1
                return "api_key=LEAK-FROM-KEY"

        key = SecretKey()
        agent = TopAgent(model=MockChatModel())

        result = agent.invoke(
            {"messages": [{"role": "user", "content": {key: "visible"}}]}
        )
        serialized = json.dumps(result, sort_keys=True)

        self.assertEqual(SecretKey.string_calls, 0)
        self.assertNotIn("LEAK-FROM-KEY", serialized)
        self.assertIn("<non-string-key:1>", serialized)
        self.assertIn("visible", serialized)

    def test_model_visible_tool_spec_has_no_handler_or_raw_secret(self) -> None:
        tool = AgentTool(
            name="safe_tool",
            description="Use safely; api_key=SPEC-SECRET",
            handler=lambda: None,
        )
        agent = TopAgent(model=MockChatModel(), tools=(tool,))

        self.assertEqual(len(agent.tool_specs), 1)
        self.assertFalse(hasattr(agent.tool_specs[0], "handler"))
        self.assertNotIn("SPEC-SECRET", agent.tool_specs[0].description)
        self.assertIn("<redacted>", agent.tool_specs[0].description)

    def test_tool_exception_message_cannot_leak_credentials(self) -> None:
        def fail() -> None:
            raise RecoverableToolError("openai_api_key=TOOL-SECRET")

        agent = TopAgent(
            model=MockChatModel(tool_sequence=("fail_safely",)),
            tools=(
                AgentTool(
                    name="fail_safely",
                    description="Fail deterministically for a redaction test.",
                    handler=fail,
                ),
            ),
            max_iterations=2,
        )

        with self.assertLogs("module.agent.agent", level=logging.WARNING) as captured:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": "run"}]}
            )
        serialized = json.dumps(result, sort_keys=True)
        self.assertNotIn("TOOL-SECRET", serialized)
        self.assertNotIn("TOOL-SECRET", "\n".join(captured.output))
        self.assertIn("<redacted>", serialized)


class CompleteLogRedactionTests(unittest.TestCase):
    def test_traceback_mapping_and_bearer_credentials_are_redacted(self) -> None:
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
                try:
                    raise RuntimeError(
                        "openai_api_key=TRACEBACK-SECRET "
                        "client_secret='TRACE BACK SECRET' "
                        r'{"password": "abc\" TRACE SECOND HALF"} '
                        'private_key="TRACE LINE1\nTRACE LINE2"'
                    )
                except RuntimeError:
                    logging.getLogger("tests.security").exception(
                        "payload=%s Authorization: Bearer HEADER-SECRET",
                        {"openai_api_key": "MAPPING-SECRET"},
                    )
                configured_handlers = list(root_logger.handlers)
                for handler in configured_handlers:
                    handler.flush()
                log_text = Path(log_path).read_text(encoding="utf-8")
        finally:
            root_logger.handlers.clear()
            for handler in configured_handlers:
                handler.close()
            root_logger.handlers.extend(original_handlers)
            root_logger.setLevel(original_level)

        combined = log_text + stderr.getvalue()
        for secret in (
            "TRACEBACK-SECRET",
            "MAPPING-SECRET",
            "HEADER-SECRET",
            "TRACE BACK SECRET",
            "TRACE SECOND HALF",
            "TRACE LINE1",
            "TRACE LINE2",
        ):
            self.assertNotIn(secret, combined)
        self.assertIn("<redacted>", combined)
        self.assertIn("Traceback", combined)


if __name__ == "__main__":
    unittest.main()

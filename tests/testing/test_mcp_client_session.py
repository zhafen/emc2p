"""Deterministic tests for McpClientSession -- a real MCP server subprocess
(the actual `emc2p-mcp` under test), with litellm scripted/mocked out.

No live model calls: `litellm.acompletion` is patched with a scripted
sequence of fake responses, so these are free, fast, and exercise the real
MCP wire protocol (tool schemas from the real server, real call_tool round
trips) without depending on any model's actual judgment -- that's what
`tests/test_human_validated/` covers instead.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from emc2p.testing.mcp_client_session import McpClientSession

REPO_ROOT = Path(__file__).parent.parent.parent
STATUS_BOARD_SCENARIO_DIR = REPO_ROOT / "tests" / "data" / "scenarios" / "status_board"
FAKE_MODEL = "deepseek/deepseek-chat"


# Plain duck-typed doubles for litellm's response shape, not litellm's own
# pydantic types -- keeps these tests from depending on litellm/pydantic.
@dataclass
class _FakeFunction:
    name: str
    arguments: str


@dataclass
class _FakeToolCall:
    id: str
    function: _FakeFunction
    type: str = "function"


@dataclass
class _FakeMessage:
    content: str | None = None
    tool_calls: list[_FakeToolCall] | None = None
    role: str = "assistant"

    def model_dump(self, exclude_none: bool = False) -> dict:
        data = {
            "role": self.role,
            "content": self.content,
            "tool_calls": [
                {"id": tc.id, "type": tc.type, "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in self.tool_calls
            ]
            if self.tool_calls
            else None,
        }
        return {k: v for k, v in data.items() if v is not None} if exclude_none else data


@dataclass
class _FakeUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: object = None


@dataclass
class _FakeChoice:
    message: _FakeMessage


@dataclass
class _FakeResponse:
    choices: list[_FakeChoice]
    model: str
    usage: _FakeUsage


def _mcp_config(tmp_path: Path) -> Path:
    config = tmp_path / ".mcp.json"
    config.write_text(json.dumps({"mcpServers": {"emc2p": {"type": "stdio", "command": "uv", "args": ["run", "emc2p-mcp"]}}}))
    return config


def _session(tmp_path: Path, *, allowed_tools: list[str] | None = None, **kwargs) -> McpClientSession:
    return McpClientSession(
        mcp_config=_mcp_config(tmp_path),
        allowed_tools=allowed_tools or ["mcp__emc2p__*"],
        cwd=REPO_ROOT,
        model=FAKE_MODEL,
        trace_dir=tmp_path / "traces",
        **kwargs,
    )


def _tool_call_response(name: str, arguments: dict, *, call_id: str = "call_1") -> _FakeResponse:
    message = _FakeMessage(content=None, tool_calls=[_FakeToolCall(id=call_id, function=_FakeFunction(name=name, arguments=json.dumps(arguments)))])
    return _FakeResponse(
        choices=[_FakeChoice(message=message)],
        model=FAKE_MODEL,
        usage=_FakeUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def _text_response(text: str) -> _FakeResponse:
    return _FakeResponse(
        choices=[_FakeChoice(message=_FakeMessage(content=text))],
        model=FAKE_MODEL,
        usage=_FakeUsage(prompt_tokens=8, completion_tokens=3, total_tokens=11),
    )


def _patched_model(*responses: _FakeResponse):
    return patch("emc2p.agents.tool_calling_loop.litellm.acompletion", new=AsyncMock(side_effect=list(responses)))


class TestRealMcpRoundTrip:
    @pytest.mark.slow
    def test_send_turn_drives_the_real_server_and_returns_final_text(self, tmp_path):
        db_path = tmp_path / "test.duckdb"
        with _patched_model(
            _tool_call_response(
                "mcp__emc2p__open_registry",
                {"database_url": f"duckdb:///{db_path}", "manifest_dir": str(STATUS_BOARD_SCENARIO_DIR)},
            ),
            _text_response("Registry opened."),
        ):
            with _session(tmp_path) as session:
                result = session.send_turn("Open the registry.")
        assert result == "Registry opened."
        assert db_path.exists()

    def test_tool_schemas_come_from_the_real_server_not_a_hand_mirror(self, tmp_path):
        with _patched_model(_text_response("done")):
            with _session(tmp_path) as session:
                session.send_turn("hello")
                names = {t["function"]["name"] for t in session._openai_tools}
        assert names == {
            "mcp__emc2p__open_registry",
            "mcp__emc2p__view_registry",
            "mcp__emc2p__view_entity",
            "mcp__emc2p__update_registry",
            "mcp__emc2p__consolidate_review",
        }

    def test_allowed_tools_filters_to_the_matching_pattern(self, tmp_path):
        with _patched_model(_text_response("done")):
            with _session(tmp_path, allowed_tools=["mcp__emc2p__view_registry"]) as session:
                session.send_turn("hello")
                names = {t["function"]["name"] for t in session._openai_tools}
        assert names == {"mcp__emc2p__view_registry"}

    @pytest.mark.slow
    def test_multiple_send_turn_calls_share_one_open_registry(self, tmp_path):
        db_path = tmp_path / "test.duckdb"
        with _patched_model(
            _tool_call_response(
                "mcp__emc2p__open_registry",
                {"database_url": f"duckdb:///{db_path}", "manifest_dir": str(STATUS_BOARD_SCENARIO_DIR)},
            ),
            _text_response("opened"),
            _tool_call_response("mcp__emc2p__view_registry", {"component_type": "description"}),
            _text_response("read back"),
        ):
            with _session(tmp_path) as session:
                session.send_turn("open it")
                second = session.send_turn("now read it back")
        assert second == "read back"

    def test_unknown_tool_name_is_reported_as_an_error_not_a_crash(self, tmp_path):
        with _patched_model(
            _tool_call_response("mcp__emc2p__not_a_real_tool", {}),
            _text_response("gave up"),
        ):
            with _session(tmp_path) as session:
                result = session.send_turn("try something bogus")
        assert result == "gave up"


class TestUsageAndTrace:
    def test_usage_summary_accumulates_across_turns(self, tmp_path):
        with _patched_model(_text_response("first"), _text_response("second")):
            with _session(tmp_path) as session:
                session.send_turn("one")
                session.send_turn("two")
                usage = session.usage_summary()
        assert usage["input_tokens"] == 16
        assert usage["output_tokens"] == 6

    def test_trace_file_records_the_turn(self, tmp_path):
        with _patched_model(_text_response("hi there")):
            with _session(tmp_path) as session:
                session.send_turn("hello")
                trace_path = session.trace_path
        lines = [json.loads(line) for line in trace_path.read_text().splitlines()]
        assert {"type": "user", "content": "hello"} in lines
        assert any(line["type"] == "assistant" and line["content"] == "hi there" for line in lines)

    def test_usage_json_written_on_close(self, tmp_path):
        with _patched_model(_text_response("hi")):
            with _session(tmp_path) as session:
                session.send_turn("hello")
                trace_path = session.trace_path
        usage_path = trace_path.with_suffix(".usage.json")
        assert usage_path.exists()
        assert json.loads(usage_path.read_text())["input_tokens"] > 0


class TestLifecycle:
    def test_close_is_safe_to_call_twice(self, tmp_path):
        with _patched_model(_text_response("hi")):
            with _session(tmp_path) as session:
                session.send_turn("hello")
            session.close()  # __exit__ already called this once

    def test_turn_timeout_fires_with_its_own_message(self, tmp_path):
        async def _hang(*args, **kwargs):
            import asyncio

            await asyncio.sleep(10)

        with patch("emc2p.agents.tool_calling_loop.litellm.acompletion", new=AsyncMock(side_effect=_hang)):
            with _session(tmp_path, turn_timeout=0.2, session_timeout=300) as session:
                with pytest.raises(pytest.fail.Exception) as exc_info:
                    session.send_turn("hello")
        assert "turn timed out after 0.2s" in str(exc_info.value)

    def test_missing_command_on_path_skips_instead_of_hanging(self, tmp_path):
        config = tmp_path / ".mcp.json"
        config.write_text(
            json.dumps({"mcpServers": {"emc2p": {"type": "stdio", "command": "definitely-not-a-real-command"}}})
        )
        session = McpClientSession(
            mcp_config=config,
            allowed_tools=["mcp__emc2p__*"],
            cwd=REPO_ROOT,
            model=FAKE_MODEL,
            trace_dir=tmp_path / "traces",
        )
        with pytest.raises(pytest.skip.Exception):
            with session:
                pass

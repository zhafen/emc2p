"""Unit tests for HeadlessSession's timeout/kill mechanics -- turn_timeout,
session_timeout, and _fail's process-termination path. No real `claude`
CLI or Anthropic API calls: a plain `cat` subprocess stands in for the
real one, so these run fast and free as part of the normal test suite.

mcp_config/allowed_tools/cwd/model are all irrelevant to what's under test
here (only __enter__ uses them, and __enter__ is bypassed entirely by
_stub_session below), so dummy values stand in for them.
"""

import subprocess
import time
from pathlib import Path

import pytest

from emc2p.testing.headless_session import (
    ClaudeHeadlessProvider,
    CopilotHeadlessProvider,
    HeadlessProvider,
    HeadlessSession,
)


class _EchoProvider(HeadlessProvider):
    name = "echo"

    @property
    def executable(self) -> str:
        return "cat"

    def build_command(
        self,
        *,
        mcp_config: str | Path,
        allowed_tools: list[str],
        model: str,
        strict_tool_isolation: bool,
        options: dict,
    ) -> list[str]:
        return ["cat"]

    def encode_user_message(self, prompt: str) -> str:
        return prompt

    def decode_result_event(self, line: str) -> dict | None:
        text = line.strip()
        if not text.startswith("RESULT:"):
            return None
        return {
            "kind": "result",
            "result": text.removeprefix("RESULT:"),
            "total_cost_usd": 0.0,
            "usage": {
                "input_tokens": 1,
                "output_tokens": 2,
                "cache_read_input_tokens": 3,
                "cache_creation_input_tokens": 4,
            },
            "duration_api_ms": 5,
        }


def _stub_session(
    tmp_path,
    *,
    allowed_tools: list[str] | None = None,
    model: str = "dummy-model",
    **kwargs,
) -> HeadlessSession:
    """A HeadlessSession with `cat` standing in for the real `claude -p`
    subprocess: reads stdin until EOF then exits promptly, unlike a bare
    `sleep`, which would ignore stdin closing and force close()'s full
    30s wait/kill fallback on every test here. Bypasses __enter__
    entirely (no shutil.which("claude") check, no real MCP config) since
    only the timeout/kill mechanics are under test, not an actual
    conversation.
    """
    session = HeadlessSession(
        mcp_config=tmp_path / ".mcp.json",
        allowed_tools=allowed_tools or ["mcp__dummy__*"],
        cwd=tmp_path,
        model=model,
        trace_dir=tmp_path,
        **kwargs,
    )
    session.trace_path = tmp_path / "trace.jsonl"
    session._session_deadline = time.monotonic() + session.session_timeout
    session._trace_file = open(session.trace_path, "w")
    session.proc = subprocess.Popen(
        ["cat"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    return session


def test_fail_terminates_a_still_running_process_without_hanging(tmp_path):
    """The bug this guards against: _fail() used to read the process's
    stderr (blocking until EOF) before ever killing it -- a deadlock,
    since EOF only arrives once the process exits, and nothing killed it
    until after that read. Confirmed live once (see docs/manifest/
    history.yaml: project_history.headless_session_fail_deadlock_fixed):
    every timeout in this harness silently hung instead of firing until
    this was fixed."""
    session = _stub_session(tmp_path)
    proc = session.proc
    assert proc.poll() is None  # still running

    start = time.monotonic()
    with pytest.raises(pytest.fail.Exception) as exc_info:
        session._fail("boom")
    elapsed = time.monotonic() - start

    assert "boom" in str(exc_info.value)
    assert elapsed < 10, f"_fail took {elapsed:.1f}s -- should return quickly, not hang"
    assert proc.poll() is not None, "the subprocess must be terminated, not left running"


def test_turn_timeout_fires_with_its_own_message(tmp_path):
    session = _stub_session(tmp_path, turn_timeout=0.2, session_timeout=300)
    start = time.monotonic()
    with pytest.raises(pytest.fail.Exception) as exc_info:
        session.send_turn("hello")
    elapsed = time.monotonic() - start

    assert "turn timed out after 0.2s" in str(exc_info.value)
    assert elapsed < 10


def test_session_timeout_fires_when_it_is_the_sooner_deadline(tmp_path):
    """send_turn's deadline is min(turn_deadline, session_deadline) -- a
    session past its own overall budget must fail with the session
    message, not silently wait out the (deliberately generous)
    turn_timeout instead."""
    session = _stub_session(tmp_path, turn_timeout=300, session_timeout=300)
    session._session_deadline = time.monotonic() - 1  # already past
    start = time.monotonic()
    with pytest.raises(pytest.fail.Exception) as exc_info:
        session.send_turn("hello")
    elapsed = time.monotonic() - start

    assert "session timed out after 300s" in str(exc_info.value)
    assert elapsed < 10


def test_close_is_safe_to_call_twice(tmp_path):
    session = _stub_session(tmp_path)
    session.close()
    session.close()  # must not raise
    assert session.proc is None


def test_send_turn_uses_provider_contract_and_accumulates_usage(tmp_path):
    session = _stub_session(
        tmp_path,
        provider=_EchoProvider(),
        strict_tool_isolation=False,
    )
    reply = session.send_turn("RESULT:hello")
    assert reply == "hello"
    assert session.usage_summary() == {
        "total_cost_usd": 0.0,
        "input_tokens": 1,
        "output_tokens": 2,
        "cache_read_input_tokens": 3,
        "cache_creation_input_tokens": 4,
        "duration_api_ms": 5,
    }


def test_copilot_result_normalization_uses_message_payload():
    provider = CopilotHeadlessProvider()
    normalized = provider.decode_result_event(
        '{"type":"turn.completed","message":{"content":[{"text":"ok"}]},"usage":{"input_tokens":2}}'
    )
    assert normalized == {
        "kind": "result",
        "result": "ok",
        "total_cost_usd": 0.0,
        "usage": {
            "input_tokens": 2,
            "output_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        },
        "duration_api_ms": 0,
    }


def test_strict_tool_isolation_prompt_fallback_for_non_strict_provider(tmp_path):
    session = _stub_session(
        tmp_path,
        provider=_EchoProvider(),
        allowed_tools=["mcp__foo__*"],
        strict_tool_isolation=True,
        turn_timeout=0.2,
        session_timeout=300,
    )
    with pytest.raises(pytest.fail.Exception) as exc_info:
        session.send_turn("hello")
    message = str(exc_info.value)
    assert "turn timed out" in message
    trace = session.trace_path.read_text()
    assert "Use only MCP tools allowed for this run." in trace
    assert "Allowed tool patterns: mcp__foo__*" in trace


def test_provider_smoke_claude_cli_gate():
    import shutil

    if shutil.which("claude") is None:
        pytest.skip("claude CLI not on PATH")
    provider = ClaudeHeadlessProvider()
    cmd = provider.build_command(
        mcp_config=".mcp.json",
        allowed_tools=["mcp__dummy__*"],
        model="claude-sonnet-4.5",
        strict_tool_isolation=True,
        options={},
    )
    assert cmd[0] == "claude"


def test_provider_smoke_copilot_cli_gate():
    import shutil

    if shutil.which("copilot") is None:
        pytest.skip("copilot CLI not on PATH")
    provider = CopilotHeadlessProvider()
    cmd = provider.build_command(
        mcp_config=".mcp.json",
        allowed_tools=["mcp__dummy__*"],
        model="gpt-5-mini",
        strict_tool_isolation=True,
        options={},
    )
    assert cmd[0] == "copilot"

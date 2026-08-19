"""Generic infrastructure for driving a real MCP server through a real MCP
client -- `HeadlessSession` spawns a provider CLI as a subprocess."""

from __future__ import annotations

import json
import os
import selectors
import subprocess
import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pytest

# --tools "" strips every built-in tool outright, not a --disallowedTools
# denylist, so a model under MCP-tool test can't reach for one.
_STRIP_BUILTIN_TOOLS_ARGS = ["--tools", ""]


class HeadlessProvider(ABC):
    """Provider contract for the CLI process `HeadlessSession` drives."""

    name = "provider"

    @property
    @abstractmethod
    def executable(self) -> str:
        pass

    @property
    def supports_strict_tool_isolation(self) -> bool:
        return False

    def skip_message(self) -> str:
        return f"{self.executable} CLI not on PATH"

    def filter_env(self, env: dict[str, str]) -> dict[str, str]:
        return env

    @abstractmethod
    def build_command(
        self,
        *,
        mcp_config: str | Path,
        allowed_tools: list[str],
        model: str,
        strict_tool_isolation: bool,
        options: dict[str, Any],
    ) -> list[str]:
        pass

    def prepare_prompt(
        self,
        prompt: str,
        *,
        allowed_tools: list[str],
        strict_tool_isolation: bool,
    ) -> str:
        if strict_tool_isolation and not self.supports_strict_tool_isolation:
            allowed = ", ".join(allowed_tools) if allowed_tools else "(none)"
            return (
                "Use only MCP tools allowed for this run. "
                f"Allowed tool patterns: {allowed}. "
                f"{prompt}"
            )
        return prompt

    def encode_user_message(self, prompt: str) -> str:
        message = {"type": "user", "message": {"role": "user", "content": prompt}}
        return json.dumps(message)

    @abstractmethod
    def decode_result_event(self, line: str) -> dict[str, Any] | None:
        """Return normalized result event dict (kind/result/error/usage/
        total_cost_usd/duration_api_ms), or None for non-result lines."""


def _normalize_usage(usage: dict[str, Any] | None) -> dict[str, int]:
    usage = usage or {}
    return {
        "input_tokens": int(usage.get("input_tokens", 0) or 0),
        "output_tokens": int(usage.get("output_tokens", 0) or 0),
        "cache_read_input_tokens": int(usage.get("cache_read_input_tokens", 0) or 0),
        "cache_creation_input_tokens": int(usage.get("cache_creation_input_tokens", 0) or 0),
    }


class ClaudeHeadlessProvider(HeadlessProvider):
    name = "claude"

    @property
    def executable(self) -> str:
        return "claude"

    @property
    def supports_strict_tool_isolation(self) -> bool:
        return True

    def filter_env(self, env: dict[str, str]) -> dict[str, str]:
        # Dropping every CLAUDE*/AI_AGENT var is load-bearing: a nested
        # claude -p can otherwise inherit and attach to this outer session.
        return {k: v for k, v in env.items() if not k.startswith("CLAUDE") and k != "AI_AGENT"}

    def build_command(
        self,
        *,
        mcp_config: str | Path,
        allowed_tools: list[str],
        model: str,
        strict_tool_isolation: bool,
        options: dict[str, Any],
    ) -> list[str]:
        command = [
            self.executable,
            "-p",
            "--input-format",
            "stream-json",
            "--output-format",
            "stream-json",
            "--verbose",
            "--mcp-config",
            str(mcp_config),
            "--strict-mcp-config",
            "--allowedTools",
            *allowed_tools,
        ]
        if strict_tool_isolation:
            command.extend(_STRIP_BUILTIN_TOOLS_ARGS)
        command.extend(["--model", model])
        return command

    def decode_result_event(self, line: str) -> dict[str, Any] | None:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return None
        if event.get("type") != "result":
            return None
        if event.get("is_error"):
            return {"kind": "error", "error": f"turn errored: {event}"}
        return {
            "kind": "result",
            "result": event["result"],
            "total_cost_usd": float(event.get("total_cost_usd", 0.0) or 0.0),
            "usage": _normalize_usage(event.get("usage")),
            "duration_api_ms": int(event.get("duration_api_ms", 0) or 0),
        }


class CopilotHeadlessProvider(HeadlessProvider):
    name = "copilot"

    def __init__(self, *, options: dict[str, Any] | None = None):
        self.options = options or {}

    @property
    def executable(self) -> str:
        return str(self.options.get("executable", "copilot"))

    def filter_env(self, env: dict[str, str]) -> dict[str, str]:
        # Prevent nested sessions from accidentally inheriting current
        # Copilot/agent runtime session wiring.
        blocked_prefixes = ("COPILOT_", "GITHUB_COPILOT_", "AI_AGENT")
        return {k: v for k, v in env.items() if not any(k.startswith(p) for p in blocked_prefixes)}

    def build_command(
        self,
        *,
        mcp_config: str | Path,
        allowed_tools: list[str],
        model: str,
        strict_tool_isolation: bool,
        options: dict[str, Any],
    ) -> list[str]:
        merged = {**self.options, **options}
        command = list(merged.get("base_command", [self.executable, "agent", "run"]))
        mcp_flag = merged.get("mcp_config_flag", "--mcp-config")
        if mcp_flag:
            command.extend([str(mcp_flag), str(mcp_config)])
        strict_mcp_flag = merged.get("strict_mcp_config_flag", "--strict-mcp-config")
        if strict_mcp_flag:
            command.append(str(strict_mcp_flag))
        model_flag = merged.get("model_flag", "--model")
        if model_flag:
            command.extend([str(model_flag), model])
        allowed_tools_flag = merged.get("allowed_tools_flag", "--allowedTools")
        if allowed_tools and allowed_tools_flag:
            command.append(str(allowed_tools_flag))
            command.extend(allowed_tools)
        if strict_tool_isolation and self.supports_strict_tool_isolation:
            command.extend([str(x) for x in merged.get("strip_builtin_tools_args", [])])
        return command

    def decode_result_event(self, line: str) -> dict[str, Any] | None:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return None

        payload = event.get("data") if isinstance(event.get("data"), dict) else event
        if not isinstance(payload, dict):
            return None

        err = payload.get("error") or payload.get("message_error")
        if payload.get("is_error") or err:
            return {"kind": "error", "error": f"turn errored: {err or payload}"}

        event_type = str(payload.get("type", "")).lower()
        complete_types = {
            "result",
            "done",
            "turn.complete",
            "turn.completed",
            "response.complete",
            "response.completed",
        }
        looks_complete = (
            event_type in complete_types
            or bool(payload.get("result"))
            or payload.get("status") in {"done", "completed", "success"}
        )
        if not looks_complete:
            return None

        result = (
            payload.get("result")
            or payload.get("output_text")
            or payload.get("assistant_response")
            or _extract_message_text(payload.get("message"))
            or ""
        )

        usage = payload.get("usage") or payload.get("token_usage")
        return {
            "kind": "result",
            "result": str(result),
            "total_cost_usd": float(payload.get("total_cost_usd", payload.get("cost_usd", 0.0)) or 0.0),
            "usage": _normalize_usage(usage),
            "duration_api_ms": int(payload.get("duration_api_ms", payload.get("duration_ms", 0)) or 0),
        }


def _extract_message_text(message: Any) -> str:
    if isinstance(message, str):
        return message
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    parts.append(str(item.get("text", "")))
            return "".join(parts)
    return ""


def _coerce_provider(provider: str | HeadlessProvider, options: dict[str, Any]) -> HeadlessProvider:
    if isinstance(provider, HeadlessProvider):
        return provider
    if provider == "claude":
        return ClaudeHeadlessProvider()
    if provider == "copilot":
        return CopilotHeadlessProvider(options=options)
    raise ValueError(f"unsupported headless provider: {provider}")


class HeadlessSession:
    """One continuously-running provider process, driven over stream-json --
    keeps a single MCP server child process alive for the whole `with` block."""

    def __init__(
        self,
        *,
        mcp_config: str | Path,
        allowed_tools: list[str],
        cwd: str | Path,
        model: str,
        provider: str | HeadlessProvider = "claude",
        provider_options: dict[str, Any] | None = None,
        strict_tool_isolation: bool = True,
        extra_env: dict[str, str] | None = None,
        trace_dir: str | Path | None = None,
        turn_timeout: float = 240,
        session_timeout: float = 120,
    ):
        """mcp_config/allowed_tools/cwd/model describe the MCP server under
        test; extra_env/trace_dir/turn_timeout/session_timeout tune the run."""
        self.mcp_config = mcp_config
        self.allowed_tools = allowed_tools
        self.cwd = cwd
        self.model = model
        self.provider_options = provider_options or {}
        self.provider = _coerce_provider(provider, self.provider_options)
        self.strict_tool_isolation = strict_tool_isolation
        self.extra_env = extra_env or {}
        self.turn_timeout = turn_timeout
        self.session_timeout = session_timeout
        self._session_deadline: float | None = None
        self.proc: subprocess.Popen | None = None
        trace_dir = Path(trace_dir) if trace_dir is not None else Path(cwd) / ".live_test_traces"
        trace_dir.mkdir(parents=True, exist_ok=True)
        self.trace_path = trace_dir / f"{time.strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}.jsonl"
        self._trace_file = None
        # Summed across every send_turn call this session makes, so a
        # caller reads one session total instead of re-deriving it from the trace.
        self.total_cost_usd: float = 0.0
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.total_cache_read_tokens: int = 0
        self.total_cache_creation_tokens: int = 0
        self.total_duration_api_ms: int = 0

    def __enter__(self) -> "HeadlessSession":
        import shutil

        if shutil.which(self.provider.executable) is None:
            pytest.skip(self.provider.skip_message())
        self._session_deadline = time.monotonic() + self.session_timeout
        self._trace_file = open(self.trace_path, "w")
        env = self.provider.filter_env(dict(os.environ))
        env.update(self.extra_env)
        command = self.provider.build_command(
            mcp_config=self.mcp_config,
            allowed_tools=self.allowed_tools,
            model=self.model,
            strict_tool_isolation=self.strict_tool_isolation,
            options=self.provider_options,
        )
        self.proc = subprocess.Popen(
            command,
            cwd=self.cwd,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        return self

    def send_turn(self, prompt: str) -> str:
        """Send one prompt and block for its narrated reply -- every raw
        line is appended to self.trace_path as it arrives, not just the result."""
        assert self.proc is not None and self.proc.stdin is not None
        prepared = self.provider.prepare_prompt(
            prompt,
            allowed_tools=self.allowed_tools,
            strict_tool_isolation=self.strict_tool_isolation,
        )
        self.proc.stdin.write(self.provider.encode_user_message(prepared) + "\n")
        self.proc.stdin.flush()

        sel = selectors.DefaultSelector()
        sel.register(self.proc.stdout, selectors.EVENT_READ)
        assert self._session_deadline is not None
        turn_deadline = time.monotonic() + self.turn_timeout
        deadline = min(turn_deadline, self._session_deadline)
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    if self._session_deadline <= turn_deadline:
                        self._fail(
                            f"session timed out after {self.session_timeout}s total "
                            "(this turn was still waiting for a reply)"
                        )
                    self._fail(f"turn timed out after {self.turn_timeout}s waiting for a reply")
                if not sel.select(timeout=remaining):
                    continue
                line = self.proc.stdout.readline()
                if not line:
                    self._fail(f"{self.provider.name} process exited before replying")
                if self._trace_file is not None:
                    self._trace_file.write(line)
                    self._trace_file.flush()
                event = self.provider.decode_result_event(line)
                if event is None:
                    continue
                if event["kind"] == "error":
                    self._fail(str(event["error"]))
                self._accumulate_usage(event)
                return str(event["result"])
        finally:
            sel.close()

    def _accumulate_usage(self, result_event: dict[str, Any]) -> None:
        self.total_cost_usd += result_event.get("total_cost_usd", 0.0) or 0.0
        usage = result_event.get("usage") or {}
        self.total_input_tokens += usage.get("input_tokens", 0) or 0
        self.total_output_tokens += usage.get("output_tokens", 0) or 0
        self.total_cache_read_tokens += usage.get("cache_read_input_tokens", 0) or 0
        self.total_cache_creation_tokens += usage.get("cache_creation_input_tokens", 0) or 0
        self.total_duration_api_ms += result_event.get("duration_api_ms", 0) or 0

    def usage_summary(self) -> dict:
        """This session's usage so far, summed across every send_turn call.
        duration_api_ms is the model's own active time, not wall-clock time."""
        return {
            "total_cost_usd": self.total_cost_usd,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "cache_read_input_tokens": self.total_cache_read_tokens,
            "cache_creation_input_tokens": self.total_cache_creation_tokens,
            "duration_api_ms": self.total_duration_api_ms,
        }

    def _fail(self, message: str) -> None:
        """close() before reading stderr, not after -- read() blocks until
        EOF, which only arrives once close() has killed the process."""
        proc = self.proc
        self.close()
        stderr = proc.stderr.read() if proc and proc.stderr else ""
        message = f"{message}\ntrace: {self.trace_path}\nstderr: {stderr}"
        pytest.fail(message)

    def close(self) -> None:
        if self._trace_file is not None:
            self._trace_file.close()
            self._trace_file = None
            # Persisted next to the trace unconditionally, so every live
            # test's spend is recoverable even if the test itself fails.
            usage_path = self.trace_path.with_suffix(".usage.json")
            usage_path.write_text(json.dumps(self.usage_summary(), indent=2))
        if self.proc is None:
            return
        proc, self.proc = self.proc, None
        if proc.stdin:
            proc.stdin.close()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()

    def __exit__(self, *exc_info) -> None:
        self.close()

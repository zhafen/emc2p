"""Generic infrastructure for driving a real MCP server through a real MCP
client -- `HeadlessSession` spawns `claude -p` as a subprocess against a
caller-supplied `.mcp.json`, exactly the way a client connecting to that
server would.

Extracted from story-simulator's own test harness: nothing here knows about
any downstream project's MCP tool names, registry backend, or domain
scenarios. A caller supplies `mcp_config`, `allowed_tools`, `cwd`, `model`,
and any `extra_env` its own MCP server(s) need (e.g. a database URL derived
from a save/workdir the caller picked) -- see story-simulator's
tests/test_human_validated.py for a downstream subclass that fixes these to
one project's own defaults.
"""

import json
import os
import selectors
import subprocess
import time
import uuid
from pathlib import Path

import pytest

# --tools "" strips every built-in Claude Code tool (Bash, Read, Write,
# Edit, Glob, Grep, WebFetch, WebSearch, Task, ...), leaving only the
# caller's own `allowed_tools` -- not just denied via --disallowedTools,
# which still leaves every other built-in tool available. Confirmed
# happening in practice (story-simulator, #-numbered issue in that repo's
# own history): a weak model, hitting an MCP tool timeout, used Glob+Read
# to go read the connected project's own source code trying to
# self-diagnose the failure instead of retrying or reporting it -- burning
# the rest of the run on unrelated exploration. A model being tested for
# how it uses one project's MCP tools has no legitimate reason to reach for
# unrelated built-in tools; removing the option outright is more robust
# than a denylist of specific tool names, which only covers what's been
# caught happening so far.
_STRIP_BUILTIN_TOOLS_ARGS = ["--tools", ""]


class HeadlessSession:
    """One continuously-running `claude -p` process, driven over stream-json.

    Unlike separate one-shot calls, this keeps a single MCP server child
    process alive for the whole `with` block, so `send_turn`'s caller can
    inspect whatever state that server manages in between turns while it's
    still holding it open, and so the test itself can pace the conversation
    rather than handing the model a single "do everything" prompt with no
    chance to inspect state along the way.
    """

    def __init__(
        self,
        *,
        mcp_config: str | Path,
        allowed_tools: list[str],
        cwd: str | Path,
        model: str,
        extra_env: dict[str, str] | None = None,
        trace_dir: str | Path | None = None,
        turn_timeout: float = 240,
        session_timeout: float = 120,
    ):
        """
        Args:
            mcp_config: Path (absolute, or relative to `cwd`) to the
                `.mcp.json` describing the MCP server(s) under test.
            allowed_tools: The `mcp__<server>__*`-style tool patterns this
                session may call -- passed straight to `claude -p
                --allowedTools`.
            cwd: Working directory `claude -p` is spawned in (its own
                `.mcp.json` resolution, and typically the repo whose MCP
                server(s) are under test).
            model: `claude -p --model` value, e.g. a specific model ID.
            extra_env: Extra environment variables the spawned process
                needs beyond a filtered copy of this process's own
                environment (see `__enter__` for what's filtered out) --
                e.g. a database URL a caller's MCP server(s) need, derived
                from whatever workdir/save the caller picked before this
                session was constructed.
            trace_dir: Directory every raw stream-json line is appended to,
                one file per session -- not just the final narrated result.
                Diagnosing live-test flakiness needs the intermediate
                thinking/tool_use/tool_result payloads, not only
                send_turn's own return value. Defaults to
                `.live_test_traces` under `cwd`. Gitignored by convention --
                these are per-run debug artifacts, not source.
            turn_timeout: Seconds to wait for a single send_turn's reply.
            session_timeout: Seconds bounding the whole session's
                wall-clock runtime (every send_turn call combined), not
                just a single one -- the backstop for a model that's
                technically still replying within each turn_timeout but
                never doing anything useful (see `send_turn`'s wandering-
                off-task note above `_STRIP_BUILTIN_TOOLS_ARGS`).
        """
        self.mcp_config = mcp_config
        self.allowed_tools = allowed_tools
        self.cwd = cwd
        self.model = model
        self.extra_env = extra_env or {}
        self.turn_timeout = turn_timeout
        self.session_timeout = session_timeout
        self._session_deadline: float | None = None
        self.proc: subprocess.Popen | None = None
        trace_dir = Path(trace_dir) if trace_dir is not None else Path(cwd) / ".live_test_traces"
        trace_dir.mkdir(parents=True, exist_ok=True)
        self.trace_path = trace_dir / f"{time.strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}.jsonl"
        self._trace_file = None
        # Accumulated across every send_turn call this session makes (a
        # session is usually several turns) -- each turn's "result" event
        # already reports its own cost/usage (see send_turn), this just
        # sums them so a caller can read one total for the whole session
        # instead of re-deriving it from the trace file after the fact.
        self.total_cost_usd: float = 0.0
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.total_cache_read_tokens: int = 0
        self.total_cache_creation_tokens: int = 0
        self.total_duration_api_ms: int = 0

    def __enter__(self) -> "HeadlessSession":
        import shutil

        if shutil.which("claude") is None:
            pytest.skip("claude CLI not on PATH")
        self._session_deadline = time.monotonic() + self.session_timeout
        self._trace_file = open(self.trace_path, "w")
        # Dropping every CLAUDE*/AI_AGENT env var (not just adding
        # extra_env on top) is load-bearing, not cosmetic -- confirmed
        # live: run from inside a Claude Code Cloud session, this
        # process's own environment carries CLAUDE_CODE_SESSION_ID (plus
        # its messaging socket/token, oauth fd, container id, ...) for
        # *this* session. Passed through via a plain `{**os.environ, ...}`,
        # the nested `claude -p` subprocess picked that session ID up and
        # attached to this exact outer session's own transcript/state
        # instead of starting a fresh, isolated one. A local terminal
        # invocation never had these vars set in the first place, which is
        # presumably why this went unnoticed until first run from a Cloud
        # session.
        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith("CLAUDE") and k != "AI_AGENT"
        }
        env.update(self.extra_env)
        self.proc = subprocess.Popen(
            [
                "claude",
                "-p",
                "--input-format",
                "stream-json",
                "--output-format",
                "stream-json",
                "--verbose",
                "--mcp-config",
                str(self.mcp_config),
                "--strict-mcp-config",
                "--allowedTools",
                *self.allowed_tools,
                *_STRIP_BUILTIN_TOOLS_ARGS,
                "--model",
                self.model,
            ],
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
        """Send one prompt on the ongoing conversation and block for its narrated reply.

        Every raw line read from the subprocess -- not just the final
        result -- is appended to `self.trace_path` as it arrives, so a
        failure (or a passing-but-suspicious run) can be inspected after
        the fact instead of only ever seeing the narrated text this method
        returns.
        """
        assert self.proc is not None and self.proc.stdin is not None
        message = {"type": "user", "message": {"role": "user", "content": prompt}}
        self.proc.stdin.write(json.dumps(message) + "\n")
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
                    self._fail("claude process exited before replying")
                if self._trace_file is not None:
                    self._trace_file.write(line)
                    self._trace_file.flush()
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "result":
                    if event.get("is_error"):
                        self._fail(f"turn errored: {event}")
                    self._accumulate_usage(event)
                    return event["result"]
        finally:
            sel.close()

    def _accumulate_usage(self, result_event: dict) -> None:
        self.total_cost_usd += result_event.get("total_cost_usd", 0.0) or 0.0
        usage = result_event.get("usage") or {}
        self.total_input_tokens += usage.get("input_tokens", 0) or 0
        self.total_output_tokens += usage.get("output_tokens", 0) or 0
        self.total_cache_read_tokens += usage.get("cache_read_input_tokens", 0) or 0
        self.total_cache_creation_tokens += usage.get("cache_creation_input_tokens", 0) or 0
        self.total_duration_api_ms += result_event.get("duration_api_ms", 0) or 0

    def usage_summary(self) -> dict:
        """This session's usage so far, summed across every send_turn call.

        `duration_api_ms` is time actually spent generating (the model's
        own "active" time), not this process's wall-clock time -- the
        metric that maps onto a usage-plan's hours-per-week allotment.
        """
        return {
            "total_cost_usd": self.total_cost_usd,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "cache_read_input_tokens": self.total_cache_read_tokens,
            "cache_creation_input_tokens": self.total_cache_creation_tokens,
            "duration_api_ms": self.total_duration_api_ms,
        }

    def _fail(self, message: str) -> None:
        # close() first, not stderr.read() first: read() blocks until EOF,
        # which only arrives once the subprocess exits -- but the process
        # is still alive at this point (that's the whole reason _fail was
        # called), and nothing kills it until close() runs. Reading before
        # closing deadlocks forever, silently swallowing every timeout this
        # method exists to enforce. close() sets self.proc to None, so the
        # proc handle is captured first for the stderr read afterward.
        proc = self.proc
        self.close()
        stderr = proc.stderr.read() if proc and proc.stderr else ""
        message = f"{message}\ntrace: {self.trace_path}\nstderr: {stderr}"
        pytest.fail(message)

    def close(self) -> None:
        if self._trace_file is not None:
            self._trace_file.close()
            self._trace_file = None
            # Persisted next to the trace unconditionally (not just when a
            # caller bothers to read usage_summary() itself) so every live
            # test's spend is recoverable after the fact, matching how the
            # raw trace itself is already kept regardless of whether a
            # given test happens to fail.
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

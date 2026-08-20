"""A `HeadlessSession`-shaped alternative that is its own MCP client and
its own agent loop, instead of shelling out to an external agentic CLI.

`HeadlessSession` drives a real MCP server through a *provider CLI*
(`claude -p`, `copilot`) that already bundles both an MCP client and an
agent loop internally -- fine for providers that ship such a CLI, but not
every litellm-supported model has one (see docs/manifest/history.yaml:
project_history.mcp_client_session_added for why this exists: DeepSeek's
own CLI, `dsh`, is a week-old one-shot-only developer preview with no
documented streaming output or usage reporting -- not a good fit for
`HeadlessProvider`'s contract). `McpClientSession` owns both halves
itself instead: `mcp.client.stdio`/`mcp.ClientSession` for the real wire
protocol (the actual tool schemas a connected client would see, not a
hand-mirrored copy), and `emc2p.agents.tool_calling_loop.
run_tool_calling_loop` (via litellm) for the model<->tool-call exchange --
so it works with any litellm-supported model, not just DeepSeek.

Same public shape as `HeadlessSession` (context manager, `send_turn`,
`usage_summary`, `trace_path`, `turn_timeout`/`session_timeout`) so it can
be dropped in wherever a downstream project's own fixture currently
constructs a `HeadlessSession`. Requires both the `agents` extra
(`litellm`) and the `mcp` extra (the `mcp` client SDK) -- `pip install
emc2p[agents,mcp]`.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import fnmatch
import json
import os
import queue
import shutil
import threading
import time
import uuid
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from emc2p.agents.tool_calling_loop import run_tool_calling_loop

DEFAULT_MAX_TOOL_ITERATIONS = 15
DEFAULT_CONNECT_TIMEOUT = 30


def _load_server_params(mcp_config: str | Path, cwd: str | Path, extra_env: dict[str, str]) -> dict[str, StdioServerParameters]:
    """Parse a `.mcp.json`'s `mcpServers` into one `StdioServerParameters` per server.

    Only the `stdio` transport is supported -- the only kind either repo's
    own `.mcp.json` uses. `extra_env` is merged onto a filtered copy of
    this process's own environment for every server, the same way
    `HeadlessSession.extra_env` applies to the single CLI process it spawns.
    """
    config = json.loads(Path(mcp_config).read_text())
    env = {**os.environ, **extra_env}
    params = {}
    for name, server in config.get("mcpServers", {}).items():
        if server.get("type", "stdio") != "stdio":
            raise ValueError(f"McpClientSession only supports stdio servers, got {server.get('type')!r} for {name!r}")
        params[name] = StdioServerParameters(
            command=server["command"],
            args=server.get("args", []),
            env={**env, **server.get("env", {})},
            cwd=str(cwd),
        )
    return params


class McpClientSession:
    """One continuously-running MCP client + agent loop, driven by litellm.

    Keeps every spawned server's stdio subprocess and `ClientSession` open
    for the whole `with` block, the same way `HeadlessSession` keeps its
    single CLI process alive across turns -- so state a server manages (an
    opened registry, ...) persists between `send_turn` calls.

    Everything MCP/litellm-side is async and runs on a single background
    task on its own event-loop thread -- connect, every `send_turn`, and
    teardown all execute inside that *same* task, not one task per call
    (`mcp`'s stdio transport opens an anyio task group at connect time, and
    anyio requires a cancel scope to be exited from the same task it was
    entered in -- confirmed live: closing it from a separate
    `run_coroutine_threadsafe` task raises "cancel scope in a different
    task"). A thread-safe queue feeds work requests into that one task; a
    `concurrent.futures.Future` per request bridges each synchronous public
    method call back to its result, so callers use this class exactly like
    `HeadlessSession` -- no `async def test_...` needed.
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
        max_tool_iterations: int = DEFAULT_MAX_TOOL_ITERATIONS,
    ):
        """mcp_config/allowed_tools/cwd describe the MCP server(s) under
        test; `model` is any litellm-recognized model string.

        Args:
            mcp_config: Path to the `.mcp.json` describing the MCP
                server(s) under test (stdio transport only).
            allowed_tools: `mcp__<server>__<tool>`-style glob patterns
                (e.g. `"mcp__emc2p__*"`). Only servers named by at least
                one pattern are spawned at all; within a spawned server,
                only matching tools are exposed to the model -- there is
                no separate built-in-tool surface to isolate from here,
                unlike a full agentic CLI, since the model only ever sees
                MCP tools in the first place.
            cwd: Working directory each server subprocess is spawned in.
            model: A litellm model string (e.g. `"deepseek/deepseek-chat"`,
                `"claude-sonnet-4-5"`) -- picks both the provider and the
                API key env var litellm looks for.
            extra_env: Extra environment variables every spawned server
                needs beyond a copy of this process's own environment.
            trace_dir: Directory this session's trace file is written
                under. Defaults to `.live_test_traces` under `cwd`, same
                convention as `HeadlessSession`.
            turn_timeout: Seconds to wait for a single `send_turn` call.
            session_timeout: Seconds bounding the whole session's
                wall-clock runtime, across every `send_turn` call combined.
            max_tool_iterations: Passed straight to
                `run_tool_calling_loop`'s own `max_iterations`, per turn.
        """
        self.mcp_config = mcp_config
        self.allowed_tools = allowed_tools
        self.cwd = cwd
        self.model = model
        self.extra_env = extra_env or {}
        self.turn_timeout = turn_timeout
        self.session_timeout = session_timeout
        self.max_tool_iterations = max_tool_iterations
        self._session_deadline: float | None = None

        trace_dir = Path(trace_dir) if trace_dir is not None else Path(cwd) / ".live_test_traces"
        trace_dir.mkdir(parents=True, exist_ok=True)
        self.trace_path = trace_dir / f"{time.strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}.jsonl"
        self._usage_log_path = self.trace_path.with_suffix(".usage_log.jsonl")
        self._trace_file = None

        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._inbox: queue.Queue = queue.Queue()
        self._sessions: dict[str, ClientSession] = {}
        self._openai_tools: list[dict[str, Any]] = []
        self._tool_lookup: dict[str, tuple[str, str]] = {}

        self.total_cost_usd: float = 0.0
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.total_cache_read_tokens: int = 0
        self.total_cache_creation_tokens: int = 0
        self.total_duration_api_ms: int = 0

    # Lifecycle

    def __enter__(self) -> "McpClientSession":
        server_params = _load_server_params(self.mcp_config, self.cwd, self.extra_env)
        wanted = {name: params for name, params in server_params.items() if self._server_is_used(name)}
        for name, params in wanted.items():
            if shutil.which(params.command) is None:
                pytest.skip(f"{params.command!r} (MCP server {name!r}) not on PATH")

        self._session_deadline = time.monotonic() + self.session_timeout
        self._trace_file = open(self.trace_path, "w")
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

        connected: concurrent.futures.Future = concurrent.futures.Future()
        asyncio.run_coroutine_threadsafe(self._worker(wanted, connected), self._loop)
        try:
            connected.result(timeout=DEFAULT_CONNECT_TIMEOUT)
        except BaseException:
            # __enter__ raising means __exit__ never runs -- this class has
            # to unwind its own thread/loop here, not rely on close().
            self._thread.join(timeout=DEFAULT_CONNECT_TIMEOUT)
            self._loop.close()
            self._loop = None
            self._thread = None
            raise
        return self

    async def _worker(self, server_params: dict[str, StdioServerParameters], connected: concurrent.futures.Future) -> None:
        """Owns the whole session's async lifetime in one task: connect, every queued turn, then teardown.

        See the class docstring for why this can't be split across
        separate `run_coroutine_threadsafe` calls instead.
        """
        stack = AsyncExitStack()
        try:
            async with stack:
                try:
                    await self._connect(stack, server_params)
                except BaseException as exc:
                    connected.set_exception(exc)
                    return
                connected.set_result(None)
                await self._serve()
        finally:
            asyncio.get_running_loop().stop()

    async def _connect(self, stack: AsyncExitStack, server_params: dict[str, StdioServerParameters]) -> None:
        for name, params in server_params.items():
            read, write = await stack.enter_async_context(stdio_client(params))
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            self._sessions[name] = session
            for tool in (await session.list_tools()).tools:
                full_name = f"mcp__{name}__{tool.name}"
                if not any(fnmatch.fnmatchcase(full_name, pattern) for pattern in self.allowed_tools):
                    continue
                self._tool_lookup[full_name] = (name, tool.name)
                self._openai_tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": full_name,
                            "description": tool.description or "",
                            "parameters": tool.inputSchema,
                        },
                    }
                )

    async def _serve(self) -> None:
        """Pull queued (coro_factory, future) work items off `self._inbox`
        (a thread-safe queue) until a `None` sentinel asks it to stop."""
        loop = asyncio.get_running_loop()
        while True:
            item = await loop.run_in_executor(None, self._inbox.get)
            if item is None:
                return
            coro_factory, future = item
            try:
                result = await coro_factory()
                if not future.cancelled():
                    future.set_result(result)
            except BaseException as exc:  # noqa: BLE001 -- reported back to the caller's own thread
                if not future.cancelled():
                    future.set_exception(exc)

    def _server_is_used(self, server_name: str) -> bool:
        prefix = f"mcp__{server_name}__"
        return any(pattern.startswith(prefix) for pattern in self.allowed_tools)

    def _submit(self, coro_factory, timeout: float | None) -> Any:
        future: concurrent.futures.Future = concurrent.futures.Future()
        self._inbox.put((coro_factory, future))
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            future.cancel()
            raise

    # Turns

    def send_turn(self, prompt: str) -> str:
        """Send one prompt on the ongoing conversation and block for its final reply.

        Every model response and tool call/result this turn makes is
        appended to `self.trace_path` as it happens, the same debugging
        role `HeadlessSession`'s raw-line trace plays.
        """
        assert self._session_deadline is not None
        turn_deadline = time.monotonic() + self.turn_timeout
        deadline = min(turn_deadline, self._session_deadline)
        session_is_tighter = self._session_deadline <= turn_deadline
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            self._fail_on_timeout(session_is_tighter, "this turn was still waiting to start")
        self._trace_file.write(json.dumps({"type": "user", "content": prompt}) + "\n")
        self._trace_file.flush()
        try:
            # `_run_turn` itself enforces the authoritative `asyncio.wait_for`
            # timeout -- this outer bound just needs a little slack past it.
            return self._submit(lambda: self._run_turn(prompt, remaining), timeout=remaining + 5)
        except (concurrent.futures.TimeoutError, asyncio.TimeoutError, TimeoutError):
            self._fail_on_timeout(session_is_tighter, "this turn was still waiting for a reply")
        except Exception as exc:  # noqa: BLE001 -- surfaced as a normal pytest failure, trace attached
            self._fail(str(exc))

    def _fail_on_timeout(self, session_is_tighter: bool, still_doing: str) -> None:
        if session_is_tighter:
            self._fail(f"session timed out after {self.session_timeout}s total ({still_doing})")
        self._fail(f"turn timed out after {self.turn_timeout}s waiting for a reply")

    async def _run_turn(self, prompt: str, timeout: float) -> str:
        return await asyncio.wait_for(
            run_tool_calling_loop(
                prompt,
                model=self.model,
                dispatch=self._dispatch,
                tools=self._openai_tools,
                max_iterations=self.max_tool_iterations,
                usage_log_path=self._usage_log_path,
                on_response=self._trace_and_accumulate,
            ),
            timeout=timeout,
        )

    async def _dispatch(self, full_name: str, arguments: dict[str, Any]) -> str:
        if full_name not in self._tool_lookup:
            return f"Error: {full_name!r} is not an allowed tool for this session."
        server_name, real_name = self._tool_lookup[full_name]
        result = await self._sessions[server_name].call_tool(real_name, arguments)
        text = "".join(getattr(block, "text", "") for block in result.content)
        self._trace_file.write(
            json.dumps({"type": "tool_result", "name": full_name, "is_error": result.isError, "content": text}) + "\n"
        )
        self._trace_file.flush()
        if result.isError:
            return f"Error: {text}"
        return text

    def _trace_and_accumulate(self, response: Any) -> None:
        message = response.choices[0].message
        tool_calls = [
            {"name": tc.function.name, "arguments": tc.function.arguments} for tc in (message.tool_calls or [])
        ]
        self._trace_file.write(
            json.dumps({"type": "assistant", "content": message.content, "tool_calls": tool_calls}) + "\n"
        )
        self._trace_file.flush()

        usage = getattr(response, "usage", None)
        try:
            import litellm

            cost = litellm.completion_cost(completion_response=response)
        except Exception:
            cost = 0.0
        self.total_cost_usd += cost or 0.0
        self.total_input_tokens += getattr(usage, "prompt_tokens", 0) or 0
        self.total_output_tokens += getattr(usage, "completion_tokens", 0) or 0
        details = getattr(usage, "prompt_tokens_details", None)
        self.total_cache_read_tokens += getattr(details, "cached_tokens", 0) or 0

    def usage_summary(self) -> dict:
        """This session's usage so far, summed across every send_turn call.

        `cache_creation_input_tokens`/`duration_api_ms` stay 0 for
        providers litellm doesn't report them for (most non-Anthropic
        ones, including DeepSeek) -- present for shape-compatibility with
        `HeadlessSession.usage_summary()`, not populated everywhere.
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
        self.close()
        pytest.fail(f"{message}\ntrace: {self.trace_path}")

    def close(self) -> None:
        if self._trace_file is not None:
            self._trace_file.close()
            self._trace_file = None
            usage_path = self.trace_path.with_suffix(".usage.json")
            usage_path.write_text(json.dumps(self.usage_summary(), indent=2))
        if self._loop is None:
            return
        loop, self._loop = self._loop, None
        # None is _serve's stop sentinel -- the worker task then tears down
        # the AsyncExitStack (same task it was built in) and stops `loop` itself.
        self._inbox.put(None)
        if self._thread is not None:
            self._thread.join(timeout=30)
            self._thread = None
        loop.close()

    def __exit__(self, *exc_info) -> None:
        self.close()

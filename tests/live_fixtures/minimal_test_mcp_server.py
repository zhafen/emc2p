"""Minimal test MCP server for tests/testing/test_trace_convergence.py.

Not part of the emc2p package or its console scripts -- spawned directly
via `uv run python tests/live_fixtures/minimal_test_mcp_server.py` from
that test's own mcp.json (see MCP_CONFIG there). Exposes `resolve`, one
tool with a `responder` argument mirroring the three generic ways an MCP
server can get a judgment call answered without knowing anything about
any particular game/workflow domain -- see docs/manifest/history.yaml:
project_history.trace_convergence_test_moved_from_a_downstream_project
for where this pattern first showed up:

- "host": the connected model answers directly, from its own accumulated
  context -- no delegation, no nested call of any kind. The control case:
  no nested trace should ever appear for this one.
- "subagent": the connected model is asked to answer via its OWN isolated
  subagent mechanism (e.g. Claude Code's Agent/Task tool), not from its
  own context. This server does nothing to make that converge -- whether
  the subagent's own tool activity lands in the same trace file is
  entirely that mechanism's own responsibility, so this is only
  meaningful for a driver that actually has such a mechanism.
- "keyed_subagent": this server itself runs a nested `run_tool_calling_loop`
  call in-process (no MCP round-trip at all), appending to
  `client_trace_path` if given -- the trace-threading pattern this whole
  test module exists to pin down.

`record_note` is a real MCP tool (not just an internal tool spec) so a
"subagent"-style nested call -- which answers through the connected
client's own subagent mechanism, entirely outside this server's control
-- has something to call that this server can observe in the shared
trace file.
"""

from __future__ import annotations

from pathlib import Path

from mcp.server.fastmcp import Context, FastMCP

from emc2p.agents.tool_calling_loop import run_tool_calling_loop

server = FastMCP("minimal-test-mcp-server")

_NOTE_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "record_note",
            "description": "Record a short note.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": False,
            },
        },
    }
]

_HOST_PREFIX = "Decide the answer yourself, from your own context -- don't delegate.\n\n"

_SUBAGENT_PREFIX = (
    "Answer this via your own subagent mechanism (e.g. Claude Code's Agent/Task "
    "tool), not from your own accumulated context: pass the text below as that "
    "subagent's entire prompt, run it in the foreground, and reply with its "
    "answer verbatim as your own reply.\n\n"
)


@server.tool()
async def record_note(text: str) -> str:
    """Record a short note. A real MCP tool, callable by a client's own
    subagent mechanism as well as directly -- see `resolve`'s "subagent"
    responder.
    """
    return "recorded"


def _dispatch(name: str, arguments: dict) -> str:
    return "recorded"


@server.tool()
async def resolve(prompt: str, responder: str, ctx: Context, client_trace_path: str | None = None) -> str:
    """Answer `prompt` the way `responder` says to -- see this module's
    own docstring for what each of "host"/"subagent"/"keyed_subagent" means.
    """
    if responder == "keyed_subagent":
        return await run_tool_calling_loop(
            prompt,
            model="deepseek/deepseek-chat",
            dispatch=_dispatch,
            tools=_NOTE_TOOL,
            trace_path=Path(client_trace_path) if client_trace_path else None,
        )
    if responder == "host":
        return _HOST_PREFIX + prompt
    if responder == "subagent":
        return _SUBAGENT_PREFIX + prompt
    raise ValueError(f"unknown responder {responder!r}")


def main() -> None:
    server.run(transport="stdio")


if __name__ == "__main__":
    main()

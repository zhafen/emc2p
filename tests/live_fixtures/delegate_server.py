"""Test-only MCP server for tests/testing/test_trace_convergence.py.

Not part of the emc2p package or its console scripts -- spawned directly
via `uv run python tests/live_fixtures/delegate_server.py` from that
test's own mcp.json (see MCP_CONFIG there). Exists purely to give that
test a real MCP tool boundary to drive a nested `run_tool_calling_loop`
call through, the same shape a keyed_subagent-style responder uses.
"""

from __future__ import annotations

from pathlib import Path

from mcp.server.fastmcp import Context, FastMCP

from emc2p.agents.tool_calling_loop import run_tool_calling_loop

server = FastMCP("trace-convergence-fixture")

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


def _dispatch(name: str, arguments: dict) -> str:
    return "recorded"


@server.tool()
async def delegate(prompt: str, ctx: Context, client_trace_path: str | None = None) -> str:
    """Answer `prompt` via a nested model call, letting it call `record_note`.

    `client_trace_path`, if given, is where that nested call's own tool
    calls/results get appended (see `run_tool_calling_loop`'s own
    `trace_path`) -- the same trace-threading pattern a keyed_subagent-style
    responder uses to keep a connected client's trace file whole.
    """
    return await run_tool_calling_loop(
        prompt,
        model="deepseek/deepseek-chat",
        dispatch=_dispatch,
        tools=_NOTE_TOOL,
        trace_path=Path(client_trace_path) if client_trace_path else None,
    )


def main() -> None:
    server.run(transport="stdio")


if __name__ == "__main__":
    main()

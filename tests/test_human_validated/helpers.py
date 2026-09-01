"""Shared infrastructure for live end-to-end tests driving emc2p-mcp via a real MCP client.

`create_session` fixes create_agent_session's constructor to this repo's
own .mcp.json -- emc2p's own dogfooding, exercising emc2p.mcp_server
directly. Always the "mcp_client" driver; DeepSeek by default.
"""

from pathlib import Path

from emc2p.testing.agent_session import AgentSession, create_agent_session

REPO_ROOT = Path(__file__).parent.parent.parent

# Every raw stream-json line from a session's turn lands here, one file
# per session -- not just the final narrated result.
LIVE_TRACE_DIR = REPO_ROOT / ".live_test_traces"

DEFAULT_LIVE_TEST_MODEL = "deepseek/deepseek-chat"

_ALLOWED_TOOLS = ["mcp__emc2p__*"]


def create_session(
    *,
    model: str = DEFAULT_LIVE_TEST_MODEL,
    turn_timeout: float = 240,
    session_timeout: float = 120,
) -> AgentSession:
    """emc2p's own live-test session, fixed to this repo's `.mcp.json`."""
    return create_agent_session(
        "mcp_client",
        mcp_config=REPO_ROOT / ".mcp.json",
        allowed_tools=_ALLOWED_TOOLS,
        cwd=REPO_ROOT,
        model=model,
        trace_dir=LIVE_TRACE_DIR,
        turn_timeout=turn_timeout,
        session_timeout=session_timeout,
    )

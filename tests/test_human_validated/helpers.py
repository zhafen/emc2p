"""Shared infrastructure for live end-to-end tests that drive emc2p-mcp
through a real MCP client.

`McpClientSession` fixes `emc2p.testing.mcp_client_session.
McpClientSession`'s generic constructor to this repo's own
`.mcp.json`/allowed-tools/cwd, the same way a downstream project's own
test suite would -- this package is emc2p's own dogfooding of that
pattern, exercising `emc2p.mcp_server` directly rather than a downstream
project's domain-specific wrapper around it. A real MCP client plus
`emc2p.agents.tool_calling_loop` (via litellm) drives the "connected"
side, so it works with any litellm-supported model -- DeepSeek by
default (DEFAULT_LIVE_TEST_MODEL) -- rather than `HeadlessSession`'s own
`claude -p` subprocess, which can only ever drive a Claude model.
"""

from pathlib import Path

from emc2p.registrar import Registrar
from emc2p.testing.mcp_client_session import McpClientSession as _Emc2pMcpClientSession

REPO_ROOT = Path(__file__).parent.parent.parent

# Every raw stream-json line from a McpClientSession's turn lands here,
# one file per session -- not just the final narrated result.
LIVE_TRACE_DIR = REPO_ROOT / ".live_test_traces"

DEFAULT_LIVE_TEST_MODEL = "deepseek/deepseek-chat"

_ALLOWED_TOOLS = ["mcp__emc2p__*"]


class McpClientSession(_Emc2pMcpClientSession):
    """emc2p's own `McpClientSession`: fixes the generic constructor to
    this repo's own `.mcp.json`/allowed-tools/cwd."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_LIVE_TEST_MODEL,
        turn_timeout: float = 240,
        session_timeout: float = 120,
    ):
        super().__init__(
            mcp_config=REPO_ROOT / ".mcp.json",
            allowed_tools=_ALLOWED_TOOLS,
            cwd=REPO_ROOT,
            model=model,
            trace_dir=LIVE_TRACE_DIR,
            turn_timeout=turn_timeout,
            session_timeout=session_timeout,
        )


def _load_registrar(db_path: Path, narration: str) -> Registrar:
    """Confirm `db_path` was actually created, then load it.

    A weak model can narrate a plausible-sounding scene without ever
    calling `open_registry` -- or call it with a subtly wrong
    `database_url` it mistyped out of the prompt's text -- so this checks
    the database file actually exists before trying to read it, rather
    than hitting a confusing raw `ibis`/duckdb error partway through.
    """
    assert db_path.exists(), (
        f"No database file exists at {db_path} -- open_registry was not "
        f"actually called for this exact database_url despite this "
        f"narration: {narration!r}"
    )
    return Registrar.load(f"duckdb:///{db_path}")

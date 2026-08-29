"""A single entry point for constructing whichever agent-session driver a
test wants: HeadlessSession (a real provider CLI subprocess -- claude or
copilot) or McpClientSession (litellm + emc2p's own MCP client, for any
other litellm-supported model). See docs/manifest/history.yaml:
project_history.mcp_client_session_added for why both exist.

AgentSession is a typing.Protocol, not a base class -- HeadlessSession and
McpClientSession already duck-type this exact shape (see
McpClientSession's own docstring: "Same public shape as HeadlessSession"),
so neither one had to change its declared base class for this to
type-check them structurally. create_agent_session is the one place that
actually picks a driver from a plain string, mirroring _coerce_provider's
own string -> HeadlessProvider resolution one level down inside
HeadlessSession itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from emc2p.testing.headless_session import HeadlessSession
from emc2p.testing.mcp_client_session import McpClientSession

Driver = Literal["claude", "copilot", "mcp_client"]


@runtime_checkable
class AgentSession(Protocol):
    """The common shape HeadlessSession and McpClientSession already share.

    Structural, not nominal: matching this doesn't require either class
    to inherit from anything new, so both already satisfy it as-is.
    """

    trace_path: Path

    def __enter__(self) -> "AgentSession": ...
    def __exit__(self, *exc_info: object) -> None: ...
    def send_turn(self, prompt: str) -> str: ...
    def usage_summary(self) -> dict: ...
    def close(self) -> None: ...


def create_agent_session(
    driver: Driver,
    *,
    mcp_config: str | Path,
    allowed_tools: list[str],
    cwd: str | Path,
    model: str,
    extra_env: dict[str, str] | None = None,
    trace_dir: str | Path | None = None,
    turn_timeout: float = 240,
    session_timeout: float = 120,
    provider_options: dict[str, Any] | None = None,
) -> AgentSession:
    """Construct whichever session class `driver` names.

    Args:
        driver: "claude"/"copilot" spawn a HeadlessSession against that
            provider CLI; "mcp_client" builds a McpClientSession instead
            (litellm-driven, for a model with no headless CLI of its own).
        mcp_config: Path to the `.mcp.json` describing the MCP server(s)
            under test.
        allowed_tools: `mcp__<server>__<tool>`-style glob patterns.
        cwd: Working directory the session runs against.
        model: A model identifier -- meaning depends on `driver` (a
            `claude -p --model` value for "claude", litellm model string
            for "mcp_client"; see each class's own docstring).
        extra_env: Extra environment variables beyond a filtered copy of
            this process's own environment.
        trace_dir: Directory this session's trace file is written under.
            Defaults to `.live_test_traces` under `cwd`.
        turn_timeout: Seconds to wait for a single `send_turn` call.
        session_timeout: Seconds bounding the whole session's wall-clock
            runtime, across every `send_turn` call combined.
        provider_options: Only meaningful for "copilot" today (see
            CopilotHeadlessProvider) -- ignored for "mcp_client".

    Returns:
        A HeadlessSession or McpClientSession -- either way, something
        satisfying AgentSession.

    Raises:
        ValueError: If `driver` isn't one of "claude", "copilot", "mcp_client".
    """
    common = dict(
        mcp_config=mcp_config, allowed_tools=allowed_tools, cwd=cwd, model=model,
        extra_env=extra_env, trace_dir=trace_dir,
        turn_timeout=turn_timeout, session_timeout=session_timeout,
    )
    if driver in ("claude", "copilot"):
        return HeadlessSession(provider=driver, provider_options=provider_options, **common)
    if driver == "mcp_client":
        return McpClientSession(**common)
    raise ValueError(f"Unknown driver {driver!r}; expected 'claude', 'copilot', or 'mcp_client'")

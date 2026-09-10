"""Live regression: a nested `run_tool_calling_loop` call's own trace only
converges with the outer session's trace file when the *connected model*
is actually told to pass the passthrough parameter -- a tool's docstring
describing that parameter is never enough, since the model has no other
way to learn a harness-internal value like its own `trace_path`.

Generic, project-agnostic reproduction of the exact gap story-simulator
hit: `start_world`'s own `client_trace_path` parameter existed and was
unit-tested at the Python level (`RegistrarSessions.set_registrar`/
`get_trace_path`, `run_tool_calling_loop`'s own `trace_path`), but no
live test ever proved a connected model would actually supply it --
because nothing in the live tests' own driving instruction ever told it
to. Uses `tests/live_fixtures/minimal_test_mcp_server.py`, a minimal test
MCP server exposing the same trace-threading shape (a tool with an
optional `client_trace_path` argument, threaded into a nested
tool-calling loop), so this doesn't depend on any downstream project's
own tool names.
"""

import json
from pathlib import Path

import pytest

pytest.importorskip("litellm", reason="requires the 'agents' extra")
pytest.importorskip("mcp", reason="requires the 'mcp' extra")

from emc2p.testing.agent_session import create_agent_session  # noqa: E402
from emc2p.testing.render_trace import parse_trace  # noqa: E402

REPO_ROOT = Path(__file__).parent.parent.parent
MCP_CONFIG = REPO_ROOT / "tests" / "live_fixtures" / "mcp.json"
TRACE_DIR = REPO_ROOT / ".live_test_traces"
_ALLOWED_TOOLS = ["mcp__minimal-test-server__*"]

_NESTED_PROMPT = (
    "Call record_note with text set to exactly 'nested note', then reply with just the word done."
)


def _delegate_instruction(client_trace_path: str | None) -> str:
    call = f'Call delegate with prompt="{_NESTED_PROMPT}"'
    if client_trace_path is not None:
        call += f" and client_trace_path={client_trace_path}"
    return call + ". Don't narrate anything back to me."


def _nested_call_present(trace_path: Path) -> bool:
    """True if `record_note` -- the nested loop's own tool -- shows up
    anywhere in `trace_path`'s parsed turns.

    `delegate` itself is the only tool the outer model calls directly;
    `record_note` can only appear if the nested `run_tool_calling_loop`
    call's own trace_path actually pointed at this same file.
    """
    turns = parse_trace(trace_path)
    return any(tc.name == "record_note" for t in turns for tc in t.tool_calls)


@pytest.mark.live
def test_nested_trace_converges_when_model_is_told_the_trace_path():
    """Success path: the driving instruction explicitly includes the outer
    session's own `trace_path`, so the connected model can actually supply
    it -- the nested call's own tool activity must then appear in the SAME
    trace file as the outer conversation.
    """
    with create_agent_session(
        "mcp_client",
        mcp_config=MCP_CONFIG,
        allowed_tools=_ALLOWED_TOOLS,
        cwd=REPO_ROOT,
        model="deepseek/deepseek-chat",
        trace_dir=TRACE_DIR,
    ) as session:
        session.send_turn(_delegate_instruction(str(session.trace_path)))
        assert _nested_call_present(session.trace_path), (
            f"record_note never showed up in {session.trace_path} -- trace "
            "convergence failed even though the model was explicitly told the trace_path"
        )


@pytest.mark.live
def test_nested_trace_does_not_converge_without_the_model_being_told():
    """Regression pin: relying on `delegate`'s own docstring alone (never
    restating `client_trace_path`'s value in the driving instruction) must
    NOT converge -- a model cannot supply a harness-internal value it was
    never given, no matter how well the parameter is documented.

    This is the exact gap that let story-simulator's client_trace_path
    plumbing ship unused for an entire live-test suite: pinning the
    failure mode here keeps it from silently recurring for any future
    trace-passthrough parameter, in this project or a downstream one.
    """
    with create_agent_session(
        "mcp_client",
        mcp_config=MCP_CONFIG,
        allowed_tools=_ALLOWED_TOOLS,
        cwd=REPO_ROOT,
        model="deepseek/deepseek-chat",
        trace_dir=TRACE_DIR,
    ) as session:
        session.send_turn(_delegate_instruction(None))
        assert not _nested_call_present(session.trace_path), (
            f"record_note showed up in {session.trace_path} even though the model "
            "was never told the trace_path -- if this now fails, the model started "
            "inferring/guessing a harness-internal value, which breaks the assumption "
            "every trace-passthrough parameter in this codebase relies on"
        )

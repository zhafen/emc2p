"""Live regression: a judgment call answered through one of the three
generic responder patterns -- "host" (the connected model decides inline),
"subagent" (the connected model delegates to its own isolated subagent
mechanism), "keyed_subagent" (this process answers it directly, in-process,
via a nested `run_tool_calling_loop` call) -- must converge its own trace
activity into the SAME trace file the outer session already writes to,
whenever that pattern has any nested activity to converge at all.

Generic, project-agnostic reproduction of a gap a downstream project's own
trace-passthrough parameter once hit (see docs/manifest/history.yaml:
project_history.trace_convergence_test_moved_from_a_downstream_project):
a passthrough parameter existed and was unit-tested at the Python level,
but no live test ever proved a connected model would actually supply it --
because nothing in the live tests' own driving instruction ever told it
to. Uses `tests/live_fixtures/minimal_test_mcp_server.py`'s `resolve` tool,
which implements all three responder patterns generically, so this
doesn't depend on any downstream project's own tool names or domain.

Parametrized over every `AgentSession` driver (claude/copilot/mcp_client)
for the patterns that make sense under any driver -- convergence is a
property of each driver's own trace-writing (see
`docs/manifest/history.yaml: project_history.
trace_file_opened_in_append_mode`), not just the one a downstream
project's test suite happens to default to. "subagent" is the exception:
it only makes sense under a driver with its own isolated subagent
mechanism, so it's tested against "claude" alone (see
`resolve`'s own docstring). A driver whose CLI isn't installed/configured
in this environment skips (see `HeadlessSession`'s own
skip-on-missing-executable behavior), rather than failing.
"""

from pathlib import Path

import pytest

pytest.importorskip("litellm", reason="requires the 'agents' extra")
pytest.importorskip("mcp", reason="requires the 'mcp' extra")

from emc2p.testing.agent_session import Driver, create_agent_session  # noqa: E402
from emc2p.testing.render_trace import parse_trace  # noqa: E402

REPO_ROOT = Path(__file__).parent.parent.parent
MCP_CONFIG = REPO_ROOT / "tests" / "live_fixtures" / "mcp.json"
TRACE_DIR = REPO_ROOT / ".live_test_traces"
_ALLOWED_TOOLS = ["mcp__minimal-test-server__*"]

_NESTED_PROMPT = (
    "Call record_note with text set to exactly 'nested note', then reply with just the word done."
)

# mcp_client has no built-in default model (see create_agent_session's own
# docstring), so every driver gets an explicit one here regardless.
_MODEL_BY_DRIVER: dict[Driver, str] = {
    "claude": "claude-haiku-4-5-20251001",
    "copilot": "gpt-4.1",
    "mcp_client": "deepseek/deepseek-chat",
}
_DRIVERS: list[Driver] = ["claude", "copilot", "mcp_client"]


def _resolve_instruction(responder: str, client_trace_path: str | None) -> str:
    call = f'Call resolve with prompt="{_NESTED_PROMPT}" and responder="{responder}"'
    if client_trace_path is not None:
        call += f" and client_trace_path={client_trace_path}"
    return call + ". Don't narrate anything back to me."


def _nested_call_present(trace_path: Path) -> bool:
    """True if `record_note` -- the nested call's own tool -- shows up
    anywhere in `trace_path`'s parsed turns, including inside a native
    subagent's own subtrace (see `render_trace._attach_subagent_subtrace`):
    "subagent" never writes into `trace_path` itself the way
    "keyed_subagent" does -- its activity only surfaces there indirectly,
    via a `task_notification` event `parse_trace` already resolves into
    each spawning `ToolCall`'s own `subtrace`.

    Matched by suffix, not equality: "keyed_subagent"'s nested
    `run_tool_calling_loop` call dispatches through the bare tool-spec
    name ("record_note", no MCP round-trip at all), but a real subagent
    calls it as an actual MCP tool, namespaced
    "mcp__minimal-test-server__record_note" like any other.
    """
    turns = parse_trace(trace_path)

    def _search(tool_calls: list) -> bool:
        for tc in tool_calls:
            if tc.name.endswith("record_note"):
                return True
            if tc.subtrace and any(_search(t.tool_calls) for t in tc.subtrace):
                return True
        return False

    return any(_search(t.tool_calls) for t in turns)


def _session(driver: Driver, *, allowed_tools: list[str] | None = None, strict_tool_isolation: bool = True):
    return create_agent_session(
        driver,
        mcp_config=MCP_CONFIG,
        allowed_tools=allowed_tools if allowed_tools is not None else _ALLOWED_TOOLS,
        cwd=REPO_ROOT,
        model=_MODEL_BY_DRIVER[driver],
        trace_dir=TRACE_DIR,
        strict_tool_isolation=strict_tool_isolation,
    )


@pytest.mark.live
@pytest.mark.parametrize("driver", _DRIVERS)
def test_keyed_subagent_converges_when_model_is_told_the_trace_path(driver: Driver):
    """Success path: the driving instruction explicitly includes the outer
    session's own `trace_path`, so the connected model can actually supply
    it -- the nested call's own tool activity must then appear in the SAME
    trace file as the outer conversation.
    """
    with _session(driver) as session:
        session.send_turn(_resolve_instruction("keyed_subagent", str(session.trace_path)))
        assert _nested_call_present(session.trace_path), (
            f"record_note never showed up in {session.trace_path} -- trace "
            "convergence failed even though the model was explicitly told the trace_path"
        )


@pytest.mark.live
@pytest.mark.parametrize("driver", _DRIVERS)
def test_keyed_subagent_does_not_converge_without_being_told(driver: Driver):
    """Regression pin: relying on `resolve`'s own docstring alone (never
    restating `client_trace_path`'s value in the driving instruction) must
    NOT converge -- a model cannot supply a harness-internal value it was
    never given, no matter how well the parameter is documented. This is
    the exact gap that let a downstream project's own client_trace_path
    plumbing ship unused for an entire live-test suite (see
    docs/manifest/history.yaml: project_history.
    trace_convergence_test_moved_from_a_downstream_project).
    """
    with _session(driver) as session:
        session.send_turn(_resolve_instruction("keyed_subagent", None))
        assert not _nested_call_present(session.trace_path), (
            f"record_note showed up in {session.trace_path} even though the model "
            "was never told the trace_path -- if this now fails, the model started "
            "inferring/guessing a harness-internal value, which breaks the assumption "
            "every trace-passthrough parameter in this codebase relies on"
        )


@pytest.mark.live
@pytest.mark.parametrize("driver", _DRIVERS)
def test_host_never_delegates(driver: Driver):
    """Control case: "host" never delegates to anything -- the connected
    model answers directly, in the outer conversation. It may well call
    `record_note` itself (that's a real top-level MCP tool, not gated
    behind delegation of any kind) -- the invariant is that nothing ever
    shows up as a *nested* subtrace, since "host" gives it no mechanism
    (no client_trace_path, no subagent instruction) to produce one.
    """
    with _session(driver) as session:
        session.send_turn(_resolve_instruction("host", None))
        turns = parse_trace(session.trace_path)
        assert not any(tc.subtrace for t in turns for tc in t.tool_calls), (
            f"a tool call in {session.trace_path} has a subtrace attached for the "
            "'host' responder -- it isn't supposed to delegate to anything"
        )


@pytest.mark.live
def test_subagent_converges_via_the_clients_own_mechanism():
    """"subagent" hands the prompt to the connected model's OWN isolated
    subagent mechanism (e.g. Claude Code's Agent/Task tool) instead of this
    server driving anything itself -- unlike "keyed_subagent", there's no
    `client_trace_path` to pass, so convergence here depends entirely on
    that mechanism's own trace-writing (confirmed reachable via a
    `task_notification` event's `output_file` -- see
    `render_trace._attach_subagent_subtrace`, first confirmed working by a
    downstream project's own regression test, referenced from
    docs/manifest/history.yaml: project_history.
    trace_convergence_matrix_generalized_to_every_responder_pattern).
    Needs the CLI's own native subagent tool available, so
    `strict_tool_isolation=False` here -- the default elsewhere in this
    module strips it along with every other built-in.

    Only "claude" has such a mechanism; skips (via `HeadlessSession`'s own
    skip-on-missing-executable behavior) if the CLI isn't available in
    this environment. Delegation itself is trust-based, same as any other
    "subagent" responder (nothing forces the model to actually use Task
    instead of answering inline) -- so this can fail flaky, not just on a
    real regression.
    """
    with _session(
        "claude", allowed_tools=[*_ALLOWED_TOOLS, "Task"], strict_tool_isolation=False
    ) as session:
        session.send_turn(_resolve_instruction("subagent", None))
        assert _nested_call_present(session.trace_path), (
            f"record_note never showed up in {session.trace_path} (including any "
            "subagent subtrace) -- the client's own subagent mechanism didn't "
            "converge its trace with the outer session"
        )

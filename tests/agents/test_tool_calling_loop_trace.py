"""Gap-revealing live test: can a subagent's own interactions be traced?

`run_tool_calling_loop` (`emc2p.agents.tool_calling_loop`) is the generic
in-process "subagent" mechanism `keyed_subagent`-style responders build
on -- a nested model + tool-calling exchange that currently runs and
returns only a final string. Nothing about that exchange (which tools it
called, with what arguments, what came back) is ever written anywhere a
caller could inspect after the fact, unlike a top-level live session
(`emc2p.testing.headless_session.HeadlessSession`/`agent_session.py`),
which always writes a `.jsonl` trace readable by
`emc2p.testing.render_trace.parse_trace`.

This test specifies the desired capability -- a `trace_path` a caller can
pass to `run_tool_calling_loop` so the subagent's own tool calls/results
land in a trace file in the same shape `render_trace` already parses --
and is expected to fail (`TypeError: unexpected keyword argument
'trace_path'`) until that capability exists. It is deliberately not
implemented as part of this change; see the project's tracking issue/PR
for closing this gap.
"""

import asyncio
from pathlib import Path

import pytest

pytest.importorskip("litellm", reason="requires the 'agents' extra")

from emc2p.agents.tool_calling_loop import run_tool_calling_loop  # noqa: E402
from emc2p.testing.render_trace import parse_trace  # noqa: E402

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

_PROMPT = (
    "Call record_note with text set to exactly 'the car parked in spot 1', "
    "then reply with just the word done."
)


@pytest.mark.live
def test_subagent_tool_calls_are_recorded_to_a_parseable_trace(tmp_path: Path):
    """A real model's tool call/result inside `run_tool_calling_loop` must
    show up in a trace file `render_trace.parse_trace` can read -- the
    same tooling already used to inspect a top-level live session.
    """
    trace_path = tmp_path / "subagent_trace.jsonl"
    dispatched: list[dict] = []

    def dispatch(name: str, arguments: dict) -> str:
        dispatched.append(arguments)
        return "recorded"

    result = asyncio.run(
        run_tool_calling_loop(
            _PROMPT,
            model="deepseek/deepseek-chat",
            dispatch=dispatch,
            tools=_NOTE_TOOL,
            trace_path=trace_path,  # not yet a real parameter -- see module docstring
        )
    )

    assert dispatched, f"model never called record_note -- final answer: {result!r}"
    assert trace_path.exists(), "run_tool_calling_loop did not write a trace file"

    turns = parse_trace(trace_path)
    traced_tool_calls = [tc for t in turns if t.kind == "assistant" for tc in t.tool_calls]
    assert any(tc.name == "record_note" for tc in traced_tool_calls), (
        f"record_note call missing from parsed trace: {traced_tool_calls!r}"
    )

    traced_results = [t for t in turns if t.kind == "tool_result"]
    assert any(t.tool_name == "record_note" and not t.is_error for t in traced_results), (
        f"record_note result missing from parsed trace: {traced_results!r}"
    )

"""Regression coverage for `emc2p.testing.render_trace`.

Two small synthetic fixtures, one per trace shape it has to normalize
(see that module's own docstring) -- no live model calls, no real
session: both shapes are just fixed JSONL text here.
"""

import base64
import json
from pathlib import Path

from emc2p.testing.render_trace import parse_trace, render_html

# The mcp_client_session.py shape: flat type/content/tool_calls/name/is_error.
_SIMPLE_TRACE = "\n".join(
    [
        json.dumps({"type": "user", "content": "do the thing"}),
        json.dumps(
            {
                "type": "assistant",
                "content": "I'll call the tool.",
                "tool_calls": [{"name": "do_thing", "arguments": json.dumps({"x": 1})}],
            }
        ),
        json.dumps({"type": "tool_result", "name": "do_thing", "is_error": False, "content": "done"}),
        json.dumps(
            {
                "type": "assistant",
                "content": "Now the failing one.",
                "tool_calls": [{"name": "do_thing", "arguments": json.dumps({"x": 2})}],
            }
        ),
        json.dumps({"type": "tool_result", "name": "do_thing", "is_error": True, "content": "boom"}),
        json.dumps({"type": "assistant", "content": "All done.", "tool_calls": []}),
    ]
)

# The raw Anthropic Messages stream-json shape: nested message.content
# blocks, a tool_use's result arriving in a *later* user event.
_ANTHROPIC_TRACE = "\n".join(
    [
        json.dumps({"type": "system", "subtype": "init"}),
        json.dumps(
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "I'll call the tool."},
                        {"type": "tool_use", "id": "toolu_1", "name": "do_thing", "input": {"x": 1}},
                    ],
                },
            }
        ),
        json.dumps(
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "is_error": False,
                            "content": [{"type": "text", "text": "done"}],
                        }
                    ],
                },
            }
        ),
        json.dumps({"type": "result", "result": "All done.", "total_cost_usd": 0.01}),
    ]
)


def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "trace.jsonl"
    p.write_text(text)
    return p


class TestParseSimpleFormat:
    def test_turn_kinds_and_order(self, tmp_path: Path):
        turns = parse_trace(_write(tmp_path, _SIMPLE_TRACE))
        assert [t.kind for t in turns] == [
            "user",
            "assistant",
            "tool_result",
            "assistant",
            "tool_result",
            "assistant",
        ]

    def test_tool_call_name_and_arguments_captured(self, tmp_path: Path):
        turns = parse_trace(_write(tmp_path, _SIMPLE_TRACE))
        assistant_turn = turns[1]
        assert len(assistant_turn.tool_calls) == 1
        assert assistant_turn.tool_calls[0].name == "do_thing"
        assert assistant_turn.tool_calls[0].arguments == {"x": 1}

    def test_tool_result_error_flag(self, tmp_path: Path):
        turns = parse_trace(_write(tmp_path, _SIMPLE_TRACE))
        assert turns[2].is_error is False
        assert turns[4].is_error is True
        assert turns[4].tool_name == "do_thing"

    def test_actor_defaults_to_empty_when_absent(self, tmp_path: Path):
        turns = parse_trace(_write(tmp_path, _SIMPLE_TRACE))
        assert all(t.actor == "" for t in turns)


class TestActorLabel:
    """`actor`, when a writer stamps it (`run_tool_calling_loop`'s own
    `trace_actor`), identifies which call context produced a turn -- e.g.
    a keyed_subagent responder's nested exchange sharing its parent's
    trace_path -- explicitly, rather than leaving a reader to infer it
    from tool-name conventions."""

    def test_actor_captured_on_every_event_type(self, tmp_path: Path):
        trace = "\n".join(
            [
                json.dumps({"type": "assistant", "content": "on it.", "tool_calls": [], "actor": "keyed_subagent"}),
                json.dumps(
                    {
                        "type": "tool_result",
                        "name": "update_registry",
                        "is_error": False,
                        "content": "ok",
                        "actor": "keyed_subagent",
                    }
                ),
            ]
        )
        turns = parse_trace(_write(tmp_path, trace))
        assert [t.actor for t in turns] == ["keyed_subagent", "keyed_subagent"]

    def test_actor_shown_in_rendered_html(self, tmp_path: Path):
        trace = json.dumps({"type": "assistant", "content": "on it.", "tool_calls": [], "actor": "keyed_subagent"})
        turns = parse_trace(_write(tmp_path, trace))
        output = render_html(turns, title="t", source_label="s")
        assert '<span class="actor-tag">keyed_subagent</span>' in output

    def test_no_actor_tag_rendered_when_absent(self, tmp_path: Path):
        """The CSS rule always ships in `<style>`; only the `<span>` marks an
        actual actor-labeled step, so that's what must stay absent."""
        turns = parse_trace(_write(tmp_path, _SIMPLE_TRACE))
        output = render_html(turns, title="t", source_label="s")
        assert '<span class="actor-tag">' not in output


class TestOriginLabel:
    """`origin`, when a writer stamps it (`run_tool_calling_loop`'s own
    `trace_origin`), identifies *why* a turn happened -- e.g. which of a
    project's several call sites constructed the prompt a keyed_subagent
    responder is answering -- distinct from `actor` (*who* is answering).
    Never folded into the prompt text itself, so it's inspectable without
    ever reaching the model's own context."""

    def test_origin_captured_on_every_event_type(self, tmp_path: Path):
        trace = "\n".join(
            [
                json.dumps({"type": "user", "content": "decide.", "origin": "resolve_event"}),
                json.dumps(
                    {"type": "assistant", "content": "on it.", "tool_calls": [], "origin": "resolve_event"}
                ),
                json.dumps(
                    {
                        "type": "tool_result",
                        "name": "update_registry",
                        "is_error": False,
                        "content": "ok",
                        "origin": "resolve_event",
                    }
                ),
            ]
        )
        turns = parse_trace(_write(tmp_path, trace))
        assert [t.origin for t in turns] == ["resolve_event"] * 3

    def test_origin_shown_in_rendered_html(self, tmp_path: Path):
        trace = json.dumps({"type": "assistant", "content": "on it.", "tool_calls": [], "origin": "plan_events"})
        turns = parse_trace(_write(tmp_path, trace))
        output = render_html(turns, title="t", source_label="s")
        assert '<span class="origin-tag">plan_events</span>' in output

    def test_no_origin_tag_rendered_when_absent(self, tmp_path: Path):
        turns = parse_trace(_write(tmp_path, _SIMPLE_TRACE))
        output = render_html(turns, title="t", source_label="s")
        assert '<span class="origin-tag">' not in output

    def test_origin_defaults_to_empty_when_absent(self, tmp_path: Path):
        turns = parse_trace(_write(tmp_path, _SIMPLE_TRACE))
        assert all(t.origin == "" for t in turns)


class TestActorNestedUnderPendingToolCall:
    """A nested call (e.g. keyed_subagent's own run_tool_calling_loop)
    shares its parent's trace_path, so its "actor"-stamped events land
    interleaved into the same flat file, between the outer host tool
    call that triggered it (e.g. advance_simulation) and that call's own
    eventual result -- the outer dispatch can't write its result until
    the nested call returns. Left flat, a reader sees the outer call
    followed by an unrelated exchange with no result in sight, which
    reads as the result having gone missing. It should nest instead,
    the same way a native subagent's own turns already do."""

    def test_interleaved_actor_run_nests_under_the_pending_call(self, tmp_path: Path):
        trace = "\n".join(
            [
                json.dumps(
                    {"type": "assistant", "content": "", "tool_calls": [{"name": "advance_simulation", "arguments": "{}"}]}
                ),
                json.dumps({"type": "assistant", "content": "on it.", "tool_calls": [], "actor": "keyed_subagent"}),
                json.dumps(
                    {
                        "type": "tool_result",
                        "name": "update_registry",
                        "is_error": False,
                        "content": "ok",
                        "actor": "keyed_subagent",
                    }
                ),
                json.dumps({"type": "tool_result", "name": "advance_simulation", "is_error": False, "content": "done"}),
            ]
        )
        turns = parse_trace(_write(tmp_path, trace))

        # The nested run doesn't show up as its own top-level turns --
        # the outer call is followed directly by its own result.
        assert [t.kind for t in turns] == ["assistant", "tool_result"]
        assert turns[1].tool_name == "advance_simulation"

        [call] = turns[0].tool_calls
        assert call.name == "advance_simulation"
        assert [t.actor for t in call.subtrace] == ["keyed_subagent", "keyed_subagent"]

    def test_nests_under_the_specific_call_still_pending_in_a_multi_call_turn(self, tmp_path: Path):
        """Two tool calls in one turn, dispatched and resulted one at a
        time -- the actor run interleaves after the first result, while
        the second call is the one actually pending, so it must nest
        there, not on the first (already-resolved) call."""
        trace = "\n".join(
            [
                json.dumps(
                    {
                        "type": "assistant",
                        "content": "",
                        "tool_calls": [
                            {"name": "view_registry", "arguments": "{}"},
                            {"name": "advance_simulation", "arguments": "{}"},
                        ],
                    }
                ),
                json.dumps({"type": "tool_result", "name": "view_registry", "is_error": False, "content": "..."}),
                json.dumps({"type": "assistant", "content": "on it.", "tool_calls": [], "actor": "keyed_subagent"}),
                json.dumps({"type": "tool_result", "name": "advance_simulation", "is_error": False, "content": "done"}),
            ]
        )
        turns = parse_trace(_write(tmp_path, trace))

        assert [t.kind for t in turns] == ["assistant", "tool_result", "tool_result"]
        view_call, advance_call = turns[0].tool_calls
        assert view_call.subtrace is None
        assert [t.actor for t in advance_call.subtrace] == ["keyed_subagent"]

    def test_actor_run_with_no_pending_call_falls_back_to_top_level(self, tmp_path: Path):
        """No outer host call was awaiting this run (e.g. the trace is
        nothing but a nested exchange) -- surface the turns directly
        rather than silently dropping them."""
        trace = json.dumps({"type": "assistant", "content": "on it.", "tool_calls": [], "actor": "keyed_subagent"})
        turns = parse_trace(_write(tmp_path, trace))
        assert [t.actor for t in turns] == ["keyed_subagent"]

    def test_actor_run_unflushed_at_end_of_file_is_not_dropped(self, tmp_path: Path):
        """The file can end mid-nested-run (e.g. captured while the
        session was still live) -- flush whatever's buffered rather than
        losing it."""
        trace = "\n".join(
            [
                json.dumps(
                    {"type": "assistant", "content": "", "tool_calls": [{"name": "advance_simulation", "arguments": "{}"}]}
                ),
                json.dumps({"type": "assistant", "content": "still going", "tool_calls": [], "actor": "keyed_subagent"}),
            ]
        )
        turns = parse_trace(_write(tmp_path, trace))
        [call] = turns[0].tool_calls
        assert [t.actor for t in call.subtrace] == ["keyed_subagent"]

    def test_rendered_summary_names_the_actor(self, tmp_path: Path):
        trace = "\n".join(
            [
                json.dumps(
                    {"type": "assistant", "content": "", "tool_calls": [{"name": "advance_simulation", "arguments": "{}"}]}
                ),
                json.dumps({"type": "assistant", "content": "on it.", "tool_calls": [], "actor": "keyed_subagent"}),
                json.dumps({"type": "tool_result", "name": "advance_simulation", "is_error": False, "content": "done"}),
            ]
        )
        turns = parse_trace(_write(tmp_path, trace))
        output = render_html(turns, title="t", source_label="s")
        assert "<summary>keyed_subagent trace</summary>" in output

    def test_rendered_summary_names_the_actor_and_origin(self, tmp_path: Path):
        trace = "\n".join(
            [
                json.dumps(
                    {"type": "assistant", "content": "", "tool_calls": [{"name": "advance_simulation", "arguments": "{}"}]}
                ),
                json.dumps(
                    {
                        "type": "user",
                        "content": "decide.",
                        "actor": "keyed_subagent",
                        "origin": "resolve_event",
                    }
                ),
                json.dumps(
                    {
                        "type": "assistant",
                        "content": "on it.",
                        "tool_calls": [],
                        "actor": "keyed_subagent",
                        "origin": "resolve_event",
                    }
                ),
                json.dumps({"type": "tool_result", "name": "advance_simulation", "is_error": False, "content": "done"}),
            ]
        )
        turns = parse_trace(_write(tmp_path, trace))
        output = render_html(turns, title="t", source_label="s")
        assert "<summary>keyed_subagent (resolve_event) trace</summary>" in output


class TestParseAnthropicFormat:
    def test_system_event_skipped_and_turn_kinds_correct(self, tmp_path: Path):
        turns = parse_trace(_write(tmp_path, _ANTHROPIC_TRACE))
        assert [t.kind for t in turns] == ["assistant", "tool_result", "final"]

    def test_tool_use_and_its_later_result_are_correlated_by_id(self, tmp_path: Path):
        turns = parse_trace(_write(tmp_path, _ANTHROPIC_TRACE))
        assistant_turn, result_turn, _final = turns
        assert assistant_turn.tool_calls[0].name == "do_thing"
        assert assistant_turn.tool_calls[0].arguments == {"x": 1}
        # The result event only carries tool_use_id, not the tool's own
        # name -- parse_trace must look it up from the earlier tool_use block.
        assert result_turn.tool_name == "do_thing"
        assert result_turn.text == "done"


class TestParseMixedFormat:
    """A nested in-process call (e.g. a keyed_subagent-style responder's
    own run_tool_calling_loop) writes the simple format via its own
    trace_path even when the surrounding session is real Anthropic
    stream-json (single_shared_trace_file) -- so one file can genuinely
    mix both shapes. Regression for the bug this exposed: parse_trace
    used to pick ONE format for the whole file (whichever the first
    matching event implied), silently dropping every event in the other
    shape.
    """

    def test_simple_format_event_survives_inside_a_stream_json_file(self, tmp_path: Path):
        mixed = "\n".join(
            [
                _ANTHROPIC_TRACE.splitlines()[1],  # the stream-json assistant/tool_use event
                json.dumps({"type": "assistant", "content": "nested call", "tool_calls": []}),
                json.dumps({"type": "tool_result", "name": "record_note", "is_error": False, "content": "ok"}),
                _ANTHROPIC_TRACE.splitlines()[2],  # the stream-json tool_result event
            ]
        )
        turns = parse_trace(_write(tmp_path, mixed))
        assert [t.kind for t in turns] == ["assistant", "assistant", "tool_result", "tool_result"]
        assert turns[1].text == "nested call"
        assert turns[2].tool_name == "record_note"
        # The stream-json tool_result must still resolve correctly too --
        # confirms the mix doesn't corrupt the anthropic side's own
        # tool_use_id -> name correlation.
        assert turns[3].tool_name == "do_thing"


class TestBase64PayloadDecoding:
    def test_a_base64_json_argument_is_decoded_and_surfaced(self, tmp_path: Path):
        token = base64.urlsafe_b64encode(json.dumps({"alias": "x", "answers": ["did the thing"]}).encode()).decode()
        trace = json.dumps(
            {
                "type": "assistant",
                "content": "resuming",
                "tool_calls": [{"name": "advance", "arguments": json.dumps({"resume_token": token})}],
            }
        )
        turns = parse_trace(_write(tmp_path, trace))
        [call] = turns[0].tool_calls
        assert len(call.decoded) == 1
        label, decoded = call.decoded[0]
        assert label == "resume_token"
        assert decoded == {"alias": "x", "answers": ["did the thing"]}

    def test_an_ordinary_short_string_argument_is_not_mistaken_for_a_payload(self, tmp_path: Path):
        trace = json.dumps(
            {
                "type": "assistant",
                "content": "x",
                "tool_calls": [{"name": "do_thing", "arguments": json.dumps({"save_dir": "/tmp/x"})}],
            }
        )
        turns = parse_trace(_write(tmp_path, trace))
        assert turns[0].tool_calls[0].decoded == []


class TestNativeSubagentSubtrace:
    """render_trace_follows_task_notification: a task_notification event
    naming an output_file gets that file's own isSidechain-tagged turns
    attached onto the matching ToolCall's subtrace -- confirmed real by
    tests/test_native_subagent_sidechain.py (story-simulator); this is the
    parsing/rendering side of that finding.
    """

    def _write_output_file(self, tmp_path: Path) -> Path:
        # A subagent's own output_file: the same anthropic stream-json
        # shape, each line isSidechain-tagged (not itself load-bearing for
        # parsing -- only tool_use_id/output_file on the *parent's* own
        # task_notification event drive the attachment).
        output_path = tmp_path / "subagent_output.jsonl"
        output_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "type": "user",
                            "isSidechain": True,
                            "agentId": "agent_1",
                            "message": {"role": "user", "content": "Compute 17 * 23."},
                        }
                    ),
                    json.dumps(
                        {
                            "type": "assistant",
                            "isSidechain": True,
                            "agentId": "agent_1",
                            "message": {
                                "role": "assistant",
                                "content": [{"type": "text", "text": "391"}],
                            },
                        }
                    ),
                ]
            )
        )
        return output_path

    def _parent_trace(self, output_path: Path) -> str:
        return "\n".join(
            [
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": "I'll delegate this."},
                                {
                                    "type": "tool_use",
                                    "id": "toolu_agent_1",
                                    "name": "Agent",
                                    "input": {"prompt": "Compute 17 * 23."},
                                },
                            ],
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "system",
                        "subtype": "task_notification",
                        "tool_use_id": "toolu_agent_1",
                        "output_file": str(output_path),
                        "status": "completed",
                        "summary": "391",
                    }
                ),
                json.dumps({"type": "result", "result": "The answer is 391."}),
            ]
        )

    def test_subtrace_attached_to_the_matching_tool_call(self, tmp_path: Path):
        output_path = self._write_output_file(tmp_path)
        turns = parse_trace(_write(tmp_path, self._parent_trace(output_path)))

        assistant_turn = turns[0]
        [agent_call] = assistant_turn.tool_calls
        assert agent_call.name == "Agent"
        assert agent_call.subtrace is not None
        assert [t.kind for t in agent_call.subtrace] == ["user", "assistant"]
        assert agent_call.subtrace[1].text == "391"

    def test_ordinary_tool_call_has_no_subtrace(self, tmp_path: Path):
        trace = json.dumps(
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "toolu_1", "name": "view_entity", "input": {}}
                    ],
                },
            }
        )
        turns = parse_trace(_write(tmp_path, trace))
        assert turns[0].tool_calls[0].subtrace is None

    def test_missing_output_file_is_ignored_not_fatal(self, tmp_path: Path):
        missing = tmp_path / "does_not_exist.jsonl"
        turns = parse_trace(_write(tmp_path, self._parent_trace(missing)))
        assert turns[0].tool_calls[0].subtrace is None

    def test_notification_for_an_unknown_tool_use_id_is_ignored(self, tmp_path: Path):
        output_path = self._write_output_file(tmp_path)
        trace = "\n".join(
            [
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {"role": "assistant", "content": [{"type": "text", "text": "no tool use here"}]},
                    }
                ),
                json.dumps(
                    {
                        "type": "system",
                        "subtype": "task_notification",
                        "tool_use_id": "toolu_never_seen",
                        "output_file": str(output_path),
                    }
                ),
                json.dumps({"type": "result", "result": "done"}),
            ]
        )
        # Must not raise even though no ToolCall matches this notification.
        turns = parse_trace(_write(tmp_path, trace))
        assert [t.kind for t in turns] == ["assistant", "final"]
        assert turns[0].tool_calls == []

    def test_rendered_html_includes_a_nested_subagent_trace_block(self, tmp_path: Path):
        output_path = self._write_output_file(tmp_path)
        turns = parse_trace(_write(tmp_path, self._parent_trace(output_path)))
        output = render_html(turns, title="t", source_label="s")
        assert "subagent trace" in output
        assert "391" in output

    def test_nested_subtrace_block_is_open_by_default(self, tmp_path: Path):
        """Collapsed by default, a nested exchange reads as if nothing
        happened between the call and its result -- it should be visible
        without a click, unlike the "arguments"/"decoded payload" blobs."""
        output_path = self._write_output_file(tmp_path)
        turns = parse_trace(_write(tmp_path, self._parent_trace(output_path)))
        output = render_html(turns, title="t", source_label="s")
        assert '<details class="blob" open><summary>subagent trace</summary>' in output

    def test_nested_subtrace_gets_a_taller_scrolling_window(self, tmp_path: Path):
        """A nested exchange can run to hundreds of turns -- it needs a
        much taller scroll window than the compact arguments/decoded-
        payload blobs it sits alongside, not the same default."""
        output_path = self._write_output_file(tmp_path)
        turns = parse_trace(_write(tmp_path, self._parent_trace(output_path)))
        output = render_html(turns, title="t", source_label="s")
        assert '<div class="blob-content subtrace-content">' in output


class TestRenderHtml:
    def test_renders_without_error_and_includes_key_content(self, tmp_path: Path):
        turns = parse_trace(_write(tmp_path, _SIMPLE_TRACE))
        output = render_html(turns, title="My Trace", source_label="trace.jsonl")
        assert "<title>My Trace</title>" in output
        assert "do_thing" in output
        assert "boom" in output
        # 6 turns rendered as 6 timeline steps.
        assert output.count('<li class="step') == 6

    def test_error_count_shown_in_the_summary_pills(self, tmp_path: Path):
        turns = parse_trace(_write(tmp_path, _SIMPLE_TRACE))
        output = render_html(turns, title="t", source_label="s")
        assert "1 error" in output

    def test_html_special_characters_in_content_are_escaped(self, tmp_path: Path):
        trace = json.dumps({"type": "user", "content": "<script>alert(1)</script> & stuff"})
        turns = parse_trace(_write(tmp_path, trace))
        output = render_html(turns, title="t", source_label="s")
        assert "<script>alert(1)</script>" not in output
        assert "&lt;script&gt;" in output

    def test_driver_shown_in_header_when_given(self, tmp_path: Path):
        """Not recoverable from the trace file's own content (an
        mcp_client outer session and a claude/copilot one write
        distinguishable shapes, but claude vs. copilot don't), so the
        caller supplies it and it's just displayed, not inferred."""
        turns = parse_trace(_write(tmp_path, _SIMPLE_TRACE))
        output = render_html(turns, title="t", source_label="s", driver="mcp_client")
        assert "driver: mcp_client" in output

    def test_no_driver_pill_when_omitted(self, tmp_path: Path):
        turns = parse_trace(_write(tmp_path, _SIMPLE_TRACE))
        output = render_html(turns, title="t", source_label="s")
        assert "driver:" not in output

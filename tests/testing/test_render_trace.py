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

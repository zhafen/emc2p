"""Tests for emc2p.agents.keyed_subagent's tool-calling loop mechanics --
no real model calls: litellm.acompletion is monkeypatched with a queue of
fake responses, since these are mechanical dispatch/iteration checks, not
a test of any model's actual judgment quality.
"""

import asyncio
import json
from types import SimpleNamespace

import pytest

pytest.importorskip("litellm", reason="requires the 'agents' extra")

from emc2p.agents.keyed_subagent import run_tool_calling_loop  # noqa: E402


class _FakeToolCall:
    def __init__(self, id: str, name: str, arguments: dict):
        self.id = id
        self.function = SimpleNamespace(name=name, arguments=json.dumps(arguments))


class _FakeMessage:
    def __init__(self, content: str | None = None, tool_calls: list | None = None):
        self.content = content
        self.tool_calls = tool_calls

    def model_dump(self, exclude_none: bool = True) -> dict:
        d = {"role": "assistant", "content": self.content, "tool_calls": self.tool_calls}
        return {k: v for k, v in d.items() if v is not None} if exclude_none else d


class _FakeResponse:
    def __init__(self, message: _FakeMessage, model: str = "test-model", usage=None):
        self.choices = [SimpleNamespace(message=message)]
        self.model = model
        self.usage = usage


def _queue_responses(monkeypatch, responses: list[_FakeResponse]):
    calls = []

    async def fake_acompletion(*, model, messages, tools):
        calls.append({"model": model, "messages": messages, "tools": tools})
        return responses[len(calls) - 1]

    monkeypatch.setattr("emc2p.agents.keyed_subagent.litellm.acompletion", fake_acompletion)
    return calls


def test_returns_text_immediately_when_no_tool_calls(monkeypatch):
    _queue_responses(monkeypatch, [_FakeResponse(_FakeMessage(content="the car parked."))])

    def dispatch(name, arguments):
        raise AssertionError("dispatch should never be called")

    result = asyncio.run(
        run_tool_calling_loop("what happened?", model="test-model", dispatch=dispatch)
    )
    assert result == "the car parked."


def test_dispatches_tool_call_then_returns_final_answer(monkeypatch):
    tool_call = _FakeToolCall("call_1", "view_entity", {"entity_id": "car_a"})
    calls = _queue_responses(
        monkeypatch,
        [
            _FakeResponse(_FakeMessage(tool_calls=[tool_call])),
            _FakeResponse(_FakeMessage(content="parked, per view_entity.")),
        ],
    )

    dispatched = []

    def dispatch(name, arguments):
        dispatched.append((name, arguments))
        return "car_a: position=driveway"

    result = asyncio.run(
        run_tool_calling_loop("what happened?", model="test-model", dispatch=dispatch)
    )

    assert result == "parked, per view_entity."
    assert dispatched == [("view_entity", {"entity_id": "car_a"})]
    assert len(calls) == 2
    # The tool result must be threaded back into the second call's messages.
    tool_messages = [m for m in calls[1]["messages"] if m.get("role") == "tool"]
    assert tool_messages == [
        {"role": "tool", "tool_call_id": "call_1", "content": "car_a: position=driveway"}
    ]


def test_dispatch_error_is_fed_back_as_tool_result_not_raised(monkeypatch):
    tool_call = _FakeToolCall("call_1", "update_registry", {"yaml_string": "bad"})
    calls = _queue_responses(
        monkeypatch,
        [
            _FakeResponse(_FakeMessage(tool_calls=[tool_call])),
            _FakeResponse(_FakeMessage(content="retried and finished.")),
        ],
    )

    def dispatch(name, arguments):
        raise ValueError("unknown component type")

    result = asyncio.run(
        run_tool_calling_loop("what happened?", model="test-model", dispatch=dispatch)
    )

    assert result == "retried and finished."
    tool_messages = [m for m in calls[1]["messages"] if m.get("role") == "tool"]
    assert tool_messages[0]["content"] == "Error: unknown component type"


def test_max_iterations_returns_last_content_instead_of_looping_forever(monkeypatch):
    never_converges = [
        _FakeResponse(_FakeMessage(tool_calls=[_FakeToolCall(f"call_{i}", "view_registry", {})]))
        for i in range(5)
    ]
    _queue_responses(monkeypatch, never_converges)

    result = asyncio.run(
        run_tool_calling_loop(
            "what happened?",
            model="test-model",
            dispatch=lambda name, arguments: "ok",
            max_iterations=2,
        )
    )
    # Never converged to a text-only message -- falls back to the last
    # response's (empty) content rather than raising or looping past the cap.
    assert result == ""


def test_usage_logged_when_path_given(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "emc2p.agents.keyed_subagent.litellm.completion_cost", lambda **kwargs: 0.0042
    )
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    _queue_responses(monkeypatch, [_FakeResponse(_FakeMessage(content="done."), usage=usage)])

    log_path = tmp_path / "usage.jsonl"
    result = asyncio.run(
        run_tool_calling_loop(
            "what happened?",
            model="test-model",
            dispatch=lambda name, arguments: "unused",
            usage_log_path=log_path,
        )
    )

    assert result == "done."
    lines = log_path.read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["model"] == "test-model"
    assert record["prompt_tokens"] == 10
    assert record["completion_tokens"] == 5
    assert record["total_tokens"] == 15
    assert record["cost_usd"] == 0.0042


def test_no_usage_log_written_when_path_omitted(monkeypatch, tmp_path):
    _queue_responses(monkeypatch, [_FakeResponse(_FakeMessage(content="done."))])
    asyncio.run(
        run_tool_calling_loop("what happened?", model="test-model", dispatch=lambda n, a: "x")
    )
    assert list(tmp_path.iterdir()) == []

"""Generic tool-calling loop for giving an LLM real, in-process access to a
Registrar -- not just a bare text completion.

A caller needing a judgment prompt answered by a model that can actually
read/write the registry itself (`view_registry`/`view_entity`/
`update_registry`/`consolidate_review`), not just narrate a decision for
something else to apply later, drives that model through this loop
instead of a single completion -- a tool-less completion structurally
cannot persist anything on its own.

Nothing here knows about any downstream project's registry schema,
session/save-dir plumbing, or prompt framing -- `REGISTRAR_TOOL_SPECS`
names only the four Registrar-shaped operations, and `run_tool_calling_loop`
takes `model` and `dispatch` as plain parameters (matching
`emc2p.testing.headless_session.HeadlessSession`'s own
caller-supplies-the-specifics style). A caller wanting `update_registry`
calls to also do something extra (e.g. re-exporting to a save directory,
or writing under a specific source key) supplies its own `dispatch`
wired to its own functions instead of using the default wiring.

Requires the `agents` extra (`litellm`), not a core emc2p dependency --
most emc2p-based projects have no need for this module at all.
"""

from __future__ import annotations

import inspect
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import litellm

# Independent of any caller's own pacing (e.g. a wall-clock turn budget):
# a safety net against a tool-call loop that never converges.
DEFAULT_MAX_ITERATIONS = 15

# Generic Registrar-shaped tool specs -- a caller is free to pass its own
# `tools`/`dispatch` instead; these are just the common default.
REGISTRAR_TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "view_registry",
            "description": (
                "View all recorded data for one component type (e.g. "
                '"position", "description", "state").'
            ),
            "parameters": {
                "type": "object",
                "properties": {"component_type": {"type": "string"}},
                "required": ["component_type"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "view_entity",
            "description": "Return every recorded component instance for a specific entity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_id": {
                        "type": "string",
                        "description": "Entity hash, alias, or unambiguous path fragment.",
                    }
                },
                "required": ["entity_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_registry",
            "description": (
                "Merge entity-first YAML into the registry. Each component needs a "
                "leading `- ` list item to attach to an existing aliased entity, e.g. "
                '"widget_1:\\n    - color:\\n        value: blue". A bare mapping (no '
                "leading `- `) is silently accepted but records nothing at all."
            ),
            "parameters": {
                "type": "object",
                "properties": {"yaml_string": {"type": "string"}},
                "required": ["yaml_string"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "consolidate_review",
            "description": (
                "Show every component type recorded so far, to review and "
                "consolidate before finishing."
            ),
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
]

# A dispatch may return its result directly or as an awaitable -- the loop
# below awaits it either way, so an async tool needs no sync wrapper.
Dispatch = Callable[[str, dict[str, Any]], "str | Any"]


async def run_tool_calling_loop(
    prompt: str,
    *,
    model: str,
    dispatch: Dispatch,
    tools: list[dict[str, Any]] = REGISTRAR_TOOL_SPECS,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    usage_log_path: Path | None = None,
    on_response: Callable[[Any], None] | None = None,
    trace_path: Path | None = None,
    trace_actor: str | None = None,
    trace_origin: str | None = None,
) -> str:
    """Answer `prompt` via `model`, letting it call `tools` (dispatched
    through `dispatch`) as many times as needed before a final text answer.

    `dispatch` is the only thing here that touches an actual registry --
    this function itself just drives the completion/tool-result exchange.
    A tool call that raises is fed back to the model as an error string
    rather than aborting the loop, so the model can retry with corrected
    arguments instead of the whole judgment call failing outright.

    Runs at most `max_iterations` rounds; past that, returns whatever text
    (possibly empty) came back from the last call rather than looping
    forever -- a caller wanting a tighter bound (e.g. a wall-clock budget)
    enforces it around this call, not inside it.

    `on_response`, if given, is called with each raw litellm response
    object right after it comes back -- for a caller that needs live
    usage/cost per call (e.g. accumulating a running total across several
    of these calls) rather than only the best-effort log `usage_log_path`
    writes to disk.

    `trace_path`, if given, appends one JSON line per assistant response
    and per tool_result -- render_trace's own "simple format" (the same
    shape `mcp_client_session.py` hand-writes for its top-level turn loop)
    -- so this call's own tool calls/arguments/results are inspectable
    afterward via `emc2p.testing.render_trace.parse_trace`, the same way a
    top-level session's trace already is. Opened in append mode, so a
    caller wanting one combined trace across several calls (e.g. a nested
    subagent call sharing its parent's own trace_path) gets that for free
    by passing the same path.

    `trace_actor`, if given, is stamped onto every event this call writes
    (an `"actor"` key alongside `"type"`) so a shared trace_path's reader
    can tell which call context produced a given step without falling
    back to guessing from tool-name conventions (e.g. an MCP dispatch
    prefix versus a bare name) -- see `render_trace`'s own Turn.actor.
    None (the default) omits the key entirely, matching a top-level
    session's own unlabeled events.

    `prompt` itself is written to `trace_path` too, as a leading "user"
    event -- matching a top-level session's own convention
    (`mcp_client_session.py` writes its turn's prompt the same way) --
    since it's what actually framed everything this call went on to do,
    not just incidental setup. Without it, a reader has no way to see
    what this call was actually asked to do short of separately hunting
    it down (e.g. story-simulator's own diagnostic
    `.write_call_trace.jsonl`, which each caller may or may not have).

    `trace_origin`, if given, is stamped onto every event this call
    writes too (an `"origin"` key, alongside `"actor"`) -- a caller-
    supplied label for *why* this call happened (e.g. which of a
    project's several call sites constructed `prompt`), distinct from
    `trace_actor`'s *who is answering* -- see `render_trace`'s own
    Turn.origin. Never folded into `prompt` itself: a caller wanting
    this recorded without it ever reaching the model's own context
    passes it here instead of concatenating it into the prompt text.
    """
    if trace_path is not None:
        event: dict[str, Any] = {"type": "user", "content": prompt}
        if trace_actor is not None:
            event["actor"] = trace_actor
        if trace_origin is not None:
            event["origin"] = trace_origin
        _append_trace_event(trace_path, event)
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    response = await litellm.acompletion(model=model, messages=messages, tools=tools)
    _log_usage(response, usage_log_path)
    if trace_path is not None:
        _append_trace_event(
            trace_path, _assistant_trace_event(response.choices[0].message, trace_actor, trace_origin)
        )
    if on_response is not None:
        on_response(response)

    for _ in range(max_iterations):
        message = response.choices[0].message
        messages.append(message.model_dump(exclude_none=True))
        tool_calls = message.tool_calls
        # Not a `break` -- this `return` ends the whole function right
        # here: the model asked for nothing further, so it's done.
        if not tool_calls:
            return message.content or ""
        for tool_call in tool_calls:
            is_error = False
            try:
                arguments = json.loads(tool_call.function.arguments or "{}")
                result = dispatch(tool_call.function.name, arguments)
                if inspect.isawaitable(result):
                    result = await result
            except Exception as exc:  # noqa: BLE001 -- feed the error back, don't crash the loop
                result = f"Error: {exc}"
                is_error = True
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": str(result)})
            if trace_path is not None:
                event = {
                    "type": "tool_result",
                    "name": tool_call.function.name,
                    "content": str(result),
                    "is_error": is_error,
                }
                if trace_actor is not None:
                    event["actor"] = trace_actor
                if trace_origin is not None:
                    event["origin"] = trace_origin
                _append_trace_event(trace_path, event)
        response = await litellm.acompletion(model=model, messages=messages, tools=tools)
        _log_usage(response, usage_log_path)
        if trace_path is not None:
            _append_trace_event(
                trace_path, _assistant_trace_event(response.choices[0].message, trace_actor, trace_origin)
            )
        if on_response is not None:
            on_response(response)

    # Reached only by falling out of the `for` loop above (no `break`
    # anywhere) once `max_iterations` is exhausted without the model stopping.
    return response.choices[0].message.content or ""


def _assistant_trace_event(
    message: Any, actor: str | None = None, origin: str | None = None
) -> dict[str, Any]:
    """render_trace's "simple format" shape for one assistant response.

    Arguments are parsed back into a dict where possible (matching what
    `render_trace._normalize_simple_format` already does on read) rather
    than left as the raw JSON string litellm hands back -- a malformed
    arguments string (the model's own mistake, not this function's to
    fix) is kept as-is instead of raising, since a trace write should
    never be what turns a recoverable model error into a hard failure.
    """
    tool_calls = []
    for tc in message.tool_calls or []:
        try:
            arguments = json.loads(tc.function.arguments or "{}")
        except json.JSONDecodeError:
            arguments = tc.function.arguments
        tool_calls.append({"name": tc.function.name, "arguments": arguments})
    event: dict[str, Any] = {"type": "assistant", "content": message.content or "", "tool_calls": tool_calls}
    if actor is not None:
        event["actor"] = actor
    if origin is not None:
        event["origin"] = origin
    return event


def _append_trace_event(trace_path: Path, event: dict[str, Any]) -> None:
    with trace_path.open("a") as f:
        f.write(json.dumps(event) + "\n")


def _log_usage(response: Any, usage_log_path: Path | None) -> None:
    """Best-effort append of this call's cost/tokens to `usage_log_path`,
    one JSON line per call. A no-op when `usage_log_path` is None.

    Never lets a logging failure take down the actual judgment call --
    this is auxiliary instrumentation, not something a caller's
    correctness should ever hinge on.
    """
    if usage_log_path is None:
        return
    try:
        usage = getattr(response, "usage", None)
        try:
            cost = litellm.completion_cost(completion_response=response)
        except Exception:
            cost = None
        record = {
            "timestamp": time.time(),
            "model": response.model,
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
            "cost_usd": cost,
        }
        with usage_log_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as exc:  # noqa: BLE001 -- deliberately broad, see docstring
        print(f"[emc2p.agents.tool_calling_loop] could not log usage: {exc}", file=sys.stderr)

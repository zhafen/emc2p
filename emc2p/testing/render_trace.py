"""Render a live-test session trace (the `.jsonl` a `trace_dir`-configured
session writes) into a readable, self-contained HTML report: one entry
per turn, tool calls paired with their results, errors flagged.

Normalizes the two trace shapes session drivers write into one common
`Turn` sequence before rendering:

- `mcp_client_session.py`'s flat format (`driver="mcp_client"`).
- The raw Anthropic Messages `stream-json` protocol `headless_session.py`
  passes through verbatim (`driver="claude"`/`"copilot"`) -- a `tool_use`
  block's result arrives in a *later* event, correlated by `tool_use_id`.

Any base64-encoded-JSON tool-call argument (e.g. story-simulator's own
`resume_token`) is decoded and shown alongside the raw call -- generic,
not keyed to any one project's token format.

CLI: `uv run python -m emc2p.testing.render_trace trace.jsonl -o report.html`
Library: `parse_trace()`/`render_html()`.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import dataclasses
import html
import json
import sys
from pathlib import Path
from typing import Any


@dataclasses.dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any] | str
    # (label, decoded value) pairs -- any argument value that turned out
    # to be base64-encoded JSON, decoded for display alongside the raw call.
    decoded: list[tuple[str, Any]] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class Turn:
    kind: str  # "user" | "assistant" | "tool_result" | "system" | "final"
    text: str = ""
    tool_calls: list[ToolCall] = dataclasses.field(default_factory=list)
    tool_name: str = ""
    is_error: bool = False


# ─── Parsing ──────────────────────────────────────────────────────────


def parse_trace(path: Path) -> list[Turn]:
    """Read a trace file and return its turns, oldest first.

    Detects which of the two shapes (see module docstring) the file
    uses; an unparseable line is skipped, not fatal.
    """
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    events: list[dict[str, Any]] = []
    for line in lines:
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if any("message" in e and isinstance(e.get("message"), dict) for e in events):
        return _normalize_anthropic_stream(events)
    return _normalize_simple_format(events)


def _normalize_simple_format(events: list[dict[str, Any]]) -> list[Turn]:
    """`mcp_client_session.py`'s own flat format."""
    turns: list[Turn] = []
    for event in events:
        kind = event.get("type")
        if kind == "user":
            turns.append(Turn(kind="user", text=event.get("content") or ""))
        elif kind == "assistant":
            tool_calls = []
            for tc in event.get("tool_calls") or []:
                args = _try_parse_json(tc.get("arguments", "")) or tc.get("arguments", "")
                tool_calls.append(ToolCall(name=tc.get("name", ""), arguments=args, decoded=_decode_blobs(args)))
            turns.append(Turn(kind="assistant", text=event.get("content") or "", tool_calls=tool_calls))
        elif kind == "tool_result":
            turns.append(
                Turn(
                    kind="tool_result",
                    text=str(event.get("content", "")),
                    tool_name=event.get("name", ""),
                    is_error=bool(event.get("is_error")),
                )
            )
    return turns


def _normalize_anthropic_stream(events: list[dict[str, Any]]) -> list[Turn]:
    """The raw Anthropic Messages `stream-json` protocol.

    `tool_names` maps `tool_use` id to name, since the later
    `tool_result` block only carries the id.
    """
    turns: list[Turn] = []
    tool_names: dict[str, str] = {}
    for event in events:
        etype = event.get("type")
        if etype == "result":
            turns.append(Turn(kind="final", text=str(event.get("result", ""))))
            continue
        message = event.get("message")
        if etype == "system" or message is None:
            continue
        content = message.get("content")
        blocks = content if isinstance(content, list) else [{"type": "text", "text": content}]
        if etype == "assistant":
            text_parts = [b.get("text", "") for b in blocks if b.get("type") == "text"]
            tool_calls = []
            for b in blocks:
                if b.get("type") != "tool_use":
                    continue
                tool_names[b.get("id", "")] = b.get("name", "")
                args = b.get("input") or {}
                tool_calls.append(ToolCall(name=b.get("name", ""), arguments=args, decoded=_decode_blobs(args)))
            turns.append(Turn(kind="assistant", text="\n".join(p for p in text_parts if p), tool_calls=tool_calls))
        elif etype == "user":
            for b in blocks:
                if b.get("type") == "tool_result":
                    result_content = b.get("content")
                    if isinstance(result_content, list):
                        result_text = "\n".join(
                            part.get("text", "") for part in result_content if part.get("type") == "text"
                        )
                    else:
                        result_text = str(result_content or "")
                    turns.append(
                        Turn(
                            kind="tool_result",
                            text=result_text,
                            tool_name=tool_names.get(b.get("tool_use_id", ""), ""),
                            is_error=bool(b.get("is_error")),
                        )
                    )
                elif b.get("type") == "text" and b.get("text"):
                    turns.append(Turn(kind="user", text=b["text"]))
    return turns


def _try_parse_json(value: Any) -> Any:
    if not isinstance(value, str):
        return None
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None


def _decode_blobs(value: Any, path: str = "") -> list[tuple[str, Any]]:
    """Recursively find every string leaf in `value` that's base64-encoded
    JSON, and return (dotted-path, decoded-value) for each.
    """
    found: list[tuple[str, Any]] = []
    if isinstance(value, dict):
        for k, v in value.items():
            found.extend(_decode_blobs(v, f"{path}.{k}" if path else k))
    elif isinstance(value, list):
        for i, v in enumerate(value):
            found.extend(_decode_blobs(v, f"{path}[{i}]"))
    elif isinstance(value, str) and len(value) >= 24:
        try:
            decoded_bytes = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
            decoded = json.loads(decoded_bytes.decode("utf-8"))
        except (binascii.Error, ValueError, UnicodeDecodeError):
            return found
        if isinstance(decoded, (dict, list)):
            found.append((path, decoded))
    return found


# ─── Rendering ────────────────────────────────────────────────────────

_STEP_LABELS = {
    "user": "user",
    "assistant": "assistant",
    "tool_result": "tool result",
    "final": "final result",
}


def _esc(text: Any) -> str:
    return html.escape(str(text), quote=False)


def _render_decoded_value(value: Any) -> str:
    """Paragraphs for a list of strings (the common "prior free-text
    answers" shape); indented JSON otherwise."""
    if isinstance(value, list) and value and all(isinstance(v, str) for v in value):
        return "".join(f"<p>{_esc(v)}</p>" for v in value)
    return f"<pre>{_esc(json.dumps(value, indent=2))}</pre>"


def _render_tool_call(tc: ToolCall) -> str:
    args_text = tc.arguments if isinstance(tc.arguments, str) else json.dumps(tc.arguments, indent=2)
    parts = [
        '<div class="tool-call">',
        f'<div class="tool-call-name">→ <span class="tool-name">{_esc(tc.name)}</span></div>',
        '<details class="blob"><summary>arguments</summary>',
        f'<div class="blob-content"><pre>{_esc(args_text)}</pre></div></details>',
    ]
    for label, decoded in tc.decoded:
        where = f" ({_esc(label)})" if label else ""
        parts.append(
            f'<details class="blob"><summary>decoded payload{where}</summary>'
            f'<div class="blob-content">{_render_decoded_value(decoded)}</div></details>'
        )
    parts.append("</div>")
    return "".join(parts)


def render_html(turns: list[Turn], *, title: str, source_label: str) -> str:
    n_errors = sum(1 for t in turns if t.kind == "tool_result" and t.is_error)
    n_tool_calls = sum(len(t.tool_calls) for t in turns)
    step_html: list[str] = []
    for i, turn in enumerate(turns, start=1):
        css_kind = "err" if turn.is_error else turn.kind
        kind_label = _STEP_LABELS.get(turn.kind, turn.kind)
        body = [f'<div class="step-kind">{_esc(kind_label)}'
                + (f' · <span class="tool-name">{_esc(turn.tool_name)}</span>' if turn.tool_name else "")
                + "</div>"]
        if turn.text:
            body.append(f'<div class="step-text">{_esc(turn.text)}</div>')
        for tc in turn.tool_calls:
            body.append(_render_tool_call(tc))
        step_html.append(
            f'<li class="step {css_kind}"><div class="step-num">{i}</div>'
            f'<div class="step-body">{"".join(body)}</div></li>'
        )

    return f"""<title>{_esc(title)}</title>
<style>
{_CSS}
</style>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Manrope:wght@500;600;700;800&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<div class="page">
  <header class="report-head">
    <div class="eyebrow">rendered live-test trace</div>
    <h1>{_esc(title)}</h1>
    <div class="meta-row">
      <span class="pill mono-pill">{_esc(source_label)}</span>
      <span class="pill">{len(turns)} entries</span>
      <span class="pill">{n_tool_calls} tool calls</span>
      {f'<span class="pill fail">{n_errors} error{"s" if n_errors != 1 else ""}</span>' if n_errors else ''}
    </div>
  </header>
  <section>
    <ol class="timeline">
      {"".join(step_html)}
    </ol>
  </section>
</div>
"""


_CSS = """
:root {
  --bg: #f5f6f9; --surface: #ffffff; --surface-2: #edeff4; --border: #dde1ea;
  --text: #1b212c; --text-muted: #5b6577; --accent: #3552d1;
  --ok: #1f9d63; --ok-soft: #e6f7ee; --err: #b8342f; --err-soft: #fdeceb;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #12151c; --surface: #1a1f29; --surface-2: #212734; --border: #2c3444;
    --text: #e6e9f0; --text-muted: #98a1b3; --accent: #8fa0ff;
    --ok: #4ade9a; --ok-soft: #17281f; --err: #ff8078; --err-soft: #331d1e;
  }
}
:root[data-theme="dark"] {
  --bg: #12151c; --surface: #1a1f29; --surface-2: #212734; --border: #2c3444;
  --text: #e6e9f0; --text-muted: #98a1b3; --accent: #8fa0ff;
  --ok: #4ade9a; --ok-soft: #17281f; --err: #ff8078; --err-soft: #331d1e;
}
* { box-sizing: border-box; }
body { background: var(--bg); color: var(--text); font-family: 'Manrope', ui-sans-serif, system-ui, sans-serif; line-height: 1.5; }
.page { max-width: 860px; margin: 0 auto; padding: 48px 24px 96px; }
header.report-head { display: flex; flex-direction: column; gap: 10px; margin-bottom: 32px; padding-bottom: 28px; border-bottom: 1px solid var(--border); }
.eyebrow { font-size: 12.5px; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; color: var(--text-muted); }
h1 { font-size: 26px; font-weight: 800; letter-spacing: -0.01em; margin: 0; text-wrap: balance; word-break: break-word; }
.meta-row { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 6px; }
.pill { display: inline-flex; align-items: center; gap: 6px; border-radius: 999px; padding: 5px 12px; font-size: 13px; font-weight: 600; border: 1px solid var(--border); background: var(--surface); color: var(--text-muted); }
.pill.fail { background: var(--err-soft); border-color: transparent; color: var(--err); }
.pill.mono-pill { font-family: 'IBM Plex Mono', ui-monospace, monospace; font-weight: 500; word-break: break-all; }
.timeline { list-style: none; margin: 0; padding: 0; position: relative; }
.timeline::before { content: ""; position: absolute; left: 17px; top: 8px; bottom: 8px; width: 1px; background: var(--border); }
.step { position: relative; padding-left: 48px; margin-bottom: 18px; }
.step:last-child { margin-bottom: 0; }
.step-num { position: absolute; left: 0; top: 0; width: 34px; height: 34px; border-radius: 50%; background: var(--surface); border: 1px solid var(--border); display: flex; align-items: center; justify-content: center; font-family: 'IBM Plex Mono', ui-monospace, monospace; font-size: 13px; font-weight: 600; color: var(--text-muted); }
.step.assistant .step-num { color: var(--accent); border-color: color-mix(in srgb, var(--accent) 40%, var(--border)); }
.step.err .step-num { color: var(--err); border-color: color-mix(in srgb, var(--err) 45%, var(--border)); background: var(--err-soft); }
.step.tool_result .step-num { color: var(--ok); border-color: color-mix(in srgb, var(--ok) 45%, var(--border)); background: var(--ok-soft); }
.step-body { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; }
.step-kind { font-size: 11.5px; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; color: var(--text-muted); margin-bottom: 6px; }
.step-kind .tool-name { text-transform: none; letter-spacing: 0; }
.tool-name { font-family: 'IBM Plex Mono', ui-monospace, monospace; color: var(--accent); }
.step.err .tool-name { color: var(--err); }
.step-text { font-size: 14.5px; white-space: pre-wrap; word-break: break-word; }
.tool-call { margin-top: 8px; }
.tool-call-name { font-size: 13.5px; margin-bottom: 4px; }
details.blob { margin-top: 8px; border: 1px solid var(--border); border-radius: 8px; background: var(--surface-2); }
details.blob > summary { cursor: pointer; padding: 7px 12px; font-size: 12px; font-weight: 600; color: var(--text-muted); list-style: none; }
details.blob > summary::-webkit-details-marker { display: none; }
details.blob > summary::before { content: "▸ "; }
details.blob[open] > summary::before { content: "▾ "; }
.blob-content { padding: 0 14px 12px; font-family: 'IBM Plex Mono', ui-monospace, monospace; font-size: 12.5px; color: var(--text); max-height: 340px; overflow-y: auto; }
.blob-content pre { margin: 0; white-space: pre-wrap; word-break: break-word; }
.blob-content p { margin: 0 0 .8em; font-family: 'Manrope', sans-serif; white-space: pre-wrap; }
.blob-content p:last-child { margin-bottom: 0; }
"""


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Render a live-test session trace to an HTML report.")
    parser.add_argument("trace_path", type=Path, help="Path to the .jsonl trace file.")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Write HTML here instead of stdout.")
    parser.add_argument("--title", default=None, help="Report title (default: the trace file's own name).")
    args = parser.parse_args(argv)

    turns = parse_trace(args.trace_path)
    title = args.title or args.trace_path.stem
    output = render_html(turns, title=title, source_label=str(args.trace_path))

    if args.output:
        args.output.write_text(output)
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()

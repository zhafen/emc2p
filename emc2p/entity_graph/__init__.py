"""Entity-graph export: a registry's entities, components and `*_eid` links
as nodes/edges, plus a self-contained mobile HTML viewer for them.

Works on any live registry (e.g. an MCP session's open Registrar), so the
view matches the running state without re-ingesting a manifest.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

TEMPLATE_PATH = Path(__file__).parent / "template.html"
DATA_PLACEHOLDER = "/*__DATA__*/null"
_SKIPPED_TYPES = ("entity_id", "invalid_field")


def _clean(v: Any) -> Any:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if hasattr(v, "item"):
        v = v.item()
    return v.isoformat() if hasattr(v, "isoformat") else v


def registry_to_graph(registry) -> dict:
    """Convert `registry` to `{"nodes": [...], "edges": [...]}`.

    One node per entity (with every component it carries); one edge per
    `*_eid` field pointing at a known entity, typed by its component type.
    """
    nodes = {
        r.value: dict(id=r.value, name=r.display_key, alias=_clean(r.display_alias),
                      path=r.path, file=_clean(r.filepath), components=[])
        for r in registry.get("entity_id").to_pandas().itertuples()
    }
    edges = []
    for t in registry.component_types:
        if t in _SKIPPED_TYPES:
            continue
        for row in registry.get(t).to_pandas().to_dict("records"):
            eid = row.pop("entity_id")
            f = {k: _clean(v) for k, v in row.items()
                 if not k.startswith("_") and k != "component_index" and _clean(v) is not None}
            if not f.get("modifier"):
                f.pop("modifier", None)
            for k, v in f.items():
                if k.endswith("_eid") and v in nodes:
                    edges.append(dict(source=eid, target=v, type=t))
            if eid in nodes:
                nodes[eid]["components"].append(dict(type=t, fields=f))
    return dict(nodes=list(nodes.values()), edges=edges)


def render_graph_html(graph: dict) -> str:
    """Embed `graph` into the viewer template, returning standalone HTML."""
    data = json.dumps(graph, default=str).replace("</", "<\\/")
    return TEMPLATE_PATH.read_text().replace(DATA_PLACEHOLDER, data)


def write_graph(registry, output_path: str | Path) -> Path:
    """Write `registry`'s graph to `output_path`: JSON if it ends in
    `.json`, otherwise the HTML viewer."""
    path = Path(output_path)
    graph = registry_to_graph(registry)
    text = json.dumps(graph, default=str) if path.suffix == ".json" else render_graph_html(graph)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path

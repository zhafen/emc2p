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


HOW_TREE_TEMPLATE_PATH = Path(__file__).parent / "how_tree.html"

# Tests in a file or directory named like this are human-validated (extra carefully
# verified, e.g. tests/test_human_validated/ or test_acceptance.py) and shown apart.
VALIDATED_TEST_PREFIXES = ("test_human_validated", "test_acceptance")


def _is_validated_location(loc: str | None) -> bool:
    """True if any path segment of `loc` (``path::name``) starts with a VALIDATED_TEST_PREFIXES entry."""
    if not loc:
        return False
    path = loc.split("::", 1)[0].split(":", 1)[-1]
    return any(Path(part).stem.startswith(VALIDATED_TEST_PREFIXES) for part in Path(path).parts)


def _py_module(key: str, filepath: str) -> str:
    """The module part of a Python entity's dotted `key`, matched against its `filepath`."""
    dotted = filepath[:-3].replace("/", ".")
    if key == dotted or key.startswith(dotted + "."):
        return dotted
    parts = key.split(".")
    for i in range(len(parts), 0, -1):
        if dotted.endswith(".".join(parts[:i])):
            return ".".join(parts[:i])
    return key


def _how_kind(key: str, filepath: str, comps: dict) -> str:
    """Rough object kind for display: concept, file, definition, module, class, function, variable."""
    if "source_file" in comps:
        loc = comps["source_file"][0].get("value") or ""
        if _clean(comps["source_file"][0].get("kind")):
            return comps["source_file"][0]["kind"]
        if "::" not in loc:
            return "file"
        return "variable" if loc.split("::", 1)[0].endswith(".py") else "definition"
    if not filepath.endswith(".py"):
        return "concept"
    if key == _py_module(key, filepath):
        return "module"
    return "class" if key.rsplit(".", 1)[-1][:1].isupper() else "function"


def registry_to_how_tree(registry, root: str | None = None) -> dict:
    """Convert `registry`'s how chain to `{"root": id, "nodes": {id: {...}}}`.

    Edges come from ``how`` components (owner -> ``value``) and ``how_link``
    components (``source`` -> ``value``), in listed order, each carrying the
    ``question`` it answers. ``test`` components (``source`` = a test,
    ``value`` = what it tests) and ``tests`` components (owned by the test,
    ``value`` = what it tests; hand-written) become each node's ``tests``
    list; the hand-written ones are also listed in ``hand_tests``. `root` is an entity key (e.g.
    ``how_chain_mission``); by default, the one entity with outgoing but no
    incoming how edges.
    """
    ids = registry.get("entity_id").to_pandas()
    info = {}
    for r in ids.itertuples():
        key = str(r.path).split(":", 1)[1] if ":" in str(r.path) else str(r.path)
        info[r.value] = dict(key=key, file=_clean(r.filepath) or "")
    comps: dict = {}
    for t in ("description", "source_file"):
        if t in registry.component_types:
            for row in registry.get(t).to_pandas().to_dict("records"):
                comps.setdefault(row["entity_id"], {}).setdefault(t, []).append(row)
    edges = []
    for t, src in (("how", "entity_id"), ("how_link", "source_eid")):
        if t in registry.component_types:
            df = registry.get(t).to_pandas()
            df = df.sort_values([c for c in ("component_index",) if c in df.columns] or df.columns[:1].tolist(), kind="stable")
            for row in df.to_dict("records"):
                if row.get(src) in info and row.get("value_eid") in info:
                    edges.append((row[src], row["value_eid"], _clean(row.get("question")), _clean(row.get("note"))))
    # Validation links: inferred ``test`` rows (source = test) and hand-written
    # ``tests`` components owned by the test itself (e.g. from its docstring).
    tests_of: dict = {}
    hand_of: dict = {}
    for t, src, hand in (("test", "source_eid", False), ("tests", "entity_id", True)):
        if t in registry.component_types:
            for row in registry.get(t).to_pandas().to_dict("records"):
                if row.get(src) in info and row.get("value_eid") in info:
                    tests_of.setdefault(row["value_eid"], []).append(row[src])
                    if hand:
                        hand_of.setdefault(row["value_eid"], []).append(row[src])
    children: dict = {}
    incoming = {}
    for s, v, q, note in edges:
        children.setdefault(s, []).append(v)
        incoming[v] = (q, note)
    if root is not None:
        root_id = next(i for i, d in info.items() if d["key"] == root)
    else:
        root_id = next(s for s in children if s not in incoming)
    nodes, stack = {}, [root_id]
    while stack:
        n = stack.pop()
        if n in nodes:
            continue
        d, c = info[n], comps.get(n, {})
        kind = _how_kind(d["key"], d["file"], c)
        if "source_file" in c:
            loc = c["source_file"][0].get("value")
        elif d["file"].endswith(".py"):
            module = _py_module(d["key"], d["file"])
            qual = d["key"][len(module) + 1:]
            loc = d["file"] + (f"::{qual}" if qual else "")
        else:
            loc = None
        desc = c.get("description", [{}])[-1].get("value") or ""
        q, note = incoming.get(n, (None, None))
        nodes[n] = dict(name=d["key"].rsplit(".", 1)[-1] if d["file"].endswith(".py") else d["key"],
                        kind=kind, description=_clean(desc), question=q, note=note, location=loc,
                        children=children.get(n, []), tests=tests_of.get(n, []),
                        hand_tests=hand_of.get(n, []), validated=_is_validated_location(loc))
        stack.extend(reversed(children.get(n, [])))
    return dict(root=root_id, nodes=nodes)


def render_how_tree_html(tree: dict, title: str = "How Chain") -> str:
    """Embed `tree` into the drill-down viewer, returning standalone HTML."""
    data = json.dumps(tree, default=str).replace("</", "<\\/")
    html = HOW_TREE_TEMPLATE_PATH.read_text().replace(DATA_PLACEHOLDER, data)
    return html.replace("<title>How Chain</title>", f"<title>{title}</title>")

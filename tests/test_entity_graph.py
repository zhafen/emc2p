"""Tests for emc2p.entity_graph on a small fake registry (fast, no ETL)."""

import json
import textwrap

import pandas as pd
import pytest

from emc2p.entity_graph import registry_to_graph, render_graph_html, write_graph


class _Table:
    def __init__(self, rows):
        self._df = pd.DataFrame(rows)

    def to_pandas(self):
        return self._df.copy()


class FakeRegistry:
    def __init__(self, tables):
        self._tables = {k: _Table(v) for k, v in tables.items()}
        self.component_types = list(tables)

    def get(self, t):
        return self._tables[t]


def _entity(eid, key):
    return dict(value=eid, display_key=key, display_alias=key, path=f"f.{key}", filepath="f.yaml")


def _registry():
    return FakeRegistry({
        "entity_id": [_entity("e1", "a"), _entity("e2", "b")],
        "description": [
            dict(entity_id="e1", value="A thing", component_index=0, _source="x", modifier=None),
            dict(entity_id="e2", value=float("nan"), component_index=0, _source="x", modifier=None),
        ],
        "parent": [dict(entity_id="e2", parent_eid="e1", component_index=0)],
        "dangling": [dict(entity_id="e1", other_eid="missing", component_index=0)],
        "invalid_field": [dict(entity_id="e1", value="ignored")],
    })


def test_nodes_carry_entity_metadata_and_components():
    nodes = {n["id"]: n for n in registry_to_graph(_registry())["nodes"]}
    assert set(nodes) == {"e1", "e2"}
    assert nodes["e1"]["name"] == "a" and nodes["e1"]["file"] == "f.yaml"
    desc = next(c for c in nodes["e1"]["components"] if c["type"] == "description")
    assert desc["fields"] == {"value": "A thing"}


def test_nan_fields_and_internal_columns_are_dropped():
    nodes = {n["id"]: n for n in registry_to_graph(_registry())["nodes"]}
    desc = next(c for c in nodes["e2"]["components"] if c["type"] == "description")
    assert desc["fields"] == {}


def test_eid_fields_become_edges_only_for_known_targets():
    assert registry_to_graph(_registry())["edges"] == [dict(source="e2", target="e1", type="parent")]


def test_invalid_field_is_skipped():
    nodes = registry_to_graph(_registry())["nodes"]
    assert all(c["type"] != "invalid_field" for n in nodes for c in n["components"])


def test_render_embeds_data_and_escapes_script_close():
    html = render_graph_html(dict(nodes=[dict(id="x", name="</script>")], edges=[]))
    assert "/*__DATA__*/null" not in html
    assert "<\\/script>" in html and '"id": "x"' in html


def test_write_graph_json_vs_html(tmp_path):
    j = write_graph(_registry(), tmp_path / "g.json")
    assert set(json.loads(j.read_text())) == {"nodes", "edges"}
    h = write_graph(_registry(), tmp_path / "sub" / "g.html")
    assert "<title>Entity Graph</title>" in h.read_text()


@pytest.mark.slow
def test_question_component_defaults_to_open(tmp_path):
    from emc2p.registrar import Registrar

    (tmp_path / "t.yaml").write_text("topic:\n    - question: Ship it?\n")
    nodes = registry_to_graph(Registrar.from_manifest([str(tmp_path)]).registry)["nodes"]
    q = next(c for n in nodes for c in n["components"] if c["type"] == "question")
    assert q["fields"] == {"value": "Ship it?", "status": "open"}


@pytest.mark.slow
def test_question_component_keeps_suggestion(tmp_path):
    from emc2p.registrar import Registrar

    (tmp_path / "t.yaml").write_text(
        "topic:\n    - question:\n          value: Ship it?\n          suggestion: Ship it\n")
    nodes = registry_to_graph(Registrar.from_manifest([str(tmp_path)]).registry)["nodes"]
    q = next(c for n in nodes for c in n["components"] if c["type"] == "question")
    assert q["fields"] == {"value": "Ship it?", "suggestion": "Ship it", "status": "open"}


@pytest.mark.slow
def test_registry_to_how_tree(tmp_path):
    from emc2p.entity_graph import registry_to_how_tree, render_how_tree_html
    from emc2p.registrar import Registrar

    (tmp_path / "t.yaml").write_text(textwrap.dedent("""\
        how:
            - parent: directed_relation
            - field:
                  value: {type: entity_ref}
                  question: {type: str}
        how_link:
            - parent: directed_relation
            - field:
                  source: {type: entity_ref}
                  value: {type: entity_ref}
                  question: {type: str}
        mission:
            - description: Root.
            - how:
                  - value: part_a
                    question: How A?
                  - value: part_b
                    question: How B?
        part_a:
            - description: A.
        part_b:
            - description: B.
        part_c:
            - description: C.
        index:
            - how_link:
                  - source: part_a
                    value: part_c
                    question: How C?
        """))
    tree = registry_to_how_tree(Registrar.from_manifest([str(tmp_path)]).registry)
    nodes = tree["nodes"]
    root = nodes[tree["root"]]
    assert root["name"] == "mission"
    assert [nodes[c]["question"] for c in root["children"]] == ["How A?", "How B?"]
    a = nodes[root["children"][0]]
    assert [nodes[c]["name"] for c in a["children"]] == ["part_c"]
    assert len(nodes) == 4 and all(n["kind"] == "concept" for n in nodes.values())
    assert "<title>X</title>" in render_how_tree_html(tree, title="X")

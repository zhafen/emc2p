"""Tests for the load_python Hamilton DAG functions."""

from pathlib import Path

import pytest

_EMC2P_SRC = Path(__file__).parent.parent.parent / "emc2p"

import emc2p.dataflows.etl.load_manifest as load_manifest
import emc2p.dataflows.etl.load_python as load_python


# Helpers

def _all_entities(result: dict) -> dict:
    merged = {}
    for entities in result.values():
        merged.update(entities)
    return merged


def _load(python_strings: dict) -> dict:
    """Run the load_python DAG end to end: parsed_asts -> module_names -> raw_entity_first_data."""
    asts = load_python.parsed_asts(python_strings)
    names = load_python.module_names(asts)
    return load_python.raw_entity_first_data(asts, names)


# _find_iacs_meta

class TestFindIacsMeta:

    def _parse_body(self, src):
        import ast
        return ast.parse(src).body

    def test_finds_module_level_assignment(self):
        stmts = self._parse_body('__iacs__ = {"solution of": "req"}')
        result = load_python._find_iacs_meta(stmts)
        assert result == {"solution of": "req"}

    def test_finds_after_docstring(self):
        stmts = self._parse_body('"""Docstring."""\n__iacs__ = {"solution of": "req"}')
        result = load_python._find_iacs_meta(stmts)
        assert result == {"solution of": "req"}

    def test_returns_none_when_absent(self):
        stmts = self._parse_body("x = 1\ny = 2")
        assert load_python._find_iacs_meta(stmts) is None

    def test_ignores_non_dict_value(self):
        stmts = self._parse_body('__iacs__ = "not a dict"')
        assert load_python._find_iacs_meta(stmts) is None

    def test_multiple_keys(self):
        stmts = self._parse_body('__iacs__ = {"solution of": "r", "work_state": "done"}')
        result = load_python._find_iacs_meta(stmts)
        assert result == {"solution of": "r", "work_state": "done"}


# parsed_asts / module_names -- the DAG nodes preceding raw_entity_first_data

class TestParsedAsts:

    def test_empty_dict_returns_empty(self):
        assert load_python.parsed_asts({}) == {}

    def test_valid_source_is_parsed(self):
        import ast
        result = load_python.parsed_asts({"mod.py": '"""Doc."""\n'})
        assert isinstance(result["mod.py"], ast.Module)

    def test_syntax_errors_are_skipped(self):
        result = load_python.parsed_asts({"bad.py": "def (:\n"})
        assert result == {}

    def test_only_failing_entries_are_dropped(self):
        result = load_python.parsed_asts({
            "good.py": '"""Doc."""\n',
            "bad.py": "def (:\n",
        })
        assert set(result.keys()) == {"good.py"}


class TestModuleNames:

    def test_empty_dict_returns_empty(self):
        assert load_python.module_names({}) == {}

    def test_derives_dotted_name_from_file_id(self):
        asts = load_python.parsed_asts({"pkg/mod.py": '"""Doc."""\n'})
        names = load_python.module_names(asts)
        assert names == {"pkg/mod.py": "pkg.mod"}

    def test_keys_match_parsed_asts(self):
        asts = load_python.parsed_asts({"a.py": '"""A."""\n', "b.py": '"""B."""\n'})
        names = load_python.module_names(asts)
        assert set(names.keys()) == set(asts.keys())


# raw_entity_first_data -- parsing a dict of raw Python source

class TestRawEntityFirstData:

    def test_empty_dict_returns_empty(self):
        assert _load({}) == {}

    def test_skips_entries_with_syntax_errors(self):
        result = _load({"bad.py": "def (:\n"})
        assert result == {}

    def test_entries_without_docstring_or_iacs_produce_no_entities(self):
        result = _load({"empty.py": "x = 1\n"})
        assert result == {}

    def test_multiple_entries_produce_separate_keys(self):
        result = _load({
            "a.py": '"""Module A."""\n',
            "b.py": '"""Module B."""\n',
        })
        assert set(result.keys()) == {"a.py", "b.py"}

    def test_ignores_files_without_python_content(self):
        """A non-Python value (e.g. accidentally-included YAML text) just fails to parse."""
        result = _load({"data.yaml": "key: value\n"})
        assert result == {}


# Entity extraction

class TestModuleEntity:

    def test_module_docstring_becomes_description(self):
        entities = _all_entities(_load({"mod.py": '"""A module docstring."""\n'}))
        key = next(k for k in entities if k.endswith("mod"))
        comps = entities[key]
        assert any(c.get("description", "").startswith("A module docstring") for c in comps)

    def test_module_iacs_meta_becomes_component(self):
        src = '"""Doc."""\n__iacs__ = {"solution of": "some_req"}\n'
        entities = _all_entities(_load({"mod.py": src}))
        key = next(k for k in entities if k.endswith("mod"))
        comps = entities[key]
        assert {"solution of": "some_req"} in comps

    def test_module_without_docstring_but_with_iacs(self):
        src = '__iacs__ = {"solution of": "req"}\n'
        entities = _all_entities(_load({"mod.py": src}))
        assert any(k.endswith("mod") for k in entities)


class TestDocstringComponentsSection:
    """Ingesting a docstring's numpy-style "Components" YAML section."""

    def test_task_example(self):
        src = (
            'def my_function():\n'
            '    """Docstring here.\n'
            '\n'
            '    Components\n'
            '    ----------\n'
            '    - todo: We need to implement this.\n'
            '    """\n'
        )
        entities = _all_entities(_load({"mod.py": src}))
        key = next(k for k in entities if k.endswith(".my_function"))
        comps = entities[key]
        assert {"description": "Docstring here."} in comps
        assert {"todo": "We need to implement this."} in comps

    def test_multiple_component_entries(self):
        src = (
            'def foo():\n'
            '    """Foo.\n'
            '\n'
            '    Components\n'
            '    ----------\n'
            '    - todo: First thing.\n'
            '    - todo: Second thing.\n'
            '    """\n'
        )
        entities = _all_entities(_load({"mod.py": src}))
        key = next(k for k in entities if k.endswith(".foo"))
        comps = entities[key]
        assert {"todo": "First thing."} in comps
        assert {"todo": "Second thing."} in comps

    def test_dict_component_entry(self):
        src = (
            'def foo():\n'
            '    """Foo.\n'
            '\n'
            '    Components\n'
            '    ----------\n'
            '    - solution of: some_requirement\n'
            '    """\n'
        )
        entities = _all_entities(_load({"mod.py": src}))
        key = next(k for k in entities if k.endswith(".foo"))
        assert {"solution of": "some_requirement"} in entities[key]

    def test_components_section_excluded_from_description(self):
        src = (
            'def foo():\n'
            '    """Foo does things.\n'
            '\n'
            '    Components\n'
            '    ----------\n'
            '    - todo: Do it.\n'
            '    """\n'
        )
        entities = _all_entities(_load({"mod.py": src}))
        key = next(k for k in entities if k.endswith(".foo"))
        desc = next(c["description"] for c in entities[key] if "description" in c)
        assert "Components" not in desc
        assert "todo" not in desc
        assert desc == "Foo does things."

    def test_no_components_section_unaffected(self):
        src = 'def foo():\n    """Just a plain docstring."""\n'
        entities = _all_entities(_load({"mod.py": src}))
        key = next(k for k in entities if k.endswith(".foo"))
        assert entities[key] == [{"description": "Just a plain docstring."}]

    def test_malformed_yaml_is_ignored(self):
        src = (
            'def foo():\n'
            '    """Foo.\n'
            '\n'
            '    Components\n'
            '    ----------\n'
            '    - todo: [unterminated\n'
            '    """\n'
        )
        entities = _all_entities(_load({"mod.py": src}))
        key = next(k for k in entities if k.endswith(".foo"))
        assert entities[key] == [{"description": "Foo."}]

    def test_non_list_yaml_is_ignored(self):
        src = (
            'def foo():\n'
            '    """Foo.\n'
            '\n'
            '    Components\n'
            '    ----------\n'
            '    just a plain string, not a list\n'
            '    """\n'
        )
        entities = _all_entities(_load({"mod.py": src}))
        key = next(k for k in entities if k.endswith(".foo"))
        assert entities[key] == [{"description": "Foo."}]

    def test_component_only_docstring_with_no_leading_description(self):
        src = (
            'def foo():\n'
            '    """\n'
            '    Components\n'
            '    ----------\n'
            '    - todo: Only a todo, no description.\n'
            '    """\n'
        )
        entities = _all_entities(_load({"mod.py": src}))
        key = next(k for k in entities if k.endswith(".foo"))
        assert entities[key] == [{"todo": "Only a todo, no description."}]

    def test_section_before_trailing_numpy_section(self):
        src = (
            'def foo():\n'
            '    """Foo.\n'
            '\n'
            '    Components\n'
            '    ----------\n'
            '    - todo: Do it.\n'
            '\n'
            '    Returns\n'
            '    -------\n'
            '    None.\n'
            '    """\n'
        )
        entities = _all_entities(_load({"mod.py": src}))
        key = next(k for k in entities if k.endswith(".foo"))
        comps = entities[key]
        assert {"todo": "Do it."} in comps
        desc = next(c["description"] for c in comps if "description" in c)
        assert "Components" not in desc
        assert "Returns" in desc
        assert "None." in desc


# _call_target -- resolving a Call node's func to a best-effort dotted name

class TestCallTarget:

    def _call_func(self, src: str):
        import ast
        call = ast.parse(src, mode="eval").body
        assert isinstance(call, ast.Call)
        return call.func

    def test_bare_name_call(self):
        assert load_python._call_target(self._call_func("foo()")) == "foo"

    def test_attribute_call(self):
        assert load_python._call_target(self._call_func("pd.DataFrame()")) == "pd.DataFrame"

    def test_self_call_drops_self(self):
        assert load_python._call_target(self._call_func("self.foo()")) == "foo"

    def test_cls_call_drops_cls(self):
        assert load_python._call_target(self._call_func("cls.foo()")) == "foo"

    def test_chained_attribute_call(self):
        assert (
            load_python._call_target(self._call_func("self.spatial.select_character()"))
            == "spatial.select_character"
        )

    def test_deep_chained_attribute_call_keeps_full_chain(self):
        assert load_python._call_target(self._call_func("a.b.c()")) == "a.b.c"

    def test_unresolvable_subscript_target_returns_none(self):
        assert load_python._call_target(self._call_func("handlers[0]()")) is None

    def test_call_result_call_returns_none(self):
        assert load_python._call_target(self._call_func("foo()()")) is None


# _collect_body -- calls/imports made directly in a body, not nested defs

class TestCollectBody:

    def _body(self, src: str):
        import ast
        import textwrap
        return ast.parse(textwrap.dedent(src)).body

    def test_collects_a_call(self):
        calls, imports = load_python._collect_body(self._body("foo()\n"))
        assert calls == ["foo"]
        assert imports == []

    def test_collects_multiple_distinct_calls_in_first_occurrence_order_deduped(self):
        # "foo" appears (and repeats) before "bar" in source -- first-
        # occurrence order, not alphabetical, is what's under test here.
        calls, _ = load_python._collect_body(self._body("foo()\nbar()\nfoo()\n"))
        assert calls == ["foo", "bar"]

    def test_does_not_descend_into_nested_function(self):
        # This body itself plays the role of an enclosing function's body --
        # `def inner` is a nested def within it, as `_collect_body` receives.
        calls, _ = load_python._collect_body(self._body(
            """
            def inner():
                inner_only_call()
            outer_call()
            """
        ))
        assert calls == ["outer_call"]

    def test_does_not_descend_into_nested_class(self):
        calls, _ = load_python._collect_body(self._body(
            """
            class Outer:
                def method(self):
                    method_only_call()
            top_level_call()
            """
        ))
        assert calls == ["top_level_call"]

    def test_collects_plain_import(self):
        _, imports = load_python._collect_body(self._body("import os\n"))
        assert imports == ["os"]

    def test_collects_dotted_import(self):
        _, imports = load_python._collect_body(self._body("import os.path\n"))
        assert imports == ["os.path"]

    def test_collects_import_from(self):
        _, imports = load_python._collect_body(self._body("from a.b import c\n"))
        assert imports == ["a.b.c"]

    def test_collects_import_from_multiple_names(self):
        _, imports = load_python._collect_body(self._body("from a.b import c, d\n"))
        assert imports == ["a.b.c", "a.b.d"]

    def test_finds_import_inside_try_block(self):
        _, imports = load_python._collect_body(self._body(
            """
            try:
                import ujson as json
            except ImportError:
                import json
            """
        ))
        assert imports == ["json", "ujson"]

    def test_call_inside_lambda_is_collected(self):
        calls, _ = load_python._collect_body(self._body("f = lambda: foo()\n"))
        assert calls == ["foo"]

    def test_excludes_a_constructor_call(self):
        # Foo() constructs a Foo, it doesn't call behavior named "Foo" --
        # only callables belong in a call-reachability trace.
        calls, _ = load_python._collect_body(self._body("Foo()\n"))
        assert calls == []

    def test_excludes_a_constructor_call_via_attribute_access(self):
        calls, _ = load_python._collect_body(self._body("self.spatial.Foo()\n"))
        assert calls == []

    def test_keeps_lowercase_calls_alongside_an_excluded_constructor(self):
        calls, _ = load_python._collect_body(self._body("bar()\nFoo()\nbaz()\n"))
        assert calls == ["bar", "baz"]


# End-to-end: calls/imports show up as component entries

class TestCallsAndImportsAsComponents:

    def test_function_call_becomes_calls_component(self):
        src = (
            'def foo():\n'
            '    """Foo."""\n'
            '    bar()\n'
        )
        entities = _all_entities(_load({"mod.py": src}))
        key = next(k for k in entities if k.endswith(".foo"))
        assert {"calls": {"value": "bar", "seq": 0}} in entities[key]

    def test_function_without_calls_has_no_calls_component(self):
        src = 'def foo():\n    """Just a plain docstring."""\n'
        entities = _all_entities(_load({"mod.py": src}))
        key = next(k for k in entities if k.endswith(".foo"))
        assert entities[key] == [{"description": "Just a plain docstring."}]

    def test_multiple_calls_become_multiple_calls_components_carrying_call_order(self):
        src = (
            'def foo():\n'
            '    """Foo."""\n'
            '    baz()\n'
            '    bar()\n'
        )
        entities = _all_entities(_load({"mod.py": src}))
        key = next(k for k in entities if k.endswith(".foo"))
        # baz() is called first in source, despite sorting after "bar"
        # alphabetically -- seq reflects call order, not alphabetical order.
        assert {"calls": {"value": "baz", "seq": 0}} in entities[key]
        assert {"calls": {"value": "bar", "seq": 1}} in entities[key]

    def test_module_level_import_becomes_imports_component(self):
        src = '"""Mod."""\nimport os\n'
        entities = _all_entities(_load({"mod.py": src}))
        key = next(k for k in entities if k.endswith("mod"))
        assert {"imports": "os"} in entities[key]

    def test_method_call_via_self_resolves_to_bare_name(self):
        src = (
            'class Foo:\n'
            '    """Foo."""\n'
            '    def run(self):\n'
            '        """Run."""\n'
            '        self.helper()\n'
        )
        entities = _all_entities(_load({"mod.py": src}))
        key = next(k for k in entities if k.endswith(".run"))
        assert {"calls": {"value": "helper", "seq": 0}} in entities[key]

    def test_constructor_call_produces_no_calls_component(self):
        # A call-reachability trace is meant to show callables, not types
        # in use -- Bar() constructs a Bar, so it's excluded.
        src = 'def foo():\n    Bar()\n'
        entities = _all_entities(_load({"mod.py": src}))
        assert entities == {}

    def test_constructor_call_alongside_a_real_call_only_the_real_one_is_kept(self):
        src = (
            'def foo():\n'
            '    """Foo."""\n'
            '    Bar()\n'
            '    baz()\n'
        )
        entities = _all_entities(_load({"mod.py": src}))
        key = next(k for k in entities if k.endswith(".foo"))
        assert {"calls": {"value": "baz", "seq": 0}} in entities[key]
        calls_values = [c["calls"]["value"] for c in entities[key] if "calls" in c]
        assert calls_values == ["baz"]

    def test_entity_with_only_calls_has_no_description(self):
        src = 'def foo():\n    bar()\n'
        entities = _all_entities(_load({"mod.py": src}))
        assert entities == {}

    def test_undocumented_function_calls_do_not_leak_onto_module_entity(self):
        """A module-level __iacs__ still creates the module entity
        (existing behavior).

        But calls inside an undocumented nested function must not be
        attributed to it -- _collect_body stops at nested FunctionDef.
        """
        src = '__iacs__ = {"solution of": "req"}\ndef foo():\n    bar()\n'
        entities = _all_entities(_load({"mod.py": src}))
        mod_key = next(k for k in entities if k.endswith("mod"))
        assert {"solution of": "req"} in entities[mod_key]
        assert not any("calls" in c for c in entities[mod_key])
        assert not any(k.endswith(".foo") for k in entities)


class TestFunctionEntity:

    def test_function_docstring_becomes_description(self):
        src = 'def foo():\n    """Foo does things."""\n    pass\n'
        entities = _all_entities(_load({"mod.py": src}))
        key = next(k for k in entities if k.endswith(".foo"))
        comps = entities[key]
        assert any("Foo does things" in c.get("description", "") for c in comps)

    def test_function_iacs_in_body(self):
        src = (
            'def foo():\n'
            '    """Doc."""\n'
            '    __iacs__ = {"solution of": "req"}\n'
        )
        entities = _all_entities(_load({"mod.py": src}))
        key = next(k for k in entities if k.endswith(".foo"))
        assert {"solution of": "req"} in entities[key]

    def test_function_without_docstring_excluded(self):
        src = 'def foo():\n    pass\n'
        entities = _all_entities(_load({"mod.py": src}))
        assert not any(k.endswith(".foo") for k in entities)


class TestClassEntity:

    def test_class_docstring_becomes_description(self):
        src = 'class MyClass:\n    """A class."""\n    pass\n'
        entities = _all_entities(_load({"mod.py": src}))
        key = next(k for k in entities if k.endswith(".MyClass"))
        assert any("A class" in c.get("description", "") for c in entities[key])

    def test_class_iacs_in_body(self):
        src = 'class MyClass:\n    __iacs__ = {"solution of": "req"}\n'
        entities = _all_entities(_load({"mod.py": src}))
        key = next(k for k in entities if k.endswith(".MyClass"))
        assert {"solution of": "req"} in entities[key]

    def test_method_uses_full_qualified_path(self):
        src = (
            'class MyClass:\n'
            '    def my_method(self):\n'
            '        """Method doc."""\n'
            '        __iacs__ = {"solution of": "req"}\n'
        )
        entities = _all_entities(_load({"mod.py": src}))
        assert any(k.endswith(".MyClass.my_method") for k in entities)
        key = next(k for k in entities if k.endswith(".MyClass.my_method"))
        assert {"solution of": "req"} in entities[key]


# Entity ID stability

class TestEntityIdStability:

    def test_same_source_produces_same_keys(self):
        python_strings = {"mod.py": '"""Doc."""\n'}
        r1 = _all_entities(_load(python_strings))
        r2 = _all_entities(_load(python_strings))
        assert set(r1.keys()) == set(r2.keys())

    def test_entity_key_uses_dotted_module_path(self):
        entities = _all_entities(_load({"pkg/mod.py": '"""Doc."""\n'}))
        assert any("pkg.mod" in k for k in entities)


# Integration with pathvalue_pairs

class TestIntegrationWithPipeline:

    def test_entity_first_dict_feeds_pathvalue_pairs(self):
        """Python entities flow through pathvalue_pairs without error."""
        src = (
            '"""Module doc."""\n'
            '__iacs__ = {"solution of": "some_req"}\n'
        )
        py_data = _load({"mod.py": src})
        pvp = load_manifest.pathvalue_pairs(py_data)
        df = pvp.to_pandas()
        assert len(df) > 0
        assert "path" in df.columns
        assert "value" in df.columns

    def test_solution_component_survives_pipeline(self):
        """A __iacs__ solution component is present after keyvalue_store."""
        src = (
            '"""Module doc."""\n'
            '__iacs__ = {"solution of": "some_req"}\n'
        )
        py_data = _load({"mod.py": src})
        pvp = load_manifest.pathvalue_pairs(py_data)
        kvs = load_manifest.keyvalue_store(pvp)
        ct = load_manifest.component_tables(kvs)
        assert "solution" in ct
        sol_df = ct["solution"].to_pandas()
        assert (sol_df["modifier"] == "of").any()
        assert (sol_df["value"] == "some_req").any()


# Integration test: load the real emc2p package

class TestLoadEmc2pPackage:
    """Load the emc2p source tree and verify the output is well-formed."""

    @pytest.fixture(scope="class")
    def emc2p_result(self):
        python_strings = load_manifest.raw_strings([_EMC2P_SRC])["raw_python_strings"]
        return _load(python_strings)

    @pytest.fixture(scope="class")
    def all_entities(self, emc2p_result):
        merged = {}
        for entities in emc2p_result.values():
            merged.update(entities)
        return merged

    def test_finds_multiple_files(self, emc2p_result):
        assert len(emc2p_result) >= 10

    def test_finds_many_entities(self, all_entities):
        assert len(all_entities) >= 50

    def test_keyed_by_py_file_path(self, emc2p_result):
        assert all(k.endswith(".py") for k in emc2p_result)

    def test_registrar_module_entity_present(self, all_entities):
        assert "emc2p.registrar" in all_entities

    def test_registrar_class_entity_present(self, all_entities):
        assert "emc2p.registrar.Registrar" in all_entities

    def test_registrar_method_entity_present(self, all_entities):
        assert "emc2p.registrar.Registrar.from_manifest" in all_entities

    def test_load_manifest_module_entity_present(self, all_entities):
        assert "emc2p.dataflows.etl.load_manifest" in all_entities

    def test_load_manifest_function_entities_present(self, all_entities):
        assert "emc2p.dataflows.etl.load_manifest.raw_entity_first_data" in all_entities
        assert "emc2p.dataflows.etl.load_manifest.registry" in all_entities

    def test_load_python_module_itself_present(self, all_entities):
        assert "emc2p.dataflows.etl.load_python" in all_entities

    def test_description_components_are_strings(self, all_entities):
        for key, comps in all_entities.items():
            for comp in comps:
                if "description" in comp:
                    assert isinstance(comp["description"], str), key
                    assert comp["description"].strip(), key

    def test_entity_values_are_lists(self, all_entities):
        for key, val in all_entities.items():
            assert isinstance(val, list), f"{key} should be a list"

    def test_all_components_are_dicts(self, all_entities):
        for key, comps in all_entities.items():
            for comp in comps:
                assert isinstance(comp, dict), f"component in {key} should be a dict"

    def test_flows_through_pipeline_without_error(self, emc2p_result):
        pvp = load_manifest.pathvalue_pairs(emc2p_result)
        kvs = load_manifest.keyvalue_store(pvp)
        ct = load_manifest.component_tables(kvs)
        assert "description" in ct
        df = ct["description"].to_pandas()
        assert len(df) >= 50

    def test_registrar_from_manifest_includes_python_entities(self):
        """Registrar.from_manifest on a dir with .py files registers Python entities."""
        from emc2p.registrar import Registrar
        a = Registrar.from_manifest(_EMC2P_SRC)
        desc = a.registry.get("description").execute()
        py_entities = desc[desc["entity_id"].isin(
            a.registry.get("entity_id").execute()
            .query("path.str.contains('.py:', regex=False)", engine="python")["value"]
        )]
        assert len(py_entities) >= 50

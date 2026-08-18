"""Tests for the load_manifest Hamilton DAG functions."""

from pathlib import Path

import ibis
import pandas as pd
import pytest

import emc2p.dataflows.etl.load_manifest as load_manifest
import emc2p.dataflows.etl.load_python as load_python
import emc2p.dataflows.etl.load_yaml as load_yaml
from emc2p.registry import Registry
from emc2p.utils import dhash


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_yaml_dir(tmp_path):
    """Create a temp directory with a minimal YAML manifest."""
    (tmp_path / "minimal.yaml").write_text(
        "my_task:\n"
        "- description: A task I need to complete.\n"
        "- parent: task\n"
    )
    return str(tmp_path)


@pytest.fixture
def multi_file_yaml_dir(tmp_path):
    """Create a temp directory with multiple YAML files including a subdirectory."""
    (tmp_path / "tasks.yaml").write_text(
        "my_task:\n"
        "- description: A task.\n"
        "- parent: task\n"
    )
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "infra.yaml").write_text(
        "my_infra:\n"
        "- description: Infrastructure.\n"
        "- solution of: my_task\n"
    )
    return str(tmp_path)


# ---------------------------------------------------------------------------
# raw_entity_first_data
# ---------------------------------------------------------------------------

def _yaml_strings(input_dirs, yaml_strings=None) -> dict:
    return load_manifest.raw_strings(input_dirs, yaml_strings=yaml_strings)["raw_yaml_strings"]


class TestRawEntityFirstData:
    """Tests for load_yaml.raw_entity_first_data (YAML source loader)."""

    def _all_entities(self, result: dict) -> dict:
        """Flatten all per-file entity dicts into one dict (for assertions)."""
        merged = {}
        for entities in result.values():
            if isinstance(entities, dict):
                merged.update(entities)
        return merged

    def test_loads_single_yaml_file(self, minimal_yaml_dir):
        result = load_yaml.raw_entity_first_data(_yaml_strings([minimal_yaml_dir]))
        assert isinstance(result, dict)
        assert "my_task" in self._all_entities(result)

    def test_loads_multiple_yaml_files(self, multi_file_yaml_dir):
        result = load_yaml.raw_entity_first_data(_yaml_strings([multi_file_yaml_dir]))
        entities = self._all_entities(result)
        assert "my_task" in entities
        assert "my_infra" in entities

    def test_loads_yaml_from_subdirectories(self, multi_file_yaml_dir):
        result = load_yaml.raw_entity_first_data(_yaml_strings([multi_file_yaml_dir]))
        assert "my_infra" in self._all_entities(result)

    def test_empty_dir_has_only_builtin(self, tmp_path):
        result = load_yaml.raw_entity_first_data(_yaml_strings([str(tmp_path)]))
        assert "builtins.components" in result
        assert len(result) == 1

    def test_always_includes_builtin(self, minimal_yaml_dir):
        result = load_yaml.raw_entity_first_data(_yaml_strings([minimal_yaml_dir]))
        assert "builtins.components" in result

    def test_preserves_raw_structure(self, minimal_yaml_dir):
        result = load_yaml.raw_entity_first_data(_yaml_strings([minimal_yaml_dir]))
        entities = self._all_entities(result)
        assert isinstance(entities["my_task"], list)
        assert {"description": "A task I need to complete."} in entities["my_task"]

    def test_keyed_by_file_path(self, minimal_yaml_dir):
        result = load_yaml.raw_entity_first_data(_yaml_strings([minimal_yaml_dir]))
        user_keys = [k for k in result if k != "builtins.components"]
        assert any(k.endswith("minimal.yaml") for k in user_keys)

    def test_accepts_single_file_path(self, tmp_path):
        yaml_file = tmp_path / "single.yaml"
        yaml_file.write_text("my_entity:\n- description: A thing.\n")
        result = load_yaml.raw_entity_first_data(_yaml_strings([str(yaml_file)]))
        assert "my_entity" in self._all_entities(result)


class TestRawEntityFirstDataMalformedComponents:
    """A component given as a bare mapping instead of a YAML list must raise,
    not silently parse into a container that flattens to zero rows."""

    def test_bare_mapping_component_raises(self):
        given = {
            "inline": "my_entity:\n    description:\n        value: A thing.\n"
        }
        with pytest.raises(ValueError, match="my_entity.description"):
            load_yaml.raw_entity_first_data(given)

    def test_error_names_the_offending_file(self):
        given = {
            "bad_file.yaml": "my_entity:\n    description:\n        value: A thing.\n"
        }
        with pytest.raises(ValueError, match="bad_file.yaml"):
            load_yaml.raw_entity_first_data(given)

    def test_well_formed_component_list_does_not_raise(self):
        given = {"inline": "my_entity:\n- description: A thing.\n"}
        result = load_yaml.raw_entity_first_data(given)
        assert result["inline"] == {"my_entity": [{"description": "A thing."}]}

    def test_legitimate_nested_entity_dict_does_not_raise(self):
        given = {
            "inline": (
                "container:\n"
                "    child_a:\n"
                "    - description: First child.\n"
                "    child_b:\n"
                "    - description: Second child.\n"
            )
        }
        result = load_yaml.raw_entity_first_data(given)
        assert result["inline"]["container"]["child_a"] == [{"description": "First child."}]
        assert result["inline"]["container"]["child_b"] == [{"description": "Second child."}]

    def test_null_placeholder_entity_does_not_raise(self):
        """A bare `key:` with nothing after the colon is a legitimate,
        componentless placeholder entity (e.g. iacs_manifest/iacs.yaml's
        `required_functionality:` block), not a malformed component list."""
        given = {
            "inline": (
                "container:\n"
                "    not_yet_fleshed_out:\n"
                "    also_a_placeholder:\n"
                "    nested:\n"
                "        still_a_placeholder:\n"
            )
        }
        result = load_yaml.raw_entity_first_data(given)
        assert result["inline"]["container"]["not_yet_fleshed_out"] is None
        assert result["inline"]["container"]["nested"]["still_a_placeholder"] is None

    def test_data_key_is_exempt_from_the_check(self):
        given = {"inline": "my_entity:\n    data: some freeform note, not a list\n"}
        result = load_yaml.raw_entity_first_data(given)
        assert result["inline"] == {"my_entity": {"data": "some freeform note, not a list"}}

    def test_empty_yaml_does_not_raise(self):
        result = load_yaml.raw_entity_first_data({"inline": ""})
        assert result == {"inline": {}}


# ---------------------------------------------------------------------------
# raw_strings — combining input_dirs with directly-provided yaml/python strings
# ---------------------------------------------------------------------------

class TestRawStrings:

    def _all_entities(self, result: dict) -> dict:
        merged = {}
        for entities in result.values():
            if isinstance(entities, dict):
                merged.update(entities)
        return merged

    def test_returns_both_fields(self, minimal_yaml_dir):
        result = load_manifest.raw_strings([minimal_yaml_dir])
        assert set(result.keys()) == {"raw_python_strings", "raw_yaml_strings"}
        assert all(isinstance(v, str) for v in result["raw_yaml_strings"].values())
        assert all(isinstance(v, str) for v in result["raw_python_strings"].values())

    def test_no_input_dirs_returns_only_given_strings_plus_builtins(self):
        given = {"inline": "my_entity:\n- description: Inline.\n"}
        result = load_manifest.raw_strings([], yaml_strings=given)
        assert result["raw_yaml_strings"]["inline"] == given["inline"]
        assert "builtins.components" in result["raw_yaml_strings"]
        assert result["raw_python_strings"] == {}

    def test_combines_input_dirs_with_given_yaml_strings(self, minimal_yaml_dir):
        given = {"inline": "my_other_task:\n- description: Also inline.\n"}
        result = load_manifest.raw_strings([minimal_yaml_dir], yaml_strings=given)
        yaml_result = result["raw_yaml_strings"]
        assert "inline" in yaml_result
        assert any(k.endswith("minimal.yaml") for k in yaml_result)

    def test_combines_input_dirs_with_given_python_strings(self, tmp_path):
        (tmp_path / "mod.py").write_text('"""A module."""\n')
        given = {"inline_mod": '"""Inline module."""\n'}
        result = load_manifest.raw_strings([tmp_path], python_strings=given)
        python_result = result["raw_python_strings"]
        assert "inline_mod" in python_result
        assert any(k.endswith("mod.py") for k in python_result)

    def test_yaml_and_python_files_both_discovered_from_same_dir(self, tmp_path):
        (tmp_path / "data.yaml").write_text("my_task:\n- description: A task.\n")
        (tmp_path / "mod.py").write_text('"""A module."""\n')
        result = load_manifest.raw_strings([tmp_path])
        assert any(k.endswith("data.yaml") for k in result["raw_yaml_strings"])
        assert any(k.endswith("mod.py") for k in result["raw_python_strings"])

    def test_given_yaml_strings_parse_correctly_downstream(self):
        given = {"inline": "my_entity:\n- description: Inline thing.\n"}
        raw = load_manifest.raw_strings([], yaml_strings=given)["raw_yaml_strings"]
        entities = load_yaml.raw_entity_first_data(raw)
        assert entities["inline"] == {"my_entity": [{"description": "Inline thing."}]}

    def test_given_python_strings_parse_correctly_downstream(self):
        given = {"inline_mod": '"""Inline module doc."""\n'}
        raw = load_manifest.raw_strings([], python_strings=given)["raw_python_strings"]
        asts = load_python.parsed_asts(raw)
        entities = load_python.raw_entity_first_data(asts, load_python.module_names(asts))
        assert {"description": "Inline module doc."} in entities["inline_mod"]["inline_mod"]

    def test_accepts_path_objects_directly(self, tmp_path):
        """input_dirs entries may be Path objects, not just strings."""
        (tmp_path / "minimal.yaml").write_text("my_task:\n- description: A task.\n")
        result = load_manifest.raw_strings([tmp_path])
        entities = load_yaml.raw_entity_first_data(result["raw_yaml_strings"])
        assert "my_task" in self._all_entities(entities)


# ---------------------------------------------------------------------------
# load_manifest — accepting yaml/python strings directly via the Hamilton driver
# ---------------------------------------------------------------------------

class TestLoadManifestAcceptsStrings:

    def _driver(self):
        from hamilton import driver, base
        return driver.Driver({}, load_manifest, adapter=base.DictResult())

    def test_yaml_strings_without_input_dirs(self):
        dr = self._driver()
        result = dr.execute(
            ["raw_entity_first_data"],
            inputs={
                "input_dirs": [],
                "yaml_strings": {"inline": "my_entity:\n- description: Inline.\n"},
            },
        )["raw_entity_first_data"]
        assert "my_entity" in result["inline"]

    def test_python_strings_without_input_dirs(self):
        dr = self._driver()
        result = dr.execute(
            ["raw_entity_first_data"],
            inputs={
                "input_dirs": [],
                "python_strings": {"inline_mod": '"""Inline module doc."""\n'},
            },
        )["raw_entity_first_data"]
        assert {"description": "Inline module doc."} in result["inline_mod"]["inline_mod"]

    def test_defaults_to_empty_when_strings_omitted(self, minimal_yaml_dir):
        dr = self._driver()
        result = dr.execute(
            ["raw_entity_first_data"], inputs={"input_dirs": [minimal_yaml_dir]}
        )["raw_entity_first_data"]
        user_keys = [k for k in result if not k.startswith("builtins.")]
        assert any(k.endswith("minimal.yaml") for k in user_keys)


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

def _make_entity_id_table():
    df = pd.DataFrame([{
        "value": "abc", "path": "test:e", "alias": "e",
        "entity_key": "e", "filepath": "test",
    }])
    return ibis.memtable(df)


def _make_component_type_table():
    df = pd.DataFrame([{
        "entity_id": "abc", "component_index": 0,
        "component_type": "description", "modifier": pd.NA,
    }])
    df["modifier"] = df["modifier"].astype(pd.StringDtype())
    return ibis.memtable(df)


class TestRegistry:

    def test_returns_registry_instance(self):
        eid = _make_entity_id_table()
        ct = _make_component_type_table()
        comps = {"description": ibis.memtable(pd.DataFrame([{"entity_id": "e1", "value": "Hello"}]))}
        result = load_manifest.registry(eid, ct, comps)
        assert isinstance(result, Registry)

    def test_registry_has_component_types(self):
        eid = _make_entity_id_table()
        ct = _make_component_type_table()
        comps = {
            "description": ibis.memtable(pd.DataFrame([{"entity_id": "e1", "value": "Hello"}])),
            "task": ibis.memtable(pd.DataFrame([{"entity_id": "e1"}])),
        }
        result = load_manifest.registry(eid, ct, comps)
        assert "description" in result.component_types
        assert "task" in result.component_types


# ---------------------------------------------------------------------------
# pathvalue_pairs
# ---------------------------------------------------------------------------

_FILE_ID = "test.yaml"


def _wrap(entities: dict) -> dict:
    """Wrap a flat entities dict in the file-keyed format expected by pathvalue_pairs."""
    return {_FILE_ID: entities}


def _path(entity_path: str) -> str:
    """Build the expected path string with the test file prefix."""
    return f"{_FILE_ID}:{entity_path}"


class TestPathvaluePairs:

    def test_returns_ibis_table(self):
        data = _wrap({"my_task": [{"description": "A task."}]})
        result = load_manifest.pathvalue_pairs(data)
        assert isinstance(result, ibis.Table)

    def test_has_path_and_value_columns(self):
        data = _wrap({"my_task": [{"description": "A task."}]})
        result = load_manifest.pathvalue_pairs(data)
        assert "path" in result.columns
        assert "value" in result.columns

    def test_scalar_component(self):
        """A dict component with a string value produces one (path, value) row."""
        data = _wrap({"my_task": [{"description": "A task."}]})
        result = load_manifest.pathvalue_pairs(data)
        df = result.to_pandas()
        expected_path = _path("my_task[0].description")
        assert expected_path in df["path"].values
        row = df[df["path"] == expected_path]
        assert row["value"].iloc[0] == "A task."

    def test_tag_component(self):
        """A bare-string tag produces a row with an empty value."""
        data = _wrap({"my_task": ["task"]})
        result = load_manifest.pathvalue_pairs(data)
        df = result.to_pandas()
        expected_path = _path("my_task[0].task")
        assert expected_path in df["path"].values
        assert df[df["path"] == expected_path]["value"].iloc[0] == ""

    def test_tag_index_preserved(self):
        """A tag at list index N shifts subsequent components to N+1."""
        data = _wrap({"my_task": ["task", {"description": "A task."}]})
        result = load_manifest.pathvalue_pairs(data)
        df = result.to_pandas()
        assert _path("my_task[1].description") in df["path"].values

    def test_dict_valued_component(self):
        """A component whose value is a dict produces one row per sub-field."""
        data = _wrap({"entity": [{"requirement": {"value": 1}}]})
        result = load_manifest.pathvalue_pairs(data)
        df = result.to_pandas()
        expected_path = _path("entity[0].requirement.value")
        assert expected_path in df["path"].values
        assert df[df["path"] == expected_path]["value"].iloc[0] == "1"

    def test_component_key_with_space(self):
        """Component keys containing spaces (e.g. 'solution of') are preserved."""
        data = _wrap({"entity": [{"solution of": "other_entity"}]})
        result = load_manifest.pathvalue_pairs(data)
        df = result.to_pandas()
        expected_path = _path("entity[0].solution of")
        assert expected_path in df["path"].values
        assert df[df["path"] == expected_path]["value"].iloc[0] == "other_entity"

    def test_nested_entity_data_prefix(self):
        """Components of a nested entity are placed under the .data[N] prefix."""
        data = _wrap({"parent": {"data": [{"description": "A parent."}]}})
        result = load_manifest.pathvalue_pairs(data)
        df = result.to_pandas()
        assert _path("parent.data[0].description") in df["path"].values

    def test_nested_entity_sub_entities_recurse(self):
        """Sub-entities of a nested entity get their own paths."""
        data = _wrap({
            "parent": {
                "data": [{"description": "A parent."}],
                "child": [{"description": "A child."}],
            }
        })
        result = load_manifest.pathvalue_pairs(data)
        df = result.to_pandas()
        assert _path("parent.data[0].description") in df["path"].values
        assert _path("parent.child[0].description") in df["path"].values

    def test_multiple_entities(self):
        """Multiple top-level entities each contribute their own paths."""
        data = _wrap({
            "req": [{"description": "A req."}],
            "infra": [{"description": "Infrastructure."}],
        })
        result = load_manifest.pathvalue_pairs(data)
        df = result.to_pandas()
        assert _path("req[0].description") in df["path"].values
        assert _path("infra[0].description") in df["path"].values

    def test_multiple_files_produce_prefixed_paths(self):
        """Each file's paths are prefixed with that file's identifier."""
        data = {
            "file_a.yaml": {"entity_a": [{"description": "A."}]},
            "file_b.yaml": {"entity_b": [{"description": "B."}]},
        }
        result = load_manifest.pathvalue_pairs(data)
        df = result.to_pandas()
        assert "file_a.yaml:entity_a[0].description" in df["path"].values
        assert "file_b.yaml:entity_b[0].description" in df["path"].values


# ---------------------------------------------------------------------------
# entity_id_table and component_type_table
# ---------------------------------------------------------------------------

def _pvp(pairs: list[tuple[str, str]]) -> ibis.Table:
    """Create a pathvalue_pairs ibis Table from (path, value) tuples."""
    return ibis.memtable(pd.DataFrame(pairs, columns=["path", "value"]))


# ---------------------------------------------------------------------------
# component_tables
# ---------------------------------------------------------------------------

class TestComponentTables:

    def _call(self, data: dict) -> dict:
        wrapped = {_FILE_ID: data}
        pvp = load_manifest.pathvalue_pairs(wrapped)
        kvs = load_manifest.keyvalue_store(pvp)
        return load_manifest.component_tables(kvs)

    def _eid(self, entity_key: str) -> str:
        return dhash(f"{_FILE_ID}:{entity_key}")

    def test_returns_dict_of_ibis_tables(self):
        data = {"entity": [{"description": "A thing."}]}
        result = self._call(data)
        assert isinstance(result, dict)
        for val in result.values():
            assert isinstance(val, ibis.Table)

    def test_keys_are_component_types(self):
        data = {
            "entity": [{"description": "A thing."}, "requirement"],
        }
        result = self._call(data)
        assert "description" in result
        assert "requirement" in result

    def test_scalar_component_has_value_column(self):
        """A simple key-value component produces a 'value' column."""
        data = {"entity": [{"description": "A thing."}]}
        result = self._call(data)
        df = result["description"].to_pandas()
        assert "value" in df.columns
        row = df[df["entity_id"] == self._eid("entity")].iloc[0]
        assert row["value"] == "A thing."

    def test_tag_component_has_value_column(self):
        """A bare-string tag produces a 'value' column with empty string."""
        data = {"entity": ["requirement"]}
        result = self._call(data)
        df = result["requirement"].to_pandas()
        assert "value" in df.columns
        row = df[df["entity_id"] == self._eid("entity")].iloc[0]
        assert row["value"] == ""

    def test_sub_field_component_has_field_columns(self):
        """A dict-valued component produces one column per sub-field."""
        data = {"entity": [{"field": {"name": "x", "type": "str"}}]}
        result = self._call(data)
        df = result["field"].to_pandas()
        assert "name" in df.columns
        assert "type" in df.columns
        row = df[df["entity_id"] == self._eid("entity")].iloc[0]
        assert row["name"] == "x"
        assert row["type"] == "str"

    def test_multi_field_component_collapsed_to_one_row(self):
        """Multiple sub-fields of the same component instance → one row."""
        data = {"cat": [{"field": {"name": "breed", "value": "orange", "type": "str"}}]}
        result = self._call(data)
        df = result["field"].to_pandas()
        assert len(df[df["entity_id"] == self._eid("cat")]) == 1

    def test_modifier_preserved(self):
        """Modifier from spine is carried into the component table."""
        data = {"entity": [{"solution of": "other"}]}
        result = self._call(data)
        df = result["solution"].to_pandas()
        row = df[df["entity_id"] == self._eid("entity")].iloc[0]
        assert row["modifier"] == "of"

    def test_entity_id_and_component_index_present(self):
        """Every component table has entity_id and component_index columns."""
        data = {"entity": [{"description": "A thing."}]}
        result = self._call(data)
        df = result["description"].to_pandas()
        assert "entity_id" in df.columns
        assert "component_index" in df.columns


# ---------------------------------------------------------------------------
# raw_csv_data
# ---------------------------------------------------------------------------

class TestRawCsvData:

    def test_loads_csv_from_directory(self, tmp_path):
        """CSV files in a directory are loaded and keyed by path identifier."""
        csv_file = tmp_path / "task.csv"
        csv_file.write_text("name,priority\nalpha,1\nbeta,2\n")
        result = load_manifest.raw_csv_data([str(tmp_path)])
        assert isinstance(result, dict)
        assert len(result) == 1
        file_id = next(iter(result))
        assert file_id.endswith("task.csv")
        df = result[file_id]
        assert list(df.columns) == ["name", "priority"]
        assert len(df) == 2

    def test_loads_explicit_csv_file(self, tmp_path):
        """A CSV file path passed directly is loaded."""
        csv_file = tmp_path / "req.csv"
        csv_file.write_text("id,text\n1,Requirement A\n")
        result = load_manifest.raw_csv_data([str(csv_file)])
        assert len(result) == 1
        df = next(iter(result.values()))
        assert "id" in df.columns

    def test_empty_directory_returns_empty_dict(self, tmp_path):
        """Directory with no CSV files returns an empty dict."""
        result = load_manifest.raw_csv_data([str(tmp_path)])
        assert result == {}

    def test_accepts_path_objects_directly(self, tmp_path):
        """input_dirs entries may be Path objects, not just strings."""
        (tmp_path / "task.csv").write_text("name\nalpha\n")
        result = load_manifest.raw_csv_data([tmp_path])
        assert len(result) == 1

    def test_recursive_subdirectory(self, tmp_path):
        """CSV files in subdirectories are discovered recursively."""
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "infra.csv").write_text("host,port\nserver1,80\n")
        result = load_manifest.raw_csv_data([str(tmp_path)])
        assert len(result) == 1

    def test_ignores_yaml_files(self, tmp_path):
        """YAML files in the directory are not loaded by raw_csv_data."""
        (tmp_path / "task.yaml").write_text("my_task:\n- description: A task.\n")
        (tmp_path / "data.csv").write_text("col\nval\n")
        result = load_manifest.raw_csv_data([str(tmp_path)])
        assert len(result) == 1

    def test_returns_dataframes(self, tmp_path):
        """Values are pd.DataFrame instances."""
        (tmp_path / "comp.csv").write_text("x,y\n1,2\n")
        result = load_manifest.raw_csv_data([str(tmp_path)])
        for df in result.values():
            assert isinstance(df, pd.DataFrame)


# ---------------------------------------------------------------------------
# csv_component_tables
# ---------------------------------------------------------------------------

class TestCsvComponentTables:

    def _make_raw(self, tmp_path, filename, content):
        f = tmp_path / filename
        f.write_text(content)
        return load_manifest.raw_csv_data([str(f)])

    def test_returns_dict_of_ibis_tables(self, tmp_path):
        raw = self._make_raw(tmp_path, "task.csv", "name\nalpha\n")
        result = load_manifest.csv_component_tables(raw)
        assert isinstance(result, dict)
        for val in result.values():
            assert isinstance(val, ibis.Table)

    def test_stem_plus_comp_suffix_is_key(self, tmp_path):
        raw = self._make_raw(tmp_path, "requirement.csv", "text\nReq A\n")
        result = load_manifest.csv_component_tables(raw)
        assert "requirement_comp" in result

    def test_entity_id_column_present(self, tmp_path):
        raw = self._make_raw(tmp_path, "task.csv", "name\nalpha\n")
        result = load_manifest.csv_component_tables(raw)
        df = result["task_comp"].to_pandas()
        assert "entity_id" in df.columns

    def test_entity_id_is_dhash_of_file_id(self, tmp_path):
        csv_file = tmp_path / "task.csv"
        csv_file.write_text("name\nalpha\n")
        raw = load_manifest.raw_csv_data([str(csv_file)])
        result = load_manifest.csv_component_tables(raw)
        df = result["task_comp"].to_pandas()
        file_id = next(iter(raw.keys()))
        expected_id = dhash(file_id)
        assert df.iloc[0]["entity_id"] == expected_id

    def test_all_rows_of_one_file_share_entity_id(self, tmp_path):
        """Every row in a CSV file is a component of the *same* file-level entity."""
        raw = self._make_raw(tmp_path, "task.csv", "name\nalpha\nbeta\n")
        result = load_manifest.csv_component_tables(raw)
        df = result["task_comp"].to_pandas()
        assert df["entity_id"].nunique() == 1

    def test_component_index_is_row_position(self, tmp_path):
        raw = self._make_raw(tmp_path, "task.csv", "name\nalpha\nbeta\n")
        result = load_manifest.csv_component_tables(raw)
        df = result["task_comp"].to_pandas()
        assert sorted(df["component_index"].tolist()) == [0, 1]

    def test_modifier_is_null(self, tmp_path):
        raw = self._make_raw(tmp_path, "task.csv", "name\nalpha\n")
        result = load_manifest.csv_component_tables(raw)
        df = result["task_comp"].to_pandas()
        assert df["modifier"].isna().all()

    def test_csv_columns_become_fields(self, tmp_path):
        raw = self._make_raw(tmp_path, "req.csv", "title,priority\nReq A,1\n")
        result = load_manifest.csv_component_tables(raw)
        df = result["req_comp"].to_pandas()
        assert "title" in df.columns
        assert "priority" in df.columns

    def test_multiple_files_same_stem_unioned(self, tmp_path):
        f1 = tmp_path / "sub1"
        f1.mkdir()
        f2 = tmp_path / "sub2"
        f2.mkdir()
        (f1 / "task.csv").write_text("name\nalpha\n")
        (f2 / "task.csv").write_text("name\nbeta\n")
        raw = load_manifest.raw_csv_data([str(tmp_path)])
        result = load_manifest.csv_component_tables(raw)
        df = result["task_comp"].to_pandas()
        assert len(df) == 2
        # Rows come from two different files, so they belong to two entities.
        assert df["entity_id"].nunique() == 2


# ---------------------------------------------------------------------------
# csv_spine
# ---------------------------------------------------------------------------

class TestCsvSpine:

    def _make_raw(self, tmp_path, filename, content):
        f = tmp_path / filename
        f.write_text(content)
        return load_manifest.raw_csv_data([str(f)])

    def test_returns_ibis_table(self, tmp_path):
        raw = self._make_raw(tmp_path, "task.csv", "name\nalpha\n")
        result = load_manifest.csv_spine(raw)
        assert isinstance(result, ibis.Table)

    def test_has_required_columns(self, tmp_path):
        raw = self._make_raw(tmp_path, "task.csv", "name\nalpha\n")
        result = load_manifest.csv_spine(raw)
        for col in ["entity_id", "entity_key", "filepath", "path"]:
            assert col in result.columns

    def test_one_row_per_file_not_per_csv_row(self, tmp_path):
        raw = self._make_raw(tmp_path, "req.csv", "text\nA\nB\nC\n")
        result = load_manifest.csv_spine(raw)
        assert len(result.to_pandas()) == 1

    def test_entity_id_uses_dhash_of_file_id(self, tmp_path):
        csv_file = tmp_path / "task.csv"
        csv_file.write_text("name\nalpha\n")
        raw = load_manifest.raw_csv_data([str(csv_file)])
        result = load_manifest.csv_spine(raw)
        df = result.to_pandas()
        file_id = next(iter(raw.keys()))
        expected_id = dhash(file_id)
        assert df.iloc[0]["entity_id"] == expected_id

    def test_entity_key_is_stem(self, tmp_path):
        raw = self._make_raw(tmp_path, "requirement.csv", "text\nReq A\n")
        result = load_manifest.csv_spine(raw)
        df = result.to_pandas()
        assert df.iloc[0]["entity_key"] == "requirement"

    def test_path_format(self, tmp_path):
        csv_file = tmp_path / "task.csv"
        csv_file.write_text("name\nalpha\n")
        raw = load_manifest.raw_csv_data([str(csv_file)])
        result = load_manifest.csv_spine(raw)
        df = result.to_pandas()
        file_id = next(iter(raw.keys()))
        expected_path = f"{file_id}:task"
        assert df.iloc[0]["path"] == expected_path

    def test_filepath_is_file_id(self, tmp_path):
        csv_file = tmp_path / "task.csv"
        csv_file.write_text("name\nalpha\n")
        raw = load_manifest.raw_csv_data([str(csv_file)])
        result = load_manifest.csv_spine(raw)
        df = result.to_pandas()
        file_id = next(iter(raw.keys()))
        assert df.iloc[0]["filepath"] == file_id


# ---------------------------------------------------------------------------
# component_tables union with csv_component_tables
# ---------------------------------------------------------------------------

class TestComponentTablesWithCsv:

    def test_csv_only_type_included(self, tmp_path):
        """CSV-only component types (named "{stem}_comp") appear in the result."""
        csv_file = tmp_path / "task.csv"
        csv_file.write_text("name\nalpha\n")
        raw = load_manifest.raw_csv_data([str(csv_file)])
        csv_ct = load_manifest.csv_component_tables(raw)

        pairs = [("file.yaml:entity[0].description", "Hi.")]
        pvp = ibis.memtable(pd.DataFrame(pairs, columns=["path", "value"]))
        kvs = load_manifest.keyvalue_store(pvp)
        result = load_manifest.component_tables(kvs, csv_ct)
        assert "description" in result
        assert "task_comp" in result

    def test_shared_type_rows_merged(self, tmp_path):
        """When YAML and CSV component types coincide (both "description_comp"), rows are concatenated."""
        csv_file = tmp_path / "description.csv"
        csv_file.write_text("value\nCSV description\n")
        raw = load_manifest.raw_csv_data([str(csv_file)])
        csv_ct = load_manifest.csv_component_tables(raw)
        assert "description_comp" in csv_ct

        pairs = [("file.yaml:entity[0].description_comp", "YAML description.")]
        pvp = ibis.memtable(pd.DataFrame(pairs, columns=["path", "value"]))
        kvs = load_manifest.keyvalue_store(pvp)
        result = load_manifest.component_tables(kvs, csv_ct)
        df = result["description_comp"].to_pandas()
        assert len(df) == 2

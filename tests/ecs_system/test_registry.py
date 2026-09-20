import ibis
import pandas as pd
import pytest

from emc2p.registry import GetterResult, Registry


class TestRegistryInitialization:
    """Tests for initializing a Registry with a connection and components."""

    def test_init_with_single_component(self):
        """Registry can be initialized with a single component table."""
        conn = ibis.duckdb.connect()
        conn.create_table(
            "description",
            {"entity_id": ["iacs", "registry"], "value": ["A tool for architects", "Stores ECS data"]},
        )
        components = {"description": conn.table("description")}

        registry = Registry(conn, components)

        assert "description" in registry.component_types
        assert "description" in registry._con.list_tables()

    def test_init_with_multiple_components(self):
        """Registry can be initialized with multiple component tables."""
        conn = ibis.duckdb.connect()
        conn.create_table(
            "description",
            {"entity_id": ["iacs"], "value": ["A tool for architects"]},
        )
        conn.create_table(
            "requirement",
            {"entity_id": ["iacs"], "type": ["functional"], "value": [1.0]},
        )
        components = {
            "description": conn.table("description"),
            "requirement": conn.table("requirement"),
        }

        registry = Registry(conn, components)

        assert "description" in registry.component_types
        assert "requirement" in registry.component_types

    def test_init_with_empty_dict_creates_empty_registry(self):
        """Registry can be initialized with an empty dict."""
        conn = ibis.duckdb.connect()
        registry = Registry(conn, {})

        assert len(registry.component_types) == 0

    def test_non_table_components_excluded_from_component_types(self):
        """Components that are raw lists (not ibis Tables) should be excluded from component_types."""
        conn = ibis.duckdb.connect()
        conn.create_table(
            "description",
            {"entity_id": ["e1"], "value": ["Hello"]},
        )
        components = {
            "description": conn.table("description"),
            "schema_comp": [{"entity_id": "description", "columns": {"value": {"type": "str"}}}],
        }

        registry = Registry(conn, components)

        assert "description" in registry.component_types
        assert "schema_comp" not in registry.component_types

    def test_schema_key_excluded_from_component_types(self):
        """The 'schema' key in components is not treated as a component type."""
        conn = ibis.duckdb.connect()
        conn.create_table(
            "description",
            {"entity_id": ["e1"], "value": ["Hello"]},
        )
        components = {
            "description": conn.table("description"),
            "schema": {"description": object},
        }

        registry = Registry(conn, components)

        assert "description" in registry.component_types
        assert "schema" not in registry.component_types


class TestRegistryView:
    """Tests for viewing components in the Registry."""

    @pytest.fixture
    def sample_registry(self):
        """Create a registry with sample data for testing."""
        conn = ibis.duckdb.connect()
        conn.create_table(
            "entity_id",
            {"value": ["iacs", "registry"], "display_alias": ["iacs", "registry"],
             "path": ["test:iacs", "test:registry"], "display_key": ["iacs", "registry"],
             "filepath": ["test", "test"]},
        )
        conn.create_table(
            "description",
            {"entity_id": ["iacs", "registry"], "value": ["A tool for architects", "Stores ECS data"]},
        )
        conn.create_table(
            "requirement",
            {"entity_id": ["iacs", "iacs"], "type": ["functional", "quality"], "value": [1.0, 0.5]},
        )
        components = {
            "entity_id": conn.table("entity_id"),
            "description": conn.table("description"),
            "requirement": conn.table("requirement"),
        }
        return Registry(conn, components)

    def test_view_returns_getter_result(self, sample_registry):
        """view() returns a GetterResult wrapping an ibis Table."""
        result = sample_registry.view("description")
        assert isinstance(result, GetterResult)
        assert isinstance(result.to_table(), ibis.Table)

    def test_view_returns_correct_data(self, sample_registry):
        """view() returns the correct data for the component."""
        result = sample_registry.view("requirement").to_pandas()

        assert "requirement.value" in result.columns
        assert "requirement.type" in result.columns

    def test_view_nonexistent_component_raises_keyerror(self, sample_registry):
        """view() raises KeyError for a component type that doesn't exist."""
        with pytest.raises(KeyError):
            sample_registry.view("nonexistent")

    def test_view_returns_copy_not_reference(self, sample_registry):
        """view().to_pandas() returns a copy to prevent accidental modification."""
        result = sample_registry.view("description").to_pandas().set_index("entity_id")
        result.loc["iacs", "description.value"] = "Modified"

        original = sample_registry.view("description").to_pandas().set_index("entity_id")
        assert original.loc["iacs", "description.value"] == "A tool for architects"


class TestRegistryViewAliases:
    """Tests for the `aliases` filter on view()/view_current()."""

    @pytest.fixture
    def sample_registry(self):
        """Same shape as TestRegistryView's fixture of the same name.

        Duplicated locally since pytest doesn't share method-scoped
        fixtures across classes.
        """
        conn = ibis.duckdb.connect()
        conn.create_table(
            "entity_id",
            {"value": ["iacs", "registry"], "display_alias": ["iacs", "registry"],
             "path": ["test:iacs", "test:registry"], "display_key": ["iacs", "registry"],
             "filepath": ["test", "test"]},
        )
        conn.create_table(
            "description",
            {"entity_id": ["iacs", "registry"], "value": ["A tool for architects", "Stores ECS data"]},
        )
        conn.create_table(
            "field",
            pd.DataFrame({"entity_id": [], "value": [], "time_dimension": []}),
        )
        components = {
            "entity_id": conn.table("entity_id"),
            "description": conn.table("description"),
            "field": conn.table("field"),
        }
        return Registry(conn, components)

    @pytest.fixture
    def registry_with_ambiguous_alias(self):
        """Two entities whose paths both contain "shared", but neither is
        aliased or hashed to exactly "shared".

        Resolving "shared" can only succeed via the substring fallback,
        and matches both.
        """
        conn = ibis.duckdb.connect()
        conn.create_table(
            "entity_id",
            {
                "value": ["e1", "e2", "e3"],
                "display_alias": ["shared_a", "shared_b", "unrelated"],
                "path": ["test:shared_a", "test:shared_b", "test:unrelated"],
                "display_key": ["shared_a", "shared_b", "unrelated"],
                "filepath": ["test", "test", "test"],
            },
        )
        conn.create_table(
            "description",
            {"entity_id": ["e1", "e2", "e3"], "value": ["First", "Second", "Third"]},
        )
        components = {
            "entity_id": conn.table("entity_id"),
            "description": conn.table("description"),
        }
        return Registry(conn, components)

    def test_aliases_filters_to_exact_match(self, sample_registry):
        result = sample_registry.view("description", "iacs").to_pandas().set_index("entity_id")
        assert list(result.index) == ["iacs"]

    def test_aliases_accepts_a_list(self, sample_registry):
        result = sample_registry.view("description", ["iacs", "registry"]).to_pandas().set_index("entity_id")
        assert set(result.index) == {"iacs", "registry"}

    def test_aliases_multi_match_includes_all_matches(self, registry_with_ambiguous_alias):
        result = registry_with_ambiguous_alias.view("description", "shared").to_pandas().set_index("entity_id")
        assert set(result.index) == {"e1", "e2"}

    def test_aliases_multi_match_warns(self, registry_with_ambiguous_alias):
        with pytest.warns(UserWarning, match="shared"):
            registry_with_ambiguous_alias.view("description", "shared").to_pandas()

    def test_aliases_zero_match_excludes_everything(self, sample_registry):
        with pytest.warns(UserWarning, match="nonexistent"):
            result = sample_registry.view("description", "nonexistent").to_pandas()
        assert len(result) == 0

    def test_aliases_none_returns_everything(self, sample_registry):
        result = sample_registry.view("description").to_pandas().set_index("entity_id")
        assert set(result.index) == {"iacs", "registry"}

    def test_aliases_applies_to_view_current_too(self, sample_registry):
        result = sample_registry.view_current("description", "iacs").to_pandas()
        assert list(result["entity_id"]) == ["iacs"]


class TestRegistryViewMultipleComponents:
    """Tests for viewing multiple components joined by entity_id."""

    @pytest.fixture
    def multi_component_registry(self):
        """Create a registry with entities having multiple component types."""
        conn = ibis.duckdb.connect()
        conn.create_table(
            "entity_id",
            {"value": ["a", "b", "c"], "display_alias": ["a", "b", "c"],
             "path": ["test:a", "test:b", "test:c"], "display_key": ["a", "b", "c"],
             "filepath": ["test", "test", "test"]},
        )
        conn.create_table(
            "description",
            {"entity_id": ["a", "b", "c"], "value": ["Desc A", "Desc B", "Desc C"]},
        )
        conn.create_table(
            "requirement",
            {"entity_id": ["a", "b"], "value": [1.0, 0.0]},
        )
        components = {
            "entity_id": conn.table("entity_id"),
            "description": conn.table("description"),
            "requirement": conn.table("requirement"),
        }
        return Registry(conn, components)

    def test_view_multiple_components_returns_dataframe(self, multi_component_registry):
        """view() with list of components returns a DataFrame via to_pandas()."""
        result = multi_component_registry.view(["description", "requirement"]).to_pandas()
        assert isinstance(result, pd.DataFrame)

    def test_view_multiple_components_inner_joins_by_entity_id(self, multi_component_registry):
        """view() with list of components inner joins by entity_id."""
        result = multi_component_registry.view(["description", "requirement"]).to_pandas().set_index("entity_id")

        # entity c has no requirement, so it should be excluded
        entity_ids = result.index.unique()
        assert "a" in entity_ids
        assert "b" in entity_ids
        assert "c" not in entity_ids

    def test_view_specific_fields(self, multi_component_registry):

        result = multi_component_registry.view(
            ["description.value", "requirement.value"]
        ).to_pandas().set_index("entity_id")
        assert result.loc["b", "description.value"] == "Desc B"
        assert result.loc["b", "requirement.value"] == 0.0

    def test_view_single_component_as_list_works(self, multi_component_registry):
        """view() with single-element list works like single string."""
        result = multi_component_registry.view(["description"]).to_pandas()
        assert isinstance(result, pd.DataFrame)
        assert "description.value" in result.columns


class TestRegistryViewDoesNotCrossJoinSameTable:
    """A component type with multiple rows per entity_id must not self-cross-join
    when several of its fields are requested together."""

    @pytest.fixture
    def multi_row_registry(self):
        conn = ibis.duckdb.connect()
        conn.create_table(
            "entity_id",
            {"value": ["a"], "display_alias": ["a"], "path": ["test:a"],
             "display_key": ["a"], "filepath": ["test"]},
        )
        conn.create_table(
            "reading",
            {"entity_id": ["a", "a"], "component_index": [0, 1],
             "modifier": pd.array([None, None], dtype=pd.StringDtype()),
             "as_of": ["2024-01-01", "2024-06-01"],
             "status": ["open", "closed"]},
        )
        components = {
            "entity_id": conn.table("entity_id"),
            "reading": conn.table("reading"),
        }
        return Registry(conn, components)

    def test_bare_component_type_keeps_rows_correlated(self, multi_row_registry):
        df = multi_row_registry.view("reading").to_pandas()
        assert len(df) == 2
        pairs = set(zip(df["reading.as_of"], df["reading.status"]))
        assert pairs == {("2024-01-01", "open"), ("2024-06-01", "closed")}

    def test_explicit_dotted_fields_keep_rows_correlated(self, multi_row_registry):
        df = multi_row_registry.view(["reading.as_of", "reading.status"]).to_pandas()
        assert len(df) == 2
        pairs = set(zip(df["reading.as_of"], df["reading.status"]))
        assert pairs == {("2024-01-01", "open"), ("2024-06-01", "closed")}


class TestRegistryViewCurrent:
    """Tests for view_current(), which collapses slowly changing dimensions."""

    @pytest.fixture
    def scd_registry(self):
        """A registry with a "status_reading" type whose "as_of" field is time_dimension."""
        conn = ibis.duckdb.connect()
        conn.create_table(
            "entity_id",
            {"value": ["def1", "e1", "e2"], "display_alias": ["status_reading", "e1", "e2"],
             "path": ["test:status_reading", "test:e1", "test:e2"],
             "display_key": ["status_reading", "e1", "e2"], "filepath": ["test", "test", "test"]},
        )
        conn.create_table(
            "field",
            {"entity_id": ["def1", "def1"], "value": ["as_of", "status"],
             "time_dimension": [True, False]},
        )
        conn.create_table(
            "status_reading",
            {"entity_id": ["e1", "e1", "e2"],
             # e1's two rows deliberately differ in component_index: SCD
             # history commonly comes from separate writes each computing it fresh.
             "component_index": [0, 1, 0],
             "modifier": pd.array([None, None, None], dtype=pd.StringDtype()),
             "as_of": ["2024-01-01", "2024-06-01", "2024-03-01"],
             "status": ["open", "closed", "open"]},
        )
        components = {
            "entity_id": conn.table("entity_id"),
            "field": conn.table("field"),
            "status_reading": conn.table("status_reading"),
        }
        return Registry(conn, components)

    def test_time_dimension_field_detected(self, scd_registry):
        assert scd_registry._time_dimension_field("status_reading") == "as_of"

    def test_non_time_dimension_component_has_no_field(self, scd_registry):
        assert scd_registry._time_dimension_field("entity_id") is None

    def test_multiple_time_dimension_fields_raises(self, scd_registry):
        """Only one time_dimension field is allowed per component type."""
        conn = scd_registry._con
        conn.create_table(
            "field",
            {"entity_id": ["def1", "def1"], "value": ["as_of", "also_as_of"],
             "time_dimension": [True, True]},
            overwrite=True,
        )
        scd_registry._components["field"] = conn.table("field")
        with pytest.raises(ValueError, match="status_reading"):
            scd_registry._time_dimension_field("status_reading")

    def test_view_current_keeps_one_row_per_entity(self, scd_registry):
        df = scd_registry.view_current("status_reading").to_pandas()
        assert sorted(df["entity_id"]) == ["e1", "e2"]

    def test_view_current_picks_max_time_dimension(self, scd_registry):
        df = scd_registry.view_current("status_reading").to_pandas()
        e1_row = df[df["entity_id"] == "e1"].iloc[0]
        assert e1_row["status_reading.as_of"] == "2024-06-01"
        assert e1_row["status_reading.status"] == "closed"

    def test_view_unchanged_still_returns_all_versions(self, scd_registry):
        """view() (unlike view_current()) does not collapse SCD history."""
        df = scd_registry.view("status_reading").to_pandas()
        as_of_values = set(df.loc[df["entity_id"] == "e1", "status_reading.as_of"])
        assert as_of_values == {"2024-01-01", "2024-06-01"}

    def test_view_current_nonexistent_component_raises_keyerror(self, scd_registry):
        with pytest.raises(KeyError):
            scd_registry.view_current("nonexistent")

    def test_seq_column_hidden_from_view_current(self, scd_registry):
        """A `_seq_{field}` column, if present, doesn't leak into view/view_current output."""
        conn = scd_registry._con
        conn.create_table(
            "status_reading",
            {"entity_id": ["e1", "e1", "e2"],
             "component_index": [0, 1, 0],
             "modifier": pd.array([None, None, None], dtype=pd.StringDtype()),
             "as_of": ["2024-01-01", "2024-06-01", "2024-03-01"],
             "status": ["open", "closed", "open"],
             "_seq_as_of": [1, 2, 1]},
            overwrite=True,
        )
        scd_registry._components["status_reading"] = conn.table("status_reading")
        assert "status_reading._seq_as_of" not in scd_registry.view_current("status_reading").to_table().columns
        assert "status_reading._seq_as_of" not in scd_registry.view("status_reading").to_table().columns

    def test_seq_column_breaks_ties_most_recent_wins(self, scd_registry):
        """Two rows tied on the time_dimension value resolve via `_seq_{field}`."""
        conn = scd_registry._con
        conn.create_table(
            "status_reading",
            {"entity_id": ["e1", "e1", "e2"],
             "component_index": [0, 1, 0],
             "modifier": pd.array([None, None, None], dtype=pd.StringDtype()),
             # e1's two rows now tie on as_of -- only _seq_as_of tells them apart.
             "as_of": ["2024-06-01", "2024-06-01", "2024-03-01"],
             "status": ["stale", "fresh", "open"],
             "_seq_as_of": [1, 2, 1]},
            overwrite=True,
        )
        scd_registry._components["status_reading"] = conn.table("status_reading")
        df = scd_registry.view_current("status_reading").to_pandas()
        e1_row = df[df["entity_id"] == "e1"].iloc[0]
        assert e1_row["status_reading.status"] == "fresh"

    def test_seq_column_falls_back_to_self_when_other_lacks_schema(self, scd_registry):
        """A later merge whose own batch doesn't redeclare status_reading's
        schema (e.g. a per-turn write that only reloads its own incremental
        data, not the full manifest) must still get a `_seq_{field}` column
        -- falling back to `self`'s (the accumulated registry's) own
        already-known schema, the same fallback time_filled_registry uses
        for backfilling values."""
        conn = ibis.duckdb.connect()
        conn.create_table("entity_id", {"value": ["e3"], "display_alias": ["e3"], "path": ["test:e3"],
                                         "display_key": ["e3"], "filepath": ["test"]})
        conn.create_table("status_reading", {"entity_id": ["e3"], "component_index": [0],
                                              "modifier": pd.array([None], dtype=pd.StringDtype()),
                                              "as_of": ["2024-09-01"], "status": ["new"]})
        other = Registry(conn, {
            "entity_id": conn.table("entity_id"),
            "status_reading": conn.table("status_reading"),
        })
        assert scd_registry._seq_column("status_reading", other) == "_seq_as_of"

    def test_no_seq_column_falls_back_gracefully(self, scd_registry):
        """A tie with no `_seq_{field}` column at all still resolves to exactly one row."""
        conn = scd_registry._con
        conn.create_table(
            "status_reading",
            {"entity_id": ["e1", "e1", "e2"],
             "component_index": [0, 1, 0],
             "modifier": pd.array([None, None, None], dtype=pd.StringDtype()),
             "as_of": ["2024-06-01", "2024-06-01", "2024-03-01"],
             "status": ["stale", "fresh", "open"]},
            overwrite=True,
        )
        scd_registry._components["status_reading"] = conn.table("status_reading")
        df = scd_registry.view_current("status_reading").to_pandas()
        assert len(df[df["entity_id"] == "e1"]) == 1

    def test_view_current_component_without_time_dimension_is_unchanged(self):
        """Component types with no time_dimension field are returned as-is."""
        conn = ibis.duckdb.connect()
        conn.create_table(
            "entity_id",
            {"value": ["e1"], "display_alias": ["e1"], "path": ["test:e1"],
             "display_key": ["e1"], "filepath": ["test"]},
        )
        # No "time_dimension" column at all — no field anywhere sets it.
        conn.create_table(
            "field",
            {"entity_id": ["e1"], "value": ["value"]},
        )
        conn.create_table(
            "description",
            {"entity_id": ["e1", "e1"], "component_index": [0, 1],
             "modifier": pd.array([None, None], dtype=pd.StringDtype()),
             "value": ["First", "Second"]},
        )
        registry = Registry(conn, {
            "entity_id": conn.table("entity_id"),
            "field": conn.table("field"),
            "description": conn.table("description"),
        })
        df = registry.view_current("description").to_pandas()
        assert len(df) == 2


class TestRegistrySafeView:
    """Tests for safe_view()/safe_view_current(), the KeyError-safe view wrappers."""

    @pytest.fixture
    def scd_registry(self):
        """A registry with a "status_reading" type whose "as_of" field is time_dimension."""
        conn = ibis.duckdb.connect()
        conn.create_table(
            "entity_id",
            {"value": ["def1", "e1"], "display_alias": ["status_reading", "e1"],
             "path": ["test:status_reading", "test:e1"],
             "display_key": ["status_reading", "e1"], "filepath": ["test", "test"]},
        )
        conn.create_table(
            "field",
            {"entity_id": ["def1", "def1"], "value": ["as_of", "status"],
             "time_dimension": [True, False]},
        )
        conn.create_table(
            "status_reading",
            {"entity_id": ["e1"], "component_index": [0],
             "modifier": pd.array([None], dtype=pd.StringDtype()),
             "as_of": ["2024-01-01"], "status": ["open"]},
        )
        components = {
            "entity_id": conn.table("entity_id"),
            "field": conn.table("field"),
            "status_reading": conn.table("status_reading"),
        }
        return Registry(conn, components)

    def test_safe_view_returns_empty_getter_result_for_unknown_component_type(self, scd_registry):
        result = scd_registry.safe_view("nonexistent")
        assert isinstance(result, GetterResult)
        assert result.to_pandas().empty

    def test_safe_view_current_returns_empty_getter_result_for_unknown_component_type(self, scd_registry):
        result = scd_registry.safe_view_current("nonexistent")
        assert isinstance(result, GetterResult)
        assert result.to_pandas().empty

    def test_safe_view_matches_view_for_known_component_type(self, scd_registry):
        df = scd_registry.safe_view("status_reading").to_pandas()
        assert sorted(df["entity_id"]) == ["e1"]

    def test_safe_view_current_matches_view_current_for_known_component_type(self, scd_registry):
        df = scd_registry.safe_view_current("status_reading").to_pandas()
        assert df.iloc[0]["status_reading.status"] == "open"

    def test_safe_view_returns_empty_dataframe_not_error_for_declared_but_dataless_type(self, scd_registry):
        schema = ibis.schema(
            {"entity_id": "string", "component_index": "int64", "modifier": "string", "value": "string"}
        )
        scd_registry.declare_schema("empty_type", schema)
        df = scd_registry.safe_view("empty_type").to_pandas()
        assert df.empty


class TestRegistryCurrentValueViaToScalar:
    """The old get_current_value's job, now view_current(...).to_scalar()."""

    @pytest.fixture
    def scd_registry(self):
        """A registry with a "status_reading" type whose "as_of" field is time_dimension."""
        conn = ibis.duckdb.connect()
        conn.create_table(
            "entity_id",
            {"value": ["def1", "e1", "e2"], "display_alias": ["status_reading", "e1", "e2"],
             "path": ["test:status_reading", "test:e1", "test:e2"],
             "display_key": ["status_reading", "e1", "e2"], "filepath": ["test", "test", "test"]},
        )
        conn.create_table(
            "field",
            {"entity_id": ["def1", "def1"], "value": ["as_of", "status"],
             "time_dimension": [True, False]},
        )
        conn.create_table(
            "status_reading",
            {"entity_id": ["e1", "e1", "e2"],
             "component_index": [0, 1, 0],
             "modifier": pd.array([None, None, None], dtype=pd.StringDtype()),
             "as_of": ["2024-01-01", "2024-06-01", "2024-03-01"],
             "status": ["open", "closed", "open"]},
        )
        components = {
            "entity_id": conn.table("entity_id"),
            "field": conn.table("field"),
            "status_reading": conn.table("status_reading"),
        }
        return Registry(conn, components)

    def test_returns_the_current_value(self, scd_registry):
        assert scd_registry.view_current("status_reading.status", "e1").to_scalar() == "closed"

    def test_nonexistent_component_type_raises_keyerror(self, scd_registry):
        with pytest.raises(KeyError):
            scd_registry.view_current("nonexistent.status", "e1")

    def test_alias_matching_nothing_returns_none(self, scd_registry):
        with pytest.warns(UserWarning, match="no_such_alias"):
            result = scd_registry.view_current("status_reading.status", "no_such_alias").to_scalar()
        assert result is None

    def test_declared_but_dataless_component_type_returns_none(self, scd_registry):
        schema = ibis.schema(
            {"entity_id": "string", "component_index": "int64", "modifier": "string", "value": "string"}
        )
        scd_registry.declare_schema("empty_type", schema)
        assert scd_registry.view_current("empty_type.value", "e1").to_scalar() is None


class TestRegistryDatabaseRoundTrip:
    """Tests for exporting/loading a Registry to/from a database via ibis.connect."""

    @pytest.fixture
    def sample_registry(self):
        conn = ibis.duckdb.connect()
        conn.create_table(
            "description",
            {"entity_id": ["iacs", "registry"], "value": ["A tool for architects", "Stores ECS data"]},
        )
        conn.create_table(
            "requirement",
            {"entity_id": ["iacs"], "type": ["functional"], "value": [1.0]},
        )
        components = {
            "description": conn.table("description"),
            "requirement": conn.table("requirement"),
        }
        return Registry(conn, components)

    def test_to_database_creates_duckdb_file(self, sample_registry, tmp_path):
        db_path = tmp_path / "registry.duckdb"
        sample_registry.to_database(db_path)
        assert db_path.exists()

    def test_from_database_recovers_component_types(self, sample_registry, tmp_path):
        db_path = tmp_path / "registry.duckdb"
        sample_registry.to_database(db_path)

        loaded = Registry.from_database(db_path)

        assert set(loaded.component_types) == set(sample_registry.component_types)

    def test_from_database_recovers_data(self, sample_registry, tmp_path):
        db_path = tmp_path / "registry.duckdb"
        sample_registry.to_database(db_path)

        loaded = Registry.from_database(db_path)

        pd.testing.assert_frame_equal(
            loaded.get("description").execute().sort_values("entity_id").reset_index(drop=True),
            sample_registry.get("description").execute().sort_values("entity_id").reset_index(drop=True),
        )


class TestRegistryFromComponentRows:
    """Tests for building a Registry directly from component-first row data."""

    def test_builds_a_registry_with_the_given_component_tables(self):
        registry = Registry.from_component_rows({
            "description": [{"entity_id": "e1", "value": "a thing"}],
        })
        df = registry.get("description").to_pandas()
        assert df["value"].tolist() == ["a thing"]

    def test_supports_multiple_component_types(self):
        registry = Registry.from_component_rows({
            "description": [{"entity_id": "e1", "value": "a thing"}],
            "requirement": [{"entity_id": "e1"}],
        })
        assert set(registry.component_types) >= {"description", "requirement"}

    def test_accepts_a_non_duckdb_backend(self):
        """Not DuckDB-specific -- any already-connected ibis backend works via `conn`."""
        registry = Registry.from_component_rows(
            {"description": [{"entity_id": "e1", "value": "a thing"}]},
            conn=ibis.sqlite.connect(),
        )
        df = registry.get("description").to_pandas()
        assert df["value"].tolist() == ["a thing"]


class TestRegistryDeclareSchema:
    """Tests for declare_schema and the get/view/view_current fallback it enables."""

    @pytest.fixture
    def sample_registry(self):
        conn = ibis.duckdb.connect()
        conn.create_table(
            "entity_id",
            {"value": ["e1"], "display_alias": ["e1"], "path": ["test:e1"], "display_key": ["e1"], "filepath": ["test"]},
        )
        # An empty (but correctly-columned) "field" table, so
        # _time_dimension_field finds no time_dimension field, not a missing key.
        conn.create_table(
            "field",
            pd.DataFrame(columns=["entity_id", "value", "time_dimension"]).astype(
                {"entity_id": "string", "value": "string", "time_dimension": "boolean"}
            ),
        )
        components = {"entity_id": conn.table("entity_id"), "field": conn.table("field")}
        return Registry(conn, components)

    def test_get_returns_declared_schema_when_no_physical_table(self, sample_registry):
        schema = ibis.schema({"entity_id": "string", "component_index": "int64", "modifier": "string", "x": "float64"})
        sample_registry.declare_schema("position", schema)
        result = sample_registry.get("position").execute()
        assert result.empty
        assert set(result.columns) == {"entity_id", "component_index", "modifier", "x"}

    def test_view_returns_empty_result_when_no_physical_table(self, sample_registry):
        schema = ibis.schema({"entity_id": "string", "component_index": "int64", "modifier": "string", "x": "float64"})
        sample_registry.declare_schema("position", schema)
        result = sample_registry.view("position.x").to_pandas()
        assert result.empty
        assert "position.x" in result.columns

    def test_view_current_returns_empty_result_when_no_physical_table(self, sample_registry):
        schema = ibis.schema({"entity_id": "string", "component_index": "int64", "modifier": "string", "x": "float64"})
        sample_registry.declare_schema("position", schema)
        result = sample_registry.view_current("position.x").to_pandas()
        assert result.empty

    def test_view_still_raises_keyerror_for_undeclared_type(self, sample_registry):
        with pytest.raises(KeyError):
            sample_registry.view("nonexistent")

    def test_known_component_types_includes_declared_but_dataless_type(self, sample_registry):
        """component_types alone can't distinguish "this type doesn't exist"
        from "this type exists but nobody's used it yet" -- known_component_types can."""
        schema = ibis.schema({"entity_id": "string", "component_index": "int64", "modifier": "string", "x": "float64"})
        sample_registry.declare_schema("position", schema)
        assert "position" not in sample_registry.component_types
        assert "position" in sample_registry.known_component_types

    def test_known_component_types_includes_data_bearing_types(self, sample_registry):
        assert "entity_id" in sample_registry.known_component_types
        assert "field" in sample_registry.known_component_types

    def test_declare_schema_is_noop_when_physical_table_exists(self, sample_registry):
        """Real data's own schema always wins over a merely declared one."""
        conn = sample_registry._con
        conn.create_table("position", {"entity_id": ["e1"], "x": [1.0]})
        sample_registry.update({"position": conn.table("position")})
        original_schema = sample_registry.get("position").schema()

        sample_registry.declare_schema("position", ibis.schema({"entity_id": "string", "y": "float64"}))

        assert sample_registry.get("position").schema() == original_schema

    def test_merge_propagates_declared_schemas(self, sample_registry):
        """A schema declared on the source registry is still known on self
        after merge, even though it has no physical table on either side."""
        other_conn = ibis.duckdb.connect()
        other_conn.create_table(
            "entity_id",
            {"value": ["e2"], "display_alias": ["e2"], "path": ["test:e2"], "display_key": ["e2"], "filepath": ["test"]},
        )
        other = Registry(other_conn, {"entity_id": other_conn.table("entity_id")})
        schema = ibis.schema({"entity_id": "string", "component_index": "int64", "modifier": "string", "x": "float64"})
        other.declare_schema("position", schema)

        sample_registry.merge(other)

        result = sample_registry.get("position").execute()
        assert result.empty
        assert set(result.columns) == {"entity_id", "component_index", "modifier", "x"}


class TestRegistryGetEntityId:
    """Tests for resolving an entity ref to its canonical entity_id hash."""

    @pytest.fixture
    def registry(self):
        conn = ibis.duckdb.connect()
        conn.create_table(
            "entity_id",
            {
                "value": [
                    "aaa111aaa111",
                    "bbb222bbb222",
                    "ccc333ccc333",
                    "fff666fff666",
                    "ddd444ddd444",
                    "eee555eee555",
                ],
                "display_alias": [
                    "water_cats",
                    "feeding_system",
                    "feeding_system.feed_cats",
                    "feeding_system.feed_cats.sub",
                    "dup_alias",
                    "dup_alias",
                ],
                "path": [
                    "examples/example.yaml:feeding_system.water_cats",
                    "examples/example.yaml:feeding_system",
                    "examples/example.yaml:feeding_system.feed_cats",
                    "examples/example.yaml:feeding_system.feed_cats.sub",
                    "examples/example.yaml:zzz.dup_alias_one",
                    "examples/example.yaml:zzz.dup_alias_two",
                ],
                "display_key": [
                    "water_cats",
                    "feeding_system",
                    "feed_cats",
                    "sub",
                    "dup_alias_one",
                    "dup_alias_two",
                ],
                "filepath": ["examples/example.yaml"] * 6,
            },
        )
        return Registry(conn, {"entity_id": conn.table("entity_id")})

    def test_resolves_exact_alias_to_hash(self, registry):
        assert registry.get_entity_id("water_cats") == "aaa111aaa111"

    def test_returns_exact_hash_unchanged(self, registry):
        assert registry.get_entity_id("aaa111aaa111") == "aaa111aaa111"

    def test_resolves_unambiguous_path_fragment(self, registry):
        """A fragment that isn't anyone's exact alias but substring-matches
        exactly one entity's path still resolves."""
        assert registry.get_entity_id("system.water_cats") == "aaa111aaa111"

    def test_returns_none_when_no_match(self, registry):
        assert registry.get_entity_id("nonexistent") is None

    def test_container_alias_resolves_despite_being_path_prefix_of_children(self, registry):
        """A container's own alias is a substring of its children's paths
        too ("feeding_system" vs "feeding_system.feed_cats", ...).

        The exact-alias match must still resolve it unambiguously rather
        than reporting it as ambiguous with its own descendants.
        """
        assert registry.get_entity_id("feeding_system") == "bbb222bbb222"

    def test_returns_none_when_substring_matches_multiple_with_no_exact_alias_hit(self, registry):
        """"feed_cats" is nobody's exact alias, but is a substring of both
        "feeding_system.feed_cats" and "feeding_system.feed_cats.sub"."""
        assert registry.get_entity_id("feed_cats") is None

    def test_returns_none_when_alias_itself_is_ambiguous(self, registry):
        """Two entities sharing the same computed alias resolve to neither."""
        assert registry.get_entity_id("dup_alias") is None


class TestGetterResult:
    """Unit tests for GetterResult's extraction methods, independent of Registry."""

    def _table(self, rows: list[dict]) -> ibis.Table:
        conn = ibis.duckdb.connect()
        if not rows:
            return conn.create_table("t", schema={"entity_id": "string", "x": "string"})
        return conn.create_table("t", pd.DataFrame(rows))

    def test_to_table_returns_the_underlying_ibis_table(self):
        table = self._table([{"entity_id": "e1", "x": "a"}])
        assert GetterResult(table).to_table() is table

    def test_to_pandas_executes_to_a_dataframe(self):
        table = self._table([{"entity_id": "e1", "x": "a"}, {"entity_id": "e2", "x": "b"}])
        df = GetterResult(table).to_pandas()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_to_dict_on_zero_rows_returns_empty_dict(self):
        assert GetterResult(self._table([])).to_dict() == {}

    def test_to_dict_on_one_row_returns_that_row(self):
        table = self._table([{"entity_id": "e1", "x": "a"}])
        assert GetterResult(table).to_dict() == {"entity_id": "e1", "x": "a"}

    def test_to_dict_on_multiple_rows_raises(self):
        table = self._table([{"entity_id": "e1", "x": "a"}, {"entity_id": "e2", "x": "b"}])
        with pytest.raises(ValueError, match="exactly one row"):
            GetterResult(table).to_dict()

    def test_to_scalar_on_zero_rows_returns_none(self):
        assert GetterResult(self._table([])).to_scalar() is None

    def test_to_scalar_on_one_row_one_data_column_returns_the_value(self):
        table = self._table([{"entity_id": "e1", "x": "a"}])
        assert GetterResult(table).to_scalar() == "a"

    def test_to_scalar_on_multiple_rows_raises(self):
        table = self._table([{"entity_id": "e1", "x": "a"}, {"entity_id": "e2", "x": "b"}])
        with pytest.raises(ValueError, match="exactly one row"):
            GetterResult(table).to_scalar()

    def test_to_scalar_on_multiple_data_columns_raises(self):
        table = self._table([{"entity_id": "e1", "x": "a", "y": "b"}])
        with pytest.raises(ValueError, match="exactly one data column"):
            GetterResult(table).to_scalar()


class TestRegistryViewEntities:
    """Tests for view_entities()/view_entities_current()/safe_view_entities*()."""

    @pytest.fixture
    def registry(self):
        """Two entities: e1 has both description and status_reading (with
        SCD history); e2 has only description -- exercises null-filling
        for the component e2 lacks.
        """
        conn = ibis.duckdb.connect()
        conn.create_table(
            "entity_id",
            {
                "value": ["e1", "e2", "def1"],
                "display_alias": ["hero", "villain", "status_reading"],
                "path": ["story:hero", "story:villain", "story:status_reading"],
                "display_key": ["hero", "villain", "status_reading"],
                "filepath": ["story"] * 3,
            },
        )
        conn.create_table(
            "field",
            {"entity_id": ["def1", "def1"], "value": ["as_of", "status"], "time_dimension": [True, False]},
        )
        conn.create_table(
            "description",
            {"entity_id": ["e1", "e2"], "value": ["Hero desc", "Villain desc"]},
        )
        conn.create_table(
            "status_reading",
            {
                "entity_id": ["e1", "e1"],
                "component_index": [0, 1],
                "modifier": pd.array([None, None], dtype=pd.StringDtype()),
                "as_of": ["2024-01-01", "2024-06-01"],
                "status": ["open", "closed"],
            },
        )
        return Registry(conn, {
            "entity_id": conn.table("entity_id"),
            "field": conn.table("field"),
            "description": conn.table("description"),
            "status_reading": conn.table("status_reading"),
        })

    def test_view_entities_returns_getter_result(self, registry):
        assert isinstance(registry.view_entities("hero"), GetterResult)

    def test_resolves_by_id(self, registry):
        # _current, not plain view_entities: "hero" has SCD history in
        # status_reading, so full-history would legitimately be >1 row here.
        assert registry.view_entities_current("e1").to_dict()["description.value"] == "Hero desc"

    def test_resolves_by_alias(self, registry):
        assert registry.view_entities_current("hero").to_dict()["description.value"] == "Hero desc"

    def test_resolves_by_path_fragment(self, registry):
        assert registry.view_entities_current("story:hero").to_dict()["description.value"] == "Hero desc"

    def test_includes_all_component_types(self, registry):
        row = registry.view_entities_current("hero").to_dict()
        assert row["description.value"] == "Hero desc"
        assert row["status_reading.status"] == "closed"

    def test_nulls_for_missing_component_not_a_dropped_row(self, registry):
        row = registry.view_entities("villain").to_dict()
        assert row["description.value"] == "Villain desc"
        assert pd.isna(row["status_reading.status"])

    def test_current_collapses_scd_history(self, registry):
        row = registry.view_entities_current("hero").to_dict()
        assert row["status_reading.as_of"] == "2024-06-01"
        assert row["status_reading.status"] == "closed"

    def test_full_history_keeps_all_rows(self, registry):
        df = registry.view_entities("hero").to_pandas()
        assert set(df["status_reading.as_of"]) == {"2024-01-01", "2024-06-01"}

    def test_multiple_entities_returns_one_row_each(self, registry):
        df = registry.view_entities(["hero", "villain"]).to_pandas()
        assert set(df["entity_id"]) == {"e1", "e2"}

    def test_unresolved_ref_raises_keyerror(self, registry):
        with pytest.raises(KeyError):
            registry.view_entities("nonexistent")

    def test_safe_view_entities_unresolved_ref_returns_empty_getter_result(self, registry):
        result = registry.safe_view_entities("nonexistent")
        assert isinstance(result, GetterResult)
        assert result.to_pandas().empty

    def test_partial_match_does_not_raise(self, registry):
        with pytest.warns(UserWarning, match="nonexistent"):
            df = registry.view_entities(["hero", "nonexistent"]).to_pandas()
        assert set(df["entity_id"]) == {"e1"}

    def test_safe_view_entities_current_does_not_swallow_multiple_time_dimension_error(self, registry):
        conn = registry._con
        conn.create_table(
            "field",
            {"entity_id": ["def1", "def1"], "value": ["as_of", "also_as_of"], "time_dimension": [True, True]},
            overwrite=True,
        )
        registry._components["field"] = conn.table("field")
        with pytest.raises(ValueError, match="status_reading"):
            registry.safe_view_entities_current("hero")

    def test_resolves_container_alias_despite_being_path_prefix_of_child(self):
        """entity resolution must pick the exact-alias match over a
        path-substring match against a different entity whose alias
        happens to be a prefix of this one's.
        """
        conn = ibis.duckdb.connect()
        conn.create_table(
            "entity_id",
            {
                "value": ["bbb222bbb222", "ccc333ccc333"],
                "display_alias": ["feeding_system", "feeding_system.feed_cats"],
                "path": [
                    "examples/example.yaml:feeding_system",
                    "examples/example.yaml:feeding_system.feed_cats",
                ],
                "display_key": ["feeding_system", "feed_cats"],
                "filepath": ["examples/example.yaml"] * 2,
            },
        )
        conn.create_table(
            "description",
            {
                "entity_id": ["bbb222bbb222", "ccc333ccc333"],
                "value": ["The feeding system.", "The feed_cats task."],
            },
        )
        registry = Registry(
            conn, {"entity_id": conn.table("entity_id"), "description": conn.table("description")}
        )
        result = registry.view_entities("feeding_system").to_dict()
        assert result["description.value"] == "The feeding system."


class TestViewEntity:
    """view_entity's markdown output is meant to be read (by a human or a
    live model deciding what to do next), so it should show only an
    entity's own data -- not the registry's internal bookkeeping about
    that data (see the fix's own commit for the motivating example: a
    live-test trace where a real entity was buried under near-identical
    "component_type" blocks and redundant resolved-hash columns)."""

    def _registry(self) -> Registry:
        return Registry.from_component_rows(
            {
                "entity_id": [{"entity_id": "e1", "value": "e1", "display_alias": "widget_a"}],
                # Bookkeeping rows the registry itself tracks per component
                # type this entity carries -- not entity data.
                "component_type": [
                    {"entity_id": "e1", "component_type": "solution", "derived": False},
                    {"entity_id": "e1", "component_type": "description", "derived": False},
                ],
                "description": [{"entity_id": "e1", "value": "A test widget."}],
                # An entity_ref field: "value" is the human-readable
                # reference, "value_eid" its derive-time-resolved hash.
                "solution": [{"entity_id": "e1", "value": "widget_b", "value_eid": "e2"}],
            }
        )

    def test_component_type_bookkeeping_is_not_shown(self):
        output = self._registry().view_entity("widget_a")
        assert "## component_type" not in output
        assert "derived" not in output

    def test_resolved_eid_is_hidden_when_its_human_readable_value_is_present(self):
        output = self._registry().view_entity("widget_a")
        assert "value: widget_b" in output
        assert "value_eid" not in output
        assert "e2" not in output

    def test_actual_component_data_still_shown(self):
        output = self._registry().view_entity("widget_a")
        assert "## description" in output
        assert "A test widget." in output
        assert "## solution" in output

    def test_eid_kept_when_no_companion_human_readable_value_present(self):
        """The suppression only fires when the plain field is actually
        there to fall back on -- an _eid-suffixed field with no companion
        is left alone rather than silently dropped."""
        registry = Registry.from_component_rows(
            {
                "entity_id": [{"entity_id": "e1", "value": "e1", "display_alias": "widget_a"}],
                "parent_eid": [{"entity_id": "e1", "parent_eid": "e2"}],
            }
        )
        output = registry.view_entity("widget_a")
        assert "parent_eid: e2" in output

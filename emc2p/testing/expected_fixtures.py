"""Generic "load an example manifest, run some transformation, compare to a
hand-written expected.py fixture" framework for human-validated tests.

Nothing here knows about any downstream project's own dataflow package
names, component schema, or examples layout -- a caller supplies its own
``examples_dir``/``expected_dir`` and, for :class:`ExpectedValueChecker`,
the dotted module prefixes its own (and any upstream, e.g. emc2p's own)
dataflow packages are registered under. The expected layout this assumes:

- ``examples_dir/<name>/`` -- one directory per example, each a loadable
  EC manifest (passed straight to ``Registrar.update``/``from_manifest``).
- ``expected_dir/<name>/<dataflow module subpath>.py`` -- optional,
  hand-written per-example fixture. Each module-level variable name is
  matched against a Hamilton node's own final path component (the part
  after the last ``.``); if present, the executed node's result must
  contain that variable's value (see :func:`assert_subset` for the
  subset-matching rules). An ``incorrect_<name>`` variable, if present, is
  asserted to *not* match -- a negative case proving the DAG would have
  caught wrong data, not just that the fixture is too lenient to fail.

Originally developed in iacs (whose own dataflows drove the only
downstream project at the time), then extracted here once emc2p's own
generic ETL/derive dataflows moved out from under a second downstream
project (story-simulator) with a different transformation mechanism of
its own (a live MCP session's turns, not an in-process Hamilton driver) --
see each project's own test_human_validated for how it plugs in.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

import pandas as pd
import pytest
from hamilton.lifecycle import NodeExecutionHook

if TYPE_CHECKING:
    from emc2p.registrar import Registrar


def example_dirs(examples_dir: Path) -> list:
    """One ``pytest.param(example_dir, id=example_dir.name)`` per
    subdirectory of ``examples_dir``, for ``@pytest.mark.parametrize``.
    """
    return [
        pytest.param(example_dir, id=example_dir.name)
        for example_dir in sorted(examples_dir.iterdir())
        if example_dir.is_dir()
    ]


def load_expected_module(expected_filepath: Path) -> ModuleType:
    """Import a per-dataflow expected file (e.g. ``etl/load_manifest.py``) as a module."""
    spec = importlib.util.spec_from_file_location(
        expected_filepath.stem, expected_filepath
    )
    expected_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(expected_module)
    return expected_module


# ─── Comparison helpers ──────────────────────────────────────────────────


def _to_pandas(value) -> pd.DataFrame | None:
    """Convert an ibis Table or pandas DataFrame to a pandas DataFrame."""
    if isinstance(value, pd.DataFrame):
        return value
    if hasattr(value, "to_pandas"):
        return value.to_pandas()
    return None


def assert_df_rows_subset(expected: pd.DataFrame, actual_value, context: str = "") -> None:
    """Assert every row of expected appears in the actual ibis Table or DataFrame."""
    if expected.empty:
        return
    actual = _to_pandas(actual_value)
    assert (
        actual is not None
    ), f"{context}: could not convert actual value to DataFrame (got {type(actual_value)})"
    common_cols = [c for c in expected.columns if c in actual.columns]
    if not common_cols:
        return

    exp_sub = expected[common_cols].copy()
    act_sub = actual[common_cols].copy()
    # Normalize columns where all non-null/non-empty values are numeric to float
    # so that e.g. 1 and 1.0 compare equal regardless of int vs float dtype.
    # Empty strings are treated as missing for this check because bare tag
    # components store "" in sub-field columns that only exist on other rows.
    for col in common_cols:
        for frame in (exp_sub, act_sub):
            as_nullable = frame[col].replace("", pd.NA)
            converted = pd.to_numeric(as_nullable, errors="coerce")
            if converted.notna().sum() == as_nullable.notna().sum():
                frame[col] = converted.astype(float)

    exp_str = exp_sub.fillna("__NULL__").astype(str).reset_index(drop=True)
    act_str = act_sub.fillna("__NULL__").astype(str).reset_index(drop=True)
    for _, row in exp_str.iterrows():
        found = (act_str == row).all(axis=1).any()
        assert found, (
            f"{context}: expected row not found in actual output.\n"
            f"  Expected: {row.to_dict()}\n"
            f"  Actual (first 10 rows):\n{act_str.head(10).to_string()}"
        )


def _manifest_item_matches(expected, actual) -> bool:
    """Return True if expected dict/scalar is a lenient subset match of actual."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        common = [k for k in expected if k in actual]
        if not common:
            return False
        return all(_manifest_values_match(expected[k], actual[k]) for k in common)
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return expected == actual
    return expected == actual


def _manifest_values_match(expected, actual) -> bool:
    if isinstance(expected, dict) and isinstance(actual, dict):
        common = [k for k in expected if k in actual]
        return all(_manifest_values_match(expected[k], actual[k]) for k in common)
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return expected == actual
    return expected == actual


def assert_manifest_subset(expected: dict, actual: dict, context: str = "") -> None:
    """Assert every entry in expected appears in actual with lenient subset semantics.

    - Top-level keys missing from actual are skipped.
    - List item matching uses overlapping-field subset semantics.
    """
    for key, exp_val in expected.items():
        if key not in actual:
            continue
        act_val = actual[key]
        ctx = f"{context}.{key}" if context else key
        if isinstance(exp_val, list) and isinstance(act_val, list):
            for exp_item in exp_val:
                found = any(
                    _manifest_item_matches(exp_item, act_item) for act_item in act_val
                )
                assert found, (
                    f"{ctx}: expected item {exp_item!r} not found in actual\n"
                    f"  Actual: {act_val!r}"
                )
        elif isinstance(exp_val, dict) and isinstance(act_val, dict):
            assert_manifest_subset(exp_val, act_val, context=ctx)
        elif isinstance(exp_val, dict):
            assert False, (
                f"{ctx}: expected dict structure but got {type(act_val).__name__} — "
                "possible hierarchical/flat format mismatch"
            )


def assert_subset(var_name: str, expected_value, actual_value) -> None:
    """Assert expected_value is contained within actual_value using subset semantics."""
    if isinstance(expected_value, pd.DataFrame):
        assert_df_rows_subset(expected_value, actual_value, context=var_name)

    elif isinstance(expected_value, dict):
        assert isinstance(
            actual_value, dict
        ), f"'{var_name}': expected dict, got {type(actual_value)}"
        for key in expected_value:
            assert key in actual_value, f"'{var_name}': key '{key}' not found in actual"
            exp_val = expected_value[key]
            act_val = actual_value[key]
            if isinstance(exp_val, pd.DataFrame):
                assert_df_rows_subset(exp_val, act_val, context=f"{var_name}[{key}]")
            elif isinstance(exp_val, dict) and isinstance(act_val, dict):
                assert_manifest_subset(exp_val, act_val, context=f"{var_name}[{key!r}]")


def assert_not_subset(var_name: str, expected_value, actual_value) -> None:
    """Assert expected_value is NOT contained within actual_value."""
    try:
        assert_subset(var_name, expected_value, actual_value)
    except AssertionError:
        return
    pytest.fail(
        f"'{var_name}' was declared as incorrect data but it matched the actual output — "
        "the DAG should have produced different data."
    )


def _normalize_df(
    df: pd.DataFrame, entity_id_df: pd.DataFrame, common_cols: list[str]
) -> pd.DataFrame:
    """Restrict to common_cols and normalize entity_id hashes for cross-registry comparison.

    Entity IDs are hashes of ``filepath:entity_path``.  Two registries loaded
    from different directories produce different hashes for the same logical
    entities, so we normalize by using only the within-file entity path.
    """
    filepath_of = entity_id_df.set_index("value")["filepath"].to_dict()
    path_of = entity_id_df.set_index("value")["path"].to_dict()

    def hash_to_path(eid):
        if pd.isna(eid):
            return eid
        fp = filepath_of.get(str(eid), "")
        p = path_of.get(str(eid), str(eid))
        return p[len(fp) + 1 :] if fp and p.startswith(fp + ":") else p

    df = df[common_cols].copy()
    for col in common_cols:
        if col == "entity_id" or col.endswith("_eid"):
            df[col] = df[col].map(hash_to_path)
    return df.sort_values(common_cols, na_position="last").reset_index(drop=True)


def assert_components_equal(
    comp: pd.DataFrame,
    loaded_comp: pd.DataFrame,
    eid_df: pd.DataFrame,
    reloaded_eid_df: pd.DataFrame,
    comp_type: str,
) -> None:
    """Assert two component tables are equal after a round trip.

    Normalizes entity_id hashes to within-file entity paths, since they are
    hashes of the source filepath and the two registries were loaded from
    different paths.
    """
    common_cols = sorted(
        c for c in (set(comp.columns) & set(loaded_comp.columns))
        # _seq_{field} (see Registry.merge), like component_index, is
        # assigned independently per load and isn't meaningfully comparable
        # across two separately-loaded registries.
        if c != "component_index" and not c.startswith("_seq_")
    )
    norm1 = _normalize_df(comp, eid_df, common_cols)
    norm2 = _normalize_df(loaded_comp, reloaded_eid_df, common_cols)

    pd.testing.assert_frame_equal(
        norm1, norm2, check_dtype=False, obj=f"component_type={comp_type!r}"
    )


def assert_registries_equal(
    registrar_a: "Registrar", registrar_b: "Registrar", skip: set[str] | None = None
) -> None:
    """Compare each component type. entity_id values are hashes of the source
    filepath, which differs between the two registrars' own load locations,
    so normalize them to within-file entity paths before comparing.
    """
    skip = {"entity_id", "component_type", "invalid_field"} | (skip or set())
    eid_df_a = registrar_a.get("entity_id").execute()
    eid_df_b = registrar_b.get("entity_id").execute()
    for comp_type in set(registrar_a.registry.component_types) - skip:
        comp = registrar_a.get(comp_type).execute()
        loaded_comp = registrar_b.get(comp_type).execute()
        assert_components_equal(comp, loaded_comp, eid_df_a, eid_df_b, comp_type)


class ExpectedValueChecker(NodeExecutionHook):
    """Checks each executed node's result against a hand-written expected value, if one exists.

    ``dataflow_module_prefixes`` names every dotted-module prefix a
    caller's own registered dataflow packages (plus any upstream ones,
    e.g. ``"emc2p.dataflows."``, that its own base ETL subdags in) can
    show up under -- the part after whichever prefix matches is used to
    find the per-dataflow expected fixture (see this module's own
    docstring for the layout).
    """

    def __init__(
        self,
        example_dir: Path,
        expected_dir: Path,
        dataflow_module_prefixes: tuple[str, ...],
    ):
        self.example_dir = example_dir
        self.expected_dir = expected_dir
        self.dataflow_module_prefixes = dataflow_module_prefixes
        self._expected_modules: dict[Path, ModuleType] = {}

    def run_before_node_execution(self, **kwargs):
        pass

    def run_after_node_execution(
        self, *, node_name: str, node_tags: dict, result, **kwargs
    ):
        source_module = node_tags.get("module")
        # Input variables don't have source modules
        if source_module is None:
            return

        dataflow_module_name = None
        for prefix in self.dataflow_module_prefixes:
            if source_module.startswith(prefix):
                dataflow_module_name = source_module[len(prefix):]
                break
        if dataflow_module_name is None:
            return
        dataflow_module_subpath = dataflow_module_name.replace(".", "/")
        expected_filepath = (
            self.expected_dir / self.example_dir.name / f"{dataflow_module_subpath}.py"
        )
        if not expected_filepath.exists():
            return

        if expected_filepath not in self._expected_modules:
            self._expected_modules[expected_filepath] = load_expected_module(
                expected_filepath
            )
        expected_module = self._expected_modules[expected_filepath]

        variable_name = node_name.rsplit(".", 1)[-1]
        if not hasattr(expected_module, variable_name):
            return
        expected_value = getattr(expected_module, variable_name)

        assert_subset(node_name, expected_value, result)

        # Check we don't have incorrect values
        incorrect_name = "incorrect_" + variable_name
        if hasattr(expected_module, incorrect_name):
            incorrect_value = getattr(expected_module, incorrect_name)
            assert_not_subset(node_name, incorrect_value, result)

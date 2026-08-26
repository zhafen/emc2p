"""Hamilton DAG for converting entity-centered data to component-centered data.

Coordinates load_yaml and load_python subdags, merges their entity-first
dictionaries, then runs the shared pipeline through to a Registry.
"""

import re
from pathlib import Path

import ibis
import ibis.expr.types as ir
import pandas as pd
from hamilton.function_modifiers import extract_fields, subdag, source

from ...registry import Registry
from ...utils import dhash
from . import load_yaml, load_python


_BUILTINS_DIRS: list[tuple[str, Path]] = [
    ("builtins", Path(__file__).parent.parent.parent / "builtins"),
]


def register_builtins_dir(path: str | Path, tag: str) -> None:
    """Register an extra directory of EC files to auto-include in every manifest load.

    Lets a downstream project (e.g. one with its own domain-specific
    builtin component definitions, alongside emc2p's generic ones)
    contribute its own always-loaded builtins directory without this
    module needing to know about that project. Each YAML file directly
    under ``path`` is included the same way emc2p's own ``builtins/``
    directory already is (see ``raw_strings``), identified as
    ``"{tag}.<stem>"`` instead of ``"builtins.<stem>"``.

    Args:
        path: Directory to scan for ``.yaml``/``.yml`` files.
        tag: Prefix used for each included file's identifier. Should be
            distinct from ``"builtins"`` and any other registered tag to
            avoid two directories' same-named files colliding.
    """
    _BUILTINS_DIRS.append((tag, Path(path)))


def builtins_tags() -> list[str]:
    """Return every registered builtins tag (``"builtins"`` plus any added
    via ``register_builtins_dir``).

    For a caller that needs to recognize builtins-sourced entities by
    their file identifier prefix (e.g. ``export_manifest``'s exclusion of
    them from manifest export).
    """
    return [tag for tag, _ in _BUILTINS_DIRS]


# Source subdags -- each produces raw_entity_first_data keyed by file_id

def _file_id(path: Path, cwd: Path) -> str:
    """Identify a file by its path relative to cwd, falling back to the full path."""
    try:
        return str(path.relative_to(cwd))
    except ValueError:
        return str(path)


@extract_fields(["raw_python_strings", "raw_yaml_strings"])
def raw_strings(
    input_dirs: list[str | Path],
    python_strings: dict[str, str] = None,
    yaml_strings: dict[str, str] = None,
) -> dict[str, dict[str, str]]:
    """Read raw YAML and Python source text from input_dirs, combined with directly-provided strings.

    Always includes all EC files from every registered builtins directory
    (see ``register_builtins_dir``) in the YAML output, each identified as
    "{tag}.<stem>" -- emc2p's own builtins directory is always registered
    as "builtins". User-provided files are identified by their path
    relative to the current working directory.

    Parameters
    ----------
    input_dirs : list[str | Path]
        A list of file or directory paths. Directories are searched
        recursively for both EC (``.yaml``/``.yml``) and Python (``.py``) files.
    python_strings : dict[str, str], optional
        A dict keyed by identifier of raw Python source text to merge in
        directly, without reading from disk. Keys read from ``input_dirs``
        take precedence over identical keys in ``python_strings``.
    yaml_strings : dict[str, str], optional
        A dict keyed by identifier of raw YAML text to merge in directly,
        without reading from disk. Keys read from ``input_dirs`` take
        precedence over identical keys in ``yaml_strings``.

    Returns
    -------
    dict[str, dict[str, str]]
        A dict with keys ``"raw_python_strings"`` and ``"raw_yaml_strings"``,
        each keyed by file identifier with raw source text as values.
    """
    cwd = Path.cwd()
    yaml_files: list[tuple[Path, str]] = []
    python_files: list[tuple[Path, str]] = []

    for item in input_dirs:
        p = Path(item)
        if p.is_file():
            if p.suffix in (".yaml", ".yml"):
                yaml_files.append((p, _file_id(p, cwd)))
            elif p.suffix == ".py":
                python_files.append((p, _file_id(p, cwd)))
        elif p.is_dir():
            for f in sorted(p.rglob("*.y*ml")):
                if f.suffix in (".yaml", ".yml"):
                    yaml_files.append((f, _file_id(f, cwd)))
            for f in sorted(p.rglob("*.py")):
                python_files.append((f, _file_id(f, cwd)))

    for tag, builtins_dir in _BUILTINS_DIRS:
        for f in sorted(builtins_dir.rglob("*.y*ml")):
            if f.suffix in (".yaml", ".yml"):
                yaml_files.append((f, f"{tag}.{f.stem}"))

    resolved_yaml_strings = dict(yaml_strings) if yaml_strings else {}
    for file_path, file_id in yaml_files:
        resolved_yaml_strings[file_id] = file_path.read_text(encoding="utf-8")

    resolved_python_strings = dict(python_strings) if python_strings else {}
    for file_path, file_id in python_files:
        try:
            resolved_python_strings[file_id] = file_path.read_text(encoding="utf-8")
        except OSError:
            continue

    return {
        "raw_python_strings": resolved_python_strings,
        "raw_yaml_strings": resolved_yaml_strings,
    }


@subdag(
    load_yaml,
    inputs={"raw_yaml_strings": source("raw_yaml_strings")},
    config={},
)
def yaml_entity_first_data(raw_entity_first_data: dict) -> dict:
    return raw_entity_first_data


@subdag(
    load_python,
    inputs={"raw_python_strings": source("raw_python_strings")},
    config={},
)
def python_entity_first_data(raw_entity_first_data: dict) -> dict:
    return raw_entity_first_data


def raw_entity_first_data(
    yaml_entity_first_data: dict,
    python_entity_first_data: dict,
) -> dict:
    """Merge entity-first dicts from all source loaders."""
    return {**yaml_entity_first_data, **python_entity_first_data}


# CSV loading (stays inline -- CSV doesn't fit the entity-first dict format)

def raw_csv_data(input_dirs: list[str | Path]) -> dict[str, pd.DataFrame]:
    """Load CSV files from a list of files or directories (user-provided only, not builtins).

    The filename stem (without extension) of each CSV file becomes the component
    type for all rows in that file. Only directories and explicit CSV file paths
    from ``input_dirs`` are searched — the builtins directory is never included.

    Parameters
    ----------
    input_dirs : list[str | Path]
        A list of CSV file paths or directory paths. Directories are searched
        recursively for CSV files.

    Returns
    -------
    dict[str, pd.DataFrame]
        A dict keyed by the file path identifier (relative to cwd when possible),
        where each value is a DataFrame of that CSV's rows.
    """
    cwd = Path.cwd()
    all_files: list[tuple[Path, str]] = []

    for item in input_dirs:
        p = Path(item)
        if p.is_file() and p.suffix == ".csv":
            all_files.append((p, _file_id(p, cwd)))
        elif p.is_dir():
            for f in sorted(p.rglob("*.csv")):
                all_files.append((f, _file_id(f, cwd)))

    result = {}
    for file_path, file_id in all_files:
        result[file_id] = pd.read_csv(file_path)
    return result


def csv_component_tables(raw_csv_data: dict[str, pd.DataFrame]) -> dict[str, ir.Table]:
    """Convert raw CSV data into component tables, one table per component type.

    Each CSV file is treated as a single entity (see ``csv_spine``); each row
    in the file becomes one instance of a ``"{stem}_comp"`` component attached
    to that entity, distinguished by ``component_index`` (the row's 0-based
    position in the file). CSV column names become field names on the
    component (analogous to sub-field components in YAML), so e.g.
    ``users.csv`` exports as::

        users:
        - users_comp:
            user_id: 1
            name: Alice Johnson
        - users_comp:
            user_id: 2
            name: Bob Smith

    When multiple CSV files share the same stem, their rows belong to their
    own (per-file) entities but are unioned into the same ``"{stem}_comp"``
    component-type table.

    Parameters
    ----------
    raw_csv_data : dict[str, pd.DataFrame]
        Mapping of file path identifier → DataFrame as returned by
        ``raw_csv_data``.

    Returns
    -------
    dict[str, ir.Table]
        Keys are ``"{stem}_comp"`` component types; each value is an ibis
        Table with columns: entity_id, component_index, modifier, and one
        column per CSV field.
    """
    per_comp_type: dict[str, list[pd.DataFrame]] = {}
    for file_id, df in raw_csv_data.items():
        stem = Path(file_id).stem
        entity_id = dhash(file_id)
        comp_type = f"{stem}_comp"
        rows = []
        for i, row in df.iterrows():
            record = {
                "entity_id": entity_id,
                "component_index": i,
                "modifier": pd.NA,
            }
            for col in df.columns:
                record[col] = row[col]
            rows.append(record)
        part_df = pd.DataFrame(rows)
        part_df["modifier"] = part_df["modifier"].astype(pd.StringDtype())
        part_df["component_index"] = part_df["component_index"].astype("int64")
        per_comp_type.setdefault(comp_type, []).append(part_df)

    result = {}
    for comp_type, dfs in per_comp_type.items():
        combined = pd.concat(dfs, ignore_index=True)
        combined["modifier"] = combined["modifier"].astype(pd.StringDtype())
        result[comp_type] = ibis.memtable(combined)
    return result


def csv_spine(raw_csv_data: dict[str, pd.DataFrame]) -> ir.Table:
    """Build one spine row per CSV file (one entity per file, not per row).

    Each CSV file is a single entity named after its filename stem — e.g.
    ``orders.csv`` becomes entity ``orders`` — with entity_id
    ``dhash(file_path_id)``. Each row in the file is a component instance
    attached to that entity (see ``csv_component_tables``), not a separate
    entity.

    Treating the whole file as one entity, rather than giving each row its
    own entity, avoids two problems a per-row entity would create: every
    row's entity_key/alias colliding (all equal to the stem), and a
    synthetic ``[row_index]`` path segment coming back as a *real*
    container entity once round-tripped through YAML export/reimport.

    Parameters
    ----------
    raw_csv_data : dict[str, pd.DataFrame]
        Mapping of file path identifier → DataFrame as returned by
        ``raw_csv_data``.

    Returns
    -------
    ir.Table
        Columns: entity_id, entity_key, filepath, path. Schema matches the
        subset of ``yaml_spine`` that ``entity_id_table`` derives from.
    """
    rows = []
    for file_id in raw_csv_data:
        stem = Path(file_id).stem
        rows.append({
            "entity_id": dhash(file_id),
            "entity_key": stem,
            "filepath": file_id,
            "path": f"{file_id}:{stem}",
        })
    spine_df = pd.DataFrame(
        rows, columns=["entity_id", "entity_key", "filepath", "path"]
    )
    for col in spine_df.columns:
        spine_df[col] = spine_df[col].astype(pd.StringDtype())
    return ibis.memtable(spine_df)


# Shared pipeline: entity-first dict -> component tables -> Registry

def _add_component_pairs(
    entity_path: str, index: int, component, result: list
) -> None:
    """Append (path, value) pairs for one component entry to result.

    Parameters
    ----------
    entity_path : str
        The dot-separated path to the owning entity, e.g. "a.b.c".
    index : int
        The 0-based position of this component in the entity's list
        (bare-string tags count toward the index).
    component : str | dict
        The raw component value.
    result : list
        Accumulator of (path, value) string tuples.
    """
    prefix = f"{entity_path}[{index}]"
    if isinstance(component, str):
        result.append((f"{prefix}.{component}", ""))
    elif isinstance(component, dict):
        key = next(iter(component))
        value = component[key]
        if isinstance(value, list):
            for j, item in enumerate(value):
                item_prefix = f"{entity_path}[{index + j}]"
                if isinstance(item, dict):
                    for sub_key, sub_val in item.items():
                        str_val = "" if sub_val is None else str(sub_val)
                        result.append((f"{item_prefix}.{key}.{sub_key}", str_val))
                else:
                    str_val = "" if item is None else str(item)
                    result.append((f"{item_prefix}.{key}", str_val))
        elif isinstance(value, dict) and value and all(
            isinstance(v, dict) for v in value.values()
        ):
            for j, (inner_key, inner_dict) in enumerate(value.items()):
                item_prefix = f"{entity_path}[{index + j}]"
                result.append((f"{item_prefix}.{key}.value", inner_key))
                for sub_key, sub_val in inner_dict.items():
                    str_val = "" if sub_val is None else str(sub_val)
                    result.append((f"{item_prefix}.{key}.{sub_key}", str_val))
        elif isinstance(value, dict):
            for sub_key, sub_val in value.items():
                str_val = "" if sub_val is None else str(sub_val)
                result.append((f"{prefix}.{key}.{sub_key}", str_val))
        else:
            str_val = "" if value is None else str(value)
            result.append((f"{prefix}.{key}", str_val))


def _flatten_to_pathvalue(data: dict, parent_path: str = "") -> list[tuple[str, str]]:
    """Recursively flatten entity-first data into (path, value) string pairs.

    For flat entities (list value) the path is ``entity[N].key``.
    For nested entities (dict value with an optional ``data`` key) the entity's
    own components are at ``entity.data[N].key`` and sub-entities are processed
    recursively.
    """
    result = []
    for entity_key, entity_value in data.items():
        entity_path = f"{parent_path}.{entity_key}" if parent_path else entity_key
        if isinstance(entity_value, list):
            for i, component in enumerate(entity_value):
                _add_component_pairs(entity_path, i, component, result)
        elif isinstance(entity_value, dict):
            for i, component in enumerate(entity_value.get("data", [])):
                _add_component_pairs(f"{entity_path}.data", i, component, result)
            sub_entities = {k: v for k, v in entity_value.items() if k != "data"}
            result.extend(_flatten_to_pathvalue(sub_entities, entity_path))
    return result


def _collect_entity_paths(data: dict, parent_path: str = "") -> list[str]:
    """Recursively collect every entity_path in entity-first data.

    Mirrors the traversal in ``_flatten_to_pathvalue``, but records an entity's
    path even when it has no components of its own — e.g. a pure container
    like ``cat_food_supply`` that only exists to group child entities and has
    no ``data`` list. Without this, such entities never appear in
    ``pathvalue_pairs``/``keyvalue_store`` and so never get a row in
    ``entity_id_table``, even though other entities (e.g. their children, via
    ``parent_eid``) reference them by hash.
    """
    result = []
    for entity_key, entity_value in data.items():
        entity_path = f"{parent_path}.{entity_key}" if parent_path else entity_key
        if isinstance(entity_value, list):
            result.append(entity_path)
        elif isinstance(entity_value, dict):
            result.append(entity_path)
            sub_entities = {k: v for k, v in entity_value.items() if k != "data"}
            result.extend(_collect_entity_paths(sub_entities, entity_path))
    return result


def yaml_spine(raw_entity_first_data: dict) -> ir.Table:
    """Build one spine row per YAML entity, including component-less containers.

    Returns
    -------
    ir.Table
        Columns: entity_id, entity_key, entity_path, filepath. Schema matches
        the corresponding subset of ``keyvalue_store`` columns, so it's a
        drop-in replacement that also covers entities with no components.
    """
    rows = []
    for file_id, entities in raw_entity_first_data.items():
        for entity_path in _collect_entity_paths(entities):
            rows.append({
                "entity_id": dhash(f"{file_id}:{entity_path}"),
                "entity_key": entity_path.rsplit(".", 1)[-1],
                "entity_path": f"{file_id}:{entity_path}",
                "filepath": file_id,
            })
    spine_df = pd.DataFrame(
        rows, columns=["entity_id", "entity_key", "entity_path", "filepath"]
    )
    for col in spine_df.columns:
        spine_df[col] = spine_df[col].astype(pd.StringDtype())
    return ibis.memtable(spine_df)


def pathvalue_pairs(raw_entity_first_data: dict) -> ir.Table:
    """Convert the raw entity-first data into a database table with two
    fields: path and value, both of type str.

    The first step in the transformation process, serving as a way to
    inspect the raw data in a tabular format before applying the more
    complex transformations.

    Each path is prefixed with the file identifier using a ':' separator, e.g.
    "examples/foo.yaml:my_entity[0].description".

    Parameters
    ----------
    raw_entity_first_data : dict
        Nested dict of the form {file_id: {entity_key: entity_data}}.

    Returns
    -------
    ir.Table
        A table with columns "path" and "value".
    """
    pairs = []
    for file_id, entities in raw_entity_first_data.items():
        for entity_path, value in _flatten_to_pathvalue(entities):
            pairs.append((f"{file_id}:{entity_path}", value))
    df = pd.DataFrame(pairs, columns=["path", "value"])
    return ibis.memtable(df)


_PATH_PATTERN = r"^(.+)\[(\d+)\]\.(.+)$"


def _with_spine_path(t: ir.Table) -> ir.Table:
    """Add a spine_path column to a table that has a 'path' column.

    Assumes all rows already match _PATH_PATTERN (pre-filter before calling).
    Also adds intermediate columns _prefix, _idx, _after, _ctf.
    """
    t = t.mutate(
        _prefix=t.path.re_extract(_PATH_PATTERN, 1),
        _idx=t.path.re_extract(_PATH_PATTERN, 2),
        _after=t.path.re_extract(_PATH_PATTERN, 3),
    )
    t = t.mutate(_ctf=t["_after"].re_extract(r"^([^.]*)", 1))
    return t.mutate(spine_path=t["_prefix"] + "[" + t["_idx"] + "]." + t["_ctf"])


def keyvalue_store(pathvalue_pairs: ir.Table) -> ir.Table:
    """Parse path-value pairs into a structured long-format table.

    One row per (entity, component, field). Centralises all path-parsing so
    that ``spine`` and ``component_tables`` are simple derivations.

    Every row's ``entity_id`` is a hash of its own ``entity_path``. An entity
    can still be attached to an existing entity's identity elsewhere in the
    registry via a ``same_as`` component, resolved later in derive (see
    ``emc2p.dataflows.derive.resolve_same_as``) rather than here.

    Returns
    -------
    ir.Table
        Columns: entity_id, entity_key, entity_path, filepath,
        component_index, component_type, modifier, spine_path, field, value.
    """
    t = pathvalue_pairs.filter(pathvalue_pairs.path.re_search(_PATH_PATTERN))
    t = _with_spine_path(t)

    t = t.mutate(
        entity_path=ibis.ifelse(
            t["_prefix"].endswith(".data"),
            t["_prefix"].substr(0, t["_prefix"].length() - 5),
            t["_prefix"],
        ),
        component_type=t["_ctf"].re_extract(r"^(\S+)", 1),
        modifier=t["_ctf"].re_extract(r"^\S+ (.+)$", 1).nullif(""),
        component_index=t["_idx"].cast("int64"),
    )
    t = t.mutate(
        entity_id=t.entity_path.hexdigest("sha256").substr(0, 12),
        entity_key=t.entity_path.re_extract(r"([^:.]+)$", 1),
        filepath=t.entity_path.re_extract(r"^([^:]+):", 1).nullif(""),
    )
    t = t.mutate(
        field=ibis.ifelse(
            t["path"] == t["spine_path"],
            ibis.literal("value"),
            t["path"].substr(t["spine_path"].length() + 1),
        )
    )
    t = t.mutate(
        field=ibis.ifelse(t["field"] == "", ibis.literal("value"), t["field"])
    )
    return t.select(
        "entity_id", "entity_key", "entity_path", "filepath",
        "component_index", "component_type", "modifier",
        "spine_path", "field", "value",
    )


def entity_id_table(yaml_spine: ir.Table, csv_spine: ir.Table = None) -> ir.Table:
    """Build one row per entity from the YAML spine and optional CSV spine.

    Uses ``yaml_spine`` (not ``keyvalue_store``) so that entities with no
    components of their own — pure containers like ``cat_food_supply`` —
    still get a row here.

    Returns
    -------
    ir.Table
        Columns: value, path, alias, entity_key, filepath.
        ``value`` is the entity hash (the entity_id); ``path`` is the full
        entity_path; ``alias`` is the human-readable display ID (last two
        dot-segments of the entity path, or just entity_key for top-level).
    """
    yaml_entities = yaml_spine.select(
        "entity_id", "entity_key", "entity_path", "filepath"
    ).distinct().to_pandas()
    yaml_entities = yaml_entities.rename(columns={"entity_id": "value", "entity_path": "path"})

    def compute_alias(row):
        entity_path = row["path"]
        entity_key = row["entity_key"]
        sep = entity_path.find(":")
        name_part = entity_path[sep + 1:] if sep != -1 else entity_path
        parts = name_part.split(".")
        return ".".join(parts[-2:]) if len(parts) >= 2 else entity_key

    yaml_entities["alias"] = yaml_entities.apply(compute_alias, axis=1)
    df = yaml_entities[["value", "path", "alias", "entity_key", "filepath"]]

    if csv_spine is not None:
        csv_df = csv_spine.to_pandas()
        csv_entity_df = csv_df[["entity_id", "entity_key", "filepath", "path"]].drop_duplicates()
        csv_entity_df = csv_entity_df.rename(columns={"entity_id": "value"})
        csv_entity_df["alias"] = csv_entity_df["entity_key"]
        csv_entity_df = csv_entity_df[["value", "path", "alias", "entity_key", "filepath"]]
        df = pd.concat([df, csv_entity_df], ignore_index=True)

    df = df[["value", "path", "alias", "entity_key", "filepath"]]
    for col in ("value", "path", "alias", "entity_key"):
        df[col] = df[col].astype(pd.StringDtype())
    df["filepath"] = df["filepath"].astype(pd.StringDtype())
    return ibis.memtable(df)



def component_type_table(
    keyvalue_store: ir.Table,
    csv_component_tables: dict[str, ir.Table] = None,
) -> ir.Table:
    """Build one row per component instance, including derived and skip_on_export flags.

    Reads explicit ``component_type`` component entries from the keyvalue_store to
    populate ``derived`` and ``skip_on_export`` columns on the metadata table.

    CSV-derived metadata comes from ``csv_component_tables`` (one row per CSV
    row, i.e. one row per ``"{stem}_comp"`` component instance) rather than
    ``csv_spine`` (one row per *file*/entity) — a whole CSV file is a single
    entity with many component instances attached, so the two are
    different granularities.

    Returns
    -------
    ir.Table
        Columns: entity_id, component_index, component_type, modifier, derived, skip_on_export.
    """
    df = keyvalue_store.execute()

    entity_keys = (
        df[["entity_id", "entity_key"]]
        .drop_duplicates(subset=["entity_id"])
        .set_index("entity_id")["entity_key"]
        .to_dict()
    )

    ct_data = df[df["component_type"] == "component_type"]
    derived_set: set[str] = set()
    skip_set: set[str] = set()
    for _, row in ct_data.iterrows():
        eid = str(row["entity_id"])
        field = str(row["field"])
        val = str(row.get("value", "")).strip().lower() in ("true", "1", "yes")
        type_name = entity_keys.get(eid, "")
        if not type_name:
            continue
        if field == "derived" and val:
            derived_set.add(type_name)
        elif field == "skip_on_export" and val:
            skip_set.add(type_name)

    meta_df = df[["entity_id", "component_index", "component_type", "modifier"]].drop_duplicates().copy()
    meta_df["derived"] = meta_df["component_type"].isin(derived_set)
    meta_df["skip_on_export"] = meta_df["component_type"].isin(skip_set)
    meta_df["modifier"] = meta_df["modifier"].astype(pd.StringDtype())
    yaml_ct = ibis.memtable(meta_df)

    if not csv_component_tables:
        return yaml_ct

    csv_rows = []
    for comp_type, table in csv_component_tables.items():
        cdf = table.to_pandas()[["entity_id", "component_index", "modifier"]].copy()
        cdf["component_type"] = comp_type
        csv_rows.append(cdf)
    csv_df = pd.concat(csv_rows, ignore_index=True)
    csv_df["derived"] = False
    csv_df["skip_on_export"] = False
    csv_df["modifier"] = csv_df["modifier"].astype(pd.StringDtype())
    csv_df["component_type"] = csv_df["component_type"].astype(pd.StringDtype())
    return ibis.union(yaml_ct, ibis.memtable(csv_df))


def component_tables(
    keyvalue_store: ir.Table,
    csv_component_tables: dict[str, ir.Table] = None,
) -> dict[str, ir.Table]:
    """Pivot the key-value store by component type, then merge with CSV component tables.

    YAML-derived tables come from ``keyvalue_store``; CSV-derived tables come
    from ``csv_component_tables``. For component types that appear in both
    sources the rows are unioned (columns are aligned; missing columns in either
    source are filled with NULL). Component types that appear only in one source
    are included as-is.

    Returns
    -------
    dict[str, ir.Table]
        Keys are component types; each value is an ibis Table with columns
        entity_id, component_index, modifier, and one column per field
        (or "value" for scalar components).
    """
    df = keyvalue_store.select(
        "entity_id", "component_index", "component_type", "modifier", "field", "value"
    ).to_pandas()

    result = {}
    for comp_type, group in df.groupby("component_type"):
        instances: dict[tuple, dict] = {}
        for _, row in group.iterrows():
            key = (row["entity_id"], row["component_index"])
            if key not in instances:
                instances[key] = {
                    "entity_id": row["entity_id"],
                    "component_index": row["component_index"],
                    "modifier": row["modifier"],
                }
            instances[key][row["field"]] = row["value"]

        comp_df = pd.DataFrame(list(instances.values()))
        comp_df["modifier"] = comp_df["modifier"].astype(pd.StringDtype())
        result[comp_type] = ibis.memtable(comp_df)

    if csv_component_tables:
        for comp_type, csv_table in csv_component_tables.items():
            if comp_type in result:
                yaml_df = result[comp_type].to_pandas()
                csv_df = csv_table.to_pandas()
                combined = pd.concat([yaml_df, csv_df], ignore_index=True)
                combined["modifier"] = combined["modifier"].astype(pd.StringDtype())
                result[comp_type] = ibis.memtable(combined)
            else:
                result[comp_type] = csv_table

    return result


def registry(
    entity_id_table: ir.Table,
    component_type_table: ir.Table,
    component_tables: dict[str, ir.Table],
) -> Registry:
    """Load the constituents of a registry into the registry object.

    Parameters
    ----------
    entity_id_table : ir.Table
        One row per entity (hash, path, value, alias, entity_key, filepath).
    component_type_table : ir.Table
        One row per component instance (entity_id, component_index, component_type, modifier).
    component_tables : dict[str, ir.Table]
        Per-component-type data tables.

    Returns
    -------
    Registry
        A registry object containing the component tables.
    """
    conn = ibis.duckdb.connect()
    conn.create_table("entity_id", entity_id_table.to_pyarrow(), overwrite=True)
    conn.create_table("component_type", component_type_table.to_pyarrow(), overwrite=True)
    components = {
        "entity_id": conn.table("entity_id"),
        "component_type": conn.table("component_type"),
    }
    for comp_type, table in component_tables.items():
        if comp_type == "component_type":
            continue  # flags already incorporated into component_type_table
        if comp_type == "entity_id":
            continue  # the spine table already comes from entity_id_table;
            # a bare `entity_id` component here would clobber it.
        conn.create_table(comp_type, table.to_pyarrow(), overwrite=True)
        components[comp_type] = conn.table(comp_type)
    return Registry(conn, components)


FINAL_VAR = "registry"

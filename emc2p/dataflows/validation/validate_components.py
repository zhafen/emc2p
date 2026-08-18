import ast
import operator

import pandas as pd
import pandera
import pandera.ibis as pa
from hamilton.function_modifiers import extract_fields, unpack_fields
import ibis
import ibis.expr.datatypes as dt
import ibis.expr.types as ir

from ...registry import Registry


INPUT_COMPONENT_TYPES = ["entity_id"]

_ARITHMETIC_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_arithmetic_expr(expr: str) -> float | None:
    """Safely evaluate a simple numeric arithmetic expression string.

    Supports ``+ - * /`` and parentheses over int/float literals, e.g.
    ``"4 / 50"`` or ``"(1 + 2) * 3"``. Returns ``None`` if ``expr`` is not a
    valid expression of that form (e.g. it references names or calls).
    """
    def _eval(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _ARITHMETIC_OPS:
            return _ARITHMETIC_OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ARITHMETIC_OPS:
            return _ARITHMETIC_OPS[type(node.op)](_eval(node.operand))
        raise ValueError(f"Unsupported expression node: {node!r}")

    try:
        tree = ast.parse(expr, mode="eval").body
        return float(_eval(tree))
    except (SyntaxError, ValueError, TypeError, ZeroDivisionError):
        return None


@extract_fields({ct: ir.Table for ct in INPUT_COMPONENT_TYPES})
def components(registry: Registry) -> dict:
    """Extract components from the registry.

    Returns all registry components (not just INPUT_COMPONENT_TYPES) so that
    the downstream ``validation_results`` node can validate the full set.
    """
    return registry._components


def field(components: dict) -> ir.Table:
    """The registry's field table, used to look up field's own self-referential schema.

    A plain passthrough by default. Callers of this dataflow (e.g. base_etl)
    may override this node with a filtered subset if a pass ever needs to
    scope the schema lookup to fewer rows than the full field table.
    """
    return components["field"]


_IACS_TO_IBIS_TYPE: dict[str, dt.DataType] = {
    "str": dt.String(),
    "bool": dt.Boolean(),
    "int": dt.Int64(),
    "float": dt.Float64(),
    "rating": dt.Float64(),
}

_IACS_TO_PYTHON_TYPE: dict[str, type] = {
    "str": str,
    "bool": bool,
    "int": int,
    "float": float,
    "rating": float,
}

_IBIS_DTYPE = {bool: "boolean", str: "string", int: "int64", float: "float64"}

_INVALID_COLS = ["entity_id", "component_index", "component_type", "field", "value", "error_type"]
_INVALID_DTYPES = {
    "entity_id": "str", "component_index": "int64", "component_type": "str",
    "field": "str", "value": "str", "error_type": "str",
}


def _isnull(val) -> bool:
    if val is None:
        return True
    if isinstance(val, (list, dict)):
        return False
    try:
        result = pd.isna(val)
        return bool(result) if isinstance(result, (bool, type(result))) else False
    except (TypeError, ValueError):
        return False


def _parse_range(val):
    if isinstance(val, list):
        return val
    if _isnull(val):
        return None
    if isinstance(val, str):
        s = val.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, list):
                    return parsed
            except (ValueError, SyntaxError):
                pass
    return None


def _resolve_numeric_string(val):
    """Evaluate a numeric string that holds a simple arithmetic expression.

    Leaves plain numbers, null, and non-arithmetic strings untouched so they
    fall through to try_cast (which nulls invalid strings for later defaulting).
    """
    if not isinstance(val, str):
        return val
    s = val.strip()
    try:
        float(s)
        return val
    except ValueError:
        pass
    result = _eval_arithmetic_expr(s)
    return val if result is None else str(result)


def _coerce_default(val, py_type: type):
    if py_type is bool:
        if isinstance(val, str):
            return val.strip().lower() in ("true", "1", "yes")
        return bool(val)
    try:
        return py_type(val)
    except (ValueError, TypeError):
        return val


def _build_component_schemas(
    ctypes, field: ir.Table, entity_id: ir.Table,
    existing_field: ir.Table = None, existing_entity_id: ir.Table = None,
) -> dict[str, dict]:
    """Build a per-component-type schema dict from ((field)) rows.

    Parameters
    ----------
    ctypes : Iterable[str]
        Component type names to build schemas for (usually the keys of
        ``components``, but any component type with field definitions
        works — including "field" itself, which defines its own schema the
        same way it defines every other component type's schema).
    field : ir.Table
        The ``field`` component table from the registry.
    entity_id : ir.Table
        One row per entity (value, path, alias, entity_key, filepath),
        used to map entity_key -> entity_id for schema lookup.
    existing_field, existing_entity_id : ir.Table, optional
        The same two tables from a registry accumulated by prior updates
        (see ``existing_registry`` on ``field_validation_results``/
        ``validation_results``), consulted so a component type's schema —
        e.g. a user manifest's own field definitions, loaded once in an
        earlier update — is found even when this batch doesn't happen to
        include the files that declare it.

    Returns
    -------
    dict[str, dict]
        Dict of component_type -> {field_name: {type, nullable, default, range}}.
        Component types with no field definitions are omitted.
    """
    df_field = field.execute()
    df_entity = entity_id.execute()
    if existing_field is not None:
        df_field = pd.concat([existing_field.execute(), df_field], ignore_index=True)
    if existing_entity_id is not None:
        df_entity = pd.concat([existing_entity_id.execute(), df_entity], ignore_index=True)

    # Map entity_key -> entity_id for entities that have field definitions
    field_entity_ids = set(df_field["entity_id"].dropna().astype(str))
    key_to_eids: dict[str, list[str]] = {}
    for _, row in df_entity[["value", "entity_key"]].dropna().drop_duplicates().iterrows():
        eid, ekey = str(row["value"]), str(row["entity_key"])
        if eid in field_entity_ids:
            key_to_eids.setdefault(ekey, []).append(eid)

    component_schemas: dict[str, dict] = {}
    for ctype in ctypes:
        schema: dict[str, dict] = {}
        for eid in key_to_eids.get(ctype, []):
            for _, row in df_field[df_field["entity_id"] == eid].iterrows():
                fname = row.get("value")
                if _isnull(fname) or str(fname) in schema:
                    continue
                fname = str(fname)
                nullable = row.get("nullable")
                schema[fname] = {
                    "type": None if _isnull(row.get("type")) else str(row["type"]),
                    "nullable": True if _isnull(nullable) else bool(nullable),
                    "default": None if _isnull(row.get("default")) else row["default"],
                    "range": _parse_range(row.get("range")),
                }
        if schema:
            component_schemas[ctype] = schema
    return component_schemas


def _validate_component(ctype: str, table: ir.Table, schema: dict) -> tuple[ir.Table, list[ir.Table]]:
    """Validate and coerce a single component table against its schema.

    Build ibis mutate chains for column addition, type casting, and default
    filling, then validate types, nullable constraints, and categorical
    ranges with :mod:`pandera.ibis` in a single lazy pass. Constraint
    violations are collected from the raised :class:`pandera.errors.SchemaErrors`
    without pulling component data into memory.

    Parameters
    ----------
    ctype : str
        The component type name (used to label any violation rows).
    table : ir.Table
        The component's data table.
    schema : dict
        Schema for this component type, as built by ``_build_component_schemas``.
        An empty schema means the table passes through unchanged.

    Returns
    -------
    tuple[ir.Table, list[ir.Table]]
        The coerced table, and a list of violation sub-tables (each with
        columns entity_id, component_index, component_type, field, value,
        error_type) — empty if there were no constraint violations.
    """
    t = table
    violation_tables: list[ir.Table] = []

    if not schema:
        return t, violation_tables

    existing = set(t.columns)
    original_existing = set(existing)
    type_map = {
        fname: _IACS_TO_PYTHON_TYPE[fschema["type"]]
        for fname, fschema in schema.items()
        if fschema["type"] in _IACS_TO_PYTHON_TYPE
    }

    # Add missing typed columns as null
    for fname, py_type in type_map.items():
        if fname not in existing:
            t = t.mutate(**{fname: ibis.null().cast(_IBIS_DTYPE[py_type])})
            existing.add(fname)

    # Resolve simple arithmetic expressions (e.g. "4 / 50") in numeric string
    # columns before casting, so they evaluate instead of failing to null.
    t_schema = t.schema()
    arithmetic_cols = [
        fname for fname, py_type in type_map.items()
        if py_type in (float, int)
        and fname in original_existing
        and t_schema[fname].is_string()
    ]
    if arithmetic_cols:
        df = t.to_pandas()
        for fname in arithmetic_cols:
            df[fname] = df[fname].map(_resolve_numeric_string)
        t = ibis.memtable(df)

    # Cast typed columns; try_cast for numeric/bool strings so empty strings become NULL.
    # Pandera ibis does not support coerce for cross-type casting so we do this manually.
    t_schema = t.schema()
    for fname, py_type in type_map.items():
        expr = t[fname]
        if py_type in (bool, float, int) and fname in original_existing and t_schema[fname].is_string():
            t = t.mutate(**{fname: expr.try_cast(_IBIS_DTYPE[py_type])})
        else:
            t = t.mutate(**{fname: expr.cast(_IBIS_DTYPE[py_type])})

    # Apply defaults after casting so literal types match
    for fname, fschema in schema.items():
        default = fschema["default"]
        if default is None or fname not in existing:
            continue
        py_type = type_map.get(fname)
        if py_type is not None:
            lit = ibis.literal(_coerce_default(default, py_type)).cast(_IBIS_DTYPE[py_type])
        else:
            lit = ibis.literal(str(default))
        col = t[fname]
        if fschema["type"] == "str":
            col = col.nullif("")
        t = t.mutate(**{fname: col.fill_null(lit)})

    # Build a pandera schema with nullable and isin constraints, then validate lazily.
    # A single validate(lazy=True) call collects all constraint violations at once
    # rather than raising on the first failure.
    pa_columns = {}
    for fname, fschema in schema.items():
        if fname not in existing:
            continue
        ibis_type = _IACS_TO_IBIS_TYPE.get(fschema["type"])
        checks = []
        frange = fschema["range"]
        if fschema["type"] == "str" and isinstance(frange, list):
            checks.append(pa.Check.isin(frange))
        pa_columns[fname] = pa.Column(
            ibis_type,
            nullable=fschema["nullable"],
            checks=checks or None,
        )

    if pa_columns:
        pa_schema = pa.DataFrameSchema(pa_columns)
        try:
            t = pa_schema.validate(t, lazy=True)
        except pandera.errors.SchemaErrors as exc:
            for err in exc.schema_errors:
                col_name = getattr(err.schema, "name", None)
                if col_name not in existing:
                    continue
                reason = err.reason_code
                if reason == pandera.errors.SchemaErrorReason.SERIES_CONTAINS_NULLS:
                    # check_output is a boolean column aligned with t: True means null
                    violation_tables.append(
                        t.filter(err.check_output).select(
                            t["entity_id"],
                            t["component_index"],
                            ibis.literal(ctype).name("component_type"),
                            ibis.literal(col_name).name("field"),
                            ibis.null().cast("string").name("value"),
                            ibis.literal("nullable").name("error_type"),
                        )
                    )
                elif reason == pandera.errors.SchemaErrorReason.DATAFRAME_CHECK:
                    # isin violation: filter rows where value is not in the allowed set
                    frange = schema[col_name]["range"]
                    violation_tables.append(
                        t.filter(t[col_name].notnull() & ~t[col_name].isin(frange)).select(
                            t["entity_id"],
                            t["component_index"],
                            ibis.literal(ctype).name("component_type"),
                            ibis.literal(col_name).name("field"),
                            t[col_name].cast("string").name("value"),
                            ibis.literal("range").name("error_type"),
                        )
                    )

    return t, violation_tables


def _union_violations(violation_tables: list[ir.Table]) -> ir.Table:
    """Union violation sub-tables into one, or an empty table with the right schema.

    Components
    ----------
    - todo: This is not visible in the DAGs right now, which is a source of confusion.
    """
    if violation_tables:
        invalid_table = violation_tables[0]
        for vt in violation_tables[1:]:
            invalid_table = invalid_table.union(vt)
    else:
        invalid_table = ibis.memtable(
            pd.DataFrame(columns=_INVALID_COLS).astype(_INVALID_DTYPES)
        )
    return invalid_table


@unpack_fields("validated_field", "invalid_field_schema")
def field_validation_results(
    components: dict, field: ir.Table, entity_id: ir.Table,
) -> tuple[ir.Table, ir.Table]:
    """Validate and coerce ((field)) against its own self-referential schema.

    The ((field)) component type defines its own schema (value, description,
    type, nullable, unique, default, range, units, time_dimension) the same
    way it defines the schema for every other component type: via ``field``
    sub-entries attached to the entity with ``entity_key == "field"`` (see
    ``data_structure.field`` in builtins). This validates and type-coerces
    the registry's actual ``field`` table against that self-referential
    schema *before* field is used to validate every other component (see
    ``validation_results``), so e.g. ``time_dimension``/``nullable``/``unique``
    are real booleans rather than raw strings by the time downstream
    consumers read them.

    Note the rows validated/coerced come from ``components["field"]`` (the
    registry's actual, complete field table), not from the ``field``
    argument — that argument may be a filtered subset (``builtin_field`` in
    the first validation pass) used only to look up field's own schema. Since
    field is never itself subclassed/extended by users, its schema is always
    fully defined in builtins, so a builtins-only subset is sufficient for
    that lookup even in the first pass. This means the complete, validated
    field table can safely be written straight back into the registry in
    every pass (see ``validation_results``), without risking the data loss a
    filtered subset would cause.

    Mirrors ``validation_results``'s per-table validation exactly (see
    ``_build_component_schemas``/``_validate_component``), just scoped to
    field's own schema.

    Parameters
    ----------
    components : dict
        All of the registry's component tables (see ``components`` above).
    field : ir.Table
        A ``field`` table used only to look up field's own schema — may be
        a filtered subset (e.g. ``builtin_field``).
    entity_id : ir.Table
        One row per entity (value, path, alias, entity_key, filepath),
        used to map entity_key -> entity_id for schema lookup.

    Returns
    -------
    tuple[ir.Table, ir.Table]
        ``(validated_field, invalid_field_schema)`` where ``validated_field``
        is the registry's "field" table, type-coerced against its own
        schema, and ``invalid_field_schema`` is an ibis Table of rows that
        failed nullable or range constraints.
    """
    schema = _build_component_schemas(["field"], field, entity_id).get("field", {})
    validated_field, violations = _validate_component("field", components["field"], schema)
    return validated_field, _union_violations(violations)


def _empty_component_schema(fields: dict) -> ibis.Schema:
    """Build the schema an empty table of a declared-but-dataless component
    type would have: its own typed fields if it has any, or the generic
    ``value`` string column a fieldless (tag) component type is given by
    the loader instead.
    """
    cols = {"entity_id": "string", "component_index": "int64", "modifier": "string"}
    if fields:
        for fname, fschema in fields.items():
            py_type = _IACS_TO_PYTHON_TYPE.get(fschema["type"])
            cols[fname] = _IBIS_DTYPE.get(py_type, "string")
    else:
        cols["value"] = "string"
    return ibis.schema(cols)


def _declared_component_types(components: dict, entity_id: ir.Table) -> set[str]:
    """Return every component type name declared via a ``component_type`` tag.

    A component type is declared by attaching a bare ``component_type`` tag
    to its own defining entity (see ``emc2p_component`` in builtins) — the
    same mechanism a fieldless tag component, with no fields of its own, is
    declared by. This is a broader net than ``_build_component_schemas``,
    which only finds types that also have ``field`` sub-entries; a fieldless
    tag type has none, but is still a legitimate component type.
    """
    if "component_type" not in components:
        return set()
    eids = set(components["component_type"].execute()["entity_id"].dropna().astype(str))
    df_entity = entity_id.execute()
    matches = df_entity[df_entity["value"].isin(eids)]
    return set(matches["entity_key"].dropna().astype(str))


@unpack_fields("validated_components", "invalid_field", "declared_schemas")
def validation_results(
    components: dict, validated_field: ir.Table, entity_id: ir.Table,
    existing_registry: Registry = None,
) -> tuple[dict, ir.Table, dict]:
    """Use the schemas defined by the ((field)) component to validate and coerce
    the data in each component, including component types that used to be
    excluded as "infrastructure" (entity_id, parent, component_type,
    invalid_field, schema) — a user-authored manifest can add invalid
    records to those tables too, so they need the same validation/coercion
    pass as every user-defined component type.

    Materialise only the schema-defining rows (O(fields) records per component
    type), then delegate to ``_validate_component`` for the actual coercion and
    constraint checking, matching exactly what ``field_validation_results`` does
    to validate ``field`` against itself. "field" itself is skipped here (it
    was already validated by ``field_validation_results``) and its
    ``validated_field`` is folded straight into the result, so it gets
    overwritten in the registry the same way as every other component (see
    ``validated_registry``) — there's nothing special about preserving the
    unvalidated version.

    Parameters
    ----------
    components : dict
        Dict of component_type -> ibis Table, for every component type in
        the registry.
    validated_field : ir.Table
        The ``field`` component table, already validated and type-coerced
        against its own schema by ``field_validation_results``.
    entity_id : ir.Table
        One row per entity (value, path, alias, entity_key, filepath),
        used to map entity_key -> entity_id for schema lookup.
    existing_registry : Registry, optional
        The registry already accumulated from prior updates (see
        ``Registrar.update``), consulted for schema lookup so a component
        type's schema is found even when this batch doesn't happen to
        include the manifest file that declares it — e.g. a small,
        hand-built YAML fragment merged onto an entity whose component
        type was defined by a manifest loaded in an earlier update. Only
        used to look up schema; the components actually validated and
        returned are always this batch's own (see ``components``).

    Returns
    -------
    tuple[dict, ir.Table, dict]
        ``(validated_components, invalid_field, declared_schemas)`` where
        ``validated_components`` is a dict of component_type -> coerced ibis
        Table, ``invalid_field`` is an ibis Table of rows that failed
        nullable or range constraints, and ``declared_schemas`` is a dict of
        component_type -> ``ibis.Schema`` for every declared component type
        (see ``_declared_component_types``) that has no data in this batch —
        e.g. a fieldless tag component before any entity has been given
        that tag — for ``validated_registry`` to register via
        ``Registry.declare_schema``, so ``get``/``view``/``view_current``
        can return an empty, correctly-typed result for it instead of
        raising.
    """
    declared_types = _declared_component_types(components, entity_id)
    existing_field = existing_registry.get("field") if existing_registry is not None else None
    existing_entity_id = existing_registry.get("entity_id") if existing_registry is not None else None
    # Built for every declared type, not just ones with data this batch: a
    # type can be fully declared (its "field" definitions loaded) with zero
    # rows of its own yet, and declared_schemas below still needs its real,
    # typed fields rather than falling back to a generic "value" column.
    component_schemas = _build_component_schemas(
        set(components.keys()) | declared_types, validated_field, entity_id, existing_field, existing_entity_id
    )

    validated_comps: dict = {"field": validated_field}
    violation_tables: list[ir.Table] = []
    for ctype, table in components.items():
        if ctype == "field":
            continue
        t, v = _validate_component(ctype, table, component_schemas.get(ctype, {}))
        validated_comps[ctype] = t
        violation_tables.extend(v)

    declared_schemas = {
        ctype: _empty_component_schema(component_schemas.get(ctype, {}))
        for ctype in declared_types
        if ctype not in components
    }

    return validated_comps, _union_violations(violation_tables), declared_schemas


def validated_registry(
    registry: Registry,
    validated_components: dict,
    invalid_field: ir.Table,
    invalid_field_schema: ir.Table,
    declared_schemas: dict,
) -> Registry:
    """Store validated components and constraint violations back into the registry."""
    registry.update({
        **validated_components,
        "invalid_field": invalid_field.union(invalid_field_schema),
    })
    for ctype, schema in declared_schemas.items():
        registry.declare_schema(ctype, schema)
    return registry


FINAL_VAR = "validated_registry"

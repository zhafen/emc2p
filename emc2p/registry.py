"""ECS Registry for storing and accessing component data."""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import ibis
import pandas as pd

from .utils import candidate_entity_ids

_TABLE_META_COLS = {"entity_id", "component_index", "modifier"}


_KNOWN_SOURCE_EXTENSIONS = (".yaml", ".yml", ".py", ".csv")


def _origin_from_filepath(filepath) -> str:
    """"builtins" (or a downstream project's own registered tag, e.g.
    "iacs") for a builtins-sourced entity, else "user-defined" for one
    from a real on-disk manifest file.

    A builtins-dir file is identified by ``load_manifest.raw_strings`` as
    ``"{tag}.{stem}"`` -- no file extension, unlike every real source
    file's own relative-path identifier (``"examples/foo.yaml"``,
    ``"manifest/requirements.yaml"``, ...). Reading the tag straight off
    that naming convention avoids importing ``load_manifest`` here, which
    would be circular (it already imports ``Registry`` from this module).
    """
    if filepath is None or (isinstance(filepath, float) and pd.isna(filepath)):
        return "user-defined"
    filepath = str(filepath)
    if filepath.endswith(_KNOWN_SOURCE_EXTENSIONS):
        return "user-defined"
    return filepath.split(".", 1)[0]


def _format_field_value(value) -> str:
    """Render a field value the way a caller reading this as plain text
    expects.

    YAML/JSON-style true/false/null rather than Python's True/False/None
    or pandas's float NaN for a missing value, none of which mean anything
    outside their own library's repr.
    """
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "null"
    return str(value)


class GetterResult:
    """Wraps one getter's data, deferring the final shape to the caller.

    .to_table()/.to_pandas() for a possibly-multi-row result;
    .to_dict()/.to_scalar() when exactly one row (and, for .to_scalar(),
    exactly one non-id column) is expected -- this is how a getter's
    result becomes "singular" rather than there being a separate method
    for it. Scope, for now: just these tabular_output_formats
    projections -- not the other four output_format leaf categories from
    docs/manifest/getter_api_combinations.md. Future work, not silently
    dropped.
    """

    def __init__(self, table: ibis.Table):
        self._table = table

    def to_table(self) -> ibis.Table:
        """The result as a lazy ibis Table."""
        return self._table

    def to_pandas(self) -> pd.DataFrame:
        """The result as an executed pandas DataFrame."""
        return self._table.execute()

    def to_dict(self) -> dict:
        """The one matching row as {column: value}.

        Returns {} if nothing matched. Raises ValueError if more than one
        row matched -- ambiguous which one "the" result is.
        """
        df = self.to_pandas()
        if df.empty:
            return {}
        if len(df) > 1:
            raise ValueError(f"Expected exactly one row, got {len(df)}.")
        return df.iloc[0].to_dict()

    def to_scalar(self) -> Any:
        """The single value in the one matching row's one data column.

        Returns None if nothing matched. Raises ValueError if more than
        one row matched, or the row has more than one non-id column
        (ambiguous which value is "the" result) -- unlike the old
        get_current_value's behavior of silently taking the last matching
        row, this is a deliberate tightening.
        """
        row = self.to_dict()
        if not row:
            return None
        data_cols = [k for k in row if k not in ("entity_id", "entity_id.display_alias")]
        if len(data_cols) != 1:
            raise ValueError(f"Expected exactly one data column, got {data_cols}.")
        return row[data_cols[0]]

    def to_exploded_yaml(self, limit: int | None = None) -> str:
        """Render the result as a vertical, YAML-style list -- one block
        per row, keyed "row 0", "row 1", ... -- instead of a wide table.

        A wide markdown/pandas table forces a reader (human or model) to
        scan across many columns to read a single row, and wraps
        unreadably in a narrow terminal/chat context once there are more
        than a handful of columns. Exploding each row into its own
        vertical `key: value` block avoids both -- every row is
        self-contained and skimmable top-to-bottom, the same shape
        `Registry.view_entity`'s own per-entity markdown output already
        uses, just applied to an arbitrary result instead of one
        entity's own data.

        Args:
            limit: Show at most this many rows, with a trailing note of
                how many more weren't shown. None (default) shows every
                row.
        """
        df = self.to_pandas()
        if df.empty:
            return "(no rows)"
        shown = df if limit is None else df.head(limit)
        blocks = []
        for i, (_, row) in enumerate(shown.iterrows()):
            lines = [f"row {i}:"]
            lines += [f"  {col}: {_format_field_value(val)}" for col, val in row.items()]
            blocks.append("\n".join(lines))
        result = "\n".join(blocks)
        if limit is not None and len(df) > limit:
            result += f"\n... ({len(df) - limit} more row(s) not shown)"
        return result


class Registry:
    """A registry that stores ECS component data as ibis tables.

    Each component type has its own table backed by a DuckDB connection.
    """

    def __init__(self, conn: ibis.BaseBackend, components: dict):
        """Initialize the registry with a connection and components.

        Args:
            conn: An ibis DuckDB backend containing the component tables.
            components: A dict mapping component type names to ibis Tables
                (or other values like dicts). Keys other than "schema" are
                treated as component types.

        Metadata:
        - todo: We likely want to store the schema with the other component types and/or just have a filter on type.
        """
        ibis.options.interactive = True
        self._con = conn
        self._components = components
        self._component_types = [
            k for k, v in components.items()
            if k != "schema" and isinstance(v, ibis.Table)
        ]
        self._schemas: dict[str, ibis.Schema] = {
            k: v.schema()
            for k, v in components.items()
            if isinstance(v, ibis.Table)
        }
        self._component_instances_cache: ibis.Table | None = None

    def update(self, components: dict) -> None:
        """Add or overwrite component tables in the registry.

        Args:
            components: Dict mapping component type names to ibis Tables.
        """
        for comp_type, table in components.items():
            if not isinstance(table, ibis.Table):
                table = ibis.memtable(table)
            table = self._resolve_null_typed_columns(comp_type, table)
            self._con.create_table(comp_type, table, overwrite=True)
            self._components[comp_type] = self._con.table(comp_type)
            self._schemas[comp_type] = self._con.table(comp_type).schema()
            if comp_type not in self._component_types and comp_type != "schema":
                self._component_types.append(comp_type)
        self._component_instances_cache = None

    def _resolve_null_typed_columns(self, comp_type: str, table: ibis.Table) -> ibis.Table:
        """Give a concrete type to any column DuckDB would otherwise reject.

        A column with no non-null value anywhere (e.g. an optional field
        nothing in this batch set) has nothing for pandas/pyarrow to infer a
        concrete dtype from, so building it as an in-memory ibis table can
        hand back a NULL-typed column -- DuckDB's own `create_table` refuses
        those outright. Prefer the type this component type is already
        known to have (its previous version in this registry always had
        real data to infer from at some point); fall back to `string` for a
        column with no prior version, matching how this codebase already
        types every other optional, no-data-yet text field (``modifier``,
        ``units``, ...).
        """
        schema = table.schema()
        null_fields = schema.null_fields
        if not null_fields:
            return table
        existing_schema = self._schemas.get(comp_type)
        casts = {}
        for fname in null_fields:
            if (
                existing_schema is not None
                and fname in existing_schema
                and not existing_schema[fname].is_null()
            ):
                dtype = existing_schema[fname]
            else:
                dtype = "string"
            casts[fname] = table[fname].cast(dtype)
        return table.mutate(**casts)

    def declare_schema(self, component_type: str, schema: ibis.Schema) -> None:
        """Register `component_type`'s schema without creating a physical table.

        Lets `get`/`view`/`view_current` return an empty, correctly-typed
        result for a component type that's declared (e.g. via a
        `component_type` tag in a loaded manifest) but has no data yet, the
        same way they already do for one that was created and later
        emptied. A no-op if a physical table for `component_type` already
        exists: real data's own schema always takes precedence over a
        merely declared one.

        Args:
            component_type: The component type this schema describes.
            schema: The columns/dtypes an empty table of this type would have.
        """
        if component_type in self._component_types:
            return
        self._schemas[component_type] = schema

    def merge(self, other: "Registry") -> None:
        """Union all component tables from another registry into this one.

        Existing component types are unioned and deduplicated; new component
        types are added directly.

        For any component type with a time_dimension field (see
        ``_time_dimension_field``), also maintains a ``_seq_{field}`` write-order
        column on that table (see ``_seq_column``) — assigned only to rows
        genuinely new to this registry, never reassigned to a row already
        present, so it records each row's relative write order stably across
        every future merge. This is what lets ``_current_table`` break a tie
        between two rows sharing the same time_dimension value
        deterministically, instead of relying on an arbitrary,
        backend-dependent window-function order.

        Args:
            other: Registry whose component tables are merged in.
        """
        for comp_type in other.component_types:
            seq_col = self._seq_column(comp_type, other)
            incoming_table = other.get(comp_type)
            if comp_type in self._component_types:
                arrow_data = incoming_table.to_pyarrow()
                tmp = f"_merge_{comp_type}"
                self._con.create_table(tmp, arrow_data, overwrite=True)
                existing = self.get(comp_type)
                incoming = self._con.table(tmp)

                # Add NULL columns for any fields present in one table but not the other
                # so that ibis union can operate on matching schemas.
                existing_cols = set(existing.columns)
                incoming_cols = set(incoming.columns)
                existing_schema = existing.schema()
                incoming_schema = incoming.schema()
                for col in incoming_cols - existing_cols:
                    existing = existing.mutate(
                        ibis.null().cast(incoming_schema[col]).name(col)
                    )
                for col in existing_cols - incoming_cols:
                    incoming = incoming.mutate(
                        ibis.null().cast(existing_schema[col]).name(col)
                    )
                if seq_col:
                    if seq_col not in existing.columns:
                        existing = existing.mutate(ibis.null().cast("int64").name(seq_col))
                    if seq_col not in incoming.columns:
                        incoming = incoming.mutate(ibis.null().cast("int64").name(seq_col))

                all_cols = sorted(existing.columns)
                if seq_col:
                    merged = self._merge_with_sequence(existing, incoming, seq_col, all_cols)
                else:
                    merged = existing.select(all_cols).union(
                        incoming.select(all_cols), distinct=True
                    )
                self.update({comp_type: merged})
                self._con.drop_table(tmp)
            else:
                if seq_col:
                    incoming_table = self._with_initial_sequence(incoming_table, seq_col)
                self.update({comp_type: incoming_table.to_pyarrow()})

        # Carry forward any of other's declared-but-dataless schemas (see
        # declare_schema) too, not just its physical tables.
        for comp_type, schema in other._schemas.items():
            if comp_type not in other.component_types:
                self.declare_schema(comp_type, schema)

    def _seq_column(self, component_type: str, other: "Registry") -> str | None:
        """Return `component_type`'s ``_seq_{field}`` write-order column name, if any.

        Only component types with a time_dimension field get one. Checked
        against ``other`` (this batch's own resolved schema) first, falling
        back to ``self`` (the accumulated registry's own schema, carried
        forward across every past merge) when ``other`` doesn't declare it
        -- the same fallback ``derive_components.time_filled_registry`` uses
        for backfilling time_dimension values, applied here to the sibling
        problem of losing tie-breaking for a component type whose schema
        was only ever declared in an earlier `update()` call, not this batch's own.

        Returns ``None`` without consulting ``_time_dimension_field`` on a
        registry that lacks a ``field`` or ``entity_id`` table -- a minimal,
        hand-built registry (e.g. in a unit test exercising ``merge`` in
        isolation) structurally cannot have declared any time_dimension
        field, so "no seq column" is the correct answer, not a special case
        to route around.
        """
        for registry in (other, self):
            if "field" not in registry._components or "entity_id" not in registry._components:
                continue
            time_field = registry._time_dimension_field(component_type)
            if time_field:
                return f"_seq_{time_field}"
        return None

    def _with_initial_sequence(self, table: ibis.Table, seq_col: str) -> ibis.Table:
        """Number every row of `table` 1..N in `seq_col`, for a component type
        this registry has no prior rows for (so nothing to preserve).

        Ordered by (entity_id, component_index) for a stable, reproducible
        result — a best-effort proxy for authoring order among rows that all
        arrive in this single merge call, not a guarantee of matching the
        exact order they were originally written across separate sessions
        (e.g. after a cold reload from an exported save, see ``merge``'s
        docstring caveat via ``_merge_with_sequence``).
        """
        order_cols = [table["entity_id"]]
        if "component_index" in table.columns:
            order_cols.append(table["component_index"])
        return table.mutate(**{seq_col: ibis.row_number().over(order_by=order_cols) + 1})

    def _merge_with_sequence(
        self, existing: ibis.Table, incoming: ibis.Table, seq_col: str, all_cols: list[str]
    ) -> ibis.Table:
        """Union `existing` and `incoming`, assigning fresh `seq_col` values
        only to rows genuinely new to `existing`.

        "Genuinely new" means: not already present in `existing` when
        compared on every column except `seq_col` — the same comparison
        ``merge``'s plain union/distinct path already uses to dedupe, just
        computed explicitly here instead of left to `union(distinct=True)`
        (which would otherwise treat two copies of an existing row as
        distinct the moment they disagree on `seq_col`, since `incoming`
        never carries one).

        A row already in `existing` keeps whatever `seq_col` value it already
        has, so a sequence number, once assigned, is never rewritten — this
        is what makes two rows' *relative* write order stable across every
        later merge. New rows are numbered starting one past `existing`'s
        current maximum, in (entity_id, component_index) order (see
        `_with_initial_sequence`) — reproducible within this merge call, but
        only a best-effort proxy for true write order across a cold reload
        from an exported save, where a whole file's worth of history arrives
        as a single "new" batch with no prior `_seq` to continue from.
        """
        non_seq_cols = [c for c in all_cols if c != seq_col]
        new_rows = incoming.select(non_seq_cols).difference(existing.select(non_seq_cols))

        existing_seq = existing.select(seq_col).to_pandas()[seq_col]
        start = int(existing_seq.max()) + 1 if existing_seq.notna().any() else 1

        order_cols = [new_rows["entity_id"]]
        if "component_index" in non_seq_cols:
            order_cols.append(new_rows["component_index"])
        new_rows = new_rows.mutate(
            **{seq_col: ibis.row_number().over(order_by=order_cols) + start}
        )

        return existing.select(all_cols).union(new_rows.select(all_cols), distinct=True)

    def to_database(self, path: str | Path) -> None:
        """Export all component tables as-is to a database.

        Uses ``ibis.connect`` so the backend is inferred from ``path``: a
        plain filesystem path with a ``.duckdb`` extension connects via
        DuckDB, while a URL such as ``"sqlite:///registry.db"`` or
        ``"postgres://user:pass@host/db"`` connects to that backend instead.

        Args:
            path: A URL or filesystem path resolvable by ``ibis.connect``.
        """
        out_con = ibis.connect(str(path))
        for comp_type in self._component_types:
            out_con.create_table(
                comp_type, self.get(comp_type).to_pyarrow(), overwrite=True
            )
        out_con.disconnect()

    @classmethod
    def from_database(cls, path: str | Path) -> "Registry":
        """Load a Registry from a database written by ``to_database``.

        Args:
            path: A URL or filesystem path resolvable by ``ibis.connect``.
        """
        con = ibis.connect(str(path))
        components = {name: con.table(name) for name in con.list_tables()}
        return cls(con, components)

    @classmethod
    def from_component_rows(
        cls, components: dict[str, list[dict]], conn: ibis.BaseBackend | None = None
    ) -> "Registry":
        """Build a Registry directly from component-first row data.

        Backed by an in-memory DuckDB connection by default, or any other
        ibis backend passed via ``conn`` (e.g. a test that specifically
        needs to exercise non-DuckDB behavior). Convenient for hand-written
        data (e.g. test fixtures) -- not how production code builds a
        Registry, which always goes through the Hamilton
        load_manifest/registrar.update() pipeline instead.

        Args:
            components: Dict mapping component type names to lists of row
                dicts. Each row dict should include "entity_id" plus any
                component fields.
            conn: An ibis backend to create the component tables in.
                Defaults to a fresh in-memory DuckDB connection.
        """
        conn = conn if conn is not None else ibis.duckdb.connect()
        comp_tables = {}
        for comp_type, rows in components.items():
            df = pd.DataFrame(rows)
            conn.create_table(comp_type, df)
            comp_tables[comp_type] = conn.table(comp_type)
        return cls(conn, comp_tables)

    def close(self) -> None:
        """Close the underlying database connection."""
        self._con.disconnect()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:  # pylint: disable=broad-except
            pass

    @property
    def component_types(self) -> list[str]:
        """Return the list of component types in the registry."""
        return list(self._component_types)

    @property
    def known_component_types(self) -> list[str]:
        """Every component type this registry knows about, whether any
        entity has written data to it yet or not -- a superset of `component_types`.

        Includes types declared via `declare_schema` (e.g. carried forward
        on merge from a loaded manifest, see `merge`'s own docstring) that
        nothing has recorded data for yet -- `component_types` alone can't
        distinguish "this type doesn't exist" from "this type exists but
        nobody's used it yet", which is exactly the gap that let a type
        like `location` go unnoticed until something actually wrote to it.
        """
        return list(self._schemas)

    def get(self, key: str):
        """Return the component table for the given component type.

        If the component type does not exist but its schema is known, returns
        an empty table with that schema. If the schema is also unknown, returns
        an empty table with only an ``entity_id`` string column.
        """
        if key in self._components:
            return self._components[key]
        if key in self._schemas:
            return ibis.memtable([], schema=self._schemas[key])
        return ibis.memtable([], schema={"entity_id": "string", "value": "string"})

    def view(
        self, component_type: str | list[str], entity: str | list[str] | None = None
    ) -> GetterResult:
        """Return the joined data for the given component type(s), as a GetterResult.

        Args:
            component_type: A component type name, a dotted "table.field"
                string, or a list of either. All results are inner-joined by
                entity_id with columns named "table.field". ``entity_id.display_alias``
                is prepended automatically unless already requested.
            entity: An entity ref, or list of them, to filter the result
                down to. Each is resolved the same way `get_entity_id` does
                (exact hash, else exact display_alias, else substring match), except
                a ref matching more than one entity is not an error here —
                every match is included, and a warning is raised (this is
                one of the few contexts where an ambiguous ref is fine, since
                the caller is filtering a view, not picking a single target
                to write to). A ref matching zero entities also warns, and
                contributes nothing to the result.

        Raises:
            KeyError: If a component type doesn't exist in the registry.
        """
        return GetterResult(self._view(component_type, self._table_or_declared, entity))

    def view_current(
        self, component_type: str | list[str], entity: str | list[str] | None = None
    ) -> GetterResult:
        """Like ``view``, but collapsed to the most recent version of each record.

        For any component type with a field flagged ``time_dimension: true`` in
        its schema (directly or via inheritance), only the row with the
        maximum time_dimension value is kept per entity_id — i.e. the current
        version of a slowly changing dimension. Component types with no
        time_dimension field are returned unchanged.

        Grouped by entity_id alone, not (entity_id, component_index,
        modifier): a component_index/modifier isn't guaranteed stable across
        the separate writes that accumulate one entity's SCD history (e.g.
        independent merges each computing their own component_index from
        scratch), so grouping on it risks splitting one entity's history into
        multiple unrelated "current" rows. SCD data is expected to be unique
        per entity per point in time, so entity_id is the only key that's
        safe to rely on.

        Args:
            component_type: Same as ``view``.
            entity: Same as ``view``.

        Raises:
            KeyError: If a component type doesn't exist in the registry.
            ValueError: If a component type has more than one time_dimension field.
        """
        return GetterResult(self._view(component_type, self._current_table, entity))

    def safe_view(
        self, component_type: str | list[str], entity: str | list[str] | None = None
    ) -> GetterResult:
        """Like `view`, but returns an empty GetterResult instead of raising
        for an unknown component type.

        A component type that's declared (see `declare_schema`) but has no
        data yet still comes back as an empty result, not an error -- only a
        component type unknown to the registry entirely is safed away.

        Args:
            component_type: Same as `view`.
            entity: Same as `view`.
        """
        try:
            return self.view(component_type, entity)
        except KeyError:
            return GetterResult(ibis.memtable([], schema={"entity_id": "string", "value": "string"}))

    def safe_view_current(
        self, component_type: str | list[str], entity: str | list[str] | None = None
    ) -> GetterResult:
        """Like `safe_view`, but resolves each field's current row per entity (see `view_current`)."""
        try:
            return self.view_current(component_type, entity)
        except KeyError:
            return GetterResult(ibis.memtable([], schema={"entity_id": "string", "value": "string"}))

    def view_entities(self, entity: str | list[str]) -> GetterResult:
        """Return all recorded component data for the given entity/entities.

        Symmetric to `view`: that fixes which components and lets entities
        vary; this fixes which entities and returns everything recorded
        about them across every component type, one row per entity, via a
        full outer join on entity_id -- an inner join would drop almost
        every entity here, since essentially none have data in every
        component table. A component missing for a given entity comes back
        as nulls for that entity's row, not a dropped row.

        `entity` is resolved via the same logic `view`'s own `entity` uses
        (see `_resolve_aliases`); required here, since choosing entities is
        this method's whole purpose.

        For a single entity's data as a flat mapping (replacing the old
        get_current_value for the common "current value of one field"
        case), pair this with GetterResult.to_dict()/.to_scalar(), e.g.
        ``view_entities_current(entity).to_dict()``.

        Raises:
            KeyError: If `entity` resolves to no entity at all. An individual
                unresolvable ref within a list just warns (see
                `_resolve_aliases`) as long as at least one ref resolves.
        """
        resolved = self._resolve_aliases(entity)
        if not resolved:
            raise KeyError(f"{entity!r} did not resolve to any entity.")
        component_type = [ct for ct in self.component_types if ct != "entity_id"]
        result = self._view(component_type, self._table_or_declared, how="outer")
        return GetterResult(result.filter(result["entity_id"].isin(resolved)))

    def view_entities_current(self, entity: str | list[str]) -> GetterResult:
        """Like `view_entities`, collapsed to current SCD value (see `view_current`)."""
        resolved = self._resolve_aliases(entity)
        if not resolved:
            raise KeyError(f"{entity!r} did not resolve to any entity.")
        component_type = [ct for ct in self.component_types if ct != "entity_id"]
        result = self._view(component_type, self._current_table, how="outer")
        return GetterResult(result.filter(result["entity_id"].isin(resolved)))

    def safe_view_entities(self, entity: str | list[str]) -> GetterResult:
        """Like `view_entities`, but returns an empty GetterResult instead
        of raising when `entity` resolves to nothing."""
        try:
            return self.view_entities(entity)
        except KeyError:
            return GetterResult(
                ibis.memtable([], schema={"entity_id": "string", "entity_id.display_alias": "string"})
            )

    def safe_view_entities_current(self, entity: str | list[str]) -> GetterResult:
        """Like `safe_view_entities`, collapsed to current SCD value (see `view_entities_current`)."""
        try:
            return self.view_entities_current(entity)
        except KeyError:
            return GetterResult(
                ibis.memtable([], schema={"entity_id": "string", "entity_id.display_alias": "string"})
            )

    def _resolve_aliases(self, aliases: str | list[str]) -> set[str]:
        """Resolve `aliases` to the union of entity_ids they match.

        Unlike `get_entity_id`, a ref matching more than one entity isn't
        collapsed to "unresolvable" here — all matches are kept, and a
        warning is raised instead of silently picking or dropping one. A ref
        matching no entity also warns, contributing nothing.
        """
        if isinstance(aliases, str):
            aliases = [aliases]
        entity_id_df = (
            self._components["entity_id"].to_pandas()
            if "entity_id" in self._components else pd.DataFrame(columns=["value"])
        )
        resolved: set[str] = set()
        for alias in aliases:
            candidates = candidate_entity_ids(alias, entity_id_df)
            if not candidates:
                warnings.warn(f"{alias!r} in `aliases` matched no entity.")
            elif len(candidates) > 1:
                warnings.warn(
                    f"{alias!r} in `aliases` matched {len(candidates)} entities "
                    f"({candidates}); including all of them."
                )
            resolved.update(candidates)
        return resolved

    def _known(self, table_name: str) -> bool:
        """True if `table_name` has a physical table or a declared schema."""
        return table_name in self._con.list_tables() or table_name in self._schemas

    def _table_or_declared(self, table_name: str) -> ibis.Table:
        """`table_name`'s physical table, or an empty one from its declared schema.

        Lets `view`/`view_current` work for a component type that's
        declared (see `declare_schema`) but has no data yet, the same way
        `get` already does — inner-joining against an always-empty table
        naturally yields the right answer (no rows) for a field that lives
        on a component type nothing has been written for.
        """
        if table_name in self._con.list_tables():
            return self._con.table(table_name)
        return self.get(table_name)

    def _view(
        self,
        component_type: str | list[str],
        table_fn,
        aliases: str | list[str] | None = None,
        how: str = "inner",
    ) -> ibis.Table:
        if isinstance(component_type, str):
            component_type = [component_type]

        if (
            "entity_id.display_alias" not in component_type
            and "entity_id" not in component_type
        ):
            component_type = ["entity_id.display_alias"] + list(component_type)

        # Resolve each entry to a (table_name, field) pair, expanding bare
        # table names to all of their non-meta fields.
        pairs: list[tuple[str, str]] = []
        for ct in component_type:
            if "." not in ct:
                if not self._known(ct):
                    raise KeyError(ct)
                t = table_fn(ct)
                skip = _TABLE_META_COLS | ({"value"} if ct == "entity_id" else set())
                pairs.extend(
                    (ct, f) for f in t.columns
                    if f not in skip and not f.startswith("_seq_")
                )
            else:
                table_name, field = ct.split(".", 1)
                if not self._known(table_name):
                    raise KeyError(table_name)
                pairs.append((table_name, field))

        # Group fields by table so that multiple fields from the same table
        # are selected together in a single pass.
        fields_by_table: dict[str, list[str]] = {}
        for table_name, field in pairs:
            fields = fields_by_table.setdefault(table_name, [])
            if field not in fields:
                fields.append(field)

        tables_to_join = []
        for table_name, fields in fields_by_table.items():
            t = table_fn(table_name)
            if table_name == "entity_id":
                cols = [t["value"].name("entity_id")]
            else:
                cols = ["entity_id"]
            cols += [t[f].name(f"{table_name}.{f}") for f in fields]
            tables_to_join.append(t.select(cols))

        result = tables_to_join[0]
        for t in tables_to_join[1:]:
            result = result.join(t, "entity_id", how=how)
            if how != "inner":
                result = result.mutate(
                    entity_id=ibis.coalesce(result["entity_id"], result["entity_id_right"])
                ).drop("entity_id_right")

        if aliases is not None:
            resolved = self._resolve_aliases(aliases)
            result = result.filter(result["entity_id"].isin(resolved))

        return result

    def _current_table(self, table_name: str) -> ibis.Table:
        """Return ``table_name`` collapsed to the latest row per entity_id,
        using its time_dimension field.

        Assumes the registry was produced by ``base_etl`` (``field`` and
        ``entity_id`` are present). Tables with no time_dimension field
        are returned unchanged.

        Two rows tied on the time_dimension value are broken by
        ``_seq_{field}`` (see ``merge``) when present, most-recent-write
        wins — so a second update in the same in-world turn doesn't lose
        to the first via an arbitrary, backend-dependent window-function
        order. A table with no ``_seq_{field}`` column (e.g. one inserted
        directly rather than through ``merge``) falls back to that
        arbitrary tie-break.

        Raises:
            ValueError: If ``table_name`` has more than one time_dimension field.
        """
        t = self._table_or_declared(table_name)
        time_field = self._time_dimension_field(table_name)
        if time_field is None or time_field not in t.columns:
            return t

        order_by = [t[time_field].desc(nulls_first=False)]
        seq_col = f"_seq_{time_field}"
        if seq_col in t.columns:
            order_by.append(t[seq_col].desc(nulls_first=False))

        ranked = t.mutate(
            _scd_rank=ibis.row_number().over(group_by="entity_id", order_by=order_by)
        )
        return ranked.filter(ranked["_scd_rank"] == 0).drop("_scd_rank")

    def _time_dimension_field(self, component_type: str) -> str | None:
        """Return the field flagged ``time_dimension: true`` for a component type, if any.

        Assumes the registry was produced by ``base_etl``, so ``field``
        (inheritance-resolved, see ``inherit_components.derived_registry``)
        and ``entity_id`` are present, and ``field["time_dimension"]`` is
        already a real bool — ``field`` is validated against its own schema
        (see ``validate_components.field_validation_results``) as part of
        ``validate_registry``, and all data is expected to reach the
        registry only by going through that pass. If this raises or behaves
        unexpectedly, the registry likely didn't go through the full
        pipeline (e.g. component tables inserted directly) — that's a bug in
        the caller, not something to work around here.

        Raises:
            ValueError: If more than one field is flagged time_dimension for
                this component type — only one is allowed.
        """
        df_field = self._components["field"].execute()
        if "time_dimension" not in df_field.columns:
            return None

        df_entity = self._components["entity_id"].execute()
        def_eids = df_entity.loc[df_entity["display_key"] == component_type, "value"]

        matches = df_field[df_field["entity_id"].isin(def_eids)]
        fields = sorted({
            str(row["value"])
            for _, row in matches.iterrows()
            if row["time_dimension"]
        })
        if len(fields) > 1:
            raise ValueError(
                f"Component type {component_type!r} has multiple time_dimension "
                f"fields {fields}; only one is allowed."
            )
        return fields[0] if fields else None

    def summarize_components(self, limit: int = 20) -> str:
        """Markdown report of every component type currently holding data, one
        section per type with its row count and up to `limit` sample rows.

        Only `component_types` (data-bearing), not the larger
        `known_component_types` -- nothing to report on a type nobody's
        written to yet. Unlike `view` (one type at a time), this covers
        everything in one call. Purely a data report -- no judgment or
        instructions attached to it; a caller wanting to prompt a reader
        toward consolidating duplicate/misplaced data on top of this (e.g.
        `validate_write`'s consolidation guidance, composed in by
        `commands.cmd_review_components`) is free to add its own.
        """
        types = sorted(self.component_types)
        if not types:
            return "No component types have any data yet -- nothing to review."
        sections = []
        for component_type in types:
            df = self.view(component_type).to_pandas()
            sample = df.head(limit).fillna("null").to_markdown(index=False)
            if len(df) > limit:
                sample += f"\n... ({len(df) - limit} more row(s) not shown)"
            sections.append(f"### {component_type} ({len(df)} row(s))\n{sample}")
        return (
            "All component types currently recorded, one section per type:\n\n"
            + "\n\n".join(sections)
        )

    def component_instances(self) -> ibis.Table:
        """Return the full, unfiltered inventory of every component
        instance currently in the registry: one row per (entity_id,
        component_index) across every component table, tagged with which
        component_type it belongs to, plus that type's own declared
        flags (``derived``/``skip_on_export``/``implicit_parent``, as of
        this writing -- see ``component_type``) broadcast onto it, and
        ``declares_type_name`` (non-null only on a ``component_type`` tag
        row itself, carried straight from ``component_type``'s own rows).

        Computed on demand from whatever's currently in ``entity_id``'s
        sibling component tables (excluding ``entity_id`` itself, the
        spine -- no entity is an "instance of" its own identity), not
        stored as one more entry in ``_components``: unlike a real
        component type, no entity "has" a component_instance. Keeping it
        a components-dict citizen (as it briefly was, see task #14) meant
        a second, easily-stale copy of every other table's own meta
        columns, and every generic "for each real component type" loop
        in this codebase needing to remember to exclude it -- close to
        the same failure mode task #14 fixed for `known_component_types`
        in the first place. This is the derived-view replacement.

        Cached after first computation; invalidated on the next
        ``update()`` (and therefore ``merge()``, which calls it
        internally), so a stale answer here is not possible -- unlike the
        stored-table version, there is exactly one source of truth (the
        registry's own other component tables) and this is always a
        fresh read of it, or a cached copy of that same fresh read.

        Returns
        -------
        ibis.Table
            Columns: entity_id, component_index, component_type, modifier,
            declares_type_name, plus one column per component_type flag.
        """
        if self._component_instances_cache is not None:
            return self._component_instances_cache

        base_schema = {
            "entity_id": "string",
            "component_index": "int64",
            "component_type": "string",
            "modifier": "string",
            "declares_type_name": "string",
        }

        defs_df = None
        flag_names: list[str] = []
        flags_by_type_name: dict[str, dict[str, bool]] = {}
        if "component_type" in self._components:
            defs_df = self._components["component_type"].execute()
            flag_names = [c for c in defs_df.columns if c not in base_schema]
            if flag_names:
                flags_by_type_name = defs_df.set_index("declares_type_name")[flag_names].to_dict(orient="index")

        parts = []
        if defs_df is not None:
            parts.append(defs_df)  # own rows, own flags, own declares_type_name -- used as-is
        for name, table in self._components.items():
            if name in ("entity_id", "component_type"):
                continue
            df = table.execute()
            if "entity_id" not in df.columns or "component_index" not in df.columns:
                continue
            part = df[["entity_id", "component_index"]].copy()
            part["modifier"] = df["modifier"] if "modifier" in df.columns else pd.NA
            part["component_type"] = name
            part["declares_type_name"] = pd.NA
            type_flags = flags_by_type_name.get(name, {})
            for flag in flag_names:
                part[flag] = type_flags.get(flag, False)
            parts.append(part)

        if not parts:
            self._component_instances_cache = ibis.memtable([], schema=base_schema)
            return self._component_instances_cache

        combined = pd.concat(parts, ignore_index=True)
        combined["modifier"] = combined["modifier"].astype(pd.StringDtype())
        combined["declares_type_name"] = combined["declares_type_name"].astype(pd.StringDtype())
        combined["component_type"] = combined["component_type"].astype(pd.StringDtype())
        for flag in flag_names:
            combined[flag] = combined[flag].astype(bool)
        self._component_instances_cache = ibis.memtable(combined)
        return self._component_instances_cache

    def component_type_overview(self) -> GetterResult:
        """Return one row per declared component type: its name
        (``declares_type_name``), description (if it has one), how many
        distinct entities carry an instance of it, and its origin (which
        builtins tag defined it, e.g. ``"builtins"``, or ``"user-defined"``
        for one declared in a real manifest file).

        Built entirely from `view`'s own join engine (anchored on
        ``component_type`` -- the type definitions table -- by putting it
        first in the field list) left-joined against ``description`` and
        ``entity_id``'s own ``display_alias``/``filepath``, rather than any
        bespoke joining logic of its own. Left, not `view`'s own inner
        default, so a type declared without ever being given its own
        description still gets a row here with a null description,
        instead of silently vanishing.

        Entity counts come from ``component_instances()`` (the full
        per-instance inventory, derived on demand rather than stored --
        see that method): grouped by its own ``component_type`` column
        (the declared type's name) and counted by distinct ``entity_id``,
        not raw row count -- an entity with multiple instances of the
        same type (e.g. SCD history) should still count once.

        Returns
        -------
        GetterResult
            Columns: entity_id, component_type.declares_type_name,
            entity_id.display_alias, entity_id.filepath,
            description.value, entity_count, origin.
        """
        empty_schema = {
            "entity_id": "string",
            "component_type.declares_type_name": "string",
            "entity_id.display_alias": "string",
            "entity_id.filepath": "string",
            "description.value": "string",
            "entity_count": "int64",
            "origin": "string",
        }
        try:
            joined = self._view(
                [
                    "component_type.declares_type_name",
                    "entity_id.display_alias",
                    "entity_id.filepath",
                    "description.value",
                ],
                self._table_or_declared,
                how="left",
            )
            df = joined.execute()
        except KeyError:
            # component_type/entity_id/description isn't known to this
            # registry at all (e.g. a minimal hand-built test registry) --
            # nothing to report, the same way `safe_view` degrades.
            df = pd.DataFrame()
        if df.empty:
            return GetterResult(ibis.memtable([], schema=empty_schema))

        instance_df = self.component_instances().execute()
        entity_counts = (
            instance_df.groupby("component_type")["entity_id"].nunique()
            if not instance_df.empty else pd.Series(dtype="int64")
        )
        df["entity_count"] = (
            df["component_type.declares_type_name"].map(entity_counts).fillna(0).astype("int64")
        )

        df["origin"] = df["entity_id.filepath"].apply(_origin_from_filepath)

        return GetterResult(ibis.memtable(df))

    def get_entity_id(self, entity_ref: str) -> str | None:
        """Resolve `entity_ref` to its canonical entity_id hash.

        The read-only counterpart to `candidate_entity_ids`, which this
        delegates to directly: it's the exact same resolution ETL uses for
        `entity_ref` fields and `same_as` targets (exact hash, else exact
        display_alias, else substring match against every entity's full path), kept
        only if it resolves to exactly one entity.

        Deliberately tolerant rather than raising — unlike `same_as` target
        resolution, where an unresolvable reference is a manifest error
        worth failing loudly on, a caller resolving an arbitrary ref (e.g.
        one it read from a still-in-flux registry) usually just wants to
        know whether a matching entity currently exists.

        Args:
            entity_ref: An entity hash, display_alias, or path fragment identifying
                the entity.

        Returns:
            The resolved entity_id hash, or `None` if `entity_ref` doesn't
            resolve to exactly one entity.
        """
        if "entity_id" not in self._components:
            return None
        entity_id_df = self._components["entity_id"].to_pandas()
        candidates = candidate_entity_ids(entity_ref, entity_id_df)
        return candidates[0] if len(candidates) == 1 else None

    def view_entity(self, entity_id: str, format: str = "markdown") -> str:
        """Return all component data for a specific entity as a formatted string.

        Queries each component type individually (like the old
        view_entity_df) rather than through view_entities' single joined
        query -- an entity with real multi-row history (SCD or otherwise)
        in more than one component type would fan out combinatorially if
        those components were cross-joined together, which is exactly
        what view_entities' full-history join does. Querying one
        component type at a time, each already filtered to this entity,
        avoids that: every historical row for every component type is
        shown, without cross-multiplying them against each other.

        Args:
            entity_id: Entity hash, display_alias, or path fragment identifying the
                entity (see `get_entity_id`).
            format: Output format — "markdown" (default, a `key: value`
                outline, not a table) or "csv".
        """
        resolved_id = self.get_entity_id(entity_id)
        by_type: dict[str, pd.DataFrame] = {}
        if resolved_id is not None:
            for comp_type in self.component_types:
                # "component_type" isn't entity data -- it's the
                # registry's own bookkeeping of which component type(s)
                # this entity declares (see
                # emc2p.dataflows.etl.load_manifest.component_type_table),
                # each shaped identically: derived/implicit_parent/
                # skip_on_export. Every other section already names its
                # own type in its "## " heading, so showing this too is
                # pure noise. (The full per-instance inventory --
                # `component_instances()` -- isn't in `component_types`
                # at all, being a derived view rather than a stored
                # component table, so it never reaches this loop.)
                if comp_type == "component_type":
                    continue
                try:
                    df = self.view(comp_type, resolved_id).to_pandas()
                except Exception:
                    continue
                if not df.empty:
                    by_type[comp_type] = df
        if not by_type:
            return f"No data found for entity {entity_id!r}."
        if format != "markdown":
            sections = [
                f"# {comp_type}\n\n{df.to_csv(index=False)}" for comp_type, df in by_type.items()
            ]
            return "\n\n".join(sections)
        lines = [f"# {entity_id}"]
        for comp_type, df in by_type.items():
            prefix = f"{comp_type}."
            for _, row in df.iterrows():
                lines.append(f"\n## {comp_type}")
                fields = {
                    (key[len(prefix):] if key.startswith(prefix) else key): value
                    for key, value in row.items()
                    if key not in ("entity_id", "entity_id.display_alias")
                }
                for field, value in fields.items():
                    # An entity_ref field (e.g. "value") resolves during
                    # derive to a companion "{field}_eid" column holding the
                    # target's raw entity_id hash (see
                    # dataflows.derive.resolve_paths) -- an internal lookup
                    # key, not something a reader can act on. Shown right
                    # next to the human-readable value it resolves (already
                    # legible on its own, whether that's an alias or plain
                    # text), it's redundant more often than not; drop it
                    # whenever that companion value is actually present.
                    if field.endswith("_eid") and fields.get(field[: -len("_eid")]) is not None:
                        continue
                    lines.append(f"- {field}: {_format_field_value(value)}")
        return "\n".join(lines)

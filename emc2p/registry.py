"""ECS Registry for storing and accessing component data."""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import ibis
import pandas as pd

from .utils import candidate_entity_ids

_TABLE_META_COLS = {"entity_id", "component_index", "modifier"}


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
        self, component_type: str | list[str], aliases: str | list[str] | None = None
    ) -> ibis.Table:
        """Return a copy of the dataframe for the given component type(s).

        Args:
            component_type: A component type name, a dotted "table.field"
                string, or a list of either. All results are inner-joined by
                entity_id with columns named "table.field". ``entity_id.alias``
                is prepended automatically unless already requested.
            aliases: An entity ref, or list of them, to filter the result
                down to. Each is resolved the same way `get_entity_id` does
                (exact hash, else exact alias, else substring match), except
                a ref matching more than one entity is not an error here —
                every match is included, and a warning is raised (this is
                one of the few contexts where an ambiguous ref is fine, since
                the caller is filtering a view, not picking a single target
                to write to). A ref matching zero entities also warns, and
                contributes nothing to the result.

        Raises:
            KeyError: If a component type doesn't exist in the registry.
        """
        return self._view(component_type, self._table_or_declared, aliases)

    def view_current(
        self, component_type: str | list[str], aliases: str | list[str] | None = None
    ) -> ibis.Table:
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
            aliases: Same as ``view``.

        Raises:
            KeyError: If a component type doesn't exist in the registry.
            ValueError: If a component type has more than one time_dimension field.
        """
        return self._view(component_type, self._current_table, aliases)

    def safe_view(
        self, component_type: str | list[str], aliases: str | list[str] | None = None
    ) -> pd.DataFrame | None:
        """Like `view`, but returns None instead of raising for an unknown component type.

        A component type that's declared (see `declare_schema`) but has no
        data yet still comes back as an empty DataFrame, not None -- only a
        component type unknown to the registry entirely is safed away.

        Args:
            component_type: Same as `view`.
            aliases: Same as `view`.
        """
        try:
            return self.view(component_type, aliases).execute()
        except KeyError:
            return None

    def safe_view_current(
        self, component_type: str | list[str], aliases: str | list[str] | None = None
    ) -> pd.DataFrame | None:
        """Like `safe_view`, but resolves each field's current row per entity (see `view_current`)."""
        try:
            return self.view_current(component_type, aliases).execute()
        except KeyError:
            return None

    def get_current_value(self, component_type: str, field: str = "value", alias: str | None = None) -> Any:
        """The current `component_type.field` value for one entity, or None.

        A safe, single-value convenience over `view_current` for the common
        "what's this entity's current X" read -- resolves to None (rather
        than raising) whenever there's nothing to return: `component_type`
        isn't loaded yet, or nothing matches `alias`. Not for multi-entity
        reads: pass a specific `alias` naming exactly one entity; use
        `view_current` directly for a whole table.

        Args:
            component_type: The component type to read.
            field: The field to read (defaults to "value", the sole field
                most single-value component types use).
            alias: An entity ref identifying exactly one entity (same
                resolution as `view`'s own `aliases`).

        Returns:
            The current value, or None if nothing is recorded yet.
        """
        df = self.safe_view_current(f"{component_type}.{field}", aliases=alias)
        if df is None or df.empty:
            return None
        return df.iloc[-1][f"{component_type}.{field}"]

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
    ) -> ibis.Table:
        if isinstance(component_type, str):
            component_type = [component_type]

        if (
            "entity_id.alias" not in component_type
            and "entity_id" not in component_type
        ):
            component_type = ["entity_id.alias"] + list(component_type)

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
            result = result.inner_join(t, "entity_id")

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
        def_eids = df_entity.loc[df_entity["entity_key"] == component_type, "value"]

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

    def view_df(
        self, component_type: str | list[str], aliases: str | list[str] | None = None
    ) -> pd.DataFrame:
        """Convenience method to return the view as a DataFrame."""
        result = self.view(component_type, aliases)
        df = result.execute()
        df = df.set_index("entity_id")
        return df

    def summarize_components(self, limit: int = 20) -> str:
        """Markdown report of every component type currently holding data, one
        section per type with its row count and up to `limit` sample rows.

        Only `component_types` (data-bearing), not the larger
        `known_component_types` -- nothing to report on a type nobody's
        written to yet. Unlike `view_df` (one type at a time), this covers
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
            df = self.view_df(component_type).reset_index()
            sample = df.head(limit).fillna("null").to_markdown(index=False)
            if len(df) > limit:
                sample += f"\n... ({len(df) - limit} more row(s) not shown)"
            sections.append(f"### {component_type} ({len(df)} row(s))\n{sample}")
        return (
            "All component types currently recorded, one section per type:\n\n"
            + "\n\n".join(sections)
        )

    def get_entity_id(self, entity_ref: str) -> str | None:
        """Resolve `entity_ref` to its canonical entity_id hash.

        The read-only counterpart to `candidate_entity_ids`, which this
        delegates to directly: it's the exact same resolution ETL uses for
        `entity_ref` fields and `same_as` targets (exact hash, else exact
        alias, else substring match against every entity's full path), kept
        only if it resolves to exactly one entity.

        Deliberately tolerant rather than raising — unlike `same_as` target
        resolution, where an unresolvable reference is a manifest error
        worth failing loudly on, a caller resolving an arbitrary ref (e.g.
        one it read from a still-in-flux registry) usually just wants to
        know whether a matching entity currently exists.

        Args:
            entity_ref: An entity hash, alias, or path fragment identifying
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

    def view_entity_df(self, entity_id: str) -> dict[str, pd.DataFrame]:
        """Return component data for a specific entity, keyed by component type.

        `entity_id` is resolved via `get_entity_id` first, so this accepts
        anything that already does — the internal hash, an exact alias, or
        an unambiguous path fragment — and returns `{}` for a ref that
        doesn't resolve to exactly one entity, the same as it already did
        for a bare unknown ref.

        Args:
            entity_id: Entity hash, alias, or path fragment identifying the entity.
        """
        resolved_id = self.get_entity_id(entity_id)
        if resolved_id is None:
            return {}
        result = {}
        for comp_type in self._component_types:
            try:
                df = self.view_df(comp_type).reset_index()
            except Exception:
                continue
            match = df[df["entity_id"] == resolved_id]
            if not match.empty:
                result[comp_type] = match.set_index("entity_id")
        return result

    def view_entity(self, entity_id: str, format: str = "markdown") -> str:
        """Return all component data for a specific entity as a formatted string.

        Args:
            entity_id: Entity hash, alias, or path fragment identifying the
                entity (see `get_entity_id`).
            format: Output format — "markdown" (default, a `key: value`
                outline, not a table) or "csv".
        """
        components = self.view_entity_df(entity_id)
        if not components:
            return f"No data found for entity {entity_id!r}."
        if format != "markdown":
            sections = [f"# {comp_type}\n\n{df.to_csv()}" for comp_type, df in components.items()]
            return "\n\n".join(sections)
        lines = [f"# {entity_id}"]
        for comp_type, df in components.items():
            prefix = f"{comp_type}."
            for _, row in df.reset_index().iterrows():
                lines.append(f"\n## {comp_type}")
                for k, v in row.items():
                    if k in ("entity_id", "entity_id.alias"):
                        continue
                    field = k[len(prefix):] if k.startswith(prefix) else k
                    lines.append(f"- {field}: {_format_field_value(v)}")
        return "\n".join(lines)

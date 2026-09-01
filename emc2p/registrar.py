"""Base class for emc2p systems."""
from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

from emc2p.etl_system import ETLSystem, resolve_dataflow

if TYPE_CHECKING:
    from emc2p.registry import Registry


class Registrar:
    """User-facing interface to a registry: loads human/agent-authored
    entries into it, and exposes it for viewing, updating, and exporting.
    """

    @classmethod
    def from_manifest(cls, manifest: str | Path | list[str | Path], time: Any = None) -> Registrar:
        """Create a Registrar with a registry loaded and validated from a manifest.

        Args:
            manifest: A directory path (or list of paths) making up the manifest.
            time: The point in time this manifest represents, e.g. a
                timestamp or date string. If given, any field flagged
                ``time_dimension: true`` that is still null after loading is
                filled with this value. See ``load_manifest``.
        """
        r = cls()
        r.load_manifest(manifest, time=time)
        return r

    @classmethod
    def load(cls, path: str | Path) -> Registrar:
        """Create a Registrar from a database written by ``save``.

        This is a straight export/import of the component tables — no ETL or
        transformation is performed. The backend is inferred from ``path``
        (see ``Registry.from_database``); a plain ``.duckdb`` path uses
        DuckDB, but any ``ibis.connect``-resolvable URL works.

        Args:
            path: A URL or filesystem path resolvable by ``ibis.connect``.
        """
        from emc2p.registry import Registry
        return cls(Registry.from_database(path))

    def __init__(self, registry: Registry | None = None):
        if registry is None:
            import ibis
            from emc2p.registry import Registry
            registry = Registry(ibis.duckdb.connect(), {})
        self._registry = registry
        self._etl = ETLSystem()
        self._dataflows: list[ModuleType] = []

    def update(
        self,
        input_dirs: str | Path | list[str | Path] | None = None,
        time: Any = None,
        **inputs,
    ) -> None:
        """Load new data and merge it into the current registry.

        The general-purpose way to grow the registry: from manifest
        directories/files, ad hoc human/agent-authored source text (e.g. an
        SCD-style position update), or both. Runs the full ETL pipeline
        (load, validate, derive); existing component types are unioned and
        deduplicated with the newly loaded data, new component types are
        added directly. The current registry is also passed into derive as
        ``existing_registry``, so a ``same_as`` component in the new data can
        target (and have its other components rebased onto) an entity from a
        prior update, not just one in this same batch.

        Args:
            input_dirs: A file path, directory path, or list of either.
            time: The point in time this data represents, e.g. a timestamp
                or date string. If given, any field flagged
                ``time_dimension: true`` in its component type's schema that
                is still null after loading is filled with this value, so
                that ``view_current`` can pick the most recent version of a
                slowly changing dimension. Only null values in the newly
                loaded components are filled; existing data is untouched.
            **inputs: Additional inputs forwarded to ``base_etl``, e.g.
                ``yaml_strings={"key": "raw yaml text"}`` or
                ``python_strings={...}``.
        """
        if isinstance(input_dirs, (str, Path)):
            input_dirs = [input_dirs]
        new_registry = self._etl.execute(
            self._base_etl_module(),
            input_dirs=input_dirs or [],
            load_time=time,
            existing_registry=self._registry,
            **inputs,
        )
        self._registry.merge(new_registry)
        new_registry.close()

    def _base_etl_module(self) -> ModuleType:
        """The dataflow module ``update()`` runs to load/validate/derive.

        Defaults to emc2p's own generic ``load -> validate -> derive ->
        validate`` pipeline. A subclass wanting extra derive steps of its
        own (e.g. a domain-specific dataflow layered on top of emc2p's
        generic ``derive_components``) overrides this to point at its own
        equivalent ``base_etl`` module instead, rather than this class
        needing to know about any downstream project's own dataflows.
        """
        from emc2p.dataflows import base_etl
        return base_etl

    def validate_write(self, yaml_string: str) -> set[str]:
        """Mechanical safety checks (see `emc2p.validate_write`) for
        `yaml_string` against this registry's own known component types.

        Run this before `update()`'ing it in, to raise on a
        component-named-as-nested-entity mistake or an unknown component
        type with a clearer, more targeted message than `update()`'s own
        ETL failure would give. (The bare-mapping check runs automatically
        inside `update()` itself, via `load_yaml`'s own
        `raw_entity_first_data` -- not duplicated here.)

        Returns the set of component types this write references, for a
        caller that wants to show/log them (e.g. a preview step).
        """
        import yaml as _yaml

        from emc2p.validate_write import (
            find_component_named_as_nested_entity,
            find_unknown_component_types,
            referenced_component_types,
        )

        known = set(self.registry.known_component_types)

        nested = find_component_named_as_nested_entity(yaml_string, known)
        if nested:
            raise ValueError(
                "A known component type's name appears as a nested dict key instead "
                "of a component-list item (missing the leading `- `), which the "
                "parser reads as a nested child entity, not a component attachment: "
                f"{', '.join(nested)}."
            )

        unknown = find_unknown_component_types(yaml_string, known)
        if unknown:
            suggestions = "; ".join(
                f"{name} (did you mean: {', '.join(close)}?)" if close else name
                for name, close in unknown.items()
            )
            raise ValueError(
                f"Unknown component type(s) referenced: {suggestions}. Define it "
                "first (a top-level entity named after the type, with its own "
                "component_type marker and field: block) if this is intentional."
            )

        parsed = _yaml.safe_load(yaml_string)
        return referenced_component_types(parsed) if isinstance(parsed, dict) else set()

    def overwrite(self, component_type: str, table) -> None:
        """Replace `component_type`'s entire table in one step, bypassing the ETL pipeline.

        The Registrar-level counterpart to the additive `update()` (which
        unions new data in through the full load/validate/derive pipeline):
        for a System that owns a component type outright and needs to
        replace it wholesale each turn — e.g. rebuilding a tag component to
        exactly match this turn's set, since a merge can only add rows,
        never drop one that no longer applies — this writes the table
        directly, the same way `update()` ultimately does, without a caller
        needing to reach past `Registrar` into `.registry` to do it.

        Args:
            component_type: The component type to replace.
            table: The replacement data, in any form `Registry.update`
                accepts (e.g. an ibis Table or a pyarrow Table).
        """
        self._registry.update({component_type: table})

    def load_manifest(
        self, manifest: str | Path | list[str | Path], time: Any = None
    ) -> None:
        """Load a manifest and merge it into the current registry.

        Thin wrapper around ``update`` for the common case of loading full
        manifest directories/files from disk.

        Args:
            manifest: A file path, directory path, or list of either.
            time: The point in time this manifest represents. See ``update``.
        """
        self.update(manifest, time=time)

    def save(self, path: str | Path) -> None:
        """Export the current registry as-is to a database.

        This is a straight export — no ETL or transformation is performed.
        The backend is inferred from ``path`` (see ``Registry.to_database``);
        a plain ``.duckdb`` path uses DuckDB, but any ``ibis.connect``-
        resolvable URL works.

        Args:
            path: A URL or filesystem path resolvable by ``ibis.connect``.
        """
        self._registry.to_database(path)

    def export_manifest(
        self, output_dir: str | Path | None = None, collapse_time_dimension: bool = True
    ) -> list[str]:
        """Export the registry back to entity-centered EC files.

        Runs the ``etl.export_manifest`` dataflow against the current
        registry, converting the relational component tables back into
        human/agent-friendly YAML entries.

        Args:
            output_dir: Directory to write the exported files to. If
                omitted, each entity is written back to its original
                source path (a "refresh" round-trip).
            collapse_time_dimension: When True (the default), a component
                type flagged ``time_dimension`` exports only its current row
                per entity. Set False for a full, uncollapsed history export
                instead -- e.g. to keep a full-history file alongside a
                collapsed current-state one.

        Returns:
            The filepaths that were written.
        """
        kwargs: dict = {"collapse_time_dimension": collapse_time_dimension}
        if output_dir is not None:
            kwargs["output_dir"] = str(output_dir)
        result = self.execute("etl.export_manifest", **kwargs)
        return result.get("exported_manifest_filepaths", [])

    def load_dataflow(self, name: str) -> None:
        """Load a dataflow module by name and attach it to the driver.

        Resolves ``name`` as a dotted path within one of the registered
        dataflow base packages (see ``emc2p.etl_system.register_dataflow_package``),
        so both top-level modules and subpackage modules are supported::

            registrar.load_dataflow("export_manifest")
            registrar.load_dataflow("audit.requirement_coverage")

        Args:
            name: Dotted module path relative to a registered dataflow base
                package (``emc2p.dataflows`` by default).

        Raises:
            ValueError: If no matching module is found.
        """
        module = resolve_dataflow(name)
        if module not in self._dataflows:
            self._dataflows.append(module)

    @property
    def registry(self) -> Registry:
        return self._registry

    @property
    def view(self):
        return self._registry.view

    @property
    def view_current(self):
        return self._registry.view_current

    @property
    def safe_view(self):
        return self._registry.safe_view

    @property
    def safe_view_current(self):
        return self._registry.safe_view_current

    @property
    def get_current_value(self):
        return self._registry.get_current_value

    @property
    def view_df(self):
        return self._registry.view_df

    @property
    def summarize_components(self):
        return self._registry.summarize_components

    @property
    def get_entity_id(self):
        return self._registry.get_entity_id

    @property
    def view_entity(self):
        return self._registry.view_entity

    @property
    def view_entity_df(self):
        return self._registry.view_entity_df

    @property
    def get(self):
        return self._registry.get

    def execute(self, final_vars: str | list[str], **inputs) -> dict[str, Any]:
        """Execute DAG nodes and return their outputs.

        Args:
            final_vars: A node name, list of node names, or dataflow module
                name (e.g. ``"export_manifest"``). When a dataflow name is
                given the module is auto-loaded and all its outputs are run.
            **inputs: Additional inputs forwarded to the Hamilton driver (e.g.
                ``output_dir="..."``) to satisfy external-input nodes.
        """
        if isinstance(final_vars, str):
            try:
                module = resolve_dataflow(final_vars)
            except ValueError:
                final_vars = [final_vars]
            else:
                if module not in self._dataflows:
                    self._dataflows.append(module)
                final_vars = self._etl.outputs(self._dataflows)

        if not final_vars:
            return {}
        return self._etl.execute(
            self._dataflows, final_vars, registry=self._registry, **inputs
        )

    @property
    def outputs(self) -> list[str]:
        return self._etl.outputs(self._dataflows)

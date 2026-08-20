"""Generic Hamilton-based dataflow execution, decoupled from Registrar."""
from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

_DATAFLOW_BASE_PACKAGES = ["emc2p.dataflows"]


def register_dataflow_package(name: str) -> None:
    """Add ``name`` to the packages searched by ``resolve_dataflow``.

    Lets a downstream project (e.g. one that defines its own domain-specific
    dataflows, alongside emc2p's generic ones) register its own dataflow
    package once, so dotted names like ``"audit.requirement_coverage"``
    resolve without every caller needing to know which package actually
    defines them. Tried in registration order, after ``emc2p.dataflows``.
    A no-op if ``name`` is already registered.

    Args:
        name: Fully-qualified package name, e.g. ``"iacs.dataflows"``.
    """
    if name not in _DATAFLOW_BASE_PACKAGES:
        _DATAFLOW_BASE_PACKAGES.append(name)


def resolve_dataflow(dataflow: ModuleType | str) -> ModuleType:
    """Resolve a dataflow module, or a dotted name relative to any
    registered dataflow base package (see ``register_dataflow_package``).

    Args:
        dataflow: A dataflow module, or a dotted path such as
            ``"etl.export_manifest"`` or ``"audit.requirement_coverage"``.

    Raises:
        ValueError: If a name is given and no matching module is found in
            any registered base package.
    """
    if isinstance(dataflow, ModuleType):
        return dataflow
    tried = []
    for base_package in _DATAFLOW_BASE_PACKAGES:
        full_name = f"{base_package}.{dataflow}"
        try:
            return importlib.import_module(full_name)
        except ImportError:
            tried.append(full_name)
    raise ValueError(f"No dataflow named {dataflow!r} found (tried {tried})")


class ETLSystem:
    """Runs Hamilton dataflows against a registry (or other inputs).

    Stateless: dataflows, adapters, and inputs (including any registry) are
    all supplied per call, since a given system rarely reruns the same
    dataflow/adapter combination.
    """

    def execute(
        self,
        dataflows: ModuleType | str | list[ModuleType | str],
        final_vars: str | list[str] | None = None,
        adapters: list | None = None,
        **inputs,
    ) -> Any:
        """Run one or more dataflows and return their output(s).

        Args:
            dataflows: A dataflow module, a dotted name relative to a
                registered dataflow base package (see
                ``register_dataflow_package``), or a list of either.
            final_vars: Node name(s) to return. Defaults to each dataflow
                module's declared ``FINAL_VAR``. A single node (whether from
                an explicit string or the single-dataflow default) is
                returned directly rather than wrapped in a dict; a list of
                node names returns a ``{name: value}`` dict.
            adapters: Extra Hamilton lifecycle adapters (e.g. for testing).
            **inputs: Runtime inputs forwarded to the Hamilton driver (e.g.
                ``registry=...``, ``input_dirs=...``).
        """
        modules = self._resolve_all(dataflows)

        if final_vars is None:
            names = [self._final_var(m) for m in modules]
            unwrap = len(names) == 1
        elif isinstance(final_vars, str):
            names = [final_vars]
            unwrap = True
        else:
            names = list(final_vars)
            unwrap = False

        if not names:
            return {}

        drv = self._build_driver(modules, adapters)
        result = drv.execute(names, inputs=inputs or None)
        return result[names[0]] if unwrap else result

    def outputs(self, dataflows: ModuleType | str | list[ModuleType | str]) -> list[str]:
        """List the non-input node names available across the given dataflows."""
        drv = self._build_driver(self._resolve_all(dataflows), adapters=None)
        return [
            v.name for v in drv.list_available_variables() if not v.is_external_input
        ]

    @staticmethod
    def _resolve_all(dataflows: ModuleType | str | list[ModuleType | str]) -> list[ModuleType]:
        if not isinstance(dataflows, list):
            dataflows = [dataflows]
        return [resolve_dataflow(d) for d in dataflows]

    @staticmethod
    def _final_var(module: ModuleType) -> str:
        try:
            return module.FINAL_VAR
        except AttributeError as e:
            raise ValueError(
                f"{module.__name__} declares no FINAL_VAR; pass final_vars explicitly"
            ) from e

    @staticmethod
    def _build_driver(modules: list[ModuleType], adapters: list | None):
        from hamilton import driver, base
        return driver.Driver(
            {},
            *modules,
            adapter=[base.DictResult(), *(adapters or [])],
            allow_module_overrides=True,
        )

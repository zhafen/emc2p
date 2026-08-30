"""Backward-compatible re-export of Registry.from_component_rows.

make_registry's actual construction logic now lives on Registry itself
(it was never test-specific -- just an alternate constructor for
component-first row data that happened to only be used by tests) -- see
docs/manifest/history.yaml. Kept here as a thin wrapper (rather than
deleted outright) so the many existing `from tests.conftest import
make_registry` / `from emc2p.testing.registry_builder import
make_registry` call sites across all three repos' test suites keep
working unchanged.

Used to also carry a lenient ``pandas.testing.assert_allclose`` monkeypatch
for comparing round-tripped component tables. That patch masked a real
export/reload fidelity gap (see docs/manifest/history.yaml:
project_history.assert_allclose_patch_replaced_by_assert_components_equal)
and has been replaced by ``emc2p.testing.expected_fixtures.
assert_components_equal``/``assert_registries_equal``, which compare every
common column instead of just ``component_type``/``modifier``.
"""

from emc2p.registry import Registry


def make_registry(components: dict[str, list[dict]]) -> Registry:
    """Create a Registry from component-first data. See Registry.from_component_rows."""
    return Registry.from_component_rows(components)

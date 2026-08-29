"""Shared test fixtures and helpers.

make_registry/the pandas.testing.assert_allclose patch now live in
emc2p.testing.registry_builder -- re-exported here (rather than every
test file importing from there directly) so the existing `from
tests.conftest import make_registry` call sites across this suite don't
all need touching. iacs's own tests/conftest.py re-exports the identical
thing for the same reason (see docs/manifest/history.yaml).
"""

from emc2p.testing.registry_builder import make_registry  # noqa: F401

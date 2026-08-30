"""Exercises emc2p's own generic dataflows through the shared
load-example-manifest / compare-to-expected.py framework
(`emc2p.testing.expected_fixtures`), the same way iacs and story-simulator
each plug their own examples/expected fixtures into it.

Only a light-touch check compared to iacs's own test_human_validated.py
(most nodes here have no expected/ fixture at all -- see this module's own
docstring in emc2p.testing.expected_fixtures for why that's fine, a node
with no fixture is simply not checked): the point of this test is proving
the shared framework itself still works against emc2p's own dataflows and
examples, not re-covering ground emc2p's other, more targeted dataflow
tests already own.
"""

from pathlib import Path

import pytest

from emc2p.registrar import Registrar
from emc2p.testing.expected_fixtures import (
    ExpectedValueChecker,
    assert_registries_equal,
    example_dirs,
)

ROOT = Path(__file__).parent.parent.parent
EXAMPLES_DIR = ROOT / "examples"
EXPECTED_DIR = ROOT / "tests" / "test_dataflows" / "expected"
DATAFLOW_MODULE_PREFIXES = ("emc2p.dataflows.",)

# Every case here loads a real example manifest end-to-end -- the Hamilton
# driver construction that goes through dominates regardless of manifest
# size (see pyproject.toml's own `slow` marker docstring).
pytestmark = pytest.mark.slow


@pytest.mark.parametrize("example_dir", example_dirs(EXAMPLES_DIR))
def test_end_to_end(example_dir: Path, tmp_path: Path):
    """Load each example manifest, export it back out, reload it, and
    confirm the reloaded registry matches the original -- comparing any
    executed node with a hand-written expected/ fixture along the way.
    """
    checker = ExpectedValueChecker(example_dir, EXPECTED_DIR, DATAFLOW_MODULE_PREFIXES)
    registrar = Registrar()

    registrar.update(input_dirs=[str(example_dir)], adapters=[checker])

    output_dir = tmp_path / example_dir.name
    registrar.execute(
        "etl.export_manifest",
        adapters=[checker],
        output_dir=str(output_dir),
    )

    reloaded_registrar = Registrar()
    reloaded_registrar.update(input_dirs=[str(output_dir)])

    assert_registries_equal(registrar, reloaded_registrar)

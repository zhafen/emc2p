"""A pytest-fixture factory for a per-test parametrizable "which strategy
answers this session's judgment calls" choice, with an environment-
variable fallback.

The resolution itself -- explicit `request.param` wins, else the given
environment variable, else the given default -- is the same for any
downstream project driving `HeadlessSession`/`McpClientSession`-based live
tests, regardless of what the resolved string actually means in that
project's own domain (story-simulator's "host"/"subagent"/"keyed_subagent"
is just one such vocabulary). Only the env var name and default are
project-specific, so those are the factory's own parameters, not baked in
here.
"""

from __future__ import annotations

import os
from typing import Callable

import pytest


def make_responder_fixture(
    env_var: str, default: str, *, name: str = "responder"
) -> Callable[[pytest.FixtureRequest], str]:
    """Return a pytest fixture resolving (explicit request.param) > (env_var) > (default).

    Usage in a downstream project's own conftest.py::

        responder = make_responder_fixture("STORY_SIM_LIVE_TEST_RESPONDER", "keyed_subagent")

    Parametrize per-test with
    ``@pytest.mark.parametrize("responder", ["host"], indirect=True)``
    rather than only being able to flip the whole suite's default via the
    environment variable. `name` controls the fixture's own registered
    name (matching the module-level variable it's normally assigned to);
    override it if the fixture should be requested under a different name
    than the local variable holding it.
    """

    def _responder(request: pytest.FixtureRequest) -> str:
        explicit = getattr(request, "param", None)
        if explicit is not None:
            return explicit
        return os.environ.get(env_var, default)

    return pytest.fixture(_responder, name=name)

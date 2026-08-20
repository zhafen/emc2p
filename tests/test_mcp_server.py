"""Tests for emc2p.mcp_server's tool functions -- no live model, no MCP
transport.

Calls each `@server.tool()`-decorated function directly with a mocked
Context, the same shape a real MCP client's session identity takes
(`ctx.request_context.session`). Mechanical dispatch/session-state checks,
not a test of any model's judgment -- see
tests/test_human_validated/test_write_accuracy.py for that.
"""

from unittest.mock import MagicMock

import pytest

from emc2p.mcp_server import open_registry, update_registry, view_entity, view_registry

STATUS_BOARD_SCENARIO_DIR = "tests/data/scenarios/status_board"


def _make_ctx():
    ctx = MagicMock()
    ctx.request_context.session = MagicMock()
    return ctx


def _db_url(tmp_path) -> str:
    return f"duckdb:///{tmp_path / 'test.duckdb'}"


def test_view_registry_without_open_registry_raises(tmp_path):
    with pytest.raises(ValueError, match="No registry open"):
        view_registry("status", _make_ctx())


def test_open_registry_creates_a_fresh_registry(tmp_path):
    result = open_registry(_db_url(tmp_path), _make_ctx())
    assert "Registry opened" in result


def test_open_registry_seeds_manifest_dir_on_a_fresh_registry(tmp_path):
    ctx = _make_ctx()
    open_registry(_db_url(tmp_path), ctx, manifest_dir=STATUS_BOARD_SCENARIO_DIR)
    assert "status" in view_registry("component_type", ctx)


def test_update_registry_merges_and_view_registry_reads_it_back(tmp_path):
    ctx = _make_ctx()
    open_registry(_db_url(tmp_path), ctx, manifest_dir=STATUS_BOARD_SCENARIO_DIR)
    update_registry("widget_a:\n    - status:\n        value: active\n", ctx)
    assert "active" in view_registry("status", ctx)


def test_view_entity_shows_every_component_on_an_entity(tmp_path):
    ctx = _make_ctx()
    open_registry(_db_url(tmp_path), ctx, manifest_dir=STATUS_BOARD_SCENARIO_DIR)
    update_registry("widget_a:\n    - status:\n        value: active\n", ctx)
    assert "active" in view_entity("widget_a", ctx)


def test_writes_to_the_same_alias_share_one_entity(tmp_path):
    """A seeded alias (from manifest_dir) and a later write to that same
    alias must land on the same entity, not two separate ones that happen
    to share a name -- see open_registry's own docstring for why."""
    ctx = _make_ctx()
    open_registry(_db_url(tmp_path), ctx, manifest_dir=STATUS_BOARD_SCENARIO_DIR)
    update_registry("widget_a:\n    - status:\n        value: active\n", ctx)
    entity_view = view_entity("widget_a", ctx)
    assert "description" in entity_view
    assert "status" in entity_view

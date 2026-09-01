"""Tests for emc2p.mcp_server.RegistrarSessions -- no live model, no MCP transport.

Most tests call the plain-session methods directly (the same ones
`keyed_subagent`-style in-process dispatch calls); a few confirm the
`mount()` wiring (Context unpacking, tool registration) separately. Not a
test of any model's judgment -- see
tests/test_human_validated/test_write_accuracy.py for that.
"""

import math
from unittest.mock import MagicMock

import pytest
from mcp.server.fastmcp import FastMCP

from emc2p.mcp_server import RegistrarSessions

STATUS_BOARD_SCENARIO_DIR = "tests/data/scenarios/status_board"


def _session():
    return MagicMock()


def _db_url(tmp_path) -> str:
    return f"duckdb:///{tmp_path / 'test.duckdb'}"


class TestRegistrarSessions:
    pytestmark = pytest.mark.slow

    def test_view_registry_without_open_raises(self, tmp_path):
        sessions = RegistrarSessions()
        with pytest.raises(ValueError, match="No registry open"):
            sessions.view_registry(_session(), "status")

    def test_open_creates_a_fresh_registry(self, tmp_path):
        sessions = RegistrarSessions()
        result = sessions.open(_session(), _db_url(tmp_path))
        assert "Registry opened" in result

    def test_open_seeds_manifest_dir_on_a_fresh_registry(self, tmp_path):
        sessions = RegistrarSessions()
        session = _session()
        sessions.open(session, _db_url(tmp_path), manifest_dir=STATUS_BOARD_SCENARIO_DIR)
        assert "status" in sessions.view_registry(session, "component_type")

    def test_update_registry_merges_and_view_registry_reads_it_back(self, tmp_path):
        sessions = RegistrarSessions()
        session = _session()
        sessions.open(session, _db_url(tmp_path), manifest_dir=STATUS_BOARD_SCENARIO_DIR)
        sessions.update_registry(session, "widget_a:\n    - status:\n        value: active\n")
        assert "active" in sessions.view_registry(session, "status")

    def test_view_entity_shows_every_component_on_an_entity(self, tmp_path):
        sessions = RegistrarSessions()
        session = _session()
        sessions.open(session, _db_url(tmp_path), manifest_dir=STATUS_BOARD_SCENARIO_DIR)
        sessions.update_registry(session, "widget_a:\n    - status:\n        value: active\n")
        assert "active" in sessions.view_entity(session, "widget_a")

    def test_writes_to_the_same_alias_share_one_entity(self, tmp_path):
        """A seeded alias (from manifest_dir) and a later write to that same
        alias must land on the same entity, not two separate ones that
        happen to share a name -- see `open`'s own docstring for why."""
        sessions = RegistrarSessions()
        session = _session()
        sessions.open(session, _db_url(tmp_path), manifest_dir=STATUS_BOARD_SCENARIO_DIR)
        sessions.update_registry(session, "widget_a:\n    - status:\n        value: active\n")
        entity_view = sessions.view_entity(session, "widget_a")
        assert "description" in entity_view
        assert "status" in entity_view

    def test_consolidate_review_includes_guidance(self, tmp_path):
        sessions = RegistrarSessions()
        session = _session()
        sessions.open(session, _db_url(tmp_path), manifest_dir=STATUS_BOARD_SCENARIO_DIR)
        sessions.update_registry(session, "widget_a:\n    - status:\n        value: active\n")
        result = sessions.consolidate_review(session)
        assert "status" in result
        assert "consolidate" in result.lower()

    def test_update_registry_rejects_bare_mapping_components(self, tmp_path):
        sessions = RegistrarSessions()
        session = _session()
        sessions.open(session, _db_url(tmp_path), manifest_dir=STATUS_BOARD_SCENARIO_DIR)
        with pytest.raises(ValueError, match="widget_a"):
            sessions.update_registry(session, "widget_a:\n    status:\n        value: active\n")


class TestDispatch:
    """dispatch() is the (name, arguments) -> str shape emc2p.agents.tool_calling_loop's
    own dispatch callback expects -- see that module's REGISTRAR_TOOL_SPECS."""

    pytestmark = pytest.mark.slow

    def _opened(self, tmp_path):
        sessions = RegistrarSessions()
        session = _session()
        sessions.open(session, _db_url(tmp_path), manifest_dir=STATUS_BOARD_SCENARIO_DIR)
        return sessions, session

    def test_dispatch_routes_view_registry_by_name(self, tmp_path):
        sessions, session = self._opened(tmp_path)
        sessions.update_registry(session, "widget_a:\n    - status:\n        value: active\n")
        assert "active" in sessions.dispatch(session, "view_registry", {"component_type": "status"})

    def test_dispatch_routes_update_registry_by_name(self, tmp_path):
        sessions, session = self._opened(tmp_path)
        sessions.dispatch(session, "update_registry", {"yaml_string": "widget_a:\n    - status:\n        value: idle\n"})
        assert "idle" in sessions.view_registry(session, "status")

    def test_dispatch_rejects_unknown_tool_name(self, tmp_path):
        sessions, session = self._opened(tmp_path)
        result = sessions.dispatch(session, "not_a_real_tool", {})
        assert "unknown tool" in result

    def test_dispatch_does_not_expose_non_tool_methods(self, tmp_path):
        """get_registrar/open/... are real methods on this instance, but
        must stay unreachable through dispatch's own by-name routing --
        the allowlist, not bare getattr, is what enforces that."""
        sessions, session = self._opened(tmp_path)
        for name in ("get_registrar", "set_registrar", "open", "dispatch", "_merge_and_sync"):
            assert "unknown tool" in sessions.dispatch(session, name, {})


class TestPreviewConfirmFlow:
    pytestmark = pytest.mark.slow

    def _opened(self, tmp_path):
        sessions = RegistrarSessions()
        session = _session()
        sessions.open(session, _db_url(tmp_path), manifest_dir=STATUS_BOARD_SCENARIO_DIR)
        return sessions, session

    def test_preview_does_not_merge(self, tmp_path):
        sessions, session = self._opened(tmp_path)
        sessions.update_registry(session, "widget_a:\n    - status:\n        value: active\n", preview=True)
        assert sessions.view_registry(session, "status").count("active") == 0

    def test_preview_returns_a_confirm_token_that_merges_on_follow_up(self, tmp_path):
        sessions, session = self._opened(tmp_path)
        preview = sessions.update_registry(
            session, "widget_a:\n    - status:\n        value: active\n", preview=True
        )
        token = _extract_token(preview)
        sessions.update_registry(session, confirm_token=token)
        assert "active" in sessions.view_registry(session, "status")

    def test_preview_lists_only_the_referenced_component_type(self, tmp_path):
        sessions, session = self._opened(tmp_path)
        preview = sessions.update_registry(
            session, "widget_a:\n    - status:\n        value: active\n", preview=True
        )
        assert "Components this write references: ['status']" in preview


def _extract_token(preview: str) -> str:
    lines = preview.splitlines()
    idx = next(i for i, line in enumerate(lines) if "confirm_token" in line and "merge it" in line)
    return lines[idx + 1].strip()


class TestExportDir:
    pytestmark = pytest.mark.slow

    def test_update_registry_exports_when_export_dir_is_set(self, tmp_path):
        sessions = RegistrarSessions()
        session = _session()
        export_dir = tmp_path / "export"
        sessions.open(
            session,
            f"duckdb:///{tmp_path / 'test.duckdb'}",
            manifest_dir=STATUS_BOARD_SCENARIO_DIR,
            export_dir=str(export_dir),
        )
        sessions.update_registry(session, "widget_a:\n    - status:\n        value: active\n")
        assert (export_dir / "registry" / "session.yaml").exists()
        assert (export_dir / "registry_history" / "session.yaml").exists()

    def test_no_export_when_export_dir_is_not_set(self, tmp_path):
        sessions = RegistrarSessions()
        session = _session()
        export_dir = tmp_path / "export"
        sessions.open(session, f"duckdb:///{tmp_path / 'test.duckdb'}", manifest_dir=STATUS_BOARD_SCENARIO_DIR)
        sessions.update_registry(session, "widget_a:\n    - status:\n        value: active\n")
        assert not export_dir.exists()

    def test_export_dirnames_are_configurable(self, tmp_path):
        sessions = RegistrarSessions(export_dirname="current", export_history_dirname="past")
        session = _session()
        export_dir = tmp_path / "export"
        sessions.open(
            session,
            f"duckdb:///{tmp_path / 'test.duckdb'}",
            manifest_dir=STATUS_BOARD_SCENARIO_DIR,
            export_dir=str(export_dir),
        )
        sessions.update_registry(session, "widget_a:\n    - status:\n        value: active\n")
        assert (export_dir / "current" / "session.yaml").exists()
        assert (export_dir / "past" / "session.yaml").exists()

    def test_preview_does_not_export(self, tmp_path):
        sessions = RegistrarSessions()
        session = _session()
        export_dir = tmp_path / "export"
        sessions.open(
            session,
            f"duckdb:///{tmp_path / 'test.duckdb'}",
            manifest_dir=STATUS_BOARD_SCENARIO_DIR,
            export_dir=str(export_dir),
        )
        sessions.update_registry(session, "widget_a:\n    - status:\n        value: active\n", preview=True)
        assert not export_dir.exists()


class TestTimeProvider:
    pytestmark = pytest.mark.slow

    def test_time_provider_backfills_time_dimension_fields(self, tmp_path):
        sessions = RegistrarSessions(time_provider=lambda registrar: 42.0)
        session = _session()
        sessions.open(session, _db_url(tmp_path), manifest_dir=STATUS_BOARD_SCENARIO_DIR)
        sessions.update_registry(
            session,
            "timed:\n"
            "    - component_type\n"
            "    - field:\n"
            "        value:\n"
            "            description: value\n"
            "            type: str\n"
            "        time:\n"
            "            description: when\n"
            "            type: float\n"
            "            nullable: true\n"
            "            time_dimension: true\n"
            "\n"
            "thing_a:\n"
            "    - timed:\n"
            "        value: hello\n",
        )
        registrar = sessions.get_registrar(session)
        assert registrar.get_current_value("timed", "time", "thing_a") == 42.0

    def test_default_time_provider_is_a_no_op(self, tmp_path):
        """No time_provider given -- time_dimension fields stay null, matching plain registrar.update() behavior."""
        sessions = RegistrarSessions()
        session = _session()
        sessions.open(session, _db_url(tmp_path), manifest_dir=STATUS_BOARD_SCENARIO_DIR)
        sessions.update_registry(
            session,
            "timed:\n"
            "    - component_type\n"
            "    - field:\n"
            "        value:\n"
            "            description: value\n"
            "            type: str\n"
            "        time:\n"
            "            description: when\n"
            "            type: float\n"
            "            nullable: true\n"
            "            time_dimension: true\n"
            "\n"
            "thing_a:\n"
            "    - timed:\n"
            "        value: hello\n",
        )
        registrar = sessions.get_registrar(session)
        value = registrar.get_current_value("timed", "time", "thing_a")
        assert value is None or math.isnan(value)


class TestMount:
    def test_mounts_all_five_tools_with_real_descriptions(self):
        server = FastMCP("test")
        RegistrarSessions().mount(server)
        tools = {t.name: t for t in server._tool_manager.list_tools()}
        assert set(tools) == {"open_registry", "view_registry", "view_entity", "update_registry", "consolidate_review"}
        for tool in tools.values():
            assert tool.description, f"{tool.name} has no description"

    def test_mounted_tool_uses_this_instance_not_a_different_one(self, tmp_path):
        """Two separately-constructed RegistrarSessions must stay independent
        -- mounting one's tools must not leak into the other's session state."""
        server = FastMCP("test")
        sessions = RegistrarSessions()
        sessions.mount(server)
        other_sessions = RegistrarSessions()

        ctx = MagicMock()
        ctx.request_context.session = MagicMock()
        tools = {t.name: t for t in server._tool_manager.list_tools()}
        tools["open_registry"].fn(_db_url(tmp_path), ctx)

        assert ctx.request_context.session in sessions._registrars
        assert ctx.request_context.session not in other_sessions._registrars

    def test_exclude_skips_named_tools(self):
        server = FastMCP("test")
        RegistrarSessions().mount(server, exclude={"open_registry"})
        tools = {t.name: t for t in server._tool_manager.list_tools()}
        assert set(tools) == {"view_registry", "view_entity", "update_registry", "consolidate_review"}

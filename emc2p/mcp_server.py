"""Minimal MCP server exposing a Registrar's read/write tools directly.

Generic testing/reference infrastructure: no downstream project's own
domain concepts (no event/simulation loop, no character/NPC module --
compare `story_simulator.mcp_server`, which wraps these same four tool
shapes around its own `start_world`/`advance_simulation` event loop).
Lets a connected model exercise real Registrar read/write accuracy over
the actual MCP wire protocol, not just in-process tool dispatch (see
`emc2p.agents.tool_calling_loop` for the in-process alternative
`keyed_subagent`-style responders use instead).

Session state (which Registrar a given MCP session is talking to) is
scoped by `ctx.request_context.session`, weak-referenced the same way
`story_simulator.core`'s own `_registrars` is -- so a session ending
doesn't leak a Registrar/its DB connection forever.
"""

from __future__ import annotations

import weakref
from pathlib import Path

from mcp.server.fastmcp import Context, FastMCP

from emc2p.registrar import Registrar

server = FastMCP(
    "emc2p",
    instructions=(
        "You have direct access to an entity-component registry. Call open_registry "
        "once, at the beginning, with the database URL you were given. Then "
        "view_registry/view_entity to look up what's already recorded, and "
        "update_registry to record new facts -- entity-first YAML, each component as a "
        "list item (a leading `- `) to attach to an entity: `widget_a:\\n    - status:\\n"
        "        value: active` merges into the same widget_a already in the registry. "
        "The same content as a bare mapping (no `- `) is silently accepted with no error, "
        "yet records nothing at all -- always use the list form. Don't invent a component "
        "type nobody has described to you -- use view_registry/view_entity to check "
        "what's already known first."
    ),
)

_registrars: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

# Every update_registry write in one session shares this source key --
# a per-call key would hash the same alias to a different entity_id on
# each write instead of merging into the same entity.
LIVE_WRITE_KEY = "live"


def _get_registrar(session) -> Registrar:
    if session not in _registrars:
        raise ValueError("No registry open for this session yet -- call open_registry first.")
    return _registrars[session]


@server.tool()
def open_registry(database_url: str, ctx: Context, manifest_dir: str | None = None) -> str:
    """Open (or create) a Registrar backed by `database_url` for this session.

    `manifest_dir`'s EC files are merged in only the first time this
    exact `database_url` is opened (a brand-new registry has no
    component types yet) -- a resumed registry's accumulated state,
    including whatever schema it already knows, is untouched. Merged
    under `LIVE_WRITE_KEY`, the same source key every later
    `update_registry` write shares -- merging it via a separate,
    permanently-fixed source key instead would hash each seeded alias
    (e.g. `widget_a`) to a *different* entity_id than the one a later
    write to that same alias gets, silently splitting one entity into two.

    Args:
        database_url: A URL or filesystem path resolvable by ibis.connect
            (e.g. "duckdb:///path/to/file.duckdb").
        manifest_dir: Directory of EC files (component type/field
            declarations, starting entities) to seed a brand-new registry
            with. Ignored if `database_url` already has any component types.
    """
    registrar = Registrar.load(database_url)
    if manifest_dir and not registrar.registry.component_types:
        seed_text = "\n".join(f.read_text() for f in sorted(Path(manifest_dir).rglob("*.y*ml")))
        registrar.update(yaml_strings={LIVE_WRITE_KEY: seed_text})
    _registrars[ctx.request_context.session] = registrar
    return f"Registry opened at {database_url}. Component types: {registrar.registry.component_types}."


@server.tool()
def view_registry(component_type: str, ctx: Context) -> str:
    """View all recorded data for one component type (e.g. "status", "object")."""
    registrar = _get_registrar(ctx.request_context.session)
    return registrar.view_df(component_type).fillna("null").to_markdown()


@server.tool()
def view_entity(entity_id: str, ctx: Context) -> str:
    """Return every recorded component instance for a specific entity.

    Args:
        entity_id: Entity hash, alias, or unambiguous path fragment.
    """
    registrar = _get_registrar(ctx.request_context.session)
    return registrar.view_entity(entity_id, format="markdown")


@server.tool()
def update_registry(yaml_string: str, ctx: Context) -> str:
    """Merge entity-first YAML into the registry, run through the same ETL as save files."""
    registrar = _get_registrar(ctx.request_context.session)
    registrar.validate_write(yaml_string)
    registrar.update(yaml_strings={LIVE_WRITE_KEY: yaml_string})
    return "Merged into the registry."


def main() -> None:
    server.run(transport="stdio")


if __name__ == "__main__":
    main()

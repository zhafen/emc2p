"""MCP server exposing a Registrar's read/write tools directly.

Generic: no downstream project's own domain concepts (no event/simulation
loop, no character/NPC module -- compare `story_simulator.mcp_server`,
which mounts these same tools alongside its own `start_world`/
`advance_simulation` event loop). Lets a connected model exercise real
Registrar read/write accuracy over the actual MCP wire protocol, not just
in-process tool dispatch (see `emc2p.agents.tool_calling_loop` for the
in-process alternative `keyed_subagent`-style responders use instead).

All session state (which Registrar a session is talking to, its optional
export directory) lives on a `RegistrarSessions` instance, not module
globals -- a caller wanting its own behavior (e.g. computing a `time` to
backfill time_dimension fields with from its own domain state, or its own
export directory names) constructs its own instance with that behavior
injected, rather than reassigning this module's own functions/dicts from
outside. `server`/the module-level tool functions below are this module's
own default instance, for standalone use; a caller mounting these tools
onto its own FastMCP server (as `story_simulator.mcp_server` does)
constructs a separate `RegistrarSessions` instead and mounts that
instance's own bound methods.
"""

from __future__ import annotations

import base64
import weakref
from pathlib import Path
from typing import Any, Callable

import yaml
from mcp.server.fastmcp import Context, FastMCP

from emc2p.dataflows.etl.load_yaml import _find_malformed_entities
from emc2p.registrar import Registrar
from emc2p.validate_write import CONSOLIDATION_GUIDANCE

INSTRUCTIONS = (
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
)

# Every write in one session shares this source key, so repeated writes to
# the same alias hash to the same entity_id instead of forking a duplicate.
SOURCE_KEY = "session"

# The plain-session methods `dispatch()` (below) allows calling by name --
# every other method/attribute on the instance stays unreachable that way.
_DISPATCHABLE_METHODS = {"view_registry", "view_entity", "update_registry", "consolidate_review"}


class RegistrarSessions:
    """Owns every open-registry session, keyed by MCP session identity so
    one instance serves every concurrently connected client.

    Also owns the pieces of behavior a caller can configure without
    reassigning this module's own internals: `time_provider` (what "now"
    means, for backfilling time_dimension fields -- generic emc2p has no
    notion of this; a caller like story-simulator computes it from its own
    world state) and the export directory names `update_registry` writes
    to when a session has one set.

    `open_tool_name` names the tool a "no registry open yet" error points
    at -- `open_registry` (this class's own, the default) unless a caller
    excludes it from its own `mount()` call in favor of a richer
    session-opening tool of its own (story-simulator's `start_world`,
    which also bootstraps Postgres and seeds/tracks an export_dir) --
    that caller passes its own tool's name here so the error a client
    actually sees names a tool that's really mounted on that server.
    """

    def __init__(
        self,
        time_provider: Callable[[Registrar], Any] | None = None,
        export_dirname: str = "registry",
        export_history_dirname: str = "registry_history",
        open_tool_name: str = "open_registry",
    ):
        self._registrars: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
        self._export_dirs: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
        self._time_provider = time_provider or (lambda registrar: None)
        self._export_dirname = export_dirname
        self._export_history_dirname = export_history_dirname
        self._open_tool_name = open_tool_name

    # Plain-session API -- usable in-process, no MCP Context needed. Each
    # `@server.tool()`-mountable method below wraps one of these.

    def get_registrar(self, session) -> Registrar:
        """Return this session's open Registrar, or raise a clear error if none is open."""
        if session not in self._registrars:
            raise ValueError(f"No registry open for this session yet -- call {self._open_tool_name} first.")
        return self._registrars[session]

    def set_registrar(self, session, registrar: Registrar, export_dir: str | Path | None = None) -> None:
        """Register an already-loaded Registrar for `session` directly,
        bypassing `open`'s own manifest_dir-seeding logic.

        For a caller that does its own richer loading (scenario seeding,
        resuming a save, ...) and just needs to hand the result to this
        instance -- e.g. story-simulator's own `start_world`, whose
        load/seed logic doesn't match `open`'s simpler "merge manifest_dir
        once if empty" behavior.
        """
        self._registrars[session] = registrar
        if export_dir:
            self._export_dirs[session] = Path(export_dir)

    def open(
        self, session, database_url: str, manifest_dir: str | None = None, export_dir: str | None = None
    ) -> str:
        """Open (or create) a Registrar backed by `database_url` for `session`.

        `manifest_dir`'s EC files are merged in only the first time this
        exact `database_url` is opened (a brand-new registry has no
        component types yet) -- a resumed registry's accumulated state,
        including whatever schema it already knows, is untouched. Merged
        under `SOURCE_KEY`, the same source key every later
        `update_registry` write for this session shares -- merging it via a
        separate, permanently-fixed source key instead would hash each
        seeded alias (e.g. `widget_a`) to a *different* entity_id than the
        one a later write to that same alias gets, silently splitting one
        entity into two.

        `export_dir`, if given, is remembered for this session: every later
        `update_registry` write re-exports the registry's current state
        there (see `_export_registry`), not just this initial seed.
        """
        registrar = Registrar.load(database_url)
        if manifest_dir and not registrar.registry.component_types:
            seed_text = "\n".join(f.read_text() for f in sorted(Path(manifest_dir).rglob("*.y*ml")))
            registrar.update(yaml_strings={SOURCE_KEY: seed_text})
        self.set_registrar(session, registrar, export_dir)
        return f"Registry opened at {database_url}. Component types: {registrar.registry.component_types}."

    def view_registry(self, session, component_type: str) -> str:
        """View all recorded data for one component type (e.g. "status", "object")."""
        registrar = self.get_registrar(session)
        return registrar.view_df(component_type).fillna("null").to_markdown()

    def view_entity(self, session, entity_id: str) -> str:
        """Return every recorded component instance for a specific entity.

        Args:
            entity_id: Entity hash, alias, or unambiguous path fragment.
        """
        registrar = self.get_registrar(session)
        return registrar.view_entity(entity_id, format="markdown")

    def consolidate_review(self, session, limit: int = 20) -> str:
        """Show every component type recorded so far, for the caller to review and consolidate.

        Unlike view_registry (one component type at a time), this covers
        everything written so far in one call, so a caller can notice the
        same fact recorded twice under different (possibly invented)
        component names, or a wrong-but-real component holding data that
        belongs under the correct one instead -- without already having to
        suspect which type to check. Nothing is changed automatically: a
        caller that finds something writes a correction via update_registry.

        Args:
            limit: Max sample rows shown per component type (default 20).
        """
        registrar = self.get_registrar(session)
        return f"{registrar.summarize_components(limit)}\n\n{CONSOLIDATION_GUIDANCE}"

    def update_registry(
        self,
        session,
        yaml_string: str | None = None,
        preview: bool = False,
        confirm_token: str | None = None,
    ) -> str:
        """Merge entity-first YAML into the registry, run through the same ETL as save files.

        Default behavior (`preview` omitted/False, `confirm_token` omitted):
        `yaml_string` merges immediately. `preview=True` opts into a
        two-round-trip alternative instead: don't merge yet, return a
        review (which component types the write references, every
        component type actually available, a stats+peek summary for each
        referenced type) plus a confirm_token. If the review looks right,
        call again with that confirm_token (no yaml_string needed) to merge
        it; if something looks off, call again with corrected yaml_string
        instead.

        A successful merge (not a preview) also re-exports the registry's
        current state if this session has an export directory set (see
        `open`'s own `export_dir`), and backfills any still-null
        time_dimension field using this instance's own `time_provider`.

        Args:
            yaml_string: Entity-first YAML content to merge (or, with
                `preview=True`, to preview before merging). Omit when
                passing `confirm_token` instead.
            preview: If True, don't merge `yaml_string` yet -- return a
                review and a confirm_token to merge it with on a follow-up
                call. Ignored if `confirm_token` is given.
            confirm_token: A token from a prior `preview=True` call --
                merges that exact previously-previewed write now.
                `yaml_string` is not needed (and ignored) in this call.
        """
        registrar = self.get_registrar(session)

        if confirm_token is not None:
            self._merge_and_sync(session, registrar, _decode_pending_write(confirm_token))
            return "Merged into the registry."

        if yaml_string is None:
            raise ValueError(
                "Provide yaml_string to merge (optionally with preview=True to review it "
                "first), or confirm_token from a prior preview to merge that write now."
            )

        if not preview:
            self._merge_and_sync(session, registrar, yaml_string)
            return "Merged into the registry."

        return self._preview(registrar, yaml_string)

    def dispatch(self, session, name: str, arguments: dict[str, Any]) -> str:
        """Route a model-shaped tool call (a name plus a flat arguments dict) to the matching method above.

        For a caller driving a model in-process via `emc2p.agents.
        tool_calling_loop.run_tool_calling_loop` instead of real MCP
        transport -- that loop's own `dispatch` contract is exactly
        `(name, arguments) -> str`, one level flatter than these methods'
        own per-tool signatures (`view_registry(session, component_type)`,
        `update_registry(session, yaml_string, ...)`, ...), so this just
        unpacks `arguments` as keyword arguments onto whichever method
        `name` picks.

        `name` is checked against an explicit allowlist first, not
        resolved via bare `getattr(self, name)` -- `name` here comes from
        a model's own output, and this instance also carries methods
        that must stay unreachable by name (`get_registrar`, `open`, the
        underscore-prefixed internals).
        """
        if name not in _DISPATCHABLE_METHODS:
            return f"Error: unknown tool {name!r}"
        return getattr(self, name)(session, **arguments)

    # Internal helpers

    def _merge_and_sync(self, session, registrar: Registrar, yaml_string: str) -> None:
        _validate_write(registrar, yaml_string)
        registrar.update(
            yaml_strings={SOURCE_KEY: yaml_string},
            time=self._time_provider(registrar),
        )
        export_dir = self._export_dirs.get(session)
        if export_dir is not None:
            self._export_registry(registrar, export_dir)

    def _export_registry(self, registrar: Registrar, export_dir: Path) -> None:
        """Re-export `registrar`'s state to `export_dir`'s collapsed/
        uncollapsed subdirectories.

        A straight, mechanical round-trip (`Registrar.export_manifest`) --
        one directory holds the current state, collapsed (a time_dimension'd
        component type shows only its current row per entity); the other
        holds the same content, uncollapsed (every row ever recorded). Each
        is a directory, not a fixed filename, since a save can be split
        across multiple files.
        """
        current_dir = export_dir / self._export_dirname
        current_yaml = current_dir / f"{SOURCE_KEY}.yaml"
        for path in registrar.export_manifest(output_dir=str(current_dir)):
            if Path(path) != current_yaml:
                Path(path).unlink(missing_ok=True)

        history_dir = export_dir / self._export_history_dirname
        history_yaml = history_dir / f"{SOURCE_KEY}.yaml"
        for path in registrar.export_manifest(output_dir=str(history_dir), collapse_time_dimension=False):
            if Path(path) != history_yaml:
                Path(path).unlink(missing_ok=True)

    def _preview(self, registrar: Registrar, yaml_string: str) -> str:
        """Validate `yaml_string` without merging it yet, and build a review + confirm_token."""
        referenced = _validate_write(registrar, yaml_string)
        known = sorted(registrar.registry.known_component_types)
        token = _encode_pending_write(yaml_string)
        summaries = "\n".join(_component_write_summary(registrar, c) for c in sorted(referenced)) or "(none referenced)"
        return (
            "Not yet merged -- review below, then confirm or correct it.\n\n"
            f"Components this write references: {sorted(referenced)}\n"
            f"All component types actually available: {known}\n\n"
            f"Stats + peek for each referenced component type:\n{summaries}\n\n"
            "If this looks right, call update_registry again with this exact "
            f"confirm_token (and no yaml_string) to merge it:\n{token}\n\n"
            "If something looks wrong -- a component you referenced isn't the one you "
            "meant, or the peek shows existing rows shaped differently than what you "
            "wrote -- call update_registry again with corrected yaml_string (and "
            "preview=True) instead of this token."
        )

    # MCP tool registration -- mounts every method above onto `server`,
    # each as a thin Context-taking wrapper around the plain-session one.

    def mount(self, server: FastMCP, exclude: set[str] | None = None) -> None:
        """Register this instance's tools onto `server`.

        `exclude` skips named tools entirely -- e.g. a caller with its own
        session-opening tool (story-simulator's `start_world`, which also
        bootstraps Postgres and wires up an export_dir) passes
        `{"open_registry"}` so a client can't silently swap out that
        already-open registrar for an unrelated one via this generic tool
        instead.
        """
        exclude = exclude or set()

        def open_registry(
            database_url: str, ctx: Context, manifest_dir: str | None = None, export_dir: str | None = None
        ) -> str:
            return self.open(ctx.request_context.session, database_url, manifest_dir, export_dir)

        def view_registry(component_type: str, ctx: Context) -> str:
            return self.view_registry(ctx.request_context.session, component_type)

        def view_entity(entity_id: str, ctx: Context) -> str:
            return self.view_entity(ctx.request_context.session, entity_id)

        def update_registry(
            yaml_string: str | None = None,
            *,
            ctx: Context,
            preview: bool = False,
            confirm_token: str | None = None,
        ) -> str:
            return self.update_registry(ctx.request_context.session, yaml_string, preview, confirm_token)

        def consolidate_review(ctx: Context, limit: int = 20) -> str:
            return self.consolidate_review(ctx.request_context.session, limit)

        # description passed explicitly -- Tool.from_function reads it once at
        # registration time, before a wrapper's own docstring could be set.
        tools = {
            "open_registry": (open_registry, self.open.__doc__),
            "view_registry": (view_registry, self.view_registry.__doc__),
            "view_entity": (view_entity, self.view_entity.__doc__),
            "update_registry": (update_registry, self.update_registry.__doc__),
            "consolidate_review": (consolidate_review, self.consolidate_review.__doc__),
        }
        for name, (fn, description) in tools.items():
            if name not in exclude:
                server.add_tool(fn, name=name, description=description)


def _validate_write(registrar: Registrar, yaml_string: str) -> set[str]:
    """The checks a write must pass before it's safe to merge (or, for the
    preview flow, safe to show a confirm token for).

    Raises on the first one that fails. Returns the set of component types
    this write actually references. The bare-mapping check (a component
    given as a dict instead of a list) is `load_yaml`'s own
    `_find_malformed_entities` -- private there since its only other caller
    is `registrar.update()` itself, reused directly here because the
    preview flow needs this same check without actually merging. The other
    two (component-named-as-nested-entity, unknown component type) are
    `registrar.validate_write`'s own job.
    """
    parsed = yaml.safe_load(yaml_string)
    if isinstance(parsed, dict):
        malformed = _find_malformed_entities(parsed)
        if malformed:
            raise ValueError(
                "Each entity's components must be a YAML list (a leading `- ` before each "
                "component name), e.g. `alias:\n    - component:\n        value: x` -- a bare "
                "mapping (`alias:\n    component:\n        value: x`) is silently ignored "
                f"rather than merged. Malformed here: {', '.join(malformed)}."
            )
    return registrar.validate_write(yaml_string)


def _component_write_summary(registrar: Registrar, component_type: str, limit: int = 3) -> str:
    """Stats + peek for one component type -- row count, plus up to `limit` existing sample rows.

    So a caller about to write this component type can see the
    shape/convention already in use for it.
    """
    try:
        df = registrar.view_df(component_type)
    except KeyError:
        return f"- {component_type}: declared, but no data written yet."
    if df.empty:
        return f"- {component_type}: declared, 0 rows written yet."
    sample = df.head(limit).fillna("null").to_markdown()
    return f"- {component_type}: {len(df)} row(s) so far. Sample:\n{sample}"


def _encode_pending_write(yaml_string: str) -> str:
    """Self-contained confirm_token for update_registry's preview flow.

    Encodes `yaml_string` directly rather than stashing it in server-side
    session state that a process restart between preview and confirm would lose.
    """
    return base64.urlsafe_b64encode(yaml_string.encode()).decode()


def _decode_pending_write(confirm_token: str) -> str:
    return base64.urlsafe_b64decode(confirm_token.encode()).decode()


server = FastMCP("emc2p", instructions=INSTRUCTIONS)
sessions = RegistrarSessions()
sessions.mount(server)


def main() -> None:
    server.run(transport="stdio")


if __name__ == "__main__":
    main()

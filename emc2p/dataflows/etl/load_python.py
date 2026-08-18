"""Hamilton DAG for loading entity-first data from Python source files."""

import ast
import re
import textwrap
from pathlib import Path

import yaml

from ...config import DOCSTRING_COMPONENTS_PATTERN

_COMPONENTS_HEADER_RE = re.compile(DOCSTRING_COMPONENTS_PATTERN, re.MULTILINE)
_SECTION_HEADER_RE = re.compile(r"^[ \t]*\S.*\n[ \t]*-{3,}[ \t]*$", re.MULTILINE)


def _split_docstring_components(docstring: str) -> tuple[str, str | None]:
    """Split a docstring into (description_text, components_yaml_text).

    Looks for a numpy-style "Components" section (see
    ``emc2p.config.DOCSTRING_COMPONENTS_PATTERN``) and pulls its body out
    separately from the rest of the docstring, which becomes the
    description text. Returns ``(docstring, None)`` unchanged if no
    Components section is present.
    """
    match = _COMPONENTS_HEADER_RE.search(docstring)
    if match is None:
        return docstring, None

    before = docstring[: match.start()].rstrip()
    rest = docstring[match.end():]
    next_header = _SECTION_HEADER_RE.search(rest)
    if next_header:
        section_text = rest[: next_header.start()]
        after = rest[next_header.start():].lstrip("\n")
    else:
        section_text = rest
        after = ""

    description = "\n\n".join(part for part in (before, after) if part)
    components_yaml = textwrap.dedent(section_text).strip()
    return description, (components_yaml or None)


def _parse_components_yaml(text: str) -> list:
    """Parse a docstring Components section into a list of component entries.

    Returns an empty list if the text fails to parse as YAML or doesn't
    parse to a list (the expected shape for a component list).
    """
    try:
        parsed = yaml.safe_load(text)
    except yaml.YAMLError:
        return []
    if not isinstance(parsed, list):
        return []
    return parsed


def _find_iacs_meta(stmts: list) -> dict | None:
    """Return the value of the first __iacs__ = {...} assignment in stmts, or None."""
    for stmt in stmts[:5]:
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and stmt.targets[0].id == "__iacs__"
            and isinstance(stmt.value, ast.Dict)
        ):
            try:
                return ast.literal_eval(stmt.value)
            except (ValueError, TypeError):
                pass
    return None


def _call_target(func: ast.expr) -> str | None:
    """Best-effort dotted name for a Call node's ``func``, or None if unresolvable.

    ``self.foo(...)``/``cls.foo(...)`` drop the leading ``self``/``cls`` (it
    isn't part of any entity's qualified name); other attribute chains and
    bare names are kept as-is for emc2p's entity_ref substring matching to
    resolve later, at derive time. Calls whose target isn't a plain name or
    attribute chain (e.g. ``handlers[0]()`` or ``foo()()``) return None.
    """
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        parts = [func.attr]
        node = func.value
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if not isinstance(node, ast.Name):
            return None
        parts.append(node.id)
        parts.reverse()
        if parts[0] in ("self", "cls"):
            parts = parts[1:]
        return ".".join(parts) if parts else None
    return None


def _is_constructor_call(target: str) -> bool:
    """Heuristic: PEP 8 class names start with an uppercase letter.

    A call-reachability trace is meant to show callables reaching other
    callables, not "this function builds one of these" -- ``TurnReplay()``
    constructs a ``TurnReplay``, it doesn't call any behavior named
    ``TurnReplay``, so it's not a real edge in that trace. Checked against
    the target's last dotted segment (``spatial.TurnReplay`` is still a
    constructor call) so an attribute-chained construction is caught too.
    """
    last_segment = target.rsplit(".", 1)[-1]
    return last_segment[:1].isupper()


class _BodyCollector(ast.NodeVisitor):
    """Collects call/import targets made directly in a node's own body.

    Does not descend into nested function/class definitions -- those get
    their own entity (if they qualify) and their own extraction via the
    outer ``_walk`` recursion, so their calls/imports must not double up
    onto an ancestor's.
    """

    def __init__(self):
        self.calls: list[str] = []
        self.imports: list[str] = []

    def visit_Call(self, node: ast.Call) -> None:
        target = _call_target(node.func)
        if target and not _is_constructor_call(target):
            self.calls.append(target)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        self.imports.extend(alias.name for alias in node.names)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            self.imports.extend(f"{node.module}.{alias.name}" for alias in node.names)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        pass

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        pass

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        pass


def _collect_body(stmts: list) -> tuple[list[str], list[str]]:
    """Return (calls in first-occurrence order deduped, sorted deduped imports)
    made directly in stmts.

    Calls keep call order (not alphabetical) because that order is real,
    caller-observable information -- see ``_make_components``'s ``seq``
    field and ``architecture_graph.build_call_reachability``'s
    ``call_sequences``, which use it to draw "this call happens before
    that one" arrows a diagram viewer couldn't otherwise recover (Mermaid
    edge order isn't call order, and multiple calls to the same target
    collapse to one edge either way). Imports have no analogous consumer,
    so they stay alphabetical for stable, easy-to-scan output.

    A constructor call (``_is_constructor_call``) is left out of ``calls``
    entirely -- a call-reachability trace is meant to follow callables
    calling other callables, and instantiating a type isn't that.

    See ``_BodyCollector`` for the "directly" scoping rule.
    """
    collector = _BodyCollector()
    for stmt in stmts:
        collector.visit(stmt)
    return list(dict.fromkeys(collector.calls)), sorted(set(collector.imports))


def _make_components(
    docstring: str | None,
    iacs_meta: dict | None,
    calls: list[str] | None = None,
    imports: list[str] | None = None,
) -> list | None:
    """Build a component list from a docstring, __iacs__ metadata, calls, and imports.

    Returns None if empty. A docstring's "Components" section (see
    ``_split_docstring_components``) is parsed as YAML and merged into the
    component list rather than the description; it is not included in
    either part of a round trip: it is excluded from the description text,
    and also excluded from manifest export (see
    ``emc2p.dataflows.etl.export_manifest``), since it lives in a Python
    source file rather than a standalone YAML file.

    Whether this construct qualifies as an entity at all is still decided
    purely by docstring/``__iacs__``/components_yaml, matching
    ``_extract_entities``'s existing rule -- ``calls``/``imports`` are only
    ever appended to an entity that already qualifies for another reason,
    never enough on their own to conjure one up (which would turn every
    undocumented helper function into a graph node).

    Each ``calls`` entry carries its own 0-indexed ``seq`` (``calls``'s
    first-occurrence position in the source, see ``_collect_body``) rather
    than being a bare scalar like ``imports`` -- that's what lets
    ``architecture_graph.build_call_reachability`` later recover "this
    entity called A before B" as its own ``call_sequences``, not just the
    unordered set of what it called.
    """
    description, components_yaml = (
        _split_docstring_components(docstring) if docstring else (None, None)
    )
    components = []
    if description:
        components.append({"description": description.strip()})
    if iacs_meta:
        for key, value in iacs_meta.items():
            components.append({key: value})
    if components_yaml:
        components.extend(_parse_components_yaml(components_yaml))
    if not components:
        return None
    for seq, target in enumerate(calls or []):
        components.append({"calls": {"value": target, "seq": seq}})
    for target in imports or []:
        components.append({"imports": target})
    return components


def _extract_entities(tree: ast.Module, module_name: str) -> dict:
    """Walk an AST module and return a flat entity-first dict.

    Entity keys are fully-qualified dotted names within the module
    (e.g. ``emc2p.dataflows.etl.load_manifest.MyClass.my_method``).
    Only constructs with a docstring or ``__iacs__`` assignment are included;
    such an entity also gets ``calls``/``imports`` components for whatever
    it calls/imports directly in its own body (see ``_collect_body``).
    """
    entities: dict = {}

    calls, imports = _collect_body(tree.body)
    components = _make_components(
        ast.get_docstring(tree), _find_iacs_meta(tree.body), calls, imports
    )
    if components is not None:
        entities[module_name] = components

    def _walk(stmts: list, prefix: str) -> None:
        for node in stmts:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qual = f"{prefix}.{node.name}"
                calls, imports = _collect_body(node.body)
                comps = _make_components(
                    ast.get_docstring(node), _find_iacs_meta(node.body), calls, imports
                )
                if comps is not None:
                    entities[qual] = comps
                _walk(node.body, qual)
            elif isinstance(node, ast.ClassDef):
                qual = f"{prefix}.{node.name}"
                calls, imports = _collect_body(node.body)
                comps = _make_components(
                    ast.get_docstring(node), _find_iacs_meta(node.body), calls, imports
                )
                if comps is not None:
                    entities[qual] = comps
                _walk(node.body, qual)

    _walk(tree.body, module_name)
    return entities


def parsed_asts(raw_python_strings: dict[str, str]) -> dict[str, ast.Module]:
    """Parse raw Python source text into AST modules, keyed by file identifier.

    Files that fail to parse (e.g. non-Python content, syntax errors) are
    silently dropped.

    Parameters
    ----------
    raw_python_strings : dict[str, str]
        A dict keyed by file identifier, where each value is raw Python
        source text.

    Returns
    -------
    dict[str, ast.Module]
        A dict keyed by file identifier, where each value is the parsed AST
        module for that file's source.
    """
    result = {}
    for file_id, source in raw_python_strings.items():
        try:
            result[file_id] = ast.parse(source, filename=file_id)
        except SyntaxError:
            continue
    return result


def module_names(parsed_asts: dict[str, ast.Module]) -> dict[str, str]:
    """Derive a dotted module name for each successfully parsed file.

    e.g. ``"emc2p/dataflows/etl/load_python.py"`` becomes
    ``"emc2p.dataflows.etl.load_python"``.
    """
    return {
        file_id: ".".join(Path(file_id).with_suffix("").parts)
        for file_id in parsed_asts
    }


def raw_entity_first_data(
    parsed_asts: dict[str, ast.Module],
    module_names: dict[str, str],
) -> dict:
    """Extract entity-first dicts from parsed ASTs, keyed by file identifier.

    Extracts entities from modules, classes, and functions that have a
    docstring or ``__iacs__`` metadata. Entity keys are fully-qualified
    dotted names derived from the file identifier (e.g.
    ``emc2p.dataflows.etl.load_manifest.raw_entity_first_data``).

    Parameters
    ----------
    parsed_asts : dict[str, ast.Module]
        A dict keyed by file identifier, where each value is the parsed AST
        module for that file's source, as returned by ``parsed_asts``.
    module_names : dict[str, str]
        A dict keyed by file identifier, where each value is that file's
        dotted module name, as returned by ``module_names``.

    Returns
    -------
    dict
        A dict keyed by file identifier, where each value is the entity-first
        dict of entities extracted from that source.
    """
    result = {}
    for file_id, tree in parsed_asts.items():
        entities = _extract_entities(tree, module_names[file_id])
        if entities:
            result[file_id] = entities
    return result


FINAL_VAR = "raw_entity_first_data"

# Getter API requirement combinations

Every relevant combination of `getter_api_requirements`' axes (see
`emc2p.yaml`), pruned per that entity's own note: some combinations aren't
relevant (e.g. a tabular format for one field on one entity), so solution
coverage should focus on what's actually likely to be used.

`implementation_details.interface`/`.implementation` aren't a branching
axis — every combination below includes both as-is (see the note on that
entity). They're omitted from the listing rather than repeated 212 times.

## Structure

Three of the six axes — `scd_data`, `error_handling`,
`entity_specification_method` — are fully independent of the other three
*and* of each other: every value of one combines meaningfully with every
value of the others, for (almost) every "shape" below. So rather than
flattening the full cross product into ~212 near-duplicate lines, this is
factored as: **shape** (`output_granularity` × `output_format` ×
`output_singularity`, pruned for relevance) × **modifiers** (`scd_data` ×
`error_handling` × `entity_specification_method`). Both factors are listed
in full below; their cross product is still every combination — nothing is
hidden behind a resolver, it's just not typed out 212 times by hand.

### The 12 orthogonal modifier combinations (apply to every shape except `raw_data`'s)

`entity_specification_method` doesn't apply to `raw_data` (see below), so
`raw_data`'s shapes cross only the 4 `scd_data` × `error_handling`
combinations, not all 12.

1. full_history + raises_errors + entity_id
2. full_history + raises_errors + entity_path
3. full_history + raises_errors + entity_alias
4. full_history + returns_empty_or_null_objects + entity_id
5. full_history + returns_empty_or_null_objects + entity_path
6. full_history + returns_empty_or_null_objects + entity_alias
7. current_data + raises_errors + entity_id
8. current_data + raises_errors + entity_path
9. current_data + raises_errors + entity_alias
10. current_data + returns_empty_or_null_objects + entity_id
11. current_data + returns_empty_or_null_objects + entity_path
12. current_data + returns_empty_or_null_objects + entity_alias

(`raw_data`'s reduced set is just 1/4/7/10 above with the
`entity_specification_method` clause dropped.)

## Shapes, by `output_granularity`

### `raw_data` — 2 shapes × 4 modifiers = 8 combinations

Access to a whole component table, unfiltered, as stored. No selection
happens, so `output_singularity` doesn't apply (there's no single item to
collapse to); no entity is targeted by id/path/alias either, so
`entity_specification_method` doesn't apply. Only a tabular format makes
sense for "the whole table."

- `raw_data` + `ibis_tables`
- `raw_data` + `pandas_dataframes`

### `components_selection` — 5 shapes × 12 modifiers = 60 combinations

Choosing which component(s) to join on `entity_id`, across entities.
Multiple components chosen is naturally tabular (entities × components).
A single component collapses one level down per `output_singularity`'s
own note, to a value-per-entity row/column format — entity-ID-indexed, so
all three `row_or_column` formats fit, `list_of_values` included.
Tabular-for-single and row/column-for-multiple are excluded as exactly
the shape mismatch the requirement's note warns about.

- `components_selection` + `multiple_version_of_output` + `ibis_tables`
- `components_selection` + `multiple_version_of_output` + `pandas_dataframes`
- `components_selection` + `single_version_of_output` + `pandas_series`
- `components_selection` + `single_version_of_output` + `dictionary_of_values`
- `components_selection` + `single_version_of_output` + `list_of_values`

### `entities_selection` — 4 shapes × 12 modifiers = 48 combinations

Symmetric to `components_selection`: multiple entities selected is
tabular (rows = entities). A single entity collapses to a row-like format
representing that one entity's values — but this time it's
component/field-name-indexed, not entity-ID-indexed, so `list_of_values`
(which "assumes IDs at same indices") doesn't fit; only `pandas_series`
and `dictionary_of_values` do.

- `entities_selection` + `multiple_version_of_output` + `ibis_tables`
- `entities_selection` + `multiple_version_of_output` + `pandas_dataframes`
- `entities_selection` + `single_version_of_output` + `pandas_series`
- `entities_selection` + `single_version_of_output` + `dictionary_of_values`

### `field_selection` — 8 shapes × 12 modifiers = 96 combinations

Choosing which field(s), within component(s), to retrieve. Two
sub-dimensions are bundled into the one `output_singularity` axis here:
how many fields, and (implicitly) how many entities the call resolves to.

Multiple fields across multiple entities is tabular. Multiple fields
resolving to one entity is a row-like format keyed by field/component
name — not ID-aligned, so `list_of_values` is excluded (same reasoning as
`entities_selection`, single).

A single field across multiple entities is ID-aligned, so all three
`row_or_column` formats fit. A single field resolving to a single entity
collapses all the way to a scalar — this is exactly the "one field on one
entity" case the requirement's own note calls irrelevant *for a tabular
format*; here it's `constant_output_formats`, not tabular, which is the
whole point.

- `field_selection` + `multiple_version_of_output` + `ibis_tables`
- `field_selection` + `multiple_version_of_output` + `pandas_dataframes`
- `field_selection` + `multiple_version_of_output` + `pandas_series`
- `field_selection` + `multiple_version_of_output` + `dictionary_of_values`
- `field_selection` + `single_version_of_output` + `pandas_series`
- `field_selection` + `single_version_of_output` + `dictionary_of_values`
- `field_selection` + `single_version_of_output` + `list_of_values`
- `field_selection` + `single_version_of_output` + `constant_output_formats`

## Total: 8 + 60 + 48 + 96 = 212 combinations

(Each also has both `implementation_details.interface` and `.implementation`
as-is, per the note that axis isn't a branching one.)

## Coverage assessment: existing candidate solutions

Both `getter_api_solutions` candidates are sketched only against
`error_handling` (their own examples: `safe_` prefix vs. `raise_errors=False`).
Neither says anything yet about the other five axes. Going through the
axes against what each *approach* (not just its one sketched example)
could plausibly extend to:

- **`error_handling`** (2 values): both candidates cover this cleanly —
  it's a boolean behavior flag, and both a name-prefix (`safe_get_...`)
  and an argument (`raise_errors=False`) express a boolean naturally.
- **`scd_data`** (2 values): not sketched by either, but both extend
  easily by the same pattern (`history_get_...` vs. `history=True`) —
  no coverage gap in kind, just in that neither wrote the example down.
- **`entity_specification_method`** (3 values): this isn't a boolean —
  it's *which kind of value* identifies the entity. `..._via_argument`
  handles it naturally (differently-typed/named kwargs, or a single
  smart argument). `..._via_method_name` also has a plausible idiom
  (`get_by_id`/`get_by_path`/`get_by_alias`), but this is the first axis
  where the method-name approach starts adding a whole word per call
  rather than a flag.
- **`output_granularity` × `output_format` × `output_singularity`**
  (the 19 shapes above): **neither candidate addresses this at all**, and
  it's where nearly all 212 combinations' variation actually lives. This
  matters because, unlike `error_handling`/`scd_data`, these three axes
  jointly determine the *return type* of the call (a table vs. a
  series/dict/list vs. a scalar) — qualitatively different objects, not a
  behavioral flag on the same object. Neither sketch's example
  generalizes to this axis group as-is:
  - `..._via_method_name`, taken literally, would need a distinct method
    name per one of the 19 shapes (e.g. `get_component_table`,
    `get_field_series`, `get_scalar_field`, ...) — plausible in isolation
    (real APIs do name methods by return shape), but stacking the other
    axes' prefixes on top of that (`safe_history_get_field_series_by_alias`)
    produces an unwieldy, hard-to-discover method per full combination —
    up to 212 names if taken to its conclusion.
  - `..._via_argument`, taken literally, would need one generic `get()`
    whose return type depends on argument values rather than the call
    signature — workable at runtime but weak for static typing/IDE
    autocomplete, since the shape isn't visible in the method name at all.

### Gap: a third candidate is needed for the shape axes

Neither existing candidate, extended straightforwardly, covers the
`output_granularity`/`output_format`/`output_singularity` group well on
its own. The natural fix is a **hybrid**: use distinct method names for
the 19 *shapes* (since those determine return type, and a handful of
well-named methods is exactly what method-name-based dispatch is good
at), and use arguments/prefixes for the three orthogonal modifiers
(`scd_data`, `error_handling`, `entity_specification_method`), since those
are flags/typed-inputs on top of a fixed return type rather than
different return types themselves. This is recorded as a new
`hybrid_method_name_and_argument` solution candidate in `emc2p.yaml`.

`implementation_details.implementation` (the mechanics: how joins,
collapsing to current-per-entity, and alias path resolution actually
work under the hood) remains unaddressed by all three candidates — that's
expected per the requirement's own framing (fitting a concrete
implementation is solution-design work, not part of picking the
interface shape).

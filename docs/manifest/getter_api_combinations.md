# Getter API requirement combinations

Every relevant combination of `getter_api_requirements`' axes (see
`emc2p.yaml`), pruned per that entity's own note: some combinations aren't
relevant (e.g. a tabular format for one field on one entity), so solution
coverage should focus on what's actually likely to be used.

`implementation_details.interface`/`.implementation` aren't a branching
axis — every combination below includes both as-is (see the note on that
entity). They're omitted from the listing rather than repeated 216 times.

`output_format` is treated at its top level only for now -- five
categories: `tabular_output_formats`, `row_or_column_output_formats`,
`constant_output_formats`, `entity_first_output_formats` (JSON-like,
nested-dictionary output keyed by entity first — the EC/entity-centered
style), and `pretty_printed_output_formats` (human-readable formatted
text) — not their individual leaf formats (`ibis_tables`,
`pandas_dataframes`, `pandas_series`, `dictionary_of_values`,
`list_of_values`). Picking a concrete leaf per category — and any
leaf-specific fit issues that come with it (e.g. `list_of_values`
"assumes IDs at same indices," so it only fits an ID-aligned collection,
not a field-name-aligned one) — is deferred to solution design.

`entity_first_output_formats` and `pretty_printed_output_formats` behave
differently from the other three categories, which is why they widen
the shape list rather than just adding one more option to it:

- `entity_first_output_formats` is a **structural alternative** to
  `tabular_output_formats` specifically — the same "many entities ×
  many values" data, re-expressed as a nested dict instead of a flat
  table. It's valid everywhere `tabular_output_formats` is (including
  `raw_data`, where no join/selection even happens), but doesn't replace
  `row_or_column_output_formats` or `constant_output_formats` (there's no
  "many entities" dimension left to nest by, once the result has already
  collapsed to one entity or one scalar).
- `pretty_printed_output_formats` is a **rendering choice orthogonal to
  shape entirely** — any of the other categories' output can be
  pretty-printed. So it's added once per `(output_granularity,
  output_singularity)` pair, alongside whatever structural formats
  already fit that pair, rather than being tied to one structural
  category.

## Structure

Three of the six axes — `scd_data`, `error_handling`,
`entity_specification_method` — are fully independent of the other three
*and* of each other: every value of one combines meaningfully with every
value of the others, for (almost) every "shape" below. So rather than
flattening the full cross product into ~216 near-duplicate lines, this is
factored as: **shape** (`output_granularity` × `output_format` category ×
`output_singularity`, pruned for relevance) × **modifiers** (`scd_data` ×
`error_handling` × `entity_specification_method`). Both factors are listed
in full below; their cross product is still every combination — nothing is
hidden behind a resolver, it's just not typed out 216 times by hand.

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

### `raw_data` — 3 shapes × 4 modifiers = 12 combinations

Access to a whole component table, unfiltered, as stored. No selection
happens, so `output_singularity` doesn't apply (there's no single item to
collapse to); no entity is targeted by id/path/alias either, so
`entity_specification_method` doesn't apply. `tabular_output_formats` and
`entity_first_output_formats` are two structural ways to represent the
same whole table; `pretty_printed_output_formats` renders either as
human-readable text.

- `raw_data` + `tabular_output_formats`
- `raw_data` + `entity_first_output_formats`
- `raw_data` + `pretty_printed_output_formats`

### `components_selection` — 5 shapes × 12 modifiers = 60 combinations

Choosing which component(s) to join on `entity_id`, across entities.
Multiple components chosen is naturally tabular (entities × components)
or entity-first (nested by entity); pretty-printed is a display
alternative to either. A single component collapses one level down per
`output_singularity`'s own note, to a value-per-entity row/column
format — entity-first doesn't apply here (there's no longer a "many
entities × many values" shape to nest), but pretty-printed still does.
Tabular/entity-first-for-single and row/column-for-multiple are excluded
as exactly the shape mismatch the requirement's note warns about.

- `components_selection` + `multiple_version_of_output` + `tabular_output_formats`
- `components_selection` + `multiple_version_of_output` + `entity_first_output_formats`
- `components_selection` + `multiple_version_of_output` + `pretty_printed_output_formats`
- `components_selection` + `single_version_of_output` + `row_or_column_output_formats`
- `components_selection` + `single_version_of_output` + `pretty_printed_output_formats`

### `entities_selection` — 5 shapes × 12 modifiers = 60 combinations

Symmetric to `components_selection`: multiple entities selected is
tabular or entity-first (rows/nesting = entities), with pretty-printed as
a display alternative. A single entity collapses to a row-like format
representing that one entity's values, still with a pretty-printed
alternative.

- `entities_selection` + `multiple_version_of_output` + `tabular_output_formats`
- `entities_selection` + `multiple_version_of_output` + `entity_first_output_formats`
- `entities_selection` + `multiple_version_of_output` + `pretty_printed_output_formats`
- `entities_selection` + `single_version_of_output` + `row_or_column_output_formats`
- `entities_selection` + `single_version_of_output` + `pretty_printed_output_formats`

### `field_selection` — 7 shapes × 12 modifiers = 84 combinations

Choosing which field(s), within component(s), to retrieve. Two
sub-dimensions are bundled into the one `output_singularity` axis here:
how many fields, and (implicitly) how many entities the call resolves to.

Multiple fields across multiple entities is tabular or entity-first.
Multiple fields resolving to one entity is a row-like format keyed by
field/component name (no "many entities" left to nest by, so no
entity-first option there). A single field across multiple entities is a
row-like format keyed by entity. A single field resolving to a single
entity collapses all the way to a scalar — this is exactly the "one
field on one entity" case the requirement's own note calls irrelevant
*for a tabular format*; here it's `constant_output_formats`, not tabular,
which is the whole point. Pretty-printed applies to both the multiple-
and single-field cases.

- `field_selection` + `multiple_version_of_output` + `tabular_output_formats`
- `field_selection` + `multiple_version_of_output` + `row_or_column_output_formats`
- `field_selection` + `multiple_version_of_output` + `entity_first_output_formats`
- `field_selection` + `multiple_version_of_output` + `pretty_printed_output_formats`
- `field_selection` + `single_version_of_output` + `row_or_column_output_formats`
- `field_selection` + `single_version_of_output` + `constant_output_formats`
- `field_selection` + `single_version_of_output` + `pretty_printed_output_formats`

## Total: 12 + 60 + 60 + 84 = 216 combinations

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
  (the 20 shapes above): **neither candidate addresses this at all**, and
  it's where nearly all 216 combinations' variation actually lives. This
  matters because, unlike `error_handling`/`scd_data`, these three axes
  jointly determine the *return type* of the call (a table vs. a nested
  dict vs. a row/column collection vs. a scalar vs. pretty-printed text)
  — qualitatively different objects, not a behavioral flag on the same
  object. Neither sketch's example generalizes to this axis group as-is:
  - `..._via_method_name`, taken literally, would need a distinct method
    name per one of the 20 shapes (e.g. `get_component_table`,
    `get_entity_first_dict`, `get_field_collection`,
    `get_pretty_printed_table`, `get_scalar_field`, ...) — plausible in
    isolation (real APIs do name methods by return shape), but stacking
    the other axes' prefixes on top of that
    (`safe_history_get_pretty_printed_field_collection_by_alias`)
    produces an unwieldy, hard-to-discover method per full combination —
    up to 216 names if taken to its conclusion.
  - `..._via_argument`, taken literally, would need one generic `get()`
    whose return type depends on argument values rather than the call
    signature — workable at runtime but weak for static typing/IDE
    autocomplete, since the shape isn't visible in the method name at all.

### Gap: a third candidate is needed for the shape axes

Neither existing candidate, extended straightforwardly, covers the
`output_granularity`/`output_format`/`output_singularity` group well on
its own. The natural fix is a **hybrid**: use distinct method names for
the 20 *shapes* (since those determine return type, and a handful of
well-named methods is exactly what method-name-based dispatch is good
at), and use arguments/prefixes for the three orthogonal modifiers
(`scd_data`, `error_handling`, `entity_specification_method`), since those
are flags/typed-inputs on top of a fixed return type rather than
different return types themselves. This is recorded as a new
`hybrid_method_name_and_argument` solution candidate in `emc2p.yaml`.
Once a concrete leaf format is picked within a shape's `output_format`
category, that choice (e.g. `ibis_tables` vs. `pandas_dataframes` within
`tabular_output_formats`) would most naturally be its own argument too,
alongside the orthogonal modifiers.

`implementation_details.implementation` (the mechanics: how joins,
collapsing to current-per-entity, and alias path resolution actually
work under the hood) remains unaddressed by all three candidates — that's
expected per the requirement's own framing (fitting a concrete
implementation is solution-design work, not part of picking the
interface shape).

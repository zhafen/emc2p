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

## Solution evaluation

`getter_api_solutions` is a set of **mechanisms for selecting one axis's
value**, not complete getter APIs in themselves. The real API will very
likely hybridize several of them, using a different mechanism per axis
(or per group of axes). This section evaluates each mechanism against
each requirement axis, as input to deciding *how* to hybridize — it
doesn't commit to one hybrid combination.

### The mechanisms

- **`method_name`** — the axis value is baked into the method's own name
  as a word/prefix/suffix, e.g. `safe_get_...`.
- **`argument`** — the axis value is an explicit named parameter on the
  call, e.g. `raise_errors=False`.
- **`chained_accessor`** — a fluent chain of narrowing sub-objects/
  properties selects the axis value before the terminal call, e.g.
  `registry.entities.current.safe.get_by_alias(...)`.
- **`type_based_dispatch`** — the *type* of a single positional argument,
  not its name or an explicit flag, determines the axis value, e.g.
  passing an `EntityAlias("foo")` object vs. a plain `entity_id` int lets
  the same `get()` infer `entity_specification_method` from what was
  handed to it.
- **`result_object`** — the getter always returns one generic/lazy
  result, and the axis value is chosen afterward via a method/property on
  that result rather than an argument to the call itself, e.g.
  `get_component(...).to_pandas()` / `.pretty_print()`.
- **`configuration_object`** — several axis choices are bundled into one
  settings/spec object, built once and reused across calls, e.g.
  `get(entity, options=GetterOptions(scd="current", raise_errors=False))`.

### Axis-by-axis fit

**`output_granularity`** (raw_data / components_selection /
entities_selection / field_selection) — the most fundamental choice:
which data dimension you're even querying along, closest to picking
which operation to run at all, since each value needs different
companion arguments.

| mechanism | fit | why |
|---|---|---|
| method_name | good | Distinct verbs read naturally: `get_raw_table()`, `get_components()`, `get_entities()`, `get_fields()`. |
| argument | poor | Burying "which fundamentally different operation" in a value (`granularity="components"`) is a classic sign it should be separate methods instead. |
| chained_accessor | good | `registry.raw` / `.components` / `.entities` / `.fields` as sub-namespaces is the typed/discoverable version of method_name — IDE autocomplete lists the four. |
| type_based_dispatch | poor | Granularity is a choice of *what operation*, not something inferable from one argument's type within a fixed operation. |
| result_object | poor | Granularity determines what other arguments are even needed, so it can't be deferred to after the call. |
| configuration_object | poor | Same problem as argument — hides the most consequential choice inside a generic bag of options. |

**`output_format` × `output_singularity`** (tabular / row_or_column /
constant / entity_first / pretty_printed; single / multiple) — the
*shape of the returned object* for an already-fixed granularity; many
formats can validly represent the same selected data, and singularity is
often a consequence of selection size rather than an independent choice.

| mechanism | fit | why |
|---|---|---|
| method_name | moderate | Workable (`get_components_as_dataframe()`), but stacking a segment per format/singularity on top of the other axes' segments gets unwieldy fast. |
| argument | good | `format=IbisTables` fits naturally since many leaf formats can share one call signature otherwise. |
| chained_accessor | moderate | `registry.components.as_pandas.get(...)` is plausible but format is usually the *last* decision, so chaining it in beforehand reads backwards. |
| type_based_dispatch | poor | No natural "argument whose type implies desired output type" distinct from just passing the type as a value (which is really `argument`). |
| result_object | good | The strongest fit of any pairing here: return one lazy/generic result and let `.to_pandas()` / `.to_dict()` / `.pretty_print()` pick the format after the fact — sidesteps the method-name stacking problem entirely. For singularity specifically, this mechanism doesn't apply (see below). |
| configuration_object | moderate | Reasonable (`options.format=...`) but no better than argument. |

Singularity specifically also has a distinctive option: it can often be
**inferred for free** from how the entity/component selector was shaped
(a single ID vs. a list of IDs already implies singular vs. plural),
rather than declared as its own explicit axis value at all — see
`type_based_dispatch`/`argument` fit below, which differs from the
format-only assessment above.

| mechanism | fit | why |
|---|---|---|
| method_name | good | Explicit and safe when the API wants to force a shape regardless of actual selection size, e.g. `get_single_component()` asserting exactly one. |
| argument | moderate | `singular=True` works but is often redundant with what the selector's own cardinality already implies. |
| chained_accessor | poor | Feels heavy for what's usually a boolean toggle. |
| type_based_dispatch | good | Passing one ID vs. a collection of IDs as the entity specifier already implies singular vs. plural for free — no separate declaration needed. |
| result_object | poor | Like granularity, singularity changes what the immediate return even contains, so it can't be deferred. |
| configuration_object | moderate | Same as argument, no strong advantage. |

**`scd_data`** (full_history / current_data) — a straightforward
boolean-ish behavior toggle on an already-fixed shape.

| mechanism | fit | why |
|---|---|---|
| method_name | moderate–good | `get_history_...()` vs. a plain default works, at the cost of one more name segment. |
| argument | good | `history=True` / `scd="current"` is very natural. |
| chained_accessor | good | `registry.history.get(...)` reads like a mode switch, works well. |
| type_based_dispatch | poor | No natural "value whose type implies history mode." |
| result_object | moderate | Once current-vs-history is decided it changes what rows even come back (a time axis or not), so deferring feels backwards — though a `.as_of(timestamp)` result method for one specific query is plausible without fully generalizing. |
| configuration_object | good | Bundles cleanly alongside `error_handling`. |

**`error_handling`** (raises_errors / returns_empty_or_null_objects) —
also a clean boolean-ish behavior toggle; this axis motivated both
original candidates.

| mechanism | fit | why |
|---|---|---|
| method_name | good | `safe_get_...()` is idiomatic (the original example). |
| argument | good | `raise_errors=False` is idiomatic (the other original example). |
| chained_accessor | moderate | `registry.safe.get(...)` is plausible but unusual for what's normally a call-site concern rather than a standing mode. |
| type_based_dispatch | poor | No natural type-implies-error-mode reading. |
| result_object | poor–moderate | By the time a result object exists, an error would already have needed not to be raised to get one; a per-field `.get_or_default()` is imaginable but doesn't generalize to the whole call. |
| configuration_object | good | Bundles cleanly alongside `scd_data`. |

**`entity_specification_method`** (entity_id / entity_path /
entity_alias) — a *type of input value* ("which kind of key"), not a
behavior flag.

| mechanism | fit | why |
|---|---|---|
| method_name | good | `get_by_id()` / `get_by_path()` / `get_by_alias()` is idiomatic and common in real APIs. |
| argument | good | Named kwargs (`entity_id=` / `entity_path=` / `entity_alias=`), or one `entity=` argument with runtime type-checking, both work. |
| chained_accessor | moderate | `registry.by_alias.get(...)` is plausible but less idiomatic than a method suffix for "which key type," since it's specific to one call rather than a broad standing mode. |
| type_based_dispatch | good | The best fit for this axis specifically: a single argument whose type (a plain `str`/`int` vs. a dedicated `EntityAlias`/`EntityPath` wrapper) determines which specification method was used, with no separate flag at all. |
| result_object | poor | Purely an input-side concern; nothing to defer to the result. |
| configuration_object | moderate | Workable but no better than argument. |

### Reading the table

No single mechanism wins across all axes, which confirms the premise:
`output_granularity` wants `method_name`/`chained_accessor` (it's really
"which operation," not a flag); `output_format` wants `result_object`;
`output_singularity` is often inferable via `type_based_dispatch` rather
than needing its own explicit value; `scd_data`/`error_handling` are
happy with `argument`/`configuration_object`/`chained_accessor` more or
less interchangeably; and `entity_specification_method` wants
`type_based_dispatch` or `method_name`. Deciding the actual hybrid — which
mechanism(s) to commit to per axis, and how the resulting call sites
read once all six choices are combined — is the next step, not resolved
here.

`implementation_details.implementation` (the mechanics: how joins,
collapsing to current-per-entity, and alias path resolution actually
work under the hood) remains unaddressed by every mechanism above —
that's expected per the requirement's own framing (fitting a concrete
implementation is solution-design work, not part of picking the
interface shape).

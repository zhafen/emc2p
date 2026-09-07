# Getter API requirement combinations

Every relevant combination of `getter_api_requirements`' axes (see
`emc2p.yaml`), pruned per that entity's own note: some combinations aren't
relevant (e.g. a tabular format for one field on one entity), so solution
coverage should focus on what's actually likely to be used.

`implementation_details.interface`/`.implementation` aren't a branching
axis — every combination below includes both as-is (see the note on that
entity). They're omitted from the listing rather than repeated 224 times.

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
  pretty-printed. So it's added once per `(data_selection,
  output_singularity)` pair, alongside whatever structural formats
  already fit that pair, rather than being tied to one structural
  category.

## Axes at a glance

The full requirement space before pruning -- every axis and its options,
not yet crossed into combinations:

| axis | options |
|---|---|
| `implementation_details` | `interface`, `implementation` (not branching -- every combination includes both as-is) |
| `data_selection` | `raw_data`, `components_selection`, `entities_selection`, `field_selection` |
| `output_format` | `tabular_output_formats`, `row_or_column_output_formats`, `constant_output_formats`, `entity_first_output_formats`, `pretty_printed_output_formats` |
| `output_singularity` | `single_version_of_output`, `multiple_version_of_output` |
| `scd_data` | `full_history`, `current_data` |
| `error_handling` | `raises_errors`, `returns_empty_or_null_objects` |
| `entity_specification_method` | `entity_id`, `entity_path`, `entity_alias` |

## Structure

Three of the six axes — `scd_data`, `error_handling`,
`entity_specification_method` — are fully independent of the other three
*and* of each other: every value of one combines meaningfully with every
value of the others, for (almost) every "shape" below. So rather than
flattening the full cross product into ~224 near-duplicate lines, this is
factored as: **shape** (`data_selection` × `output_format` category ×
`output_singularity`, pruned for relevance) × **modifiers** (`scd_data` ×
`error_handling` × `entity_specification_method`). Both factors are listed
in full below; their cross product is still every combination — nothing is
hidden behind a resolver, it's just not typed out 224 times by hand.

### The 12 orthogonal modifier combinations (apply to every shape except two of `raw_data`'s three)

`entity_specification_method` doesn't apply to whole-table `raw_data`
access (see below), so `raw_data`'s `tabular_output_formats` and
`pretty_printed_output_formats` shapes cross only the 4 `scd_data` ×
`error_handling` combinations, not all 12. `raw_data` +
`entity_first_output_formats` is the exception -- see below.

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

## Shapes, by `data_selection`

### `raw_data` — 3 shapes, 20 combinations

Access to a whole component table, unfiltered, as stored. No selection
happens, so `output_singularity` doesn't apply anywhere in this
granularity (there's no single item to collapse to).

For `tabular_output_formats` and `pretty_printed_output_formats`, no
entity is targeted by id/path/alias either -- the whole table comes back
as one flat object -- so `entity_specification_method` doesn't apply:
just the 4 `scd_data` × `error_handling` combinations each.

`entity_first_output_formats` is the exception. Nesting by entity isn't
only a re-expression of the same whole table -- it's naturally also how
a caller looks up *one entity's* full raw record across every component,
by id/path/alias, rather than choosing a component and getting the whole
column. So `entity_specification_method` does apply here, and this shape
crosses all 12 modifiers like every other entity-scoped shape below.

- `raw_data` + `tabular_output_formats` -- 4 combinations (`scd_data` × `error_handling` only)
- `raw_data` + `pretty_printed_output_formats` -- 4 combinations (`scd_data` × `error_handling` only)
- `raw_data` + `entity_first_output_formats` -- 12 combinations (all three modifier axes)

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

## Total: 20 + 60 + 60 + 84 = 224 combinations

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

Each mechanism's per-axis fit is assessed below, but a mechanism also
carries *overall* tradeoffs independent of which axis it's applied to --
these hold across the whole API, not just for one axis's shape:

- **`method_name`** — the axis value is baked into the method's own name
  as a word/prefix/suffix, e.g. `safe_get_...`.
  - **Pros:** fully discoverable via autocomplete -- the whole behavior
    is visible right where you look for the method; the call site is
    self-documenting with nothing to inspect but the name; each name can
    carry its own concrete return-type annotation with no generics or
    overloads; an invalid axis-value combination can't be constructed by
    accident, since each name already *is* one specific valid combination.
  - **Cons:** doesn't scale -- one method per combination is exactly the
    combinatorial-explosion problem this whole exercise exists to avoid;
    shared logic across near-identical method names needs internal
    delegation to avoid duplication; adding a new axis value means adding
    or renaming methods across the whole surface; the choice can't be
    parameterized at runtime (a caller with a variable holding "which
    format" can't use it without a name-keyed dispatch table, which just
    re-introduces the indirection this mechanism was meant to avoid).
- **`argument`** — the axis value is an explicit named parameter on the
  call, e.g. `raise_errors=False`.
  - **Pros:** scales additively -- a new axis value is just a new valid
    value for an existing parameter, no new methods; trivially
    parameterizable at runtime; a familiar, conventional idiom, discoverable
    via signature/parameter hints; several axes combine into one call
    without any method proliferation.
  - **Cons:** the return type isn't visible in the method name, so static
    typing/autocomplete on the *result* needs `@overload`/`Literal` tricks
    to recover what a distinct method name gives for free; nonsensical
    combinations of argument values become constructible unless validated
    at runtime; many kwargs at once crowd the call site and read less
    clearly at a glance than a descriptive name.
- **`chained_accessor`** — a fluent chain of narrowing sub-objects/
  properties selects the axis value before the terminal call, e.g.
  `registry.entities.current.safe.get_by_alias(...)`.
  - **Pros:** reads like a sentence, with autocomplete guiding each step;
    spreads axes across an object graph instead of piling them into one
    signature; each intermediate object can expose only what's valid from
    there, so a wrong combination may be structurally impossible to write
    rather than just discouraged.
  - **Cons:** needs a whole tree of intermediate wrapper classes/objects
    to build and maintain; chain order is an arbitrary-but-consequential
    design decision callers have to learn; awkward to parameterize
    dynamically (picking a chain segment by name needs `getattr`-style
    indirection); verbose for a one-off call compared to a flat method or
    kwarg.
- **`type_based_dispatch`** — the *type* of a single positional argument,
  not its name or an explicit flag, determines the axis value, e.g.
  passing an `EntityAlias("foo")` object vs. a plain `entity_id` int lets
  the same `get()` infer `entity_specification_method` from what was
  handed to it.
  - **Pros:** zero extra syntax at the call site -- the value the caller
    already has to construct *is* the signal, nothing more to remember;
    rules out a class of mismatched-flag bugs (can't pass an id while
    claiming it's a path); scales to new "kinds of value" by adding
    wrapper types, no signature changes.
  - **Cons:** poor discoverability -- nothing in the signature or
    autocomplete lists which wrapper types exist or what they mean, only
    docs do; more upfront ceremony for the caller (constructing
    `EntityAlias(...)` instead of passing a plain value); relies on
    runtime `isinstance` dispatch, which strains under subclassing or
    ambiguous plain types; only applies where an axis is inherently "a
    kind of input value" -- most axes aren't.
- **`result_object`** — the getter always returns one generic/lazy
  result, and the axis value is chosen afterward via a method/property on
  that result rather than an argument to the call itself, e.g.
  `get_component(...).to_pandas()` / `.pretty_print()`.
  - **Pros:** decouples fetching from formatting, avoiding a combinatorial
    explosion for whichever axis it takes on; enables genuine laziness --
    the underlying data need not be materialized in a given shape until
    something asks for it; a well-worn, familiar pattern (`requests.Response`,
    ORM querysets).
  - **Cons:** adds an indirection step to every call, even the common
    case (`.to_x()` always required); the result class has to implement
    and maintain a conversion method per leaf format; an error surfaced at
    the conversion step can be confusing about whether the fetch or the
    conversion is at fault; only shines for axes that are genuinely about
    *format* -- awkward elsewhere per the tables below.
- **`configuration_object`** — several axis choices are bundled into one
  settings/spec object, built once and reused across calls, e.g.
  `get(entity, options=GetterOptions(scd="current", raise_errors=False))`.
  - **Pros:** reusable across many calls -- build once, share across a
    whole pass instead of repeating kwargs; serializable/inspectable as
    one object (loggable, diffable, passable across boundaries); groups
    related settings that tend to travel together.
  - **Cons:** extra ceremony for a one-off call -- constructing an object
    to set a single flag; if plain kwargs are *also* still allowed for the
    same axes, there are now two ways to do the same thing; a shared,
    mutable options object risks action-at-a-distance bugs if reused
    without realizing a caller upstream mutated it.

### A naming refinement: `get_` vs. `view_` (vs. `raw_`)

`method_name`'s word choice for `data_selection`/`output_singularity` has
been a generic pluralization so far (`get_entity` vs. `get_entities`).
[entt](https://github.com/skypjack/entt) and other ECS frameworks already
have a vocabulary for exactly this distinction, and it's worth borrowing
rather than reinventing: `registry.get<T>(entity)` fetches one component
for one already-known entity, while `registry.view<T...>()` returns a
filterable, iterable range over *many* entities matching a component set
-- and `registry.storage<T>()` is the raw, unfiltered underlying pool for
one component type, with no entity filtering at all.

That maps almost exactly onto three things this document already treats
as distinct: `output_singularity=single_version_of_output` (**get**),
`output_singularity=multiple_version_of_output` (**view**), and
`data_selection=raw_data` (**raw**/**storage**) -- the one granularity
where no singularity axis applies at all, i.e. exactly entt's "raw pool,"
not a filtered view of it.

Recommendation: when `method_name` is chosen for `output_singularity`,
use `view_` rather than pluralizing `get_`; keep `get_` for the singular
case; and consider `raw_`/`storage_` for `raw_data`'s own method names
regardless of which mechanism carries its other axes. Concretely, the
verb comes from `output_singularity` (or `raw_data`'s special case), and
`data_selection`'s own `method_name` contribution is only ever the
resource noun (`entities`, `components`, `fields`), never the verb.

This isn't just cosmetic -- it resolves a redundancy flagged earlier.
`data_selection=chained_accessor` combined with
`output_singularity=method_name` used to produce
`registry.entities.safe_get_entities_by_alias(...)`, saying "entities"
twice because the chain already supplied the noun and pluralized `get_`
supplied it again. With the verb reassigned to `view_` and the noun left
to `data_selection` alone, the same combination now reads
`registry.entities.safe_view_by_alias(...)` -- no repetition, and it
reads more like idiomatic ECS code besides. The candidate table below
uses this convention throughout.

### Axis-by-axis fit

**`data_selection`** (raw_data / components_selection /
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
`data_selection` wants `method_name`/`chained_accessor` (it's really
"which operation," not a flag); `output_format` wants `result_object`;
`output_singularity` is often inferable via `type_based_dispatch` rather
than needing its own explicit value; `scd_data`/`error_handling` are
happy with `argument`/`configuration_object`/`chained_accessor` more or
less interchangeably; and `entity_specification_method` wants
`type_based_dispatch` or `method_name`. Deciding the actual hybrid — which
mechanism(s) to commit to per axis, and how the resulting call sites
read once all six choices are combined — is the next step, not resolved
here.

### Candidate final solutions

Crossing each axis's own good-fit mechanisms (from the tables above)
gives every viable per-axis mechanism assignment: 2 (`data_selection`)
&times; 2 (`output_format`) &times; 2 (`output_singularity`) &times; 3
(`scd_data`) &times; 3 (`error_handling`) &times; 3
(`entity_specification_method`) = **216** candidate final solutions.
"Good fit" here means literally rated `good` in the tables above --
`good–moderate` and `moderate–poor` boundary cells
(`scd_data`/`method_name` and `error_handling`/`result_object`) are
excluded, not rounded up.

Every row below renders the *same* fixed underlying request -- get
several entities' current-data tabular record, specified by alias,
without raising errors -- so only the mechanism varies from row to row,
never the scenario. That isolates what each hybrid actually reads like
at a call site, independent of which specific values it's called with.

Legend: `MN`=method_name, `ARG`=argument, `CHA`=chained_accessor,
`TBD`=type_based_dispatch, `RES`=result_object,
`CFG`=configuration_object.

Rendered with the `get_`/`view_` naming convention above: whenever
`output_singularity=method_name`, the verb is `view_` (this fixed
scenario is always the plural case); `data_selection=method_name`
contributes only the resource noun (`entities`). An earlier draft of
this table pluralized `get_` for both roles at once and produced a
redundant `registry.entities.safe_get_entities_by_alias(...)` for
`data_selection=chained_accessor` + `output_singularity=method_name`
(row #109) -- saying "entities" twice. Splitting verb from noun this way
resolves it: that row now reads `registry.entities.safe_view_by_alias(...)`.

| # | data_selection | output_format | output_singularity | scd_data | error_handling | entity_specification_method | example |
|---|---|---|---|---|---|---|---|
| 1 | MN | ARG | MN | ARG | MN | MN | `registry.safe_view_entities_by_alias(["hero", "villain"], format="tabular", scd="current")` |
| 2 | MN | ARG | MN | ARG | MN | ARG | `registry.safe_view_entities(format="tabular", scd="current", entity_alias=["hero", "villain"])` |
| 3 | MN | ARG | MN | ARG | MN | TBD | `registry.safe_view_entities([EntityAlias("hero"), EntityAlias("villain")], format="tabular", scd="current")` |
| 4 | MN | ARG | MN | ARG | ARG | MN | `registry.view_entities_by_alias(["hero", "villain"], format="tabular", scd="current", raise_errors=False)` |
| 5 | MN | ARG | MN | ARG | ARG | ARG | `registry.view_entities(format="tabular", scd="current", raise_errors=False, entity_alias=["hero", "villain"])` |
| 6 | MN | ARG | MN | ARG | ARG | TBD | `registry.view_entities([EntityAlias("hero"), EntityAlias("villain")], format="tabular", scd="current", raise_errors=False)` |
| 7 | MN | ARG | MN | ARG | CFG | MN | `registry.view_entities_by_alias(["hero", "villain"], format="tabular", scd="current", options=GetterOptions(raise_errors=False))` |
| 8 | MN | ARG | MN | ARG | CFG | ARG | `registry.view_entities(format="tabular", scd="current", entity_alias=["hero", "villain"], options=GetterOptions(raise_errors=False))` |
| 9 | MN | ARG | MN | ARG | CFG | TBD | `registry.view_entities([EntityAlias("hero"), EntityAlias("villain")], format="tabular", scd="current", options=GetterOptions(raise_errors=False))` |
| 10 | MN | ARG | MN | CHA | MN | MN | `registry.current.safe_view_entities_by_alias(["hero", "villain"], format="tabular")` |
| 11 | MN | ARG | MN | CHA | MN | ARG | `registry.current.safe_view_entities(format="tabular", entity_alias=["hero", "villain"])` |
| 12 | MN | ARG | MN | CHA | MN | TBD | `registry.current.safe_view_entities([EntityAlias("hero"), EntityAlias("villain")], format="tabular")` |
| 13 | MN | ARG | MN | CHA | ARG | MN | `registry.current.view_entities_by_alias(["hero", "villain"], format="tabular", raise_errors=False)` |
| 14 | MN | ARG | MN | CHA | ARG | ARG | `registry.current.view_entities(format="tabular", raise_errors=False, entity_alias=["hero", "villain"])` |
| 15 | MN | ARG | MN | CHA | ARG | TBD | `registry.current.view_entities([EntityAlias("hero"), EntityAlias("villain")], format="tabular", raise_errors=False)` |
| 16 | MN | ARG | MN | CHA | CFG | MN | `registry.current.view_entities_by_alias(["hero", "villain"], format="tabular", options=GetterOptions(raise_errors=False))` |
| 17 | MN | ARG | MN | CHA | CFG | ARG | `registry.current.view_entities(format="tabular", entity_alias=["hero", "villain"], options=GetterOptions(raise_errors=False))` |
| 18 | MN | ARG | MN | CHA | CFG | TBD | `registry.current.view_entities([EntityAlias("hero"), EntityAlias("villain")], format="tabular", options=GetterOptions(raise_errors=False))` |
| 19 | MN | ARG | MN | CFG | MN | MN | `registry.safe_view_entities_by_alias(["hero", "villain"], format="tabular", options=GetterOptions(scd="current"))` |
| 20 | MN | ARG | MN | CFG | MN | ARG | `registry.safe_view_entities(format="tabular", entity_alias=["hero", "villain"], options=GetterOptions(scd="current"))` |
| 21 | MN | ARG | MN | CFG | MN | TBD | `registry.safe_view_entities([EntityAlias("hero"), EntityAlias("villain")], format="tabular", options=GetterOptions(scd="current"))` |
| 22 | MN | ARG | MN | CFG | ARG | MN | `registry.view_entities_by_alias(["hero", "villain"], format="tabular", raise_errors=False, options=GetterOptions(scd="current"))` |
| 23 | MN | ARG | MN | CFG | ARG | ARG | `registry.view_entities(format="tabular", raise_errors=False, entity_alias=["hero", "villain"], options=GetterOptions(scd="current"))` |
| 24 | MN | ARG | MN | CFG | ARG | TBD | `registry.view_entities([EntityAlias("hero"), EntityAlias("villain")], format="tabular", raise_errors=False, options=GetterOptions(scd="current"))` |
| 25 | MN | ARG | MN | CFG | CFG | MN | `registry.view_entities_by_alias(["hero", "villain"], format="tabular", options=GetterOptions(scd="current", raise_errors=False))` |
| 26 | MN | ARG | MN | CFG | CFG | ARG | `registry.view_entities(format="tabular", entity_alias=["hero", "villain"], options=GetterOptions(scd="current", raise_errors=False))` |
| 27 | MN | ARG | MN | CFG | CFG | TBD | `registry.view_entities([EntityAlias("hero"), EntityAlias("villain")], format="tabular", options=GetterOptions(scd="current", raise_errors=False))` |
| 28 | MN | ARG | TBD | ARG | MN | MN | `registry.safe_get_entities_by_alias(["hero", "villain"], format="tabular", scd="current")` |
| 29 | MN | ARG | TBD | ARG | MN | ARG | `registry.safe_get_entities(format="tabular", scd="current", entity_alias=["hero", "villain"])` |
| 30 | MN | ARG | TBD | ARG | MN | TBD | `registry.safe_get_entities([EntityAlias("hero"), EntityAlias("villain")], format="tabular", scd="current")` |
| 31 | MN | ARG | TBD | ARG | ARG | MN | `registry.get_entities_by_alias(["hero", "villain"], format="tabular", scd="current", raise_errors=False)` |
| 32 | MN | ARG | TBD | ARG | ARG | ARG | `registry.get_entities(format="tabular", scd="current", raise_errors=False, entity_alias=["hero", "villain"])` |
| 33 | MN | ARG | TBD | ARG | ARG | TBD | `registry.get_entities([EntityAlias("hero"), EntityAlias("villain")], format="tabular", scd="current", raise_errors=False)` |
| 34 | MN | ARG | TBD | ARG | CFG | MN | `registry.get_entities_by_alias(["hero", "villain"], format="tabular", scd="current", options=GetterOptions(raise_errors=False))` |
| 35 | MN | ARG | TBD | ARG | CFG | ARG | `registry.get_entities(format="tabular", scd="current", entity_alias=["hero", "villain"], options=GetterOptions(raise_errors=False))` |
| 36 | MN | ARG | TBD | ARG | CFG | TBD | `registry.get_entities([EntityAlias("hero"), EntityAlias("villain")], format="tabular", scd="current", options=GetterOptions(raise_errors=False))` |
| 37 | MN | ARG | TBD | CHA | MN | MN | `registry.current.safe_get_entities_by_alias(["hero", "villain"], format="tabular")` |
| 38 | MN | ARG | TBD | CHA | MN | ARG | `registry.current.safe_get_entities(format="tabular", entity_alias=["hero", "villain"])` |
| 39 | MN | ARG | TBD | CHA | MN | TBD | `registry.current.safe_get_entities([EntityAlias("hero"), EntityAlias("villain")], format="tabular")` |
| 40 | MN | ARG | TBD | CHA | ARG | MN | `registry.current.get_entities_by_alias(["hero", "villain"], format="tabular", raise_errors=False)` |
| 41 | MN | ARG | TBD | CHA | ARG | ARG | `registry.current.get_entities(format="tabular", raise_errors=False, entity_alias=["hero", "villain"])` |
| 42 | MN | ARG | TBD | CHA | ARG | TBD | `registry.current.get_entities([EntityAlias("hero"), EntityAlias("villain")], format="tabular", raise_errors=False)` |
| 43 | MN | ARG | TBD | CHA | CFG | MN | `registry.current.get_entities_by_alias(["hero", "villain"], format="tabular", options=GetterOptions(raise_errors=False))` |
| 44 | MN | ARG | TBD | CHA | CFG | ARG | `registry.current.get_entities(format="tabular", entity_alias=["hero", "villain"], options=GetterOptions(raise_errors=False))` |
| 45 | MN | ARG | TBD | CHA | CFG | TBD | `registry.current.get_entities([EntityAlias("hero"), EntityAlias("villain")], format="tabular", options=GetterOptions(raise_errors=False))` |
| 46 | MN | ARG | TBD | CFG | MN | MN | `registry.safe_get_entities_by_alias(["hero", "villain"], format="tabular", options=GetterOptions(scd="current"))` |
| 47 | MN | ARG | TBD | CFG | MN | ARG | `registry.safe_get_entities(format="tabular", entity_alias=["hero", "villain"], options=GetterOptions(scd="current"))` |
| 48 | MN | ARG | TBD | CFG | MN | TBD | `registry.safe_get_entities([EntityAlias("hero"), EntityAlias("villain")], format="tabular", options=GetterOptions(scd="current"))` |
| 49 | MN | ARG | TBD | CFG | ARG | MN | `registry.get_entities_by_alias(["hero", "villain"], format="tabular", raise_errors=False, options=GetterOptions(scd="current"))` |
| 50 | MN | ARG | TBD | CFG | ARG | ARG | `registry.get_entities(format="tabular", raise_errors=False, entity_alias=["hero", "villain"], options=GetterOptions(scd="current"))` |
| 51 | MN | ARG | TBD | CFG | ARG | TBD | `registry.get_entities([EntityAlias("hero"), EntityAlias("villain")], format="tabular", raise_errors=False, options=GetterOptions(scd="current"))` |
| 52 | MN | ARG | TBD | CFG | CFG | MN | `registry.get_entities_by_alias(["hero", "villain"], format="tabular", options=GetterOptions(scd="current", raise_errors=False))` |
| 53 | MN | ARG | TBD | CFG | CFG | ARG | `registry.get_entities(format="tabular", entity_alias=["hero", "villain"], options=GetterOptions(scd="current", raise_errors=False))` |
| 54 | MN | ARG | TBD | CFG | CFG | TBD | `registry.get_entities([EntityAlias("hero"), EntityAlias("villain")], format="tabular", options=GetterOptions(scd="current", raise_errors=False))` |
| 55 | MN | RES | MN | ARG | MN | MN | `registry.safe_view_entities_by_alias(["hero", "villain"], scd="current").to_table()` |
| 56 | MN | RES | MN | ARG | MN | ARG | `registry.safe_view_entities(scd="current", entity_alias=["hero", "villain"]).to_table()` |
| 57 | MN | RES | MN | ARG | MN | TBD | `registry.safe_view_entities([EntityAlias("hero"), EntityAlias("villain")], scd="current").to_table()` |
| 58 | MN | RES | MN | ARG | ARG | MN | `registry.view_entities_by_alias(["hero", "villain"], scd="current", raise_errors=False).to_table()` |
| 59 | MN | RES | MN | ARG | ARG | ARG | `registry.view_entities(scd="current", raise_errors=False, entity_alias=["hero", "villain"]).to_table()` |
| 60 | MN | RES | MN | ARG | ARG | TBD | `registry.view_entities([EntityAlias("hero"), EntityAlias("villain")], scd="current", raise_errors=False).to_table()` |
| 61 | MN | RES | MN | ARG | CFG | MN | `registry.view_entities_by_alias(["hero", "villain"], scd="current", options=GetterOptions(raise_errors=False)).to_table()` |
| 62 | MN | RES | MN | ARG | CFG | ARG | `registry.view_entities(scd="current", entity_alias=["hero", "villain"], options=GetterOptions(raise_errors=False)).to_table()` |
| 63 | MN | RES | MN | ARG | CFG | TBD | `registry.view_entities([EntityAlias("hero"), EntityAlias("villain")], scd="current", options=GetterOptions(raise_errors=False)).to_table()` |
| 64 | MN | RES | MN | CHA | MN | MN | `registry.current.safe_view_entities_by_alias(["hero", "villain"]).to_table()` |
| 65 | MN | RES | MN | CHA | MN | ARG | `registry.current.safe_view_entities(entity_alias=["hero", "villain"]).to_table()` |
| 66 | MN | RES | MN | CHA | MN | TBD | `registry.current.safe_view_entities([EntityAlias("hero"), EntityAlias("villain")]).to_table()` |
| 67 | MN | RES | MN | CHA | ARG | MN | `registry.current.view_entities_by_alias(["hero", "villain"], raise_errors=False).to_table()` |
| 68 | MN | RES | MN | CHA | ARG | ARG | `registry.current.view_entities(raise_errors=False, entity_alias=["hero", "villain"]).to_table()` |
| 69 | MN | RES | MN | CHA | ARG | TBD | `registry.current.view_entities([EntityAlias("hero"), EntityAlias("villain")], raise_errors=False).to_table()` |
| 70 | MN | RES | MN | CHA | CFG | MN | `registry.current.view_entities_by_alias(["hero", "villain"], options=GetterOptions(raise_errors=False)).to_table()` |
| 71 | MN | RES | MN | CHA | CFG | ARG | `registry.current.view_entities(entity_alias=["hero", "villain"], options=GetterOptions(raise_errors=False)).to_table()` |
| 72 | MN | RES | MN | CHA | CFG | TBD | `registry.current.view_entities([EntityAlias("hero"), EntityAlias("villain")], options=GetterOptions(raise_errors=False)).to_table()` |
| 73 | MN | RES | MN | CFG | MN | MN | `registry.safe_view_entities_by_alias(["hero", "villain"], options=GetterOptions(scd="current")).to_table()` |
| 74 | MN | RES | MN | CFG | MN | ARG | `registry.safe_view_entities(entity_alias=["hero", "villain"], options=GetterOptions(scd="current")).to_table()` |
| 75 | MN | RES | MN | CFG | MN | TBD | `registry.safe_view_entities([EntityAlias("hero"), EntityAlias("villain")], options=GetterOptions(scd="current")).to_table()` |
| 76 | MN | RES | MN | CFG | ARG | MN | `registry.view_entities_by_alias(["hero", "villain"], raise_errors=False, options=GetterOptions(scd="current")).to_table()` |
| 77 | MN | RES | MN | CFG | ARG | ARG | `registry.view_entities(raise_errors=False, entity_alias=["hero", "villain"], options=GetterOptions(scd="current")).to_table()` |
| 78 | MN | RES | MN | CFG | ARG | TBD | `registry.view_entities([EntityAlias("hero"), EntityAlias("villain")], raise_errors=False, options=GetterOptions(scd="current")).to_table()` |
| 79 | MN | RES | MN | CFG | CFG | MN | `registry.view_entities_by_alias(["hero", "villain"], options=GetterOptions(scd="current", raise_errors=False)).to_table()` |
| 80 | MN | RES | MN | CFG | CFG | ARG | `registry.view_entities(entity_alias=["hero", "villain"], options=GetterOptions(scd="current", raise_errors=False)).to_table()` |
| 81 | MN | RES | MN | CFG | CFG | TBD | `registry.view_entities([EntityAlias("hero"), EntityAlias("villain")], options=GetterOptions(scd="current", raise_errors=False)).to_table()` |
| 82 | MN | RES | TBD | ARG | MN | MN | `registry.safe_get_entities_by_alias(["hero", "villain"], scd="current").to_table()` |
| 83 | MN | RES | TBD | ARG | MN | ARG | `registry.safe_get_entities(scd="current", entity_alias=["hero", "villain"]).to_table()` |
| 84 | MN | RES | TBD | ARG | MN | TBD | `registry.safe_get_entities([EntityAlias("hero"), EntityAlias("villain")], scd="current").to_table()` |
| 85 | MN | RES | TBD | ARG | ARG | MN | `registry.get_entities_by_alias(["hero", "villain"], scd="current", raise_errors=False).to_table()` |
| 86 | MN | RES | TBD | ARG | ARG | ARG | `registry.get_entities(scd="current", raise_errors=False, entity_alias=["hero", "villain"]).to_table()` |
| 87 | MN | RES | TBD | ARG | ARG | TBD | `registry.get_entities([EntityAlias("hero"), EntityAlias("villain")], scd="current", raise_errors=False).to_table()` |
| 88 | MN | RES | TBD | ARG | CFG | MN | `registry.get_entities_by_alias(["hero", "villain"], scd="current", options=GetterOptions(raise_errors=False)).to_table()` |
| 89 | MN | RES | TBD | ARG | CFG | ARG | `registry.get_entities(scd="current", entity_alias=["hero", "villain"], options=GetterOptions(raise_errors=False)).to_table()` |
| 90 | MN | RES | TBD | ARG | CFG | TBD | `registry.get_entities([EntityAlias("hero"), EntityAlias("villain")], scd="current", options=GetterOptions(raise_errors=False)).to_table()` |
| 91 | MN | RES | TBD | CHA | MN | MN | `registry.current.safe_get_entities_by_alias(["hero", "villain"]).to_table()` |
| 92 | MN | RES | TBD | CHA | MN | ARG | `registry.current.safe_get_entities(entity_alias=["hero", "villain"]).to_table()` |
| 93 | MN | RES | TBD | CHA | MN | TBD | `registry.current.safe_get_entities([EntityAlias("hero"), EntityAlias("villain")]).to_table()` |
| 94 | MN | RES | TBD | CHA | ARG | MN | `registry.current.get_entities_by_alias(["hero", "villain"], raise_errors=False).to_table()` |
| 95 | MN | RES | TBD | CHA | ARG | ARG | `registry.current.get_entities(raise_errors=False, entity_alias=["hero", "villain"]).to_table()` |
| 96 | MN | RES | TBD | CHA | ARG | TBD | `registry.current.get_entities([EntityAlias("hero"), EntityAlias("villain")], raise_errors=False).to_table()` |
| 97 | MN | RES | TBD | CHA | CFG | MN | `registry.current.get_entities_by_alias(["hero", "villain"], options=GetterOptions(raise_errors=False)).to_table()` |
| 98 | MN | RES | TBD | CHA | CFG | ARG | `registry.current.get_entities(entity_alias=["hero", "villain"], options=GetterOptions(raise_errors=False)).to_table()` |
| 99 | MN | RES | TBD | CHA | CFG | TBD | `registry.current.get_entities([EntityAlias("hero"), EntityAlias("villain")], options=GetterOptions(raise_errors=False)).to_table()` |
| 100 | MN | RES | TBD | CFG | MN | MN | `registry.safe_get_entities_by_alias(["hero", "villain"], options=GetterOptions(scd="current")).to_table()` |
| 101 | MN | RES | TBD | CFG | MN | ARG | `registry.safe_get_entities(entity_alias=["hero", "villain"], options=GetterOptions(scd="current")).to_table()` |
| 102 | MN | RES | TBD | CFG | MN | TBD | `registry.safe_get_entities([EntityAlias("hero"), EntityAlias("villain")], options=GetterOptions(scd="current")).to_table()` |
| 103 | MN | RES | TBD | CFG | ARG | MN | `registry.get_entities_by_alias(["hero", "villain"], raise_errors=False, options=GetterOptions(scd="current")).to_table()` |
| 104 | MN | RES | TBD | CFG | ARG | ARG | `registry.get_entities(raise_errors=False, entity_alias=["hero", "villain"], options=GetterOptions(scd="current")).to_table()` |
| 105 | MN | RES | TBD | CFG | ARG | TBD | `registry.get_entities([EntityAlias("hero"), EntityAlias("villain")], raise_errors=False, options=GetterOptions(scd="current")).to_table()` |
| 106 | MN | RES | TBD | CFG | CFG | MN | `registry.get_entities_by_alias(["hero", "villain"], options=GetterOptions(scd="current", raise_errors=False)).to_table()` |
| 107 | MN | RES | TBD | CFG | CFG | ARG | `registry.get_entities(entity_alias=["hero", "villain"], options=GetterOptions(scd="current", raise_errors=False)).to_table()` |
| 108 | MN | RES | TBD | CFG | CFG | TBD | `registry.get_entities([EntityAlias("hero"), EntityAlias("villain")], options=GetterOptions(scd="current", raise_errors=False)).to_table()` |
| 109 | CHA | ARG | MN | ARG | MN | MN | `registry.entities.safe_view_by_alias(["hero", "villain"], format="tabular", scd="current")` |
| 110 | CHA | ARG | MN | ARG | MN | ARG | `registry.entities.safe_view(format="tabular", scd="current", entity_alias=["hero", "villain"])` |
| 111 | CHA | ARG | MN | ARG | MN | TBD | `registry.entities.safe_view([EntityAlias("hero"), EntityAlias("villain")], format="tabular", scd="current")` |
| 112 | CHA | ARG | MN | ARG | ARG | MN | `registry.entities.view_by_alias(["hero", "villain"], format="tabular", scd="current", raise_errors=False)` |
| 113 | CHA | ARG | MN | ARG | ARG | ARG | `registry.entities.view(format="tabular", scd="current", raise_errors=False, entity_alias=["hero", "villain"])` |
| 114 | CHA | ARG | MN | ARG | ARG | TBD | `registry.entities.view([EntityAlias("hero"), EntityAlias("villain")], format="tabular", scd="current", raise_errors=False)` |
| 115 | CHA | ARG | MN | ARG | CFG | MN | `registry.entities.view_by_alias(["hero", "villain"], format="tabular", scd="current", options=GetterOptions(raise_errors=False))` |
| 116 | CHA | ARG | MN | ARG | CFG | ARG | `registry.entities.view(format="tabular", scd="current", entity_alias=["hero", "villain"], options=GetterOptions(raise_errors=False))` |
| 117 | CHA | ARG | MN | ARG | CFG | TBD | `registry.entities.view([EntityAlias("hero"), EntityAlias("villain")], format="tabular", scd="current", options=GetterOptions(raise_errors=False))` |
| 118 | CHA | ARG | MN | CHA | MN | MN | `registry.current.entities.safe_view_by_alias(["hero", "villain"], format="tabular")` |
| 119 | CHA | ARG | MN | CHA | MN | ARG | `registry.current.entities.safe_view(format="tabular", entity_alias=["hero", "villain"])` |
| 120 | CHA | ARG | MN | CHA | MN | TBD | `registry.current.entities.safe_view([EntityAlias("hero"), EntityAlias("villain")], format="tabular")` |
| 121 | CHA | ARG | MN | CHA | ARG | MN | `registry.current.entities.view_by_alias(["hero", "villain"], format="tabular", raise_errors=False)` |
| 122 | CHA | ARG | MN | CHA | ARG | ARG | `registry.current.entities.view(format="tabular", raise_errors=False, entity_alias=["hero", "villain"])` |
| 123 | CHA | ARG | MN | CHA | ARG | TBD | `registry.current.entities.view([EntityAlias("hero"), EntityAlias("villain")], format="tabular", raise_errors=False)` |
| 124 | CHA | ARG | MN | CHA | CFG | MN | `registry.current.entities.view_by_alias(["hero", "villain"], format="tabular", options=GetterOptions(raise_errors=False))` |
| 125 | CHA | ARG | MN | CHA | CFG | ARG | `registry.current.entities.view(format="tabular", entity_alias=["hero", "villain"], options=GetterOptions(raise_errors=False))` |
| 126 | CHA | ARG | MN | CHA | CFG | TBD | `registry.current.entities.view([EntityAlias("hero"), EntityAlias("villain")], format="tabular", options=GetterOptions(raise_errors=False))` |
| 127 | CHA | ARG | MN | CFG | MN | MN | `registry.entities.safe_view_by_alias(["hero", "villain"], format="tabular", options=GetterOptions(scd="current"))` |
| 128 | CHA | ARG | MN | CFG | MN | ARG | `registry.entities.safe_view(format="tabular", entity_alias=["hero", "villain"], options=GetterOptions(scd="current"))` |
| 129 | CHA | ARG | MN | CFG | MN | TBD | `registry.entities.safe_view([EntityAlias("hero"), EntityAlias("villain")], format="tabular", options=GetterOptions(scd="current"))` |
| 130 | CHA | ARG | MN | CFG | ARG | MN | `registry.entities.view_by_alias(["hero", "villain"], format="tabular", raise_errors=False, options=GetterOptions(scd="current"))` |
| 131 | CHA | ARG | MN | CFG | ARG | ARG | `registry.entities.view(format="tabular", raise_errors=False, entity_alias=["hero", "villain"], options=GetterOptions(scd="current"))` |
| 132 | CHA | ARG | MN | CFG | ARG | TBD | `registry.entities.view([EntityAlias("hero"), EntityAlias("villain")], format="tabular", raise_errors=False, options=GetterOptions(scd="current"))` |
| 133 | CHA | ARG | MN | CFG | CFG | MN | `registry.entities.view_by_alias(["hero", "villain"], format="tabular", options=GetterOptions(scd="current", raise_errors=False))` |
| 134 | CHA | ARG | MN | CFG | CFG | ARG | `registry.entities.view(format="tabular", entity_alias=["hero", "villain"], options=GetterOptions(scd="current", raise_errors=False))` |
| 135 | CHA | ARG | MN | CFG | CFG | TBD | `registry.entities.view([EntityAlias("hero"), EntityAlias("villain")], format="tabular", options=GetterOptions(scd="current", raise_errors=False))` |
| 136 | CHA | ARG | TBD | ARG | MN | MN | `registry.entities.safe_get_by_alias(["hero", "villain"], format="tabular", scd="current")` |
| 137 | CHA | ARG | TBD | ARG | MN | ARG | `registry.entities.safe_get(format="tabular", scd="current", entity_alias=["hero", "villain"])` |
| 138 | CHA | ARG | TBD | ARG | MN | TBD | `registry.entities.safe_get([EntityAlias("hero"), EntityAlias("villain")], format="tabular", scd="current")` |
| 139 | CHA | ARG | TBD | ARG | ARG | MN | `registry.entities.get_by_alias(["hero", "villain"], format="tabular", scd="current", raise_errors=False)` |
| 140 | CHA | ARG | TBD | ARG | ARG | ARG | `registry.entities.get(format="tabular", scd="current", raise_errors=False, entity_alias=["hero", "villain"])` |
| 141 | CHA | ARG | TBD | ARG | ARG | TBD | `registry.entities.get([EntityAlias("hero"), EntityAlias("villain")], format="tabular", scd="current", raise_errors=False)` |
| 142 | CHA | ARG | TBD | ARG | CFG | MN | `registry.entities.get_by_alias(["hero", "villain"], format="tabular", scd="current", options=GetterOptions(raise_errors=False))` |
| 143 | CHA | ARG | TBD | ARG | CFG | ARG | `registry.entities.get(format="tabular", scd="current", entity_alias=["hero", "villain"], options=GetterOptions(raise_errors=False))` |
| 144 | CHA | ARG | TBD | ARG | CFG | TBD | `registry.entities.get([EntityAlias("hero"), EntityAlias("villain")], format="tabular", scd="current", options=GetterOptions(raise_errors=False))` |
| 145 | CHA | ARG | TBD | CHA | MN | MN | `registry.current.entities.safe_get_by_alias(["hero", "villain"], format="tabular")` |
| 146 | CHA | ARG | TBD | CHA | MN | ARG | `registry.current.entities.safe_get(format="tabular", entity_alias=["hero", "villain"])` |
| 147 | CHA | ARG | TBD | CHA | MN | TBD | `registry.current.entities.safe_get([EntityAlias("hero"), EntityAlias("villain")], format="tabular")` |
| 148 | CHA | ARG | TBD | CHA | ARG | MN | `registry.current.entities.get_by_alias(["hero", "villain"], format="tabular", raise_errors=False)` |
| 149 | CHA | ARG | TBD | CHA | ARG | ARG | `registry.current.entities.get(format="tabular", raise_errors=False, entity_alias=["hero", "villain"])` |
| 150 | CHA | ARG | TBD | CHA | ARG | TBD | `registry.current.entities.get([EntityAlias("hero"), EntityAlias("villain")], format="tabular", raise_errors=False)` |
| 151 | CHA | ARG | TBD | CHA | CFG | MN | `registry.current.entities.get_by_alias(["hero", "villain"], format="tabular", options=GetterOptions(raise_errors=False))` |
| 152 | CHA | ARG | TBD | CHA | CFG | ARG | `registry.current.entities.get(format="tabular", entity_alias=["hero", "villain"], options=GetterOptions(raise_errors=False))` |
| 153 | CHA | ARG | TBD | CHA | CFG | TBD | `registry.current.entities.get([EntityAlias("hero"), EntityAlias("villain")], format="tabular", options=GetterOptions(raise_errors=False))` |
| 154 | CHA | ARG | TBD | CFG | MN | MN | `registry.entities.safe_get_by_alias(["hero", "villain"], format="tabular", options=GetterOptions(scd="current"))` |
| 155 | CHA | ARG | TBD | CFG | MN | ARG | `registry.entities.safe_get(format="tabular", entity_alias=["hero", "villain"], options=GetterOptions(scd="current"))` |
| 156 | CHA | ARG | TBD | CFG | MN | TBD | `registry.entities.safe_get([EntityAlias("hero"), EntityAlias("villain")], format="tabular", options=GetterOptions(scd="current"))` |
| 157 | CHA | ARG | TBD | CFG | ARG | MN | `registry.entities.get_by_alias(["hero", "villain"], format="tabular", raise_errors=False, options=GetterOptions(scd="current"))` |
| 158 | CHA | ARG | TBD | CFG | ARG | ARG | `registry.entities.get(format="tabular", raise_errors=False, entity_alias=["hero", "villain"], options=GetterOptions(scd="current"))` |
| 159 | CHA | ARG | TBD | CFG | ARG | TBD | `registry.entities.get([EntityAlias("hero"), EntityAlias("villain")], format="tabular", raise_errors=False, options=GetterOptions(scd="current"))` |
| 160 | CHA | ARG | TBD | CFG | CFG | MN | `registry.entities.get_by_alias(["hero", "villain"], format="tabular", options=GetterOptions(scd="current", raise_errors=False))` |
| 161 | CHA | ARG | TBD | CFG | CFG | ARG | `registry.entities.get(format="tabular", entity_alias=["hero", "villain"], options=GetterOptions(scd="current", raise_errors=False))` |
| 162 | CHA | ARG | TBD | CFG | CFG | TBD | `registry.entities.get([EntityAlias("hero"), EntityAlias("villain")], format="tabular", options=GetterOptions(scd="current", raise_errors=False))` |
| 163 | CHA | RES | MN | ARG | MN | MN | `registry.entities.safe_view_by_alias(["hero", "villain"], scd="current").to_table()` |
| 164 | CHA | RES | MN | ARG | MN | ARG | `registry.entities.safe_view(scd="current", entity_alias=["hero", "villain"]).to_table()` |
| 165 | CHA | RES | MN | ARG | MN | TBD | `registry.entities.safe_view([EntityAlias("hero"), EntityAlias("villain")], scd="current").to_table()` |
| 166 | CHA | RES | MN | ARG | ARG | MN | `registry.entities.view_by_alias(["hero", "villain"], scd="current", raise_errors=False).to_table()` |
| 167 | CHA | RES | MN | ARG | ARG | ARG | `registry.entities.view(scd="current", raise_errors=False, entity_alias=["hero", "villain"]).to_table()` |
| 168 | CHA | RES | MN | ARG | ARG | TBD | `registry.entities.view([EntityAlias("hero"), EntityAlias("villain")], scd="current", raise_errors=False).to_table()` |
| 169 | CHA | RES | MN | ARG | CFG | MN | `registry.entities.view_by_alias(["hero", "villain"], scd="current", options=GetterOptions(raise_errors=False)).to_table()` |
| 170 | CHA | RES | MN | ARG | CFG | ARG | `registry.entities.view(scd="current", entity_alias=["hero", "villain"], options=GetterOptions(raise_errors=False)).to_table()` |
| 171 | CHA | RES | MN | ARG | CFG | TBD | `registry.entities.view([EntityAlias("hero"), EntityAlias("villain")], scd="current", options=GetterOptions(raise_errors=False)).to_table()` |
| 172 | CHA | RES | MN | CHA | MN | MN | `registry.current.entities.safe_view_by_alias(["hero", "villain"]).to_table()` |
| 173 | CHA | RES | MN | CHA | MN | ARG | `registry.current.entities.safe_view(entity_alias=["hero", "villain"]).to_table()` |
| 174 | CHA | RES | MN | CHA | MN | TBD | `registry.current.entities.safe_view([EntityAlias("hero"), EntityAlias("villain")]).to_table()` |
| 175 | CHA | RES | MN | CHA | ARG | MN | `registry.current.entities.view_by_alias(["hero", "villain"], raise_errors=False).to_table()` |
| 176 | CHA | RES | MN | CHA | ARG | ARG | `registry.current.entities.view(raise_errors=False, entity_alias=["hero", "villain"]).to_table()` |
| 177 | CHA | RES | MN | CHA | ARG | TBD | `registry.current.entities.view([EntityAlias("hero"), EntityAlias("villain")], raise_errors=False).to_table()` |
| 178 | CHA | RES | MN | CHA | CFG | MN | `registry.current.entities.view_by_alias(["hero", "villain"], options=GetterOptions(raise_errors=False)).to_table()` |
| 179 | CHA | RES | MN | CHA | CFG | ARG | `registry.current.entities.view(entity_alias=["hero", "villain"], options=GetterOptions(raise_errors=False)).to_table()` |
| 180 | CHA | RES | MN | CHA | CFG | TBD | `registry.current.entities.view([EntityAlias("hero"), EntityAlias("villain")], options=GetterOptions(raise_errors=False)).to_table()` |
| 181 | CHA | RES | MN | CFG | MN | MN | `registry.entities.safe_view_by_alias(["hero", "villain"], options=GetterOptions(scd="current")).to_table()` |
| 182 | CHA | RES | MN | CFG | MN | ARG | `registry.entities.safe_view(entity_alias=["hero", "villain"], options=GetterOptions(scd="current")).to_table()` |
| 183 | CHA | RES | MN | CFG | MN | TBD | `registry.entities.safe_view([EntityAlias("hero"), EntityAlias("villain")], options=GetterOptions(scd="current")).to_table()` |
| 184 | CHA | RES | MN | CFG | ARG | MN | `registry.entities.view_by_alias(["hero", "villain"], raise_errors=False, options=GetterOptions(scd="current")).to_table()` |
| 185 | CHA | RES | MN | CFG | ARG | ARG | `registry.entities.view(raise_errors=False, entity_alias=["hero", "villain"], options=GetterOptions(scd="current")).to_table()` |
| 186 | CHA | RES | MN | CFG | ARG | TBD | `registry.entities.view([EntityAlias("hero"), EntityAlias("villain")], raise_errors=False, options=GetterOptions(scd="current")).to_table()` |
| 187 | CHA | RES | MN | CFG | CFG | MN | `registry.entities.view_by_alias(["hero", "villain"], options=GetterOptions(scd="current", raise_errors=False)).to_table()` |
| 188 | CHA | RES | MN | CFG | CFG | ARG | `registry.entities.view(entity_alias=["hero", "villain"], options=GetterOptions(scd="current", raise_errors=False)).to_table()` |
| 189 | CHA | RES | MN | CFG | CFG | TBD | `registry.entities.view([EntityAlias("hero"), EntityAlias("villain")], options=GetterOptions(scd="current", raise_errors=False)).to_table()` |
| 190 | CHA | RES | TBD | ARG | MN | MN | `registry.entities.safe_get_by_alias(["hero", "villain"], scd="current").to_table()` |
| 191 | CHA | RES | TBD | ARG | MN | ARG | `registry.entities.safe_get(scd="current", entity_alias=["hero", "villain"]).to_table()` |
| 192 | CHA | RES | TBD | ARG | MN | TBD | `registry.entities.safe_get([EntityAlias("hero"), EntityAlias("villain")], scd="current").to_table()` |
| 193 | CHA | RES | TBD | ARG | ARG | MN | `registry.entities.get_by_alias(["hero", "villain"], scd="current", raise_errors=False).to_table()` |
| 194 | CHA | RES | TBD | ARG | ARG | ARG | `registry.entities.get(scd="current", raise_errors=False, entity_alias=["hero", "villain"]).to_table()` |
| 195 | CHA | RES | TBD | ARG | ARG | TBD | `registry.entities.get([EntityAlias("hero"), EntityAlias("villain")], scd="current", raise_errors=False).to_table()` |
| 196 | CHA | RES | TBD | ARG | CFG | MN | `registry.entities.get_by_alias(["hero", "villain"], scd="current", options=GetterOptions(raise_errors=False)).to_table()` |
| 197 | CHA | RES | TBD | ARG | CFG | ARG | `registry.entities.get(scd="current", entity_alias=["hero", "villain"], options=GetterOptions(raise_errors=False)).to_table()` |
| 198 | CHA | RES | TBD | ARG | CFG | TBD | `registry.entities.get([EntityAlias("hero"), EntityAlias("villain")], scd="current", options=GetterOptions(raise_errors=False)).to_table()` |
| 199 | CHA | RES | TBD | CHA | MN | MN | `registry.current.entities.safe_get_by_alias(["hero", "villain"]).to_table()` |
| 200 | CHA | RES | TBD | CHA | MN | ARG | `registry.current.entities.safe_get(entity_alias=["hero", "villain"]).to_table()` |
| 201 | CHA | RES | TBD | CHA | MN | TBD | `registry.current.entities.safe_get([EntityAlias("hero"), EntityAlias("villain")]).to_table()` |
| 202 | CHA | RES | TBD | CHA | ARG | MN | `registry.current.entities.get_by_alias(["hero", "villain"], raise_errors=False).to_table()` |
| 203 | CHA | RES | TBD | CHA | ARG | ARG | `registry.current.entities.get(raise_errors=False, entity_alias=["hero", "villain"]).to_table()` |
| 204 | CHA | RES | TBD | CHA | ARG | TBD | `registry.current.entities.get([EntityAlias("hero"), EntityAlias("villain")], raise_errors=False).to_table()` |
| 205 | CHA | RES | TBD | CHA | CFG | MN | `registry.current.entities.get_by_alias(["hero", "villain"], options=GetterOptions(raise_errors=False)).to_table()` |
| 206 | CHA | RES | TBD | CHA | CFG | ARG | `registry.current.entities.get(entity_alias=["hero", "villain"], options=GetterOptions(raise_errors=False)).to_table()` |
| 207 | CHA | RES | TBD | CHA | CFG | TBD | `registry.current.entities.get([EntityAlias("hero"), EntityAlias("villain")], options=GetterOptions(raise_errors=False)).to_table()` |
| 208 | CHA | RES | TBD | CFG | MN | MN | `registry.entities.safe_get_by_alias(["hero", "villain"], options=GetterOptions(scd="current")).to_table()` |
| 209 | CHA | RES | TBD | CFG | MN | ARG | `registry.entities.safe_get(entity_alias=["hero", "villain"], options=GetterOptions(scd="current")).to_table()` |
| 210 | CHA | RES | TBD | CFG | MN | TBD | `registry.entities.safe_get([EntityAlias("hero"), EntityAlias("villain")], options=GetterOptions(scd="current")).to_table()` |
| 211 | CHA | RES | TBD | CFG | ARG | MN | `registry.entities.get_by_alias(["hero", "villain"], raise_errors=False, options=GetterOptions(scd="current")).to_table()` |
| 212 | CHA | RES | TBD | CFG | ARG | ARG | `registry.entities.get(raise_errors=False, entity_alias=["hero", "villain"], options=GetterOptions(scd="current")).to_table()` |
| 213 | CHA | RES | TBD | CFG | ARG | TBD | `registry.entities.get([EntityAlias("hero"), EntityAlias("villain")], raise_errors=False, options=GetterOptions(scd="current")).to_table()` |
| 214 | CHA | RES | TBD | CFG | CFG | MN | `registry.entities.get_by_alias(["hero", "villain"], options=GetterOptions(scd="current", raise_errors=False)).to_table()` |
| 215 | CHA | RES | TBD | CFG | CFG | ARG | `registry.entities.get(entity_alias=["hero", "villain"], options=GetterOptions(scd="current", raise_errors=False)).to_table()` |
| 216 | CHA | RES | TBD | CFG | CFG | TBD | `registry.entities.get([EntityAlias("hero"), EntityAlias("villain")], options=GetterOptions(scd="current", raise_errors=False)).to_table()` |

`implementation_details.implementation` (the mechanics: how joins,
collapsing to current-per-entity, and alias path resolution actually
work under the hood) remains unaddressed by every mechanism above —
that's expected per the requirement's own framing (fitting a concrete
implementation is solution-design work, not part of picking the
interface shape).

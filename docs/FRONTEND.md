# Frontend

The Monitoring page is plain ES modules with no build step and no framework.
This document records how it is organised, what was cleaned up when it was
extracted into `frontend/`, and whether it is worth moving to React.

## Module boundaries

```text
frontend/
  index.html          the single page: Home · Medication Safety · Monitoring
  css/app.css         shared frame
  css/stream.css      the Monitoring styles, including the organ-pulse keyframes
  js/shell.js         Home, view switching, guided-demo transition
  js/stream/
    api.js       REST + SSE client. The only module that talks to the server.
    store.js     the state object S, the event reducers and the derived selectors.
    format.js    esc() and fmtNum(): the two text helpers every view needs.
    chart.js     the patient-trajectory chart. Pure: state in, SVG string out.
    signals.js   Signals & forecasts: the single-patient indicator list.
    report.js    the report builder and the printed research report.
    ward.js      Ward order: the multi-patient reading order, observations only.
    anatomy.js   builds the figure from organ-layout.json and applies organ state.
    render.js    all DOM writing. Pure functions of the store.
    main.js      lifecycle: boot, SSE wiring, controls, navigation, deep links.
  assets/anatomy-motion/   the baseplate, the organ artwork and the layout manifest
```

Flask serves this tree at `/static`, so `frontend/css/app.css` is
`/static/css/app.css`. There is no build step and no toolchain: the page loads
the modules directly, and a contract test asserts that every asset the page names
also exists on disk.

The dependency direction is one-way and has no cycles:

```text
main.js ──▶ api.js
   │  └────▶ store.js ──▶ format.js
   ├───────▶ render.js ──▶ store.js, chart.js, format.js
   └───────▶ anatomy.js
```

`api.js` never touches the DOM. `store.js` never touches the DOM. `render.js`
never fetches. `main.js` is the only module that wires events, and the only one
that both reads the network and writes state.

## Rendering

Every state change calls `scheduleRender()` in `main.js`, which coalesces into a
single `requestAnimationFrame`; that frame runs `renderAll()` once, which paints
the command bar, the anchor band, the view bar, the rail, the workspace, the
lower panel and the drawer, and then hands the anatomy scene one update object.
A burst of SSE events at 20× therefore produces one paint per frame, not one per
event.

Two deliberate partial renders exist, and both are correct rather than
duplication:

- `renderDrawer()` alone, when only the provenance drawer changed. Repainting the
  whole page would close nothing but would cost a full pass for a panel change.
- `renderWorkspace()` alone in the signal-filter input handler, which then
  restores the caret position. Going through the full pass would still rebuild
  the input; the direct call keeps the restore adjacent to the rebuild.

The anatomy scene keeps one pulse timer per organ and clears it before starting
the next, so a high-frequency stream cannot queue pulses or cut a new animation
short. Streamed events, the live snapshot and a historical run are three separate
sequence namespaces, so switching between them never replays a pulse that already
happened and never suppresses a genuine later one.

## State

`store.js` holds one object, `S`. The server is the source of truth: events
update only what they carry, and everything derived — ages, per-organ counts,
workspace splits — is computed at render time from the anatomy and functions
documents.

Live and historical state are kept apart on purpose:

| Live | Historical anchor |
|---|---|
| `S.series`, `S.variables`, `S.live`, `S.organEvents` | `S.anchor.{series, variables, cells, organEvents}` |

`currentSeries()`, `currentVariables()`, `currentCells()`, `currentRun()` and
`currentOrganEvents()` choose by mode, so `render.js` and `anatomy.js` never
decide. Returning to the live head is `S.anchor = null`: the live state was never
touched, so it comes straight back. New SSE events may only update live state.

This is the one place where two copies of similar-looking data are intentional.
Collapsing them would reintroduce the bug the separation exists to prevent: an
open historical anchor being overwritten by live traffic.

## The four views

The view bar offers `Anatomy`, `Functions`, `Signals & forecasts` and `Ward
order`. The first three are one patient; the fourth is many. They are kept apart
on purpose, in code, in copy and in tests:

- **Signals & forecasts** (`signals.js`, selectors `indicatorRows()` /
  `indicatorView()` in `store.js`) is a result list: one row per mapped
  indicator, the indicator's name first and largest, then the current result,
  its flag, the reference interval it was compared against, the previous result
  with its change, the forecast at one chosen horizon, and when it was
  observed. That column order is the order a result report states things (CLIA
  42 CFR 493.1291(c) and (d)); FHIR R4 puts the categorical assessment beside
  the value, so the flag sits next to the result rather than at the far left.
  The page has three sections that never share an indicator — `Needs
  attention`, `Other observed indicators`, and a folded `Not yet observed (N)`
  — and one row unfolds at a time into every horizon, the basis of the
  statement, the data-quality line, a full trajectory chart with the supplied
  range drawn as a band, and folded provenance. Rows carry no miniature plot:
  a row states its numbers, and the chart is in the detail. It reads only
  through the mode selectors, so a historical snapshot shows that run's inputs
  and a ward stay opened from the ward order shows that stay's own ledger. It
  supports a search, a sort, a forecast-horizon control and a
  `Print / PDF` button (`@media print` in `stream.css`, landscape A4, the
  report head in place of the chrome). *Attention* is the observation's own
  statement and nothing else: a `referenceRange` the observation supplies,
  compared in the unit it was written in with inclusive limits, or a coded
  `interpretation` the source made (the backend passes both through untouched
  on `/state` and `/timeline`, see `observation_statements()` in
  `timeline.py`, together with the stated `input_value`/`input_unit` and the
  mapping's `derivation`). With neither the row says `Range not supplied`; with
  a range in a unit the value was not stated in it says `Range unavailable for
  this unit`; in both cases nothing is flagged and no threshold is written into
  the front end. A model point is compared against the same range only in a
  unit it can be read in (its own, or the stated unit when the mapping calls
  the conversion the identity) and then reads `Projected above/below supplied
  range`; a carried-forward cell, abstention included, never does.

  The default order is the reading order (`TIER` in `store.js`): a
  source-coded critical reading, an observed value outside its supplied range,
  a model forecast projected outside the range, a new observation, a result
  inside its range, a result with no range, a row with only baselines or
  withheld cells, and last what was never observed. Inside a tier: most recent
  first, then by name. A current abnormality always sorts before a projected
  one, and the tier reflects a projection at any horizon — hiding one because
  the reader is comparing at another horizon would under-report it.

  **The two colours.** A measured result outside its supplied interval is
  `--observed-abnormal` (deep red), bold, marked `▲`/`▼`, ruled down the left
  of its row and sorted first. A model forecast that leaves the same interval
  is `--projected-attention` (amber), marked `△`/`▽`, hollow and dashed in the
  charts, counted apart and never called an alert — a page that flags
  everything is read as flagging nothing (AHRQ). Every state carries a mark and
  a word as well as a colour (WCAG 2.2 SC 1.4.1), and both tokens pass WCAG AA
  on their own tint and on the page ground; `test_stream_ui_contract.py`
  computes the contrast rather than trusting the swatch.

  Where the supplied ranges come from: `backend/oslian_demo/monitoring/
  reference_ranges.py` is the single registry (MedlinePlus Medical
  Encyclopedia, NLM/NIH, one URL and review date per interval, the unit the
  source states and why the observation's unit is the same number). The
  synthetic fixture builder writes each interval into the laboratory
  observations it applies to and an `interpretation` derived from it, marked
  `derived_from_supplied_reference_range`; the fixture document carries the
  whole provenance block. Sex-, age- or context-dependent intervals (creatinine,
  hemoglobin, hematocrit, glucose) and vital signs are deliberately not
  supplied, and the reason is written in the same registry.
- **The report** (`report.js`). `Print / PDF` opens a preview, not a print
  dialog: `openReport()` deep-copies the whole view-model (rows, their
  trajectories, the organ states, the counts and the header facts), so a
  report does not move while the stream runs underneath it; `Refresh report`
  takes a new snapshot and keeps the builder's choices. The builder chooses a
  scope (full, needs-attention only, or selected indicators), a forecast
  horizon, and three optional sections (trend charts and the anatomy overview
  on by default, the provenance appendix off). Only the preview calls
  `window.print()`. The document is titled *Oslian Monitoring Research Report*,
  names its reporting system as a research demonstration and claims no
  performing laboratory; it carries the header a result report has to carry,
  the results requiring attention, the other observed results, Oslian's own
  forecast outlook (which separates a model forecast from a carried-forward
  value, an abstention, a missing and a degraded cell), trend charts only where
  they earn the paper, an anatomy overview drawn by the same `buildAnatomy()`
  on the frozen organ states, and the limitations. It prints A4 landscape in
  five pages, repeats the table head after a break and never splits a row.

- **Ward order** (`ward.js`, `/api/ward`) answers *which stay should I review
  first*. It is observation-only, runs no model, needs no attached runtime, and
  its bands come from the published cutpoints in `ranking.py` — a different
  instrument for a different question. Those cutpoints are never read by the
  single-patient view, and the single-patient attention words are never used on
  the ward. A row opens the stay as its own session (`retrospective:<case id>`);
  the board, the selected row, the open bands and the scroll position survive
  the trip and come back on return.

**Organ attention** is one pure function, `organAttentionState(organ, rows)` in
`store.js`, with `organAttentionMap()` and `organsByAttention()` over it. It
aggregates an organ's indicators into a state — `abnormal` (a current result
outside a supplied range), else `projected`, else `new`, else `observed`, else
`none` — plus the counts and the indicator ids behind it. The organ rail sorts
by that state and says which indicators put the organ there; the figure draws
it (`fxDanger`, `fxAmber`, the existing teal pulse) without moving or
reshaping any artwork, and a selected organ keeps the state it has because the
selection ring is drawn on the hit area, not on the wash; the organ workspace
orders its indicators by it; the printed anatomy overview reads the frozen copy
of it. The response filters are defined once for the document
(`ensureFilters()`), because the page and the report both reference them by id.

A shared head that declined to move (`fallback_reason:
shared_jepa_head_abstained`, status `persistence`) is shown everywhere as
`Carried forward — model abstained`: it is counted with the baselines, drawn as
a baseline and never in the model's blue, and is never projected anywhere. The
badge is read from the status and the reason, never from `forecast === observed`,
because the frozen backbone can equal the observation by coincidence and still be
a real prediction.

## The patient trajectory

The metric page draws one chart, `trajectoryChart()` in `chart.js`, from one
selector, `trajectoryFor(variable, horizons)` in `store.js`. The selector reads
everything through the mode selectors (`currentSeries`, `currentRun`,
`currentCells`), so the same code draws the live head and a historical snapshot
and cannot mix them. It returns:

- `points` — the observed series, each a ledger row;
- `cutoff_ms` — the run's `input_cutoff`, which is where the future starts;
- `forecasts` — the run's cells for this variable, each with the `basis` the run
  actually used (`observed_value` at `cutoff − measurement_age_hours`); a cell
  with no observed value has no basis, so it gets no lead line and no delta;
- `kind` — what the future region shows: `model`, `persistence`, `none` (no
  target × horizon cell in the registry), `unavailable` (every horizon withheld)
  or `no-run`.

The chart is one linear time axis split at the cutoff into **OBSERVED HISTORY**
(white ground, solid ink line, filled circles) and **FUTURE FORECAST** (pale
blue ground, dashed blue path from the basis observation, hollow diamonds with
`1h · 68.5` labels). A carried-forward cell is a grey dashed horizontal baseline
with hollow squares, captioned *Baseline only · not a model forecast*; a
withheld cell draws nothing and the region says *Forecast unavailable* with a
reader-facing reason (`unavailableReason()` in `format.js`; the audit reason
stays in provenance); a variable with no runtime cell says *No forecast model
for this metric*. The pale region is the future, never an uncertainty band —
no interval is ever drawn, and the legend says so.

The chart is drawn at the workspace's real pixel width (`chartWidth()` in
`render.js`; a resize redraws), so text stays at its true size at 1440, 1280
and 375 and the phone layout is one column with no sideways scroll. The SVG is
a `role="group"` with a `<title>` and a `<desc>` that states the same facts the
picture shows; every observed and forecast point is a focusable `role="button"`
with its own `aria-label`, and the tooltip is clamped inside the chart card.

Below the chart, the comparison summary lists the last observed value and each
horizon's forecast with its signed change against that cell's own basis; a
baseline row says it is a baseline, a withheld row says it is unavailable, and
no colour or word rates a change as good or bad. The window, horizon and
run-history controls come next, and the run details and provenance fold away
last. The header badge reads **LIVE HEAD** or **HISTORICAL SNAPSHOT**; a
snapshot also puts the band *Frozen at <cutoff> · Later observations are not
included* above the page. New SSE events only ever touch the live state, so an
open snapshot cannot change under a running stream.

`tests/test_stream_trajectory.py` runs `chart.js` and `store.js` under Node
and asserts these invariants on the SVG it gets back.

## What the tidy changed

When this tree was extracted, `esc()` was defined identically in `render.js`
and `chart.js`, and `fmtNum()` twice — once exported from `store.js` and once as
a private near-copy in
`chart.js` (the same function without the optional `decimals` argument). Both now
live in `format.js` and there is one definition of each. Output is unchanged: no
call site passed `decimals` to the chart copy.

No other duplicated state or duplicated DOM update was found. The audit was:
every top-level `function` / `const` name across the six modules checked for
collisions (only `esc` and `fmtNum` collided), every `renderAll` /
`renderWorkspace` / `renderDrawer` call site reviewed, and the store's fields
checked for two names holding the same value.

Visual output, organ positions and interaction behaviour are unchanged.

## Is React worth the next phase?

**No — not on the evidence this page provides today.**

The three things that make a framework pay for itself are component lifecycle,
routing, and shared state across independently-mounted subtrees. This page has a
weak case on all three:

- **Lifecycle.** There is exactly one long-lived resource, the SSE connection,
  opened once in `main.js` and reconnected by `api.js`. There are no per-component
  subscriptions to tear down, so there is no lifecycle problem to solve.
- **Routing.** Navigation is one hash deep-link parsed once at boot
  (`#stream?mode=…&view=…&organ=…&metric=…&run=…`) and updated in place. A router
  would replace 30 lines with a dependency.
- **Shared state.** One store object, one reducer entry point (`applyEvent`), one
  coalesced render. That is the shape a Redux-style app converges on anyway,
  minus the framework.

Against that, the costs are concrete: a build step where there is none today (and
the demo currently runs offline with no toolchain at all); a second rendering
model layered over `anatomy.js`, which manipulates a large SVG scene graph
directly and would need an escape hatch on day one; and re-verification of the
locked anatomy behaviour — organ placement, the pulse timers, the live/anchor
isolation — all of which are currently pinned by source-level contract tests that
read these exact files.

The honest signals that would change the answer, none of which hold now:

1. `render.js` passing roughly 1000 lines, or a second full page appearing that
   shares the store.
2. Interactive state that must be co-located with markup — inline editing,
   multi-step forms, drag-and-drop — rather than the current read-mostly view
   over server state.
3. More than one long-lived resource per view, so that subscription lifetime
   genuinely needs owning.
4. A team large enough that the informal module contract stops holding.

If a rewrite ever does happen, the order that preserves the most value is:
`chart.js` first (already pure: state in, SVG out), then `render.js` panel by
panel, with `anatomy.js` kept as an imperative scene wrapped in a single
component. `store.js` and `api.js` would survive untouched.

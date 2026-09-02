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
    chart.js     the shared time-series chart. Pure: state in, SVG string out.
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

/* In-memory view state for the monitoring page plus the event reducers.
 * The server is the source of truth: events only update what they carry, and
 * everything derived (ages, per-organ counts, workspace splits) is computed
 * at render time from the anatomy and functions documents. */

import { fmtNum } from './format.js';

export const S = {
  mode: 'synthetic',      // source key: synthetic | retrospective
  sources: [],            // /api/demo-streams/sources
  sessionId: null,
  session: null,          // last stream.state / session state (includes .source describe())
  sessionStampedAt: 0,
  patient: null,
  clock: { latest_clinical_event_time: null, latest_received_at: null, model_input_watermark: null, live_run_id: null },
  runtime: null,
  capabilities: null,
  health: null,
  anatomy: null,
  organLayout: null,      // /static/assets/anatomy-motion/organ-layout.json
  functions: null,
  variables: {},          // variable -> latest state
  series: {},             // variable -> [{effective_time, value, ...}] (live head)
  live: { run: null, cells: [] },
  anchor: null,           // { run, cells, series, variables, organEvents } when viewing a historical run
  runs: [],
  runInFlight: null,
  lastRunFailed: null,
  view: 'anatomy',        // anatomy | functions | data | ward
  // Ward attention order. Many stays, observations only: it shares nothing with
  // the session above it, which is why it survives an unattached runtime.
  ward: null,             // /api/ward payload
  wardPatient: null,      // selected row id; null = the top of the order
  wardOpenBands: {},      // band key -> open, only where the reader overrode the default
  wardError: null,
  wardScroll: null,       // { rail, page } kept when the reader leaves the ward view, put back on return
  selectedOrgan: 'kidneys',
  selectedFunction: 'renal_urinary',
  metric: null,           // variable id when the metric workspace is open
  chartWindow: 'recent',  // recent | full
  chartHorizons: null,    // null = all, else Set of horizon hours
  showSecondary: false,
  // Signals & forecasts (the single-patient indicator list)
  dataQuery: '',
  dataSort: 'attention',  // attention | organ | name | latest | age
  dataExpanded: null,     // the one indicator whose detail panel is open
  dataShowUnobserved: false, // the "Not yet observed" section, folded by default
  // "Other observed indicators": null means "whatever this viewport should do" -
  // open on a desktop, folded on a phone. Once the reader has said, the answer is
  // kept, so a stream event never reopens or recloses it under them.
  dataShowOther: null,
  forecastHorizon: null,  // the horizon every Forecast column compares at; null = best covered
  // The report. `frozen` is the snapshot the preview was opened on: the page keeps
  // streaming underneath and the open report does not move until it is refreshed.
  report: null,           // { scope, indicators, horizon, charts, anatomy, provenance, frozen, id, generatedAt }
  evidenceOpen: false,
  runDetailsOpen: false, // the metric page's run/provenance details, folded below the chart
  connection: 'connecting',
  lastSequence: 0,
  organEvents: {},        // organ id -> { at, sequence } from the event's own organ_ids
  unassetedOrganEvents: {},// organ id -> count, for canonical organs with no visual asset
  freshVariables: {},
  freshRun: 0,
  runFlow: 0,             // timestamp of the last model.run.started (for the flow transition)
  feed: [],
  mappingFailures: 0,
  drawer: null,
  error: null,
};

const FEED_CAP = 40;

/** Canonical organs the layout manifest carries with no detailed vector asset.
 *  Their events are kept and counted; nothing is drawn for them. */
export function organWithoutAsset(id) {
  if (!S.organLayout) return false;
  return S.organLayout.organs.some((o) => o.organ === id && o.visual_asset_missing);
}
export const missingAssetOrgans = () => (S.organLayout
  ? [...new Set(S.organLayout.organs.filter((o) => o.visual_asset_missing).map((o) => o.organ))] : []);

/** Clears the LIVE organ state only. A historical anchor owns its own state and is
 *  dropped whole (`S.anchor = null`) by a reset; emptying it here would blank an open
 *  anchor whenever a snapshot reload runs, which happens on any amendment. */
export function clearOrganState() {
  S.organEvents = {};
  S.unassetedOrganEvents = {};
}

/** A server organ block turned into the map the figure reads.
 *  `origin` names the state's own sequence space ('snapshot' for the live head's
 *  snapshot, 'anchor' for a historical run) and marks it as restored, so it lights
 *  the organs without replaying pulses that already happened. */
export function organStateFrom(organs, origin) {
  const out = {};
  Object.entries(organs || {}).forEach(([id, o]) => {
    if (!o || !o.at) return;
    out[id] = { at: o.at, sequence: o.sequence || 0, restored: origin };
  });
  return out;
}

/** Live organ state from the server snapshot, so a reload lights the organs that
 *  were lit. `organs` is the /state document's own routing - the same approved map
 *  the events use - and it replaces the live state outright. */
export function applySnapshotOrgans(organs) {
  clearOrganState();
  S.organEvents = organStateFrom(organs, 'snapshot');
  Object.entries(organs || {}).forEach(([id, o]) => {
    if (o && o.at && organWithoutAsset(id)) S.unassetedOrganEvents[id] = o.count || 0;
  });
}

/** The organ state the figure must show right now: a historical anchor shows the
 *  run's own snapshot, the live head shows what the stream has delivered. New SSE
 *  events only ever touch `S.organEvents`, so an open anchor cannot be polluted. */
export const currentOrganEvents = () => (S.anchor ? S.anchor.organEvents || {} : S.organEvents);

export function resetSessionState() {
  Object.assign(S, {
    sessionId: null, session: null, patient: null,
    clock: { latest_clinical_event_time: null, latest_received_at: null, model_input_watermark: null, live_run_id: null },
    variables: {}, series: {}, live: { run: null, cells: [] }, anchor: null, runs: [], runInFlight: null,
    lastRunFailed: null, metric: null, lastSequence: 0, organEvents: {}, unassetedOrganEvents: {},
    freshVariables: {}, freshRun: 0, runFlow: 0,
    feed: [], mappingFailures: 0, drawer: null, error: null,
  });
}

export const sourceInfo = () => (S.session && S.session.source) || {};
export const timeAxis = () => sourceInfo().time_axis || 'calendar';
export function onsetMs() {
  const src = sourceInfo();
  const iso = src.epoch || (S.patient && S.patient.encounter_start);
  return iso ? new Date(iso).getTime() : 0;
}

export const currentCells = () => (S.anchor ? S.anchor.cells : S.live.cells);
export const currentRun = () => (S.anchor ? S.anchor.run : S.live.run);
export const currentVariables = () => (S.anchor ? S.anchor.variables : S.variables);
export const currentSeries = () => (S.anchor ? S.anchor.series : S.series);
export const clinicalNow = () => (S.anchor ? S.anchor.run.input_cutoff : S.clock.latest_clinical_event_time);

export function cellsForVariable(variable, cells) {
  return (cells || []).filter((c) => (c.target === 'hypoxemia_event' ? 'o2sat' : c.target) === variable);
}

/** Every catalog cell (from capabilities) attributed to a variable, whether or not a run exists. */
export function catalogCellsForVariable(variable) {
  if (!S.capabilities) return [];
  return S.capabilities.cells.filter((c) => (c.target === 'hypoxemia_event' ? 'o2sat' : c.target) === variable);
}

const HOUR = 3600000;

/** Everything the patient-trajectory chart and its comparison summary read, for one
 *  variable, from whichever head is open: the observed series, the run's input cutoff,
 *  and each forecast cell with the observation the run actually used as its basis.
 *
 *  Every field comes through the mode selectors, so a historical anchor yields that
 *  run's frozen inputs and cells, never the live ones. Nothing is imputed: a cell
 *  with no `observed_value` has no basis, so it gets no lead line and no delta.
 *
 *  `kind` is what the future region shows:
 *    'no-run'       no completed run yet: observations only, no cutoff
 *    'none'         no target × horizon cell for this variable in the pinned registry
 *    'model'        at least one horizon is a model point forecast
 *    'persistence'  every drawn horizon is the last value carried forward
 *    'unavailable'  every horizon is withheld (missing or degraded)
 *  Cells whose target is a derived event (hypoxemia_event on the SpO₂ chart) are
 *  listed as `derived` and never plotted on the variable's own axis. */
export function trajectoryFor(variable, horizons) {
  const run = currentRun();
  const cutoff_ms = run ? Date.parse(run.input_cutoff) : null;
  const points = (currentSeries()[variable] || []).map((p) => ({
    ms: Date.parse(p.effective_time), value: p.value, unit: p.unit, resource_id: p.resource_id,
    version_id: p.version_id, received_at: p.received_at,
  }));
  const own = (c) => c.target === variable;
  const catalog = catalogCellsForVariable(variable);
  const runCells = run ? cellsForVariable(variable, currentCells()) : [];
  const keep = (c) => !horizons || horizons.has(c.horizon_hours);
  const forecasts = runCells.filter(own).filter(keep).map((c) => {
    const hasBasis = c.observed_value !== null && c.observed_value !== undefined
      && c.measurement_age_hours !== null && c.measurement_age_hours !== undefined;
    const basis_ms = hasBasis ? cutoff_ms - c.measurement_age_hours * HOUR : null;
    const matched = hasBasis ? points.find((p) => Math.abs(p.ms - basis_ms) < 1000) : null;
    // the chart reads the same projection rule the list does, so a point drawn
    // amber and a row marked "projected" can never disagree
    const fa = forecastAttention(c, currentVariables()[variable]);
    return {
      cell: c.cell, horizon_hours: c.horizon_hours, due_ms: cutoff_ms + c.horizon_hours * HOUR,
      forecast_point: c.forecast_point, status: c.status, source_model: c.source_model,
      fallback_reason: c.fallback_reason, unit: c.unit, run_id: run.run_id,
      projected: fa.projected, projectedSide: fa.side,
      basis: hasBasis ? { ms: basis_ms, value: c.observed_value, resource_id: matched ? matched.resource_id : null } : null,
    };
  }).sort((a, b) => a.horizon_hours - b.horizon_hours);
  const derived = runCells.filter((c) => !own(c));
  let kind;
  if (!run) kind = 'no-run';
  else if (!catalog.some(own)) kind = 'none';
  else if (forecasts.some((f) => f.status === 'model')) kind = 'model';
  else if (forecasts.some((f) => f.status === 'persistence')) kind = 'persistence';
  else kind = 'unavailable';
  return { variable, run, cutoff_ms, points, forecasts, derived, kind,
    catalogHorizons: Array.from(new Set(catalog.filter(own).map((c) => c.horizon_hours))).sort((a, b) => a - b),
    frozen: !!S.anchor };
}

/* ── Indicators: one view-model for the list, the organs and the report ──
 *
 * Every screen that says something about a value reads this file and nothing
 * else: the indicator list, the organ rail and figure, the functional groups and
 * the printed report all call `indicatorRows()` and the selectors below it.
 * There is one comparison rule, written once.
 *
 * What a value is compared against is the observation's own statement, never
 * this page's (FHIR R4 Observation: `referenceRange` is "guidance on how to
 * interpret the value by comparison to a normal or recommended range";
 * `interpretation` is "a categorical assessment of an observation value"):
 *   `referenceRange` — an interval the observation supplies, compared only in
 *     the unit it was written in (the observation's own stated unit, or the
 *     canonical unit when the mapping calls the conversion the identity), with
 *     inclusive limits, exactly as R4 defines low and high;
 *   `interpretation` — a coded reading. A reading the source coded stands on its
 *     own; a reading a fixture derived from the supplied range says so in its
 *     provenance, never outranks that range and is never critical.
 * With neither, the row says the range was not supplied; with a range in a unit
 * the value was not stated in, it says so. Nothing is flagged in either case and
 * no threshold is written here.
 *
 * Ordering follows the alert-fatigue evidence (AHRQ): a current, sourced
 * abnormality first, a model forecast that leaves the same range next, mere
 * novelty after that, and everything else quiet. Only the first three are
 * "needs attention"; a new value or a normal one is never an alert. */
const OUTSIDE_HIGH = new Set(['H', 'HH', 'HU', '>']);
const OUTSIDE_LOW = new Set(['L', 'LL', 'LU', '<']);
const OUTSIDE_OTHER = new Set(['A', 'AA']);
const CRITICAL_CODES = new Set(['HH', 'LL', 'AA']);
const NORMAL_CODE = 'N';
export const DERIVED_INTERPRETATION = 'derived_from_supplied_reference_range';
const FRESH_MS = 30000;   // a "new observation" for ordering: arrived in the last half minute

/** The one reference range that applies to this observation, or null: a range
 *  qualified by population or age is used only when it is the sole range. */
function suppliedRange(state) {
  const ranges = (state && state.reference_range) || [];
  const general = ranges.filter((r) => !r.applies_to && !r.age);
  const pick = general.length === 1 ? general[0] : (!general.length && ranges.length === 1 ? ranges[0] : null);
  if (!pick) return null;
  const low = Number.isFinite(pick.low) ? pick.low : null;
  const high = Number.isFinite(pick.high) ? pick.high : null;
  if (low === null && high === null) return null;
  // a range with no unit of its own is in the observation's own unit
  const unit = pick.low_unit || pick.high_unit || state.input_unit || state.unit || null;
  return { low, high, unit, text: pick.text || null, qualified: !!(pick.applies_to || pick.age) };
}

/** The supplied range expressed in `unit`, or null when it was written in another. */
function rangeIn(state, unit) {
  const range = suppliedRange(state);
  if (!range || !unit || range.unit !== unit) return null;
  return range;
}

// R4: low and high are inclusive, so a value on the line is inside the interval
const sideOf = (value, range) => ((range.low !== null && value < range.low) ? 'below'
  : (range.high !== null && value > range.high) ? 'above' : null);

const readingOf = (codes) => (codes.some((c) => OUTSIDE_HIGH.has(c)) ? 'above'
  : codes.some((c) => OUTSIDE_LOW.has(c)) ? 'below'
    : codes.some((c) => OUTSIDE_OTHER.has(c)) ? 'outside'
      : codes.includes(NORMAL_CODE) ? 'within' : null);

/** What the latest observation says about its own value.
 *    kind     none | unsupplied | unit | within | outside
 *    side     above | below | null
 *    basis    referenceRange | interpretation | null
 *    derived  the reading was arithmetic on the supplied range, not a source reading
 *    critical a source-coded critical reading (HH, LL, AA); never from a derived one */
export function observationAttention(state) {
  if (!state) return { kind: 'none', side: null, basis: null, derived: false, critical: false, codes: [] };
  const interp = state.interpretation || [];
  const sourceCodes = interp.filter((c) => c.provenance !== DERIVED_INTERPRETATION).map((c) => c.code);
  const derivedCodes = interp.filter((c) => c.provenance === DERIVED_INTERPRETATION).map((c) => c.code);
  const critical = sourceCodes.some((c) => CRITICAL_CODES.has(c));
  const codes = sourceCodes.concat(derivedCodes);
  const inputUnit = state.input_unit || state.unit;
  const inputValue = Number.isFinite(state.input_value) ? state.input_value : state.value;
  let range = rangeIn(state, inputUnit), value = inputValue;
  if (!range) { range = rangeIn(state, state.unit); value = state.value; }
  if (range) {
    const side = sideOf(value, range);
    return { kind: side ? 'outside' : 'within', side, basis: 'referenceRange', range, value, unit: range.unit,
      derived: derivedCodes.length > 0 && sourceCodes.length === 0, critical, codes };
  }
  const reading = readingOf(sourceCodes);
  if (reading) {
    return { kind: reading === 'within' ? 'within' : 'outside', side: reading === 'above' || reading === 'below' ? reading : null,
      basis: 'interpretation', derived: false, critical, codes };
  }
  const supplied = suppliedRange(state);
  if (supplied) {
    // fail closed: a range in a unit the value was not stated in is not compared,
    // and a reading derived from it is not trusted in its place
    return { kind: 'unit', side: null, basis: null, derived: false, critical, codes, range: supplied,
      note: `a range in ${supplied.unit} was supplied and is not compared with a value in ${state.unit}` };
  }
  return { kind: 'unsupplied', side: null, basis: null, derived: false, critical, codes };
}

/** A model point forecast against the same supplied range, in a unit it can be
 *  read in. Only a `model` cell with a value can be projected anywhere: a
 *  carried-forward value repeats the observation and a withheld cell has none. */
export function forecastAttention(cell, state) {
  if (!cell || cell.status !== 'model' || !Number.isFinite(cell.forecast_point)) return { projected: false, side: null };
  let range = rangeIn(state, cell.unit);
  // a cell is written in the canonical unit; a range in the observation's own unit
  // applies to it only when the mapping says the conversion was the identity (the
  // same number under another unit name) - the mapping's statement, not this page's
  if (!range && state && state.derivation === 'identity' && cell.unit === state.unit) range = rangeIn(state, state.input_unit);
  if (!range) return { projected: false, side: null };
  const side = sideOf(cell.forecast_point, range);
  return { projected: !!side, side, range };
}

const ABSTAINED = 'shared_jepa_head_abstained';

/* The reading order of every indicator list on this page. Only the first three
 * are "needs attention"; the rest are ordinary rows in a quiet order. */
export const TIER = { critical: 1, outside: 2, projected: 3, fresh: 4, within: 5, unsupplied: 6, quiet: 7, unobserved: 8 };
export const ATTENTION_TIERS = [TIER.critical, TIER.outside, TIER.projected];

export function indicatorTier(row) {
  if (!row.state) return TIER.unobserved;
  const a = row.attention;
  if (a.critical) return TIER.critical;
  if (a.kind === 'outside') return TIER.outside;             // observed, and outside what it was compared against
  if (row.forecasts.some((f) => f.projected)) return TIER.projected;
  if (row.fresh) return TIER.fresh;
  if (a.kind === 'within') return TIER.within;
  // nothing to compare it against: quiet if it has no model forecast either
  return row.forecasts.some((f) => f.status === 'model') ? TIER.unsupplied : TIER.quiet;
}

/** Every horizon the pinned registry knows, with how many indicators it covers
 *  for this patient, and the one that covers the most (ties: the shorter). */
export function horizonCoverage() {
  const counts = new Map();
  ((S.capabilities && S.capabilities.cells) || []).forEach((c) => {
    counts.set(c.horizon_hours, (counts.get(c.horizon_hours) || 0) + 1);
  });
  const horizons = [...counts.entries()].map(([hours, count]) => ({ hours, count })).sort((a, b) => a.hours - b.hours);
  const best = horizons.reduce((top, h) => (!top || h.count > top.count ? h : top), null);
  return { horizons, best: best ? best.hours : null };
}

/** The horizon every forecast column compares at: the reader's choice while it
 *  still exists in this registry, else the best-covered one. */
export function activeHorizon() {
  const { horizons, best } = horizonCoverage();
  if (S.forecastHorizon !== null && horizons.some((h) => h.hours === S.forecastHorizon)) return S.forecastHorizon;
  return best;
}

/** The observation before the latest one, from the same series the chart draws.
 *  A superseded or retracted version never reaches this series, so the previous
 *  result is the previous *valid* result. Null when there is only one. */
function previousResult(points, state) {
  if (!state || points.length < 2) return null;
  const last = points[points.length - 1];
  const prior = points[points.length - 2];
  if (!prior || !Number.isFinite(prior.value)) return null;
  const change = Number.isFinite(last.value) ? last.value - prior.value : null;
  return { value: prior.value, unit: prior.unit || state.unit, ms: Date.parse(prior.effective_time), change };
}

/** One row per mapped indicator, from whichever head is open. */
export function indicatorRows() {
  const units = S.anatomy ? S.anatomy.variable_units : {};
  const organsByVariable = {};
  if (S.anatomy) S.anatomy.organs.forEach((o) => o.variables.forEach((v) => { (organsByVariable[v] = organsByVariable[v] || []).push(o.label); }));
  const organIdsByVariable = {};
  if (S.anatomy) S.anatomy.organs.forEach((o) => o.variables.forEach((v) => { (organIdsByVariable[v] = organIdsByVariable[v] || []).push(o.id); }));
  const vars = currentVariables(), cells = currentCells(), run = currentRun(), allSeries = currentSeries();
  const now = clinicalNow();
  const wall = Date.now();
  const horizon = activeHorizon();
  return Object.keys(units).map((variable) => {
    const state = vars[variable] || null;
    const own = (c) => c.target === variable;
    const catalogHorizons = Array.from(new Set(catalogCellsForVariable(variable).filter(own).map((c) => c.horizon_hours))).sort((a, b) => a - b);
    const forecasts = (run ? cellsForVariable(variable, cells).filter(own) : []).slice()
      .sort((a, b) => a.horizon_hours - b.horizon_hours)
      .map((c) => {
        const fa = forecastAttention(c, state);
        return { ...c, abstained: c.status === 'persistence' && c.fallback_reason === ABSTAINED,
          projected: fa.projected, projectedSide: fa.side };
      });
    const attention = observationAttention(state);
    const points = (allSeries[variable] || []).filter((p) => Number.isFinite(p.value));
    const ageHours = state && now ? (Date.parse(now) - Date.parse(state.effective_time)) / HOUR : null;
    const row = {
      variable, unit: units[variable], organs: organsByVariable[variable] || [],
      organIds: organIdsByVariable[variable] || [], state, ageHours,
      series: points, catalogHorizons, forecasts, attention, run,
      range: attention.range || suppliedRange(state) || null,
      previous: previousResult(points, state),
      fresh: !!state && !S.anchor && (S.freshVariables[variable] || 0) > wall - FRESH_MS,
      horizon,
      atHorizon: horizon === null ? null : (forecasts.find((f) => f.horizon_hours === horizon) || null),
      hasHorizonCell: catalogHorizons.includes(horizon),
      modelCount: forecasts.filter((f) => f.status === 'model').length,
      carriedCount: forecasts.filter((f) => f.status === 'persistence').length,
      abstainedCount: forecasts.filter((f) => f.abstained).length,
      withheldCount: forecasts.filter((f) => f.status !== 'model' && f.status !== 'persistence').length,
      projectedSide: (forecasts.find((f) => f.projected) || {}).projectedSide || null,
    };
    row.tier = indicatorTier(row);
    row.flagged = ATTENTION_TIERS.includes(row.tier);
    return row;
  });
}

const byTime = (a, b) => ((b.state ? b.state.effective_time : '')).localeCompare(a.state ? a.state.effective_time : '');

/** The page's three sections and the counts its summary states. `labelOf`
 *  renders an indicator's display name; ordering inside a tier is the most
 *  recent observation first, then that name, so it is stable across renders. */
export function indicatorView(labelOf) {
  const name = labelOf || ((v) => String(v));
  const all = indicatorRows();
  const q = String(S.dataQuery || '').trim().toLowerCase();
  const matches = (r) => !q || name(r.variable).toLowerCase().includes(q) || r.variable.includes(q)
    || r.organs.join(' ').toLowerCase().includes(q);
  const byName = (a, b) => name(a.variable).localeCompare(name(b.variable));
  const organ = (r) => (r.organs[0] || 'zz');
  const sorters = {
    attention: (a, b) => (a.tier - b.tier) || byTime(a, b) || byName(a, b),
    organ: (a, b) => organ(a).localeCompare(organ(b)) || byName(a, b),
    name: byName,
    latest: (a, b) => byTime(a, b) || byName(a, b),
    age: (a, b) => ((a.ageHours ?? 1e9) - (b.ageHours ?? 1e9)) || byName(a, b),
  };
  const sort = sorters[S.dataSort] || sorters.attention;
  const shown = all.filter(matches);
  const observed = all.filter((r) => r.state);
  return {
    all,
    attention: shown.filter((r) => r.flagged).sort(sort),
    other: shown.filter((r) => r.state && !r.flagged).sort(sort),
    unobserved: shown.filter((r) => !r.state).sort(byName),
    matched: shown.length,
    total: all.length,
    observed: observed.length,
    critical: observed.filter((r) => r.attention.critical).length,
    outside: observed.filter((r) => r.attention.kind === 'outside').length,
    above: observed.filter((r) => r.attention.side === 'above').length,
    below: observed.filter((r) => r.attention.side === 'below').length,
    projected: observed.filter((r) => r.forecasts.some((f) => f.projected)).length,
    fresh: observed.filter((r) => r.fresh).length,
    supplied: observed.filter((r) => r.attention.kind === 'outside' || r.attention.kind === 'within').length,
    unsupplied: observed.filter((r) => r.attention.kind === 'unsupplied' || r.attention.kind === 'unit').length,
    modelCells: all.reduce((n, r) => n + r.modelCount, 0),
    carriedCells: all.reduce((n, r) => n + r.carriedCount, 0),
    abstainedCells: all.reduce((n, r) => n + r.abstainedCount, 0),
    withheldCells: all.reduce((n, r) => n + r.withheldCount, 0),
    horizon: activeHorizon(),
    horizons: horizonCoverage().horizons,
  };
}

/* ── organ attention ──────────────────────────────────────────────────────
 * One function decides what an organ is showing, and the rail, the figure, the
 * organ workspace and the printed anatomy overview all read it. An organ is
 * never given a state its own indicators do not carry, and the words are the
 * indicator words: outside a supplied range, projected outside, new. */
export function organAttentionState(organ, rows) {
  const own = rows.filter((r) => (organ.variables || []).includes(r.variable));
  const observed = own.filter((r) => r.state);
  const abnormal = observed.filter((r) => r.attention.kind === 'outside');
  const projected = observed.filter((r) => !abnormal.includes(r) && r.forecasts.some((f) => f.projected));
  const fresh = observed.filter((r) => r.fresh);
  const latest = observed.reduce((top, r) => (!top || r.state.effective_time > top ? r.state.effective_time : top), null);
  const tier = abnormal.some((r) => r.attention.critical) ? TIER.critical
    : abnormal.length ? TIER.outside
      : projected.length ? TIER.projected
        : fresh.length ? TIER.fresh
          : observed.length ? TIER.within : TIER.unobserved;
  return {
    id: organ.id, label: organ.label, tier,
    state: abnormal.length ? 'abnormal' : projected.length ? 'projected' : fresh.length ? 'new'
      : observed.length ? 'observed' : 'none',
    abnormal: abnormal.length, projected: projected.length, fresh: fresh.length,
    observed: observed.length, mapped: (organ.variables || []).length, latest,
    above: abnormal.filter((r) => r.attention.side === 'above').map((r) => r.variable),
    below: abnormal.filter((r) => r.attention.side === 'below').map((r) => r.variable),
    abnormalIds: abnormal.map((r) => r.variable),
    projectedIds: projected.map((r) => r.variable),
    freshIds: fresh.map((r) => r.variable),
  };
}

/** Every organ's state, keyed by organ id, from one pass over the indicators. */
export function organAttentionMap(rows) {
  const out = {};
  if (!S.anatomy) return out;
  const all = rows || indicatorRows();
  S.anatomy.organs.forEach((organ) => { out[organ.id] = organAttentionState(organ, all); });
  return out;
}

/** Canonical organ order, then attention: an organ with a current abnormality
 *  comes first, ties broken by the most recent observation and the canonical order. */
export function organsByAttention(rows) {
  if (!S.anatomy) return [];
  const map = organAttentionMap(rows);
  return S.anatomy.organs.map((organ, index) => ({ organ, attention: map[organ.id], index }))
    .sort((a, b) => (a.attention.tier - b.attention.tier)
      || String(b.attention.latest || '').localeCompare(String(a.attention.latest || ''))
      || (a.index - b.index));
}

/** The same order inside one workspace: a list of variables, abnormal first. */
export function variablesByAttention(variables, rows) {
  const all = rows || indicatorRows();
  const index = new Map(all.map((r) => [r.variable, r]));
  return variables.slice().sort((a, b) => {
    const ra = index.get(a), rb = index.get(b);
    return ((ra ? ra.tier : TIER.unobserved) - (rb ? rb.tier : TIER.unobserved))
      || (ra && rb ? byTime(ra, rb) : 0)
      || variables.indexOf(a) - variables.indexOf(b);
  });
}

/** Workspace split for an organ or a function: primary / secondary / related variables plus cross-links. */
export function workspaceFor(kind, id) {
  if (!S.anatomy || !S.functions) return null;
  const groups = S.functions.groups;
  if (kind === 'organ') {
    const organ = S.anatomy.organs.find((o) => o.id === id);
    if (!organ) return null;
    const primaryId = S.functions.organ_primary_function[id];
    const fn = groups.find((g) => g.id === primaryId) || null;
    const own = new Set(organ.variables);
    const primary = fn ? fn.primary.filter((v) => own.has(v)) : [];
    const secondary = fn ? fn.secondary.filter((v) => own.has(v)) : [];
    const taken = new Set([...primary, ...secondary]);
    const related = organ.variables.filter((v) => !taken.has(v));
    const functions = groups.filter((g) => g.organs.some((o) => o.id === id)).map((g) => ({ id: g.id, label: g.label }));
    return { kind, id, label: organ.label, primary, secondary, related, organs: [{ id, label: organ.label }],
      functions, primaryFunction: fn ? { id: fn.id, label: fn.label } : null, mappedCount: organ.variables.length,
      relatedNote: related.length ? 'other variables attributed to this organ' : '' };
  }
  const fn = groups.find((g) => g.id === id);
  if (!fn) return null;
  return { kind, id, label: fn.label, primary: fn.primary, secondary: fn.secondary, related: fn.related,
    organs: fn.organs, functions: [], module: fn.module, mappedCount: fn.primary.length + fn.secondary.length,
    relatedNote: 'from other functional modules' };
}

function pushFeed(item) {
  S.feed.unshift(item);
  if (S.feed.length > FEED_CAP) S.feed.length = FEED_CAP;
}
const maxIso = (a, b) => (!a ? b : !b ? a : a > b ? a : b);

export function applyEvent(type, data, sequence) {
  if (sequence && sequence <= S.lastSequence) return null;
  if (sequence) S.lastSequence = sequence;
  const now = Date.now();
  switch (type) {
    case 'stream.state': {
      const { reason, ...state } = data;
      S.session = state;
      S.sessionStampedAt = now;
      if (reason === 'reset') {
        Object.assign(S, { live: { run: null, cells: [] }, anchor: null, variables: {}, series: {}, lastRunFailed: null, runInFlight: null,
          clock: { latest_clinical_event_time: null, latest_received_at: null, model_input_watermark: null, live_run_id: null } });
        // a reset empties the delivered timeline, so every organ goes back to neutral
        // at once rather than staying lit until the snapshot returns
        clearOrganState();
        pushFeed({ at: data.emitted_at, kind: 'state', title: 'Stream reset', detail: `epoch ${state.epoch} · ledger and run history retained` });
        return 'reset';
      }
      if (reason !== 'step') pushFeed({ at: data.emitted_at, kind: 'state', title: `Stream ${state.status}`, detail: `${state.speed}× · ${state.cursor}/${state.total_events} delivered` });
      return 'state';
    }
    case 'fhir.resource.accepted': {
      S.clock.latest_received_at = maxIso(S.clock.latest_received_at, data.received_at);
      if (S.session && typeof data.sequence === 'number') {
        S.session = { ...S.session, cursor: Math.max(S.session.cursor || 0, data.sequence),
          remaining_events: Math.max(0, (S.session.total_events || 0) - data.sequence) };
      }
      const res = data.resource || {};
      const codes = ((res.code || {}).coding || []).map((c) => c.code).join(', ');
      let title = `${res.resourceType} ${res.id} v${(res.meta || {}).versionId}`;
      let detail = `LOINC ${codes} · ${res.status}`;
      if (data.delivery_duplicate) { title = 'Duplicate delivery ignored'; detail = `${res.id} already delivered in this epoch`; }
      else if (data.ledger_duplicate) detail += ' · already in ledger (idempotent)';
      pushFeed({ at: data.received_at, kind: data.delivery_duplicate ? 'dup' : 'accepted', title, detail, resource_id: res.id, version_id: (res.meta || {}).versionId });
      return 'accepted';
    }
    case 'fhir.resource.rejected':
      pushFeed({ at: data.received_at, kind: 'warn', title: data.version_conflict ? 'Resource version conflict · not imported' : 'Resource rejected by validation', detail: (data.validation_issues || []).join('; ') });
      return 'rejected';
    case 'observation.normalized': {
      // Organ routing comes from the event. The backend already resolved it through
      // the approved organ map (oslian_demo/monitoring/routing.py); the browser never re-derives
      // an organ from a variable name.
      const organs = new Set(data.organ_ids || []);
      (data.values || []).forEach((v) => {
        const prev = S.variables[v.variable];
        const next = { variable: v.variable, value: v.value, unit: v.unit, effective_time: v.effective_time, received_at: data.received_at,
          resource_id: data.resource_id, version_id: data.version_id, count: (prev ? prev.count : 0) + (data.superseding ? 0 : 1), age_hours: 0,
          // what the observation says about itself travels with the event, exactly as on the snapshot
          input_value: v.input_value, input_unit: v.input_unit, derivation: v.derivation,
          interpretation: data.interpretation || [], reference_range: data.reference_range || [] };
        S.variables[v.variable] = prev && prev.effective_time > v.effective_time ? { ...prev, count: next.count } : next;
        const list = S.series[v.variable] || (S.series[v.variable] = []);
        const idx = list.findIndex((p) => p.resource_id === data.resource_id);
        const point = { effective_time: v.effective_time, value: v.value, unit: v.unit, resource_id: data.resource_id, version_id: data.version_id, received_at: data.received_at };
        if (idx >= 0) list[idx] = point; else list.push(point);
        list.sort((a, b) => (a.effective_time < b.effective_time ? -1 : a.effective_time > b.effective_time ? 1 : 0));
        S.clock.latest_clinical_event_time = maxIso(S.clock.latest_clinical_event_time, v.effective_time);
        S.freshVariables[v.variable] = now;
        (v.organ_ids || []).forEach((o) => organs.add(o));
      });
      // The organ's clock is the event's own effective time, not the browser's, so a
      // replay reproduces exactly the same organ states at the same stream position.
      const at = data.effective_time || data.received_at;
      organs.forEach((o) => {
        S.organEvents[o] = { at, sequence: sequence || 0 };
        if (organWithoutAsset(o)) S.unassetedOrganEvents[o] = (S.unassetedOrganEvents[o] || 0) + 1;
      });
      if (data.superseding && !(data.values || []).length) {
        pushFeed({ at: data.emitted_at, kind: 'warn', title: 'Resource retracted', detail: `${data.resource_id} v${data.version_id} entered-in-error · removed from timeline` });
        return 'superseded';
      }
      const v = (data.values || [])[0];
      if (v) pushFeed({ at: data.received_at, kind: 'normalized', title: `${v.variable} = ${fmtNum(v.value)} ${v.unit}`, detail: `${v.derivation}${data.superseding ? ' · amended version supersedes' : ''} · effective ${data.effective_time}`, resource_id: data.resource_id, version_id: data.version_id });
      return 'normalized';
    }
    case 'observation.unmapped':
      S.mappingFailures += 1;
      pushFeed({ at: data.emitted_at, kind: 'warn', title: `Kept in ledger, not modelled: ${data.reason}`, detail: `${data.resource_id} v${data.version_id} · ${data.detail}`, resource_id: data.resource_id, version_id: data.version_id });
      return 'unmapped';
    case 'model.run.started':
      S.runInFlight = data;
      S.runFlow = now;
      return 'run-started';
    case 'model.run.completed': {
      const { cells, ...run } = data;
      S.live = { run, cells: cells || [] };
      S.clock.model_input_watermark = run.input_cutoff;
      S.clock.live_run_id = run.run_id;
      S.runInFlight = null;
      S.lastRunFailed = null;
      S.freshRun = now;
      upsertRun({ ...run, status: 'completed' });
      pushFeed({ at: run.completed_at, kind: 'run', title: `Run completed · ${run.runtime_mode}`, detail: `cutoff ${run.input_cutoff} · ${run.input_resource_count} resources · model ${run.summary.model_cells} · persistence ${run.summary.persistence_cells} · missing ${run.summary.missing_cells} · degraded ${run.summary.degraded_cells}` });
      return 'run-completed';
    }
    case 'model.run.failed':
      S.runInFlight = null;
      S.lastRunFailed = data;
      upsertRun({ ...data, status: 'failed' });
      pushFeed({ at: data.completed_at, kind: 'warn', title: 'Run failed · fail closed', detail: data.error });
      return 'run-failed';
    default:
      return null;
  }
}

function upsertRun(run) {
  const idx = S.runs.findIndex((r) => r.run_id === run.run_id);
  if (idx >= 0) S.runs[idx] = { ...S.runs[idx], ...run };
  else S.runs.unshift(run);
}

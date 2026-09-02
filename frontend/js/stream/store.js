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
  view: 'anatomy',        // anatomy | functions | signals | cells
  selectedOrgan: 'kidneys',
  selectedFunction: 'renal_urinary',
  metric: null,           // variable id when the metric workspace is open
  chartWindow: 'recent',  // recent | full
  chartHorizons: null,    // null = all, else Set of horizon hours
  showSecondary: false,
  signalQuery: '',
  signalSort: 'organ',    // organ | name | latest | age
  evidenceOpen: false,
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
      pushFeed({ at: data.received_at, kind: 'warn', title: 'Resource rejected by validation', detail: (data.validation_issues || []).join('; ') });
      return 'rejected';
    case 'observation.normalized': {
      // Organ routing comes from the event. The backend already resolved it through
      // the approved organ map (oslian_demo/monitoring/routing.py); the browser never re-derives
      // an organ from a variable name.
      const organs = new Set(data.organ_ids || []);
      (data.values || []).forEach((v) => {
        const prev = S.variables[v.variable];
        const next = { variable: v.variable, value: v.value, unit: v.unit, effective_time: v.effective_time, received_at: data.received_at,
          resource_id: data.resource_id, version_id: data.version_id, count: (prev ? prev.count : 0) + (data.superseding ? 0 : 1), age_hours: 0 };
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

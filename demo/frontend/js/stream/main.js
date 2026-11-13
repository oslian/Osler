/* Monitoring page: boot, SSE wiring, controls, navigation, metric workspace, provenance.
 *
 * Boot: list sources -> create/get the session for the chosen mode
 * (idempotent) -> load discovery documents and the server snapshot -> open
 * SSE from the sequence the session reported, so a refresh continues without
 * replaying history the snapshot already reflects. */

import { connectEvents, getJSON, postJSON } from './api.js';
import { S, applyEvent, applySnapshotOrgans, organStateFrom, resetSessionState } from './store.js';
import { buildAnatomy } from './anatomy.js';
import { renderAll, renderDrawer, renderWorkspace, fmtClinical, fmtReceived, label } from './render.js';
import { openReport } from './report.js';
import { fmtNum } from './format.js';

let anatomyBuilt = null;
let channel = null;
let booted = null;
let raf = 0;
const REDUCED = !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);

/* Two render paths, because they answer to different clocks.
 *
 * A person clicking a tab must see it immediately, so a user action renders on
 * the next frame and nothing else. Stream events are a firehose - a 20x burst
 * delivers faster than anyone can read - so they are floored at STREAM_RENDER_MS
 * between full passes. Throttling here loses nothing: every event has already
 * been folded into the store by the time the render runs, so the next pass draws
 * the newest observations, organ states and forecast, just not once per event.
 * A trailing pass is always scheduled, so the last event of a burst is drawn. */
const STREAM_RENDER_MS = 120;
let renderTimer = 0;
let lastRenderAt = 0;

function paint() {
  raf = 0;
  renderTimer = 0;
  lastRenderAt = performance.now();
  renderAll(anatomyBuilt);
  restoreWardPosition();
}

function scheduleRender() {
  if (raf) return;
  raf = requestAnimationFrame(paint);
}

function scheduleStreamRender() {
  if (raf || renderTimer) return;
  const wait = STREAM_RENDER_MS - (performance.now() - lastRenderAt);
  if (wait <= 0) { scheduleRender(); return; }
  renderTimer = setTimeout(() => { renderTimer = 0; scheduleRender(); }, wait);
}

async function loadSnapshot() {
  const pid = S.session.patient_id;
  const q = `?session_id=${encodeURIComponent(S.sessionId)}`;
  const [state, forecasts, history, timeline] = await Promise.all([
    getJSON(`/api/patients/${pid}/state${q}`), getJSON(`/api/patients/${pid}/forecasts${q}`),
    getJSON(`/api/patients/${pid}/forecast-history${q}`), getJSON(`/api/patients/${pid}/timeline${q}`),
  ]);
  S.patient = state.patient;
  S.clock = state.clock;
  S.runtime = state.runtime;
  S.variables = state.variables;
  // the snapshot carries the organ routing for everything already delivered, so a
  // refresh or a jump back into a session restores the organ states instead of
  // starting dark and waiting for the next event
  applySnapshotOrgans(state.organs);
  S.mappingFailures = state.mapping_failures;
  S.session = state.session;
  S.sessionStampedAt = Date.now();
  S.series = timeline.series;
  S.live = forecasts.run ? { run: forecasts.run, cells: forecasts.cells } : { run: null, cells: [] };
  S.runs = history.runs;
  const latest = history.runs[0];
  S.lastRunFailed = latest && latest.status === 'failed' ? latest : null;
}

async function openSession(mode) {
  if (channel) { channel.close(); channel = null; }
  resetSessionState();
  S.mode = mode;
  // Throws on a rejected source key; openWardStay relies on that to report a
  // broken link instead of leaving the reader on a session that never opened.
  const session = await postJSON('/api/demo-streams', { source: mode });
  S.sessionId = session.session_id;
  S.session = session;
  S.sessionStampedAt = Date.now();
  S.lastSequence = session.last_event_sequence || 0;
  await loadSnapshot();
  openChannel();
  scheduleRender();
}

async function boot() {
  const params = hashParams();
  const [sources, capabilities, health, anatomy, functions, organLayout] = await Promise.all([
    getJSON('/api/demo-streams/sources'), getJSON('/api/model/capabilities'), getJSON('/api/model/health'),
    getJSON('/api/model/anatomy'), getJSON('/api/model/functions'),
    getJSON('/static/assets/anatomy-motion/organ-layout.json'),
  ]);
  S.sources = sources.sources;
  S.capabilities = capabilities;
  S.health = health;
  S.anatomy = anatomy;
  S.organLayout = organLayout;
  S.functions = functions;
  // A ward stay is opened as `retrospective:<case id>`; the sources listing only
  // carries the two base keys, so the prefixed form is matched on its base.
  const knownMode = (m) => S.sources.some((s) => s.source_key === String(m).split(':')[0]);
  if (params.mode && knownMode(params.mode)) S.mode = params.mode;
  // `signals` and `cells` were the two views Signals & forecasts replaced; an old link still lands
  const LEGACY_VIEWS = { signals: 'data', cells: 'data' };
  const view = LEGACY_VIEWS[params.view] || params.view;
  if (view && ['anatomy', 'functions', 'data', 'ward'].includes(view)) S.view = view;
  if (params.organ && anatomy.organs.some((o) => o.id === params.organ)) S.selectedOrgan = params.organ;
  if (params.function && functions.groups.some((g) => g.id === params.function)) S.selectedFunction = params.function;
  anatomyBuilt = buildAnatomy(document.getElementById('anatomyHost'), anatomy.organs, organLayout, selectOrgan);
  await openSession(S.mode);
  if (params.metric && anatomy.variable_units[params.metric]) S.metric = params.metric;
  if (params.run) await anchorRun(params.run);
  if (params.lower === '1') S.lowerOpen = true;
  if (params.patient) S.wardPatient = params.patient;
  if (S.view === 'ward') openWard();
  scheduleRender();
}

/* #stream?mode=retrospective&view=functions&function=renal_urinary&organ=kidneys&metric=creatinine&run=<id>
 * #stream?view=ward&patient=<ward row id>   -- a row links the same way an organ does
 * #stream?view=data&mode=retrospective:<case id>  -- one ward stay's own Signals & forecasts */
function hashParams() {
  const hash = String(window.location.hash || '');
  const query = hash.indexOf('?') >= 0 ? hash.slice(hash.indexOf('?') + 1) : '';
  const out = {};
  new URLSearchParams(query).forEach((value, key) => { out[key] = value; });
  return out;
}

function openChannel() {
  S.connection = 'connecting';
  channel = connectEvents(S.sessionId, S.lastSequence, {
    onStatus(status) { S.connection = status; scheduleRender(); },
    onEvent(type, data, sequence) {
      const hint = applyEvent(type, data, sequence);
      if (hint === 'reset' || hint === 'superseded') loadSnapshot().then(scheduleRender);
      scheduleStreamRender();
    },
  });
}

/* Fetched the first time the view is opened, not during boot: the ward shares
 * nothing with the session, so it must not delay the stream and must still work
 * when the runtime is unattached. */
async function loadWard() {
  try {
    S.wardError = null;
    S.ward = await getJSON('/api/ward');
  } catch (e) {
    S.ward = null;
    S.wardError = e.message;
  }
  scheduleRender();
}

function openWard() { if (!S.ward && !S.wardError) loadWard(); }

/* Leaving the ward keeps where the reader was. The board, the selected row and
 * the open bands stay in the store (nothing in a session reset touches them);
 * the scroll positions are noted here and put back on the ward's next paint. */
function rememberWardPosition() {
  if (S.view !== 'ward') return;
  const list = document.getElementById('railList');
  S.wardScroll = { rail: list ? list.scrollTop : 0, page: window.scrollY || 0 };
}
function restoreWardPosition() {
  if (S.view !== 'ward' || !S.wardScroll || !S.ward) return;
  const list = document.getElementById('railList');
  if (list) list.scrollTop = S.wardScroll.rail;
  window.scrollTo(0, S.wardScroll.page);
  S.wardScroll = null;
}
function switchView(next) {
  if (!next || next === S.view) { S.metric = null; scheduleRender(); return; }
  rememberWardPosition();
  S.view = next;
  S.metric = null;
  if (next === 'ward') openWard();
  scheduleRender();
}

/* Open one ward stay in the stream view.
 *
 * The board is left in the store, so going back to it costs no request and the
 * order is exactly where the reader left it. A case id nobody has answers 404,
 * and that surfaces as an error rather than silently opening somebody else. */
async function openWardStay(patientId) {
  if (!patientId) return;
  try {
    S.error = null;
    rememberWardPosition();
    await openSession(`retrospective:${patientId}`);
    S.view = 'anatomy';
    S.metric = null;
  } catch (e) {
    S.error = `Could not open this stay — ${e.message}`;
  }
  scheduleRender();
}

function selectOrgan(id) { S.selectedOrgan = id; S.view = 'anatomy'; S.metric = null; S.showSecondary = false; scheduleRender(); }
function selectFunction(id) { S.selectedFunction = id; S.view = 'functions'; S.metric = null; S.showSecondary = false; scheduleRender(); }

async function control(action, payload) {
  try {
    S.error = null;
    const state = await postJSON(`/api/demo-streams/${S.sessionId}/${action}`, payload);
    S.session = { ...S.session, ...state };
    S.sessionStampedAt = Date.now();
  } catch (e) { S.error = e.message; }
  scheduleRender();
}

async function anchorRun(runId) {
  const pid = S.session.patient_id;
  const q = `?session_id=${encodeURIComponent(S.sessionId)}&run_id=${encodeURIComponent(runId)}`;
  const [forecasts, timeline] = await Promise.all([getJSON(`/api/patients/${pid}/forecasts${q}`), getJSON(`/api/patients/${pid}/timeline${q}`)]);
  // Returning to the live head drops the anchor, and with it the run's organ state;
  // the live state was never touched, so it comes straight back.
  if (forecasts.anchor === 'live_head') S.anchor = null;
  else S.anchor = { run: forecasts.run, cells: forecasts.cells, series: timeline.series, variables: timeline.state,
    organEvents: organStateFrom(forecasts.organs, 'anchor') };
  S.drawer = null;
  scheduleRender();
}

async function openCellProvenance(cellName) {
  const cells = S.anchor ? S.anchor.cells : S.live.cells;
  const run = S.anchor ? S.anchor.run : S.live.run;
  const cell = cells.find((c) => c.cell === cellName);
  if (!cell || !run) return;
  S.drawer = { kind: 'cell', cell, run, history: null };
  renderDrawer();
  const history = await getJSON(`/api/patients/${S.session.patient_id}/forecast-history?session_id=${encodeURIComponent(S.sessionId)}&cell=${encodeURIComponent(cellName)}`);
  if (S.drawer && S.drawer.kind === 'cell' && S.drawer.cell.cell === cellName) {
    S.drawer.history = (history.cell_history || []).filter((h) => h.run_status === 'completed');
    renderDrawer();
  }
}

async function openObservationProvenance(variable, resourceId, versionId) {
  const vars = S.anchor ? S.anchor.variables : S.variables;
  const state = vars[variable];
  const rid = resourceId || (state && state.resource_id), vid = versionId || (state && state.version_id);
  if (!rid) return;
  S.drawer = { kind: 'observation', variable, raw: null };
  renderDrawer();
  const raw = await getJSON(`/api/fhir/Observation/${encodeURIComponent(rid)}/_history/${encodeURIComponent(vid)}`);
  if (S.drawer && S.drawer.kind === 'observation' && S.drawer.variable === variable) { S.drawer.raw = raw; renderDrawer(); }
}

/* chart tooltips: hover and keyboard focus alike; click opens provenance. The tip
 * is clamped inside the chart card on every side, so it never leaves the chart. */
function showTip(target) {
  const tip = document.getElementById('chartTip');
  const wrap = target.closest('.chart-wrap');
  if (!tip || !wrap) return;
  const d = target.dataset;
  const when = fmtClinical(new Date(Number(d.ms)).toISOString());
  const value = Number(d.value);
  const unit = S.anatomy ? (S.anatomy.variable_units[S.metric] || '') : '';
  if (d.kind === 'observed') {
    tip.innerHTML = `<b>${when}</b> · observed<br><span class="mono">${fmtNum(value)} ${unit}</span> · ${d.resource} v${d.version}${d.received ? '<br>received ' + fmtReceived(d.received) : ''}<br><u>click for FHIR provenance</u>`;
  } else {
    const what = d.status === 'persistence' ? 'baseline only · last value carried forward · not a model forecast' : 'model forecast';
    const basis = d.basisValue !== undefined ? `<br>basis ${fmtNum(Number(d.basisValue))} ${unit} at ${fmtClinical(new Date(Number(d.basisMs)).toISOString())}` : '';
    tip.innerHTML = `<b>${when}</b> · ${d.horizon}h ${what}<br><span class="mono">${fmtNum(value)} ${unit}</span> · ${d.source}${basis}<br>run ${String(d.run).slice(0, 8)} · <u>click for provenance</u>`;
  }
  const r = target.getBoundingClientRect(), w = wrap.getBoundingClientRect();
  tip.hidden = false;
  const tw = tip.offsetWidth, th = tip.offsetHeight;
  const left = Math.min(Math.max(4, r.left - w.left + r.width / 2 - tw / 2), Math.max(4, w.width - tw - 4));
  const above = r.top - w.top - th - 10;
  tip.style.left = `${left}px`;
  tip.style.top = `${above >= 4 ? above : Math.min(r.bottom - w.top + 10, Math.max(4, w.height - th - 4))}px`;
}
function hideTip() { const tip = document.getElementById('chartTip'); if (tip) tip.hidden = true; }

document.addEventListener('click', (e) => {
  const pt = e.target.closest ? e.target.closest('.chart-pt') : null;
  if (pt) {
    if (pt.dataset.kind === 'observed') openObservationProvenance(S.metric, pt.dataset.resource, pt.dataset.version);
    else openCellProvenance(pt.dataset.cell);
    return;
  }
  const t = e.target.closest ? e.target.closest('[data-act]') : null;
  if (!t || !S.sessionId) return;
  const act = t.getAttribute('data-act');
  if (act === 'start') control('start');
  else if (act === 'pause') control('pause');
  else if (act === 'step') control('step');
  else if (act === 'reset') control('reset');
  else if (act === 'speed') control('speed', { speed: Number(t.getAttribute('data-speed')) });
  else if (act === 'mode') { const m = t.getAttribute('data-mode'); if (m !== S.mode) openSession(m); }
  else if (act === 'view') switchView(t.getAttribute('data-view'));
  else if (act === 'select-ward-patient') { S.wardPatient = t.getAttribute('data-patient'); scheduleRender(); }
  else if (act === 'open-ward-stay') openWardStay(t.getAttribute('data-patient'));
  else if (act === 'ward-band') {
    const key = t.getAttribute('data-band');
    S.wardOpenBands = { ...S.wardOpenBands, [key]: t.getAttribute('aria-expanded') !== 'true' };
    scheduleRender();
  }
  else if (act === 'select-organ') selectOrgan(t.getAttribute('data-organ'));
  else if (act === 'select-function') selectFunction(t.getAttribute('data-function'));
  else if (act === 'goto-organ') selectOrgan(t.getAttribute('data-organ'));
  else if (act === 'goto-function') selectFunction(t.getAttribute('data-function'));
  else if (act === 'open-metric') { S.metric = t.getAttribute('data-variable'); S.chartHorizons = null; scheduleRender(); }
  else if (act === 'close-metric') { S.metric = null; scheduleRender(); }
  else if (act === 'chart-window') { S.chartWindow = t.getAttribute('data-window'); scheduleRender(); }
  else if (act === 'chart-horizon') {
    const h = t.getAttribute('data-horizon');
    if (h === 'all') S.chartHorizons = null;
    else { const set = new Set(S.chartHorizons || []); const n = Number(h); if (set.has(n)) set.delete(n); else set.add(n); S.chartHorizons = set.size ? set : null; }
    scheduleRender();
  }
  else if (act === 'toggle-secondary') { S.showSecondary = !S.showSecondary; scheduleRender(); }
  else if (act === 'toggle-evidence') { S.evidenceOpen = !S.evidenceOpen; scheduleRender(); }
  else if (act === 'toggle-run-details') { S.runDetailsOpen = !S.runDetailsOpen; scheduleRender(); }
  else if (act === 'toggle-lower') { S.lowerOpen = !S.lowerOpen; scheduleRender(); }
  else if (act === 'data-sort') { S.dataSort = t.getAttribute('data-sort'); scheduleRender(); }
  else if (act === 'forecast-horizon') { S.forecastHorizon = Number(t.getAttribute('data-horizon')); scheduleRender(); }
  // the report is a snapshot: opening it freezes the data, and only Refresh re-takes it
  else if (act === 'report-open') { S.report = openReport(null); scheduleRender(); }
  else if (act === 'report-refresh') { S.report = openReport(S.report); scheduleRender(); }
  else if (act === 'report-close') { S.report = null; scheduleRender(); }
  else if (act === 'report-print') window.print();
  else if (act === 'report-horizon') { S.report = { ...S.report, horizon: Number(t.getAttribute('data-horizon')) }; scheduleRender(); }
  // one row unfolded at a time: opening another closes the last, so the list never balloons
  else if (act === 'data-expand') { const v = t.getAttribute('data-variable'); S.dataExpanded = S.dataExpanded === v ? null : v; scheduleRender(); }
  else if (act === 'data-unobserved') { S.dataShowUnobserved = !S.dataShowUnobserved; scheduleRender(); }
  else if (act === 'data-other') { S.dataShowOther = !(t.getAttribute('aria-expanded') === 'true'); scheduleRender(); }
  else if (act === 'anchor-run') anchorRun(t.getAttribute('data-run'));
  else if (act === 'anchor-live') { S.anchor = null; S.drawer = null; scheduleRender(); }
  else if (act === 'prov-cell') openCellProvenance(t.getAttribute('data-cell'));
  else if (act === 'prov-obs') openObservationProvenance(t.getAttribute('data-variable'));
  else if (act === 'drawer-close') { S.drawer = null; renderDrawer(); }
});
/* The report builder's own controls: radios, checkboxes and the picker. They
 * change the report the reader is looking at, never the frozen snapshot in it. */
document.addEventListener('change', (e) => {
  const t = e.target.closest ? e.target.closest('[data-act]') : null;
  if (!t || !S.report) return;
  const act = t.getAttribute('data-act');
  if (act === 'report-scope') { S.report = { ...S.report, scope: t.value }; scheduleRender(); }
  else if (act === 'report-toggle') { S.report = { ...S.report, [t.getAttribute('data-section')]: t.checked }; scheduleRender(); }
  else if (act === 'report-pick') {
    const v = t.getAttribute('data-variable');
    const picked = new Set(S.report.indicators);
    if (t.checked) picked.add(v); else picked.delete(v);
    S.report = { ...S.report, indicators: [...picked] };
    scheduleRender();
  }
});

document.addEventListener('input', (e) => {
  const t = e.target.closest ? e.target.closest('[data-act]') : null;
  if (!t) return;
  const act = t.getAttribute('data-act');
  if (act === 'data-query') {
    S.dataQuery = t.value;
    const host = document.getElementById('workspace');
    const pos = t.selectionStart;
    renderWorkspace();
    const again = host && host.querySelector('[data-act="data-query"]');
    if (again) { again.focus(); try { again.setSelectionRange(pos, pos); } catch (_) { /* ignore */ } }
  } else if (act === 'scrub-run') {
    const runs = S.runs.filter((r) => r.status === 'completed' && (!S.session || r.epoch === S.session.epoch)).slice().reverse();
    const run = runs[Number(t.value)];
    if (!run) return;
    if (run.run_id === (S.live.run && S.live.run.run_id)) { S.anchor = null; scheduleRender(); }
    else anchorRun(run.run_id);
  }
});
document.addEventListener('mouseover', (e) => { const pt = e.target.closest ? e.target.closest('.chart-pt') : null; if (pt) showTip(pt); });
document.addEventListener('mouseout', (e) => { const pt = e.target.closest ? e.target.closest('.chart-pt') : null; if (pt) hideTip(); });
document.addEventListener('focusin', (e) => { const pt = e.target.closest ? e.target.closest('.chart-pt') : null; if (pt) showTip(pt); });
document.addEventListener('focusout', (e) => { const pt = e.target.closest ? e.target.closest('.chart-pt') : null; if (pt) hideTip(); });
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    if (S.drawer) { S.drawer = null; renderDrawer(); }
    else if (S.report) { S.report = null; scheduleRender(); }
    else if (S.metric) { S.metric = null; scheduleRender(); }
    else if (S.dataExpanded) { S.dataExpanded = null; scheduleRender(); }
    return;
  }
  const pt = e.target.closest ? e.target.closest('.chart-pt') : null;
  if (pt && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); pt.click(); return; }
  // a table row that acts as a button (Signals & forecasts) opens on Enter or Space too
  const rowBtn = e.target.closest ? e.target.closest('[role="button"][data-act]') : null;
  if (rowBtn && rowBtn.tagName !== 'BUTTON' && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); rowBtn.click(); }
});

// the chart is drawn at the workspace's real width, so a resize redraws it
let resizeTimer = 0;
window.addEventListener('resize', () => {
  if (!S.metric) return;
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => { hideTip(); scheduleRender(); }, 120);
});

// while running, the simulated clock and ages advance between events
setInterval(() => { if (S.session && S.session.status === 'running' && !REDUCED) scheduleRender(); }, 1000);

window.OslianStream = {
  open() {
    if (!booted) {
      booted = boot().catch((e) => {
        S.error = e.message;
        const host = document.getElementById('streamStrip');
        if (host) host.innerHTML = `<div class="notice"><span class="n">Stream unavailable</span><span class="d">${String(e.message).replace(/[<>]/g, '')}</span></div>`;
      });
    }
    return booted;
  },
  state: S,
  // the built figure, so a QA driver can read organ state without scraping the DOM
  get anatomy() { return anatomyBuilt; },
};
if (document.body.getAttribute('data-view') === 'stream') window.OslianStream.open();

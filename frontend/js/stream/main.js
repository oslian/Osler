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
  if (params.mode && S.sources.some((s) => s.source_key === params.mode)) S.mode = params.mode;
  if (params.view && ['anatomy', 'functions', 'signals', 'cells'].includes(params.view)) S.view = params.view;
  if (params.organ && anatomy.organs.some((o) => o.id === params.organ)) S.selectedOrgan = params.organ;
  if (params.function && functions.groups.some((g) => g.id === params.function)) S.selectedFunction = params.function;
  anatomyBuilt = buildAnatomy(document.getElementById('anatomyHost'), anatomy.organs, organLayout, selectOrgan);
  await openSession(S.mode);
  if (params.metric && anatomy.variable_units[params.metric]) S.metric = params.metric;
  if (params.run) await anchorRun(params.run);
  if (params.lower === '1') S.lowerOpen = true;
  scheduleRender();
}

/* #stream?mode=retrospective&view=functions&function=renal_urinary&organ=kidneys&metric=creatinine&run=<id> */
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

/* chart tooltips: hover and keyboard focus alike; click opens provenance */
function showTip(target) {
  const tip = document.getElementById('chartTip');
  const wrap = target.closest('.chart-wrap');
  if (!tip || !wrap) return;
  const d = target.dataset;
  const when = fmtClinical(new Date(Number(d.ms)).toISOString());
  const value = Number(d.value);
  const unit = S.anatomy ? (S.anatomy.variable_units[S.metric] || '') : '';
  tip.innerHTML = d.kind === 'observed'
    ? `<b>${when}</b> · observed<br><span class="mono">${value} ${unit}</span> · ${d.resource} v${d.version}${d.received ? '<br>received ' + fmtReceived(d.received) : ''}<br><u>click for FHIR provenance</u>`
    : `<b>${when}</b> · ${d.horizon}h ${d.status === 'persistence' ? 'last value carried forward · not a model prediction' : 'model forecast'}<br><span class="mono">${value} ${unit}</span> · ${d.source}<br>run ${String(d.run).slice(0, 8)} · <u>click for provenance</u>`;
  const r = target.getBoundingClientRect(), w = wrap.getBoundingClientRect();
  tip.hidden = false;
  const left = Math.min(Math.max(8, r.left - w.left + r.width / 2 - 110), w.width - 230);
  tip.style.left = `${left}px`;
  tip.style.top = `${Math.max(4, r.top - w.top - tip.offsetHeight - 10)}px`;
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
  else if (act === 'view') { S.view = t.getAttribute('data-view'); S.metric = null; scheduleRender(); }
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
  else if (act === 'toggle-lower') { S.lowerOpen = !S.lowerOpen; scheduleRender(); }
  else if (act === 'signal-sort') { S.signalSort = t.getAttribute('data-sort'); scheduleRender(); }
  else if (act === 'anchor-run') anchorRun(t.getAttribute('data-run'));
  else if (act === 'anchor-live') { S.anchor = null; S.drawer = null; scheduleRender(); }
  else if (act === 'prov-cell') openCellProvenance(t.getAttribute('data-cell'));
  else if (act === 'prov-obs') openObservationProvenance(t.getAttribute('data-variable'));
  else if (act === 'drawer-close') { S.drawer = null; renderDrawer(); }
});
document.addEventListener('input', (e) => {
  const t = e.target.closest ? e.target.closest('[data-act]') : null;
  if (!t) return;
  const act = t.getAttribute('data-act');
  if (act === 'signal-query') {
    S.signalQuery = t.value;
    const host = document.getElementById('workspace');
    const pos = t.selectionStart;
    renderWorkspace();
    const again = host && host.querySelector('[data-act="signal-query"]');
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
    else if (S.metric) { S.metric = null; scheduleRender(); }
    return;
  }
  const pt = e.target.closest ? e.target.closest('.chart-pt') : null;
  if (pt && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); pt.click(); }
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

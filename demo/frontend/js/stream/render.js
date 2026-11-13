/* DOM rendering for the monitoring page. Pure functions of the store; event
 * wiring lives in main.js. Wording rules: the mode ("synthetic", "not a live
 * feed" / "retrospective replay") stays on screen; persistence, missing and
 * degraded are data boundaries, never alerts; nothing is drawn that is not a
 * ledger row or a persisted run. */

import { S, catalogCellsForVariable, cellsForVariable, clinicalNow, currentCells, currentRun, currentSeries,
  currentOrganEvents, currentVariables, indicatorRows, observationAttention, onsetMs, organAttentionMap,
  organWithoutAsset, organsByAttention, sourceInfo, timeAxis, trajectoryFor, variablesByAttention,
  workspaceFor } from './store.js';
import { chartLegend, makeTimeFormatter, sparkline, trajectoryChart } from './chart.js';
import { renderWardRail, renderWardWorkspace } from './ward.js';
import { renderDataView } from './signals.js';
import { mountReportAnatomy, renderReport } from './report.js';
import { esc, fmtNum, DATA_VIEW, FLAG, FORECAST_LABEL, NO_FORECAST_MODEL, OBSERVED_HISTORY,
  INTERVAL_NOTE, PROJECTED, RANGE, STATUS_MARK, TRAJECTORY, UPDATING_FORECAST, flagOf, forecastLabel,
  fmtForecastValue, fmtInterval, forecastNote, isAbstained, unavailableReason } from './format.js';

const el = (id) => document.getElementById(id);
const LABELS = {
  heart_rate: 'Heart rate', map: 'Mean arterial pressure', o2sat: 'SpO₂', respiratory_rate: 'Respiratory rate',
  temperature: 'Temperature', glucose: 'Glucose', creatinine: 'Creatinine', bun: 'BUN', urine_output: 'Urine output',
  potassium: 'Potassium', sodium: 'Sodium', chloride: 'Chloride', bicarbonate: 'Bicarbonate', hemoglobin: 'Hemoglobin',
  hematocrit: 'Hematocrit', platelets: 'Platelets', wbc: 'White cells', lactate: 'Lactate', anion_gap: 'Anion gap',
  ph: 'Arterial pH', alt: 'ALT', ast: 'AST', alkaline_phos: 'Alkaline phosphatase', bilirubin: 'Total bilirubin',
  bilirubin_direct: 'Direct bilirubin', inr: 'INR', ptt: 'aPTT', bnp: 'BNP', troponin_i: 'Troponin I', ck_mb: 'CK-MB',
  cpk: 'Creatine kinase', ldh: 'LDH', crp: 'CRP', crp_hs: 'hs-CRP', esr: 'ESR', tsh: 'TSH', free_t4: 'Free T4',
  uric_acid: 'Uric acid', ionized_calcium: 'Ionized calcium', serum_osmolality: 'Serum osmolality',
  total_protein: 'Total protein', hypoxemia_event: 'Hypoxemia event (SpO₂ ≤ 92%)',
};
export const label = (v) => LABELS[v] || String(v || '').replace(/_/g, ' ').replace(/^./, (c) => c.toUpperCase());
const SOURCE_LABELS = { frozen_whole_body_jepa: 'frozen JEPA', unified_shared_jepa_target_head: 'shared target head',
  multisource_shared_jepa_target_head: 'multi-source head', calibrated_event_risk_head: 'event-risk head',
  patient_persistence_anchor: 'persistence anchor', stale_target_abstained: 'stale observation',
  event_risk_abstained: 'event head abstained', none: 'none', missing: 'missing' };
const STATUS_WORDS = FORECAST_LABEL;

const fmtTime = () => makeTimeFormatter(timeAxis(), onsetMs());
export function fmtClinical(iso) { return iso ? fmtTime().short(new Date(iso).getTime()) : null; }
export function fmtReceived(iso) {
  if (!iso) return null;
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  const p = (n) => String(n).padStart(2, '0');
  return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
}
function ageHours(effective, now) {
  if (!effective || !now) return null;
  return (new Date(now).getTime() - new Date(effective).getTime()) / 3600000;
}
export function fmtAge(h) {
  if (h === null || h === undefined) return '—';
  if (h < 1) return `${Math.round(h * 60)} min`;
  return `${h.toFixed(1)} h`;
}
const statusChip = (status) => `<span class="status ${esc(status)}"><i></i>${esc(STATUS_WORDS[status] || status)}</span>`;
/** A cell's badge, read from its status and its reason: a shared head that abstained
 *  is "carried forward — model abstained", styled as every other baseline, never as a model. */
export const cellChip = (c) => `<span class="status ${esc(c.status)}"${isAbstained(c) ? ' data-abstained="true"' : ''}><i></i>${esc(forecastLabel(c))}</span>`;
const unitOf = (v) => (S.anatomy && S.anatomy.variable_units[v]) || '';
const isRetro = () => sourceInfo().mode === 'retrospective';

/* ── command bar ────────────────────────────────────────────────────────── */
export function renderCommandBar() {
  const host = el('streamStrip');
  if (!host) return;
  const s = S.session || {};
  const src = sourceInfo();
  const rt = S.runtime || {};
  const conn = S.connection;
  let statusWord = { running: 'Running', paused: 'Paused', idle: 'Idle', completed: 'Completed' }[s.status] || s.status || 'Idle';
  let statusClass = s.status === 'running' ? 'ok' : 'na';
  if (conn === 'reconnecting' || conn === 'connecting') { statusWord = conn === 'connecting' ? 'Connecting' : 'Reconnecting'; statusClass = 'warn'; }
  else if (conn === 'closed') { statusWord = 'Disconnected'; statusClass = 'bad'; }
  const speed = Number(s.speed || 1);
  /* Which speeds exist, and which of them a reader should see, is the source's
   * statement about itself - not a rule written into this page. A simulated
   * fixture offers demo speeds; a stress speed is declared separately and lives
   * behind a disclosure so nobody mistakes it for a clinical rate; a real
   * hospital feed would declare no speeds at all and this renders nothing. */
  const playback = src.playback || {};
  const speedButton = (opt) => `<button type="button" data-act="speed" data-speed="${opt.value}"`
    + ` aria-pressed="${speed === Number(opt.value)}"${opt.note ? ` title="${esc(opt.note)}"` : ''}>`
    + `${esc(opt.label)}</button>`;
  const speedOptions = playback.speeds || [];
  const speedGroup = speedOptions.length
    ? `<div class="segmented" role="group" aria-label="Demo speed">${speedOptions.map(speedButton).join('')}</div>`
    : '';
  const devOptions = playback.developer_speeds || [];
  const developerSpeeds = devOptions.length ? `
      <details class="dev-tools"${devOptions.some((o) => speed === Number(o.value)) ? ' open' : ''}>
        <summary>Developer / QA</summary>
        <div class="dev-body">
          <div class="segmented" role="group" aria-label="Stress test speed">${devOptions.map(speedButton).join('')}</div>
          <p class="note">${esc(devOptions[0].note || '')} It exercises the ingest, scheduler and SSE
             path under load; it says nothing about how fast a hospital delivers data.</p>
        </div>
      </details>` : '';
  const counts = S.live.run ? S.live.run.summary : null;
  const runtimeTag = rt.runtime_attached
    ? '<span class="tag">runtime attached · bounded research demo</span>'
    : rt.runtime_mode === 'release'
      ? '<span class="tag caution">runtime not attached · release validation failed</span>'
      : '<span class="tag caution">runtime not attached · gated</span>';
  const bounded = rt.runtime_attached
    ? 'bounded research demo · research-only · 90% intervals only where calibrated · no clinical alert · not a formal or clinically validated model'
    : 'research-only · no clinical alert';
  const output = S.lastRunFailed
    ? '<span class="status degraded"><i></i>runtime failure · no current forecast</span>'
    : counts
      ? `<span class="counts"><span class="m" title="predicted by the model">model forecast ${counts.model_cells}</span>`
        + `<span class="p" title="not a model prediction: the last observed value, repeated">carried forward ${counts.persistence_cells}</span>`
        + (counts.missing_cells ? `<span class="x" title="the model returned no value">no forecast ${counts.missing_cells}</span>` : '')
        + (counts.degraded_cells ? `<span class="d" title="the model runtime is not attached">withheld ${counts.degraded_cells}</span>` : '')
        + '</span>'
      : '<span class="meta">no forecast yet</span>';
  const retro = isRetro();
  const modeTag = retro ? '<span class="tag graphite">Retrospective replay</span>' : '<span class="tag">Synthetic FHIR R4</span>';
  const context = retro && src.context
    ? `${esc(src.provenance || '')} · ${esc(src.context.age || '')} y · ${esc(src.context.unit_type || '')} · not live monitoring`
    : 'Not a live Epic or hospital feed · synthetic fixture · no diagnosis, causal claim or clinical alert';
  const rows = retro && src.rows ? src.rows : null;
  const revealed = rows ? rows.filter((r) => (s.cursor || 0) >= rows.slice(0, r.row).reduce((n, x) => n + x.variables.length, 0)).length : null;
  const clocks = retro
    ? `<div class="clock"><div class="k">Replay clock · case time</div><div class="v${S.clock.latest_clinical_event_time ? '' : ' na'}">${esc(fmtClinical(S.clock.latest_clinical_event_time) || 'none yet')}</div></div>
       <div class="clock"><div class="k">Revealed</div><div class="v">${revealed} of ${rows.length} observations</div></div>
       <div class="clock"><div class="k">Causal anchor · watermark</div><div class="v${S.clock.model_input_watermark ? '' : ' na'}">${esc(fmtClinical(S.clock.model_input_watermark) || 'no run yet')}</div></div>
       <div class="clock"><div class="k">Withheld</div><div class="v">${rows.length - revealed} future row${rows.length - revealed === 1 ? '' : 's'}</div></div>`
    : `<div class="clock"><div class="k">Latest clinical event</div><div class="v${S.clock.latest_clinical_event_time ? '' : ' na'}">${esc(fmtClinical(S.clock.latest_clinical_event_time) || 'none yet')}</div></div>
       <div class="clock"><div class="k">Last received</div><div class="v${S.clock.latest_received_at ? '' : ' na'}">${esc(fmtReceived(S.clock.latest_received_at) || 'none yet')}</div></div>
       <div class="clock"><div class="k">Model input watermark</div><div class="v${S.clock.model_input_watermark ? '' : ' na'}">${esc(fmtClinical(S.clock.model_input_watermark) || 'no run yet')}</div></div>
       <div class="clock"><div class="k">Delivered / remaining</div><div class="v">${esc(s.cursor ?? 0)} / ${esc(s.remaining_events ?? '—')} · ${esc(fmtSim(simElapsed(s)))} at ${esc(speed)}×</div></div>`;
  host.innerHTML = `
    <div class="cmd-id">
      <div class="cmd-title"><h1>${esc(src.display_name || (S.patient ? S.patient.display : 'Patient'))}</h1>${modeTag}<span class="stat ${statusClass}"><i></i>${esc(statusWord)}${s.status === 'running' ? ' · ' + esc(speed) + '×' : ''}</span>${S.runInFlight ? `<span class="stat na run-flow"><i></i>${UPDATING_FORECAST}</span>` : ''}</div>
    </div>
    <div class="cmd-clocks" aria-label="Clocks">${clocks}</div>
    <div class="cmd-transport">
      <div class="transport">
        ${s.status === 'running'
          ? '<button class="btn sm primary" type="button" data-act="pause">❚❚ Pause</button>'
          : `<button class="btn sm primary" type="button" data-act="start"${s.status === 'completed' ? ' disabled' : ''}>▶ ${s.status === 'paused' ? 'Resume' : 'Play'}</button>`}
        ${playback.step === false ? '' : `<button class="btn sm" type="button" data-act="step"${s.status === 'completed' || (s.remaining_events || 0) === 0 ? ' disabled' : ''}>${esc(src.step_label || 'Next event')} ▸</button>`}
        <button class="btn sm" type="button" data-act="reset">Reset</button>
        ${speedGroup}
      </div>
      ${developerSpeeds}
      <div class="queue">epoch ${esc(s.epoch ?? '—')} · session <span class="mono">${esc((s.session_id || '').slice(0, 8))}</span> · SSE ${esc(conn)}</div>
    </div>
    <div class="cmd-sub"><span class="cmd-runtime"><span class="mono">oslian/Model ${esc((rt.model_commit || '').slice(0, 7) || '—')}</span> ${runtimeTag} ${output}</span><span class="cmd-context">${context}</span><span class="cmd-bounded">${esc(bounded)}</span></div>`;
}
function simElapsed(s) {
  const base = Number(s.sim_elapsed_seconds) || 0;
  if (s.status !== 'running' || !S.sessionStampedAt) return base;
  return base + ((Date.now() - S.sessionStampedAt) / 1000) * (Number(s.speed) || 1);
}
function fmtSim(seconds) {
  const s = Math.max(0, Math.floor(Number(seconds) || 0));
  const p = (n) => String(n).padStart(2, '0');
  return `${p(Math.floor(s / 60))}:${p(s % 60)}`;
}

/* ── anchor band + view bar ─────────────────────────────────────────────── */
export function renderAnchorBand() {
  const host = el('anchorBand');
  if (!host) return;
  if (!S.anchor) { host.hidden = true; host.innerHTML = ''; return; }
  const run = S.anchor.run;
  host.hidden = false;
  // A historical snapshot is one run's frozen inputs and cells. The stream keeps
  // flowing into the live state underneath; nothing here follows it.
  host.innerHTML = `<div class="anchor-main"><span class="eyebrow">${TRAJECTORY.snapshot}</span>
    <span class="anchor-line">Frozen at <span class="mono">${esc(fmtClinical(run.input_cutoff))}</span> · Later observations are not included</span>
    <span class="anchor-detail">Historical anchor · not the live head · run <span class="mono">${esc(run.run_id.slice(0, 8))}</span> · epoch ${esc(run.epoch)} · ${esc(run.input_resource_count)} resources · ${esc(run.runtime_mode)} · signals, charts and cells are frozen at this cutoff</span></div>
    <button class="btn sm" type="button" data-act="anchor-live">Return to live head</button>`;
}

export function renderViewBar() {
  const host = el('viewBar');
  if (!host) return;
  const modes = S.sources.length ? S.sources : [{ source_key: 'synthetic', label: 'Live synthetic stream' }, { source_key: 'retrospective', label: 'Retrospective replay' }];
  const views = [['anatomy', 'Anatomy'], ['functions', 'Functions'], ['data', DATA_VIEW.title], ['ward', 'Ward order']];
  // A ward stay is its own source key (`retrospective:<case id>`). The bar names it,
  // so a reader can see whose data every view below is showing.
  const src = sourceInfo();
  const stay = String(S.mode).includes(':')
    ? `<button type="button" data-act="mode" data-mode="${esc(S.mode)}" data-stay="${esc(src.case_id || '')}" aria-pressed="true">Ward stay · ${esc(src.display_name || String(S.mode).split(':')[1])}</button>`
    : '';
  host.innerHTML = `
    <div class="segmented sans" role="group" aria-label="Data mode">${modes.map((m) => `<button type="button" data-act="mode" data-mode="${esc(m.source_key)}" aria-pressed="${S.mode === m.source_key}">${esc(m.label)}</button>`).join('')}${stay}</div>
    <div class="segmented sans" role="group" aria-label="View">${views.map(([k, l]) => `<button type="button" data-act="view" data-view="${k}" aria-pressed="${S.view === k}">${l}</button>`).join('')}</div>
    <span class="meta">${S.anchor ? 'historical snapshot' : 'live head'} · epoch ${esc((S.session || {}).epoch ?? '—')}${isRetro() ? ' · same timeline, chart and runtime contract as the live stream' : ''}</span>
    <button class="disc right" type="button" data-act="toggle-lower" aria-expanded="${S.lowerOpen ? 'true' : 'false'}">History &amp; events ${S.lowerOpen ? '⌃' : '⌄'}</button>`;
}

/* ── per-organ / per-function summaries ─────────────────────────────────── */
/** Has the stream touched this organ inside the live window, on the stream's own clock? */
function freshOrgan(id) {
  const ev = currentOrganEvents()[id];
  const now = clinicalNow();
  if (!ev || !now) return false;
  const ageH = (Date.parse(now) - Date.parse(ev.at)) / 3600000;
  return ageH >= 0 && ageH <= 2;
}

export function organSummary() {
  const byOrgan = {};
  if (!S.anatomy) return byOrgan;
  const vars = currentVariables();
  const cells = currentCells();
  const attention = organAttentionMap();
  S.anatomy.organs.forEach((o) => {
    const organCells = o.variables.flatMap((v) => cellsForVariable(v, cells));
    byOrgan[o.id] = { label: o.label, attention: attention[o.id], signals: o.variables.filter((v) => vars[v]).length, mapped: o.variables.length,
      cells: o.cells.length, modelCells: organCells.filter((c) => c.status === 'model').length,
      boundaryCells: organCells.filter((c) => c.status !== 'model').length,
      fresh: freshOrgan(o.id), noAsset: organWithoutAsset(o.id) };
  });
  return byOrgan;
}
function functionSummary(g) {
  const vars = currentVariables();
  const own = [...g.primary, ...g.secondary];
  const cells = own.flatMap((v) => cellsForVariable(v, currentCells()));
  return { observed: own.filter((v) => vars[v]).length, mapped: own.length, cells: g.cells.length,
    model: cells.filter((c) => c.status === 'model').length, fresh: own.some((v) => (S.freshVariables[v] || 0) > Date.now() - 1200) };
}

/* ── rail ────────────────────────────────────────────────────────────────── */
export function renderRail() {
  const main = el('streamMain');
  const list = el('railList');
  const head = el('railHead');
  if (!main || !list) return;
  main.setAttribute('data-view', S.view);
  main.classList.toggle('report-open', !!S.report);
  main.classList.toggle('metric-open', !!S.metric);
  main.classList.toggle('frozen', !!S.anchor);
  main.classList.toggle('paused', !!(S.session && S.session.status !== 'running'));
  if (S.report) { head.innerHTML = ''; list.innerHTML = ''; return; }
  if (S.view === 'anatomy') {
    const summary = organSummary();
    const rows = indicatorRows();
    const ordered = organsByAttention(rows);
    const needing = ordered.filter((o) => o.attention.abnormal).length;
    head.innerHTML = `<h2 class="h-sec">Anatomy</h2><span class="meta">${S.metric ? esc(summary[S.selectedOrgan] ? summary[S.selectedOrgan].label : '')
      : (needing ? `${needing} organ${needing === 1 ? '' : 's'} with a result outside its supplied range, first` : 'anatomical organs only · no result is outside a supplied range')}</span>`;
    list.innerHTML = ordered.map(({ organ: o, attention: a }) => {
      const s = summary[o.id];
      const noArt = s.noAsset ? ' <span class="noart" title="no detailed vector asset exists for this organ; its events are still recorded">no visual asset</span>' : '';
      // what put this organ at the top, in the indicator list's own words
      const names = (ids) => ids.map((v) => label(v)).join(' · ');
      const why = a.abnormal
        ? `<span class="why">${names(a.above.map((v) => v))}${a.above.length ? ' high' : ''}${a.above.length && a.below.length ? ' · ' : ''}${names(a.below)}${a.below.length ? ' low' : ''}</span>`
        : a.projected ? `<span class="why">${names(a.projectedIds)} projected outside</span>`
          : '';
      const line = a.abnormal
        ? `<span class="n attn">${a.abnormal} observed result${a.abnormal === 1 ? '' : 's'} outside range</span>`
        : a.projected ? `<span class="n proj">${a.projected} projected outside range</span>`
          : `<span class="n">${s.signals}/${s.mapped} signal${s.mapped === 1 ? '' : 's'}</span>`;
      return `<li><button type="button" data-act="select-organ" data-organ="${esc(o.id)}" aria-pressed="${S.selectedOrgan === o.id}"
        class="organ-row ${s.signals ? 'has' : 'dim'}" data-attention="${esc(a.state)}"><i></i>
        <span class="t">${esc(o.label)}${a.abnormal ? '<b class="needs">Needs attention</b>' : ''}${!a.abnormal && a.fresh ? ' <span class="new">● new</span>' : ''}${noArt}</span>
        ${line}${why}</button></li>`;
    }).join('');
  } else if (S.view === 'functions') {
    head.innerHTML = `<h2 class="h-sec">Clinical functions</h2><span class="meta">${S.metric ? '' : '11 modules from the release checkpoint · demo-approved primary/related split v1'}</span>`;
    list.innerHTML = S.functions ? S.functions.groups.map((g) => {
      const s = functionSummary(g);
      const pct = s.mapped ? Math.round((s.observed / s.mapped) * 100) : 0;
      return `<li><button type="button" data-act="select-function" data-function="${esc(g.id)}" aria-pressed="${S.selectedFunction === g.id}" class="fn ${s.observed ? 'has' : 'dim'}"><span class="t">${esc(g.label)}${s.fresh ? ' <span class="new">● new</span>' : ''}</span><span class="n">${s.observed}/${s.mapped} · ${s.cells ? s.cells + ' cells' : 'no cell'}</span><span class="bar"><i style="width:${pct}%"></i></span></button></li>`;
    }).join('') : '';
  } else if (S.view === 'ward') {
    renderWardRail(head, list);
  } else {
    head.innerHTML = '';
    list.innerHTML = '';
  }
}

/* ── workspace (organ / function) ───────────────────────────────────────── */
function signalTile(v, opts = {}) {
  const vars = currentVariables();
  const state = vars[v];
  const unit = unitOf(v);
  const row = (opts.rows || []).find((r) => r.variable === v) || null;
  const cells = catalogCellsForVariable(v);
  const liveCells = cellsForVariable(v, currentCells());
  const horizons = cells.map((c) => c.horizon_hours).sort((a, b) => a - b);
  const fcText = !horizons.length ? NO_FORECAST_MODEL : (() => {
    const statuses = new Set(liveCells.map((c) => c.status));
    const word = statuses.size === 0 ? UPDATING_FORECAST
      : statuses.has('model') ? (statuses.size === 1 ? FORECAST_LABEL.model : `${FORECAST_LABEL.model} + carried forward`)
      : statuses.has('persistence') ? FORECAST_LABEL.persistence
      : statuses.has('degraded') ? FORECAST_LABEL.degraded : FORECAST_LABEL.missing;
    return `${horizons.map((h) => h + 'h').join(' · ')} · <span class="${statuses.has('model') ? 'm' : 'p'}">${esc(word)}</span>`;
  })();
  if (!state) {
    return `<div class="tile empty${opts.compact ? ' compact' : ''}"><div class="t">${esc(label(v))}</div><div class="v na">no observation yet</div><div class="d">Mapped · unit ${esc(unit)} · nothing delivered in this stream</div><div class="fc">${fcText}</div></div>`;
  }
  const age = ageHours(state.effective_time, clinicalNow());
  const maxAge = S.capabilities ? S.capabilities.forward_fill_max_age_hours : 6;
  const stale = age !== null && age > maxAge;
  const fresh = (S.freshVariables[v] || 0) > Date.now() - 900 && !S.anchor;
  const series = currentSeries()[v] || [];
  // the same flag, the same words and the same deep red as the indicator list
  const key = row ? flagOf(row) : 'unsupplied';
  const flag = FLAG[key];
  const abnormal = !!row && row.attention.kind === 'outside';
  const projected = !!row && !abnormal && row.forecasts.some((f) => f.projected);
  const flagLine = `<div class="tile-flag flag-${key}"><i class="mk" aria-hidden="true">${flag.mark}</i>${flag.text}`
    + `${row && row.range ? ` <span class="why">${esc(fmtNum(row.range.low))}–${esc(fmtNum(row.range.high))} ${esc(row.range.unit)}</span>` : ''}</div>`
    + (projected ? `<div class="tile-proj"><i class="mk" aria-hidden="true">${PROJECTED[row.projectedSide].mark}</i>${PROJECTED[row.projectedSide].text}</div>` : '');
  return `<button type="button" class="tile${fresh ? ' fresh' : ''}${opts.compact ? ' compact' : ''}${abnormal ? ' abnormal' : ''}${projected ? ' projected' : ''}" data-act="open-metric" data-variable="${esc(v)}" aria-label="Open ${esc(label(v))} chart, ${esc(flag.full)}">
    ${fresh ? '<span class="newv">● new value</span>' : ''}
    <div class="t">${esc(label(v))}</div>
    <div class="v">${esc(fmtNum(state.value))}<small>${esc(state.unit)}</small></div>
    ${flagLine}
    <div class="d">${esc(fmtClinical(state.effective_time))} · age <span class="${stale ? 'stale' : ''}">${esc(fmtAge(age))}${stale ? ' · beyond ' + maxAge + ' h forward-fill' : ''}</span> · ${esc(state.count)} obs</div>
    <div class="spark-label">${OBSERVED_HISTORY}</div>
    ${sparkline(series.map((p) => ({ value: p.value })), 240, 44)}
    <div class="fc">${fcText}</div>
    <span class="open">Open chart →</span>
  </button>`;
}
function signalRow(v, rows) {
  const row = (rows || []).find((r) => r.variable === v) || null;
  const state = currentVariables()[v];
  const unit = unitOf(v);
  const cells = catalogCellsForVariable(v);
  const liveCells = cellsForVariable(v, currentCells());
  const horizons = cells.map((c) => c.horizon_hours).sort((a, b) => a - b).map((h) => h + 'h').join(' · ') || 'no runtime cell';
  const status = liveCells.length ? statusChip(liveCells.some((c) => c.status === 'model') ? 'model' : liveCells[0].status) : (cells.length ? `<span class="note">${UPDATING_FORECAST}</span>` : `<span class="note">${NO_FORECAST_MODEL} · observations only</span>`);
  if (!state) {
    return `<div class="row dim"><div><div class="t">${esc(label(v))}</div><div class="d">mapped · unit ${esc(unit)} · nothing delivered in this stream</div></div><div></div><div class="v na">no observation</div><div class="fc">${esc(horizons)}</div><div class="fc">${status}</div></div>`;
  }
  const age = ageHours(state.effective_time, clinicalNow());
  const fresh = (S.freshVariables[v] || 0) > Date.now() - 900 && !S.anchor;
  const key = row ? flagOf(row) : 'unsupplied';
  const abnormal = !!row && row.attention.kind === 'outside';
  return `<button type="button" class="row${fresh ? ' fresh' : ''}${abnormal ? ' abnormal' : ''}" data-act="open-metric" data-variable="${esc(v)}" aria-label="Open ${esc(label(v))} chart, ${esc(FLAG[key].full)}">
    <div><div class="t">${esc(label(v))}</div><div class="d"><span class="flag flag-${key}"><i class="mk" aria-hidden="true">${FLAG[key].mark}</i>${FLAG[key].text}</span> · ${esc(fmtClinical(state.effective_time))} · age ${esc(fmtAge(age))}</div></div>
    ${sparkline((currentSeries()[v] || []).map((p) => ({ value: p.value })), 96, 24)}
    <div class="v">${esc(fmtNum(state.value))}<small> ${esc(state.unit)}</small></div>
    <div class="fc">${esc(horizons)}</div>
    <div class="fc">${status}</div>
  </button>`;
}

function evidenceTable(variables, title) {
  const cells = variables.flatMap((v) => cellsForVariable(v, currentCells()));
  const run = currentRun();
  const body = !run ? '<p class="empty-note">No model run yet.</p>'
    : !cells.length ? '<p class="empty-note">No target × horizon cell in the pinned registry is attributed here.</p>'
    : `<div class="scroll-x"><table class="cells-table"><thead><tr><th>Cell</th><th>Observed</th><th>Forecast</th><th>90% interval</th><th>Status</th><th>Source</th><th></th></tr></thead><tbody>${cells.map((c) => cellRow(c)).join('')}</tbody></table></div>${runFooter(run)}`;
  return `<div class="evidence${S.evidenceOpen ? ' open' : ''}">
    <button class="h" type="button" data-act="toggle-evidence" aria-expanded="${S.evidenceOpen}"><b>Forecast evidence · ${esc(title)}</b><span class="note">${cells.length} cell${cells.length === 1 ? '' : 's'}${run ? ` · run <span class="mono">${esc(run.run_id.slice(0, 8))}</span> · cutoff ${esc(fmtClinical(run.input_cutoff))}` : ''}</span><span class="x">${S.evidenceOpen ? 'Hide' : 'Show'} cells ${S.evidenceOpen ? '⌃' : '⌄'}</span></button>
    ${S.evidenceOpen ? body : ''}
  </div>`;
}

export function renderWorkspace() {
  const host = el('workspace');
  if (!host) return;
  if (S.report) { renderReport(host); mountReportAnatomy(); return; }
  if (S.metric) { renderMetric(host); return; }
  if (S.view === 'data') { renderDataView(host); return; }
  if (S.view === 'ward') { renderWardWorkspace(host); return; }
  const spec = S.view === 'functions' ? workspaceFor('function', S.selectedFunction) : workspaceFor('organ', S.selectedOrgan);
  if (!spec) { host.innerHTML = '<p class="empty-note">Loading…</p>'; return; }
  const rows = indicatorRows();
  const vars = currentVariables();
  const observed = [...spec.primary, ...spec.secondary, ...spec.related].filter((v) => vars[v]).length;
  const cells = [...spec.primary, ...spec.secondary].flatMap((v) => catalogCellsForVariable(v)).length;
  const links = spec.kind === 'organ'
    ? (spec.functions.length ? `Functions: ${spec.functions.map((f) => `<button class="disc" type="button" data-act="goto-function" data-function="${esc(f.id)}">${esc(f.label)}</button>`).join(' · ')}` : 'No functional group attributes a variable to this organ')
    : (spec.organs.length ? `Anatomy: ${spec.organs.map((o) => `<button class="disc" type="button" data-act="goto-organ" data-organ="${esc(o.id)}">${esc(o.label)}</button>`).join(' · ')}` : '');
  const secondaryBlock = spec.secondary.length
    ? `<button class="disc" type="button" data-act="toggle-secondary" aria-expanded="${S.showSecondary}">${S.showSecondary ? 'Hide' : 'Show'} ${spec.secondary.length} more mapped signal${spec.secondary.length === 1 ? '' : 's'} ${S.showSecondary ? '⌃' : '⌄'}</button>${S.showSecondary ? `<div class="rows">${variablesByAttention(spec.secondary, rows).map((v) => signalRow(v, rows)).join('')}</div>` : ''}`
    : '';
  host.innerHTML = `
    <div class="ws-h"><h2>${esc(spec.label)}</h2><span class="sub">${observed} of ${spec.primary.length + spec.secondary.length + spec.related.length} mapped signals observed · ${cells ? cells + ' forecast cell' + (cells === 1 ? '' : 's') : 'observations only · no runtime cell'}${S.anchor ? ' · frozen at historical anchor' : ''}${spec.module ? ` · module <span class="mono">${esc(spec.module)}</span>` : ''}</span><span class="fn">${links}</span></div>
    <div class="sec">Primary indicators${spec.primaryFunction ? ` · <span class="soft">${esc(spec.primaryFunction.label)}</span>` : ''}</div>
    ${spec.primary.length ? `<div class="tiles">${variablesByAttention(spec.primary, rows).map((v) => signalTile(v, { rows })).join('')}</div>` : '<p class="empty-note">No primary indicator is attributed here in the approved map. Nothing is invented.</p>'}
    ${secondaryBlock}
    ${spec.related.length ? `<div class="sec">Related indicators · <span class="soft">${esc(spec.relatedNote)}</span></div><div class="rows">${variablesByAttention(spec.related, rows).map((v) => signalRow(v, rows)).join('')}</div>` : ''}
    ${evidenceTable([...spec.primary, ...spec.secondary], spec.label)}
    <p class="foot">Grouping is navigation and explanation only; it never states that an organ or function is normal or abnormal. Persistence is a baseline fallback, not a prediction · 90% intervals only where the release calibrated the cell · causal_claim_allowed: false.</p>`;
}

/* ── metric workspace (level 3) ─────────────────────────────────────────── */
/** Signed change of a forecast against the observation its cell actually used.
 *  No basis, no delta: nothing is guessed from a neighbouring point. */
function deltaText(f) {
  if (!f.basis || !Number.isFinite(f.forecast_point) || !Number.isFinite(f.basis.value)) return '';
  const d = f.forecast_point - f.basis.value;
  const sign = d > 0 ? '+' : d < 0 ? '−' : '±';
  // the difference carries the same precision as the values it is taken from
  const decimals = Math.max(...[f.forecast_point, f.basis.value].map((v) => (fmtNum(v).split('.')[1] || '').length));
  const abs = fmtNum(Math.abs(d), decimals);
  const pct = f.basis.value !== 0 ? ` (${sign}${(Math.abs(d) / Math.abs(f.basis.value) * 100).toFixed(1)}%)` : '';
  return `${sign}${abs}${pct}`;
}

/** Last observed against each horizon, from the same run the chart draws. */
function comparisonSummary(t, v) {
  const unit = unitOf(v);
  const basis = t.forecasts.map((f) => f.basis).find(Boolean);
  const lastPoint = t.points.length ? t.points[t.points.length - 1] : null;
  const observed = basis || lastPoint;
  const observedNote = basis ? 'the observation this run used' : (lastPoint ? 'latest in view' : '');
  const rows = [];
  rows.push(`<div class="cmp-row head"><span class="k">Last observed</span><span class="v mono">${observed ? `${esc(fmtNum(observed.value))} <small>${esc(unit)}</small>` : '—'}</span><span class="d">${observed ? esc(`${fmtClinical(new Date(observed.ms).toISOString())} · ${observedNote}`) : 'no observation yet'}</span></div>`);
  if (t.kind === 'no-run') rows.push(`<div class="cmp-row"><span class="k">Forecast</span><span class="v na">${TRAJECTORY.noRun}</span><span class="d"></span></div>`);
  else if (t.kind === 'none') rows.push(`<div class="cmp-row"><span class="k">Forecast</span><span class="v na">${TRAJECTORY.noModel}</span><span class="d">observations only</span></div>`);
  else t.forecasts.forEach((f) => {
    if (f.status === 'model' && Number.isFinite(f.forecast_point)) {
      rows.push(`<div class="cmp-row" data-status="model"><span class="k">${f.horizon_hours}h forecast</span><span class="v mono">${esc(fmtNum(f.forecast_point))} <small>${esc(unit)}</small></span><span class="d mono">${f.basis ? esc(deltaText(f)) : 'no basis · delta not computed'}</span></div>`);
    } else if (f.status === 'persistence' && Number.isFinite(f.forecast_point)) {
      rows.push(`<div class="cmp-row" data-status="persistence"${isAbstained(f) ? ' data-abstained="true"' : ''}><span class="k">${f.horizon_hours}h baseline</span><span class="v mono">${esc(fmtNum(f.forecast_point))} <small>${esc(unit)}</small></span><span class="d">${isAbstained(f) ? `${esc(forecastLabel(f))} · ` : ''}${TRAJECTORY.baselineOnly}</span></div>`);
    } else {
      rows.push(`<div class="cmp-row" data-status="${esc(f.status)}"><span class="k">${f.horizon_hours}h</span><span class="v na">${TRAJECTORY.unavailable}</span><span class="d">${esc(unavailableReason(f.status, f.fallback_reason))}</span></div>`);
    }
  });
  return `<section class="compare" aria-label="Comparison summary"><h3>Comparison summary</h3>${rows.join('')}<p class="note">Change is measured against the observation each forecast cell used. It is a difference, not a clinical judgement. A persistence baseline repeats that observation and is not a model forecast.</p></section>`;
}

/** The chart's pixel width: the workspace minus its own and the card's padding, so
 *  the SVG draws at true text size instead of scaling a fixed box. */
function chartWidth(host) {
  const cs = window.getComputedStyle(host);
  const inner = host.clientWidth - parseFloat(cs.paddingLeft || 0) - parseFloat(cs.paddingRight || 0);
  const narrow = window.innerWidth < 768;
  return Math.max(280, Math.floor(inner - (narrow ? 18 : 34)));
}

export function renderMetric(host) {
  const v = S.metric;
  const state = currentVariables()[v];
  const run = currentRun();
  const t = trajectoryFor(v, S.chartHorizons);
  const fresh = (S.freshVariables[v] || 0) > Date.now() - 900 && !S.anchor && t.points.length ? t.points[t.points.length - 1].ms : null;
  const width = chartWidth(host);
  const height = Math.round(Math.min(380, Math.max(300, width * 0.32)));
  const headline = S.anchor ? TRAJECTORY.snapshot : TRAJECTORY.liveHead;
  const chart = trajectoryChart({ traj: t, unit: unitOf(v), label: label(v), axis: timeAxis(), onset_ms: onsetMs(),
    window: S.chartWindow, fresh_ms: fresh, width, height, headline });
  const back = S.view === 'data' ? DATA_VIEW.title
    : S.view === 'functions' ? `${esc((workspaceFor('function', S.selectedFunction) || {}).label || 'Function')}`
      : `${esc((workspaceFor('organ', S.selectedOrgan) || {}).label || 'Organ')}`;
  const cellsHere = cellsForVariable(v, currentCells());
  const age = state ? ageHours(state.effective_time, clinicalNow()) : null;
  const runs = S.runs.filter((r) => r.status === 'completed' && (!S.session || r.epoch === S.session.epoch)).slice().reverse();
  const anchorIdx = S.anchor ? runs.findIndex((r) => r.run_id === S.anchor.run.run_id) : runs.length - 1;
  const source = run
    ? `${OBSERVED_HISTORY} from the ledger · forecast cells from run <span class="mono">${esc(run.run_id.slice(0, 8))}</span> · cutoff ${esc(fmtClinical(run.input_cutoff))}`
    : `${OBSERVED_HISTORY} from the ledger · ${TRAJECTORY.noRun}`;
  const modeNote = S.anchor ? "frozen at this run's cutoff" : (S.session && S.session.status === 'running' ? 'live follow' : 'paused: not following');
  const badge = `<span class="head-badge ${S.anchor ? 'snapshot' : 'live'}" data-head="${S.anchor ? 'snapshot' : 'live'}">${headline}</span>`;
  // what the observation says about its own value, in the same words as the data view
  const att = observationAttention(state);
  const attKey = att.kind === 'outside' ? (att.side || 'outside') : att.kind === 'within' ? 'within' : att.kind === 'unit' ? 'unit' : att.kind === 'unsupplied' ? 'unsupplied' : null;
  const attLine = attKey ? `<span class="att-line st-${attKey}"><i class="mk" aria-hidden="true">${STATUS_MARK[attKey]}</i>${RANGE[attKey]}${att.range ? ` · ${esc(fmtNum(att.range.low))}–${esc(fmtNum(att.range.high))} ${esc(att.range.unit)} supplied with the observation` : ''}</span>` : '';
  host.innerHTML = `
    <div class="crumb"><button class="disc" type="button" data-act="close-metric">← ${back}</button><span>›</span><span>${esc(label(v))}</span></div>
    <div class="metric-h">
      <div><h2>${esc(label(v))}</h2><div class="note">${esc(unitOf(v))}${cellsHere.length ? ` · ${cellsHere.length} forecast cell${cellsHere.length === 1 ? '' : 's'}` : ` · ${NO_FORECAST_MODEL} · observations only`}</div>${attLine}</div>
      ${state ? `<div class="big">${esc(fmtNum(state.value))}<small>${esc(state.unit)}</small></div>
      <dl class="kv"><dt>Effective</dt><dd>${esc(fmtClinical(state.effective_time))}</dd><dt>Received</dt><dd>${esc(fmtReceived(state.received_at) || '—')}</dd><dt>Age</dt><dd>${esc(fmtAge(age))}</dd><dt>Observations</dt><dd>${esc(state.count)}</dd></dl>` : '<div class="note">No observation yet for this signal.</div>'}
    </div>
    <section class="traj" aria-label="${TRAJECTORY.title}">
      <div class="traj-head">
        <div><h3>${TRAJECTORY.title}</h3><span class="note">${source} · ${esc(modeNote)}</span></div>
        ${badge}
      </div>
      <div class="chart-wrap">${chart.empty ? '<p class="empty-note">Nothing to plot yet: no observation of this signal has been delivered. Nothing is imputed.</p>' : chart.svg}<div class="chart-tip" id="chartTip" role="status" hidden></div></div>
      ${chart.empty ? '' : chartLegend(t)}
      <p class="foot">Every observed point is a ledger row; every forecast point is a persisted run. ${chart.window_points < chart.total_points ? `Showing ${chart.window_points} of ${chart.total_points} points (recent window).` : ''}</p>
    </section>
    <div class="traj-below">
      ${comparisonSummary(t, v)}
      <section class="traj-controls" aria-label="Chart controls">
        <div class="chips"><span class="note">Window</span>${[['recent', 'recent 12h'], ['full', 'full stream']].map(([k, l]) => `<button class="chip" type="button" data-act="chart-window" data-window="${k}" aria-pressed="${S.chartWindow === k}">${l}</button>`).join('')}</div>
        ${t.catalogHorizons.length ? `<div class="chips"><span class="note">Horizons</span><button class="chip" type="button" data-act="chart-horizon" data-horizon="all" aria-pressed="${!S.chartHorizons}">all</button>${t.catalogHorizons.map((h) => `<button class="chip" type="button" data-act="chart-horizon" data-horizon="${h}" aria-pressed="${!!(S.chartHorizons && S.chartHorizons.has(h))}">${h}h</button>`).join('')}</div>` : ''}
        ${runs.length > 1 ? `<div class="scrub"><label for="runScrub" class="note">Run history</label><input id="runScrub" type="range" min="0" max="${runs.length - 1}" value="${Math.max(0, anchorIdx)}" data-act="scrub-run" aria-label="Select a past model run as the historical snapshot"><span class="mono note">${anchorIdx >= 0 ? esc(fmtClinical(runs[anchorIdx].input_cutoff)) : ''} · ${S.anchor ? 'historical snapshot' : 'live head'}</span></div>` : ''}
        ${S.anchor ? '<button class="chip" type="button" data-act="anchor-live">Return to live head</button>' : ''}
      </section>
    </div>
    <section class="run-details${S.runDetailsOpen ? ' open' : ''}" aria-label="Run details and provenance">
      <button class="h" type="button" data-act="toggle-run-details" aria-expanded="${S.runDetailsOpen}"><b>Run details &amp; provenance</b><span class="note">${run ? `run <span class="mono">${esc(run.run_id.slice(0, 8))}</span> · ${esc(run.runtime_mode)} · ${cellsHere.length} cell${cellsHere.length === 1 ? '' : 's'}` : TRAJECTORY.noRun}</span><span class="x">${S.runDetailsOpen ? 'Hide' : 'Show'} ${S.runDetailsOpen ? '⌃' : '⌄'}</span></button>
      ${S.runDetailsOpen ? `<div class="run-details-body">
        <div class="side"><h3>Forecast cells · ${esc(label(v))}</h3>${cellsHere.length ? cellsHere.map((c) => `<div class="hz"><span class="h">${c.horizon_hours}h</span>${cellChip(c)}<span class="pt${c.forecast_point === null ? ' null' : ''}">${esc(fmtForecastValue(c))}${fmtInterval(c) ? ` <small>[${esc(fmtInterval(c))}]</small>` : ''}</span><button class="disc" type="button" data-act="prov-cell" data-cell="${esc(c.cell)}">provenance</button>${c.target !== v ? `<span class="note derived">${esc(label(c.target))} · derived event, not drawn on this axis</span>` : ''}</div>`).join('') : `<p class="note">${catalogCellsForVariable(v).length ? UPDATING_FORECAST + ' No run yet for this session.' : NO_FORECAST_MODEL + ': this signal has no target × horizon cell in the pinned registry, so only observations are shown.'}</p>`}
          ${cellsHere.some((c) => c.fallback_reason) ? `<p class="note">${esc(Array.from(new Set(cellsHere.map((c) => c.fallback_reason).filter(Boolean))).join(' · '))}</p>` : ''}
          <p class="note">${INTERVAL_NOTE}</p></div>
        ${run ? `<div class="side"><h3>Run</h3><dl class="kv"><dt>Run</dt><dd class="mono">${esc(run.run_id.slice(0, 8))}</dd><dt>Cutoff</dt><dd>${esc(fmtClinical(run.input_cutoff))}</dd><dt>Input</dt><dd>${esc(run.input_resource_count)} resources</dd><dt>Commit</dt><dd class="mono">${esc((run.model_commit || '').slice(0, 7))}</dd><dt>Runtime</dt><dd>${esc(run.runtime_mode)}</dd><dt>Duration</dt><dd>${esc(run.duration_ms ?? '—')} ms</dd></dl><p class="note">causal_claim_allowed: false</p></div>` : ''}
        ${state ? `<div class="side"><h3>Latest observation</h3><dl class="kv"><dt>Resource</dt><dd class="mono">${esc(state.resource_id)} v${esc(state.version_id)}</dd></dl><button class="disc" type="button" data-act="prov-obs" data-variable="${esc(v)}">FHIR provenance ⌄</button></div>` : ''}
      </div>` : ''}
    </section>`;
}

/* ── forecast evidence rows (organ and function workspaces) ─────────────── */
function cellRow(c, extra = '') {
  const point = c.forecast_point === null || c.forecast_point === undefined ? '<span class="nullv">null</span>' : `<span class="pt">${esc(fmtForecastValue(c))}</span>`;
  const range = fmtInterval(c);
  const interval = range ? `<span class="pt">${esc(range)}</span>` : `<span class="nullv" title="${esc(c.interval_status || '')}">—</span>`;
  const observed = c.observed_value === null || c.observed_value === undefined ? '<span class="nullv">—</span>' : esc(fmtNum(c.observed_value));
  const age = c.measurement_age_hours === null || c.measurement_age_hours === undefined ? '' : ` <span class="why">age ${esc(fmtAge(c.measurement_age_hours))}</span>`;
  return `<tr><td class="cell">${esc(c.cell)}</td><td class="num">${observed} <span class="why">${esc(c.unit || '')}</span>${age}</td><td class="num">${point}</td><td class="num">${interval}</td><td>${cellChip(c)}<div class="why">${esc(forecastNote(c))}${c.fallback_reason ? ` · ${esc(c.fallback_reason)}` : ''}</div></td><td class="src" title="${esc(c.source_model)}">${esc(SOURCE_LABELS[c.source_model] || c.source_model)}</td>${extra}<td><button class="disc" type="button" data-act="prov-cell" data-cell="${esc(c.cell)}">Provenance ⌄</button></td></tr>`;
}
function runFooter(run) {
  const note = run.runtime_mode === 'fixture'
    ? 'Runtime not attached: every cell is degraded with a null point. Nothing is fabricated.'
    : 'Persistence baseline = the patient\'s latest observed value; not a model prediction and not clinical validation. A 90% interval appears only where the release calibrated that cell; event cells are calibrated probabilities.';
  return `<p class="cells-foot">Run <span class="mono">${esc(run.run_id.slice(0, 8))}</span> · cutoff ${esc(fmtClinical(run.input_cutoff))} · ${esc(run.input_resource_count)} resources · commit <span class="mono">${esc((run.model_commit || '').slice(0, 7))}</span> · ${esc(note)} · causal_claim_allowed: false.</p>`;
}

/* ── lower panel: run history + feed ────────────────────────────────────── */
export function renderLower() {
  const host = el('lowerPanel');
  if (!host) return;
  host.hidden = !S.lowerOpen;
  if (!S.lowerOpen) return;
  const liveId = S.live.run ? S.live.run.run_id : null;
  const runs = S.runs.map((r) => {
    const anchored = S.anchor && S.anchor.run.run_id === r.run_id;
    const failed = r.status === 'failed';
    const sum = r.summary ? `model ${r.summary.model_cells} · persistence ${r.summary.persistence_cells} · missing ${r.summary.missing_cells} · degraded ${r.summary.degraded_cells}` : (failed ? esc(r.error || 'failed') : r.status);
    return `<li><button type="button" data-act="anchor-run" data-run="${esc(r.run_id)}" aria-pressed="${anchored ? 'true' : 'false'}"${failed ? ' disabled' : ''}><span class="mono">${esc(fmtReceived(r.completed_at || r.started_at) || '—')}</span><span>cutoff ${esc(fmtClinical(r.input_cutoff))} · epoch ${esc(r.epoch)}${failed ? ' · <span class="failed">failed</span>' : ''}</span><span>${r.run_id === liveId ? '<span class="live">live head</span>' : ''}</span><span class="d">${sum} · ${esc(r.runtime_mode)} · ${esc(r.input_resource_count ?? '')} resources</span></button></li>`;
  }).join('');
  const feed = S.feed.map((f) => `<li class="${esc(f.kind)}"><span class="mono">${esc(fmtReceived(f.at) || '')}</span><b>${esc(f.title)}</b><span class="d">${esc(f.detail || '')}</span></li>`).join('');
  host.innerHTML = `
    <section aria-label="Model run history"><div class="panel-h"><h2 class="h-sec">Model run history</h2><span class="meta">${S.runs.length} run${S.runs.length === 1 ? '' : 's'} persisted · select one to anchor the view</span></div>${S.runs.length ? `<ol class="runs">${runs}</ol>` : '<p class="empty-note">No run yet.</p>'}</section>
    <section aria-label="Recent stream events"><div class="panel-h"><h2 class="h-sec">Stream events</h2><span class="meta">newest first · ${esc(S.mappingFailures)} kept-but-unmapped</span></div>${feed ? `<ol class="feed">${feed}</ol>` : '<p class="empty-note">Waiting for the first event.</p>'}</section>`;
}

/* ── provenance drawer ──────────────────────────────────────────────────── */
export function renderDrawer() {
  const host = el('provDrawer');
  const body = el('provBody');
  if (!host || !body) return;
  if (!S.drawer) { host.setAttribute('data-open', '0'); body.innerHTML = ''; return; }
  host.setAttribute('data-open', '1');
  const d = S.drawer;
  const rt = S.runtime || {};
  if (d.kind === 'cell') {
    const c = d.cell, run = d.run;
    body.innerHTML = `<div class="prov"><div class="h-sec">Cell ${esc(c.cell)}</div>
      <dl class="kv"><dt>Status</dt><dd>${cellChip(c)}</dd><dt>Forecast</dt><dd class="mono">${esc(fmtForecastValue(c))}${c.target === 'hypoxemia_event' ? '' : ` ${esc(c.unit || '')}`}</dd><dt>Observed</dt><dd class="mono">${esc(c.observed_value === null ? '—' : fmtNum(c.observed_value))} ${esc(c.unit || '')} · age ${esc(fmtAge(c.measurement_age_hours))} · ${esc(c.observations)} obs</dd><dt>Source model</dt><dd class="mono">${esc(c.source_model)}</dd><dt>Fallback reason</dt><dd>${esc(c.fallback_reason || 'none')}</dd><dt>90% interval</dt><dd>${fmtInterval(c) ? `<span class="mono">${esc(fmtInterval(c))} ${esc(c.unit || '')}</span> · ` : 'none · '}${esc(c.interval_status)}</dd><dt>Causal claim</dt><dd class="mono">causal_claim_allowed: ${esc(String(c.causal_claim_allowed))}</dd></dl>
      <div class="h-sec">Run</div>
      <dl class="kv"><dt>Run id</dt><dd class="mono">${esc(run.run_id)}</dd><dt>Input cutoff</dt><dd class="mono">${esc(run.input_cutoff)}</dd><dt>Watermark</dt><dd class="mono">${esc(run.input_watermark || '—')}</dd><dt>Input</dt><dd class="mono">${esc(run.input_resource_count)} resources · sha256 ${esc((run.input_hash || '').slice(0, 16))}…</dd><dt>Model commit</dt><dd class="mono">${esc(run.model_commit)}</dd><dt>Registry</dt><dd class="mono">${esc(run.registry_sha256)}</dd><dt>Runtime</dt><dd>${esc(run.runtime_mode)} · attached ${esc(String(rt.runtime_attached))}</dd><dt>Duration</dt><dd class="mono">${esc(run.duration_ms ?? '—')} ms</dd></dl>
      <div class="h-sec">Point history for ${esc(c.cell)}</div>
      ${d.history ? `<ol class="hist">${d.history.map((h) => `<li><span class="mono">${esc(fmtClinical(h.input_cutoff))}</span><span>${statusChip(h.status)}</span><span class="mono">${esc(h.forecast_point === null ? 'null' : fmtNum(h.forecast_point))}</span></li>`).join('')}</ol>` : '<p class="note">Loading…</p>'}</div>`;
  } else if (d.kind === 'observation') {
    const raw = d.raw, res = raw ? raw.resource : null;
    const coding = res ? ((res.code || {}).coding || []).map((c) => `${c.system} | ${c.code}${c.display ? ' · ' + c.display : ''}`).join('<br>') : '';
    const canon = raw && raw.canonical && raw.canonical[0];
    body.innerHTML = `<div class="prov"><div class="h-sec">${esc(label(d.variable))}</div>
      ${res ? `<dl class="kv"><dt>Resource</dt><dd class="mono">${esc(res.resourceType)}/${esc(res.id)} · version ${esc((res.meta || {}).versionId)} · ${esc(res.status)}</dd><dt>Coding</dt><dd class="mono">${coding}</dd><dt>Effective</dt><dd class="mono">${esc(res.effectiveDateTime || (res.effectivePeriod ? res.effectivePeriod.start + ' → ' + res.effectivePeriod.end : '—'))}</dd><dt>Issued</dt><dd class="mono">${esc(res.issued || '—')}</dd><dt>Source lastUpdated</dt><dd class="mono">${esc((res.meta || {}).lastUpdated || '—')}</dd><dt>Received</dt><dd class="mono">${esc(raw.first_received_at)}</dd>${canon ? `<dt>Unit conversion</dt><dd class="mono">${esc(fmtNum(canon.input_value))} ${esc(canon.input_unit)} → ${esc(fmtNum(canon.value))} ${esc(canon.unit)} · ${esc(canon.derivation)} · ${esc(canon.mapping_version)}</dd>` : ''}</dl>
      <div class="h-sec">Raw FHIR resource (unmodified)</div><pre>${esc(JSON.stringify(res, null, 1))}</pre>` : '<p class="note">Loading…</p>'}</div>`;
  }
}

export function renderAll(anatomyBuilt) {
  renderCommandBar();
  renderAnchorBand();
  renderViewBar();
  renderRail();
  renderWorkspace();
  renderLower();
  renderDrawer();
  // A boundary cell only says something about one organ while the runtime is actually
  // producing forecasts. With the runtime unattached every cell is degraded, which is a
  // property of the deployment, already stated in the header, and not a per-organ signal.
  const run = currentRun();
  const producing = !!(run && run.summary && (run.summary.model_cells > 0 || run.summary.persistence_cells > 0));
  if (anatomyBuilt) anatomyBuilt.update({ selected: S.selectedOrgan, byOrgan: organSummary(),
    organEvents: currentOrganEvents(), streamNow: clinicalNow(), runtimeProducing: producing,
    frozen: !!S.anchor, paused: !!(S.session && S.session.status !== 'running'),
    attention: organAttentionMap() });
}

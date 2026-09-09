/* The Oslian Monitoring Research Report: a builder, a frozen preview, and the
 * document that is actually printed.
 *
 * `Print / PDF` no longer prints whatever the page happens to be showing. It
 * opens a preview built from a snapshot taken at that moment: the stream keeps
 * running underneath, and an open report does not move until the reader asks it
 * to (`Refresh report`). Everything in the document comes from that frozen copy,
 * so what is on screen is what is on paper.
 *
 * The header carries what any result report has to carry - CLIA 42 CFR
 * 493.1291(c): who the subject is, where the result came from, the report date,
 * what was measured, the result with its units and interpretation, and (d) the
 * pertinent reference interval - and the context FHIR R4 DiagnosticReport names:
 * status, effective time, issued time, and the atomic results the report is made
 * of. Oslian is not a testing laboratory and this is not a laboratory report:
 * the title says so, the reporting system is named as a research demonstration,
 * and no performer, no conclusion and no clinical alert is claimed.
 *
 * The forecast outlook is Oslian's own section and is kept apart from the
 * results: a model point forecast, a carried-forward value, a head that
 * abstained and a withheld cell are four different things and are printed as
 * four different things.
 */

import { S, activeHorizon, currentRun, indicatorView, organAttentionMap, sourceInfo, trajectoryFor } from './store.js';
import { buildAnatomy } from './anatomy.js';
import { chartLegend, trajectoryChart } from './chart.js';
import { esc, fmtNum, BUILDER, FLAG, NO_FORECAST_MODEL, POINT_ONLY_NOTE, PROJECTED, REPORT, TRAJECTORY,
  forecastState, flagOf, isAbstained, unavailableReason } from './format.js';
import { fmtAge, fmtClinical, label } from './render.js';
import { rangeText } from './signals.js';

const mark = (glyph) => `<i class="mk" aria-hidden="true">${glyph}</i>`;
const plain = (value) => JSON.parse(JSON.stringify(value === undefined ? null : value));

/** The snapshot the report is built from: a deep copy, so nothing the stream
 *  does afterwards can reach an open report. */
export function freezeReport() {
  const view = indicatorView(label);
  const run = currentRun();
  const src = sourceInfo();
  const rows = view.all.filter((r) => r.state).map((r) => plain({
    ...r, traj: trajectoryFor(r.variable, null), name: label(r.variable), flag: flagOf(r),
  }));
  return {
    rows,
    unobserved: view.all.filter((r) => !r.state).map((r) => ({ variable: r.variable, name: label(r.variable) })),
    organs: plain(organAttentionMap(view.all)),
    organOrder: S.anatomy ? S.anatomy.organs.map((o) => ({ id: o.id, label: o.label })) : [],
    counts: { observed: view.observed, total: view.total, outside: view.outside, projected: view.projected,
      unsupplied: view.unsupplied, modelCells: view.modelCells, carriedCells: view.carriedCells,
      abstainedCells: view.abstainedCells, withheldCells: view.withheldCells },
    meta: {
      patient: src.display_name || (S.patient ? S.patient.display : 'Patient'),
      patientId: (S.session && S.session.patient_id) || '',
      sourceLabel: src.label || '', sourceSystem: src.source_system || '', fixture: src.fixture_version || '',
      mode: String(S.mode).includes(':') ? `retrospective stay · case ${src.case_id || ''}` : (src.mode === 'retrospective' ? 'retrospective replay' : 'live synthetic stream'),
      provenance: src.provenance || '',
      anchored: !!S.anchor,
      head: S.anchor ? `Historical anchor · run ${S.anchor.run.run_id.slice(0, 8)}` : 'Live snapshot',
      cutoff: run ? fmtClinical(run.input_cutoff) : null,
      clinicalNow: S.clock.latest_clinical_event_time ? fmtClinical(S.clock.latest_clinical_event_time) : null,
      runId: run ? run.run_id : null, runtimeMode: run ? run.runtime_mode : null,
      runtimeAttached: !!(S.runtime && S.runtime.runtime_attached),
      modelCommit: (S.runtime && S.runtime.model_commit) || '',
      registry: (S.capabilities && S.capabilities.registry_sha256) || '',
      epoch: (S.session && S.session.epoch) ?? null,
    },
  };
}

/** Open the preview on a fresh snapshot; the builder's choices survive a refresh. */
export function openReport(previous) {
  const horizon = activeHorizon();
  const stamp = new Date();
  return {
    scope: (previous && previous.scope) || 'full',
    indicators: (previous && previous.indicators) || [],
    horizon: previous && previous.horizon !== undefined ? previous.horizon : horizon,
    charts: previous ? previous.charts : true,
    anatomy: previous ? previous.anatomy : true,
    provenance: previous ? previous.provenance : false,
    frozen: freezeReport(),
    generatedAt: stamp.toISOString(),
    id: `OSL-${stamp.toISOString().slice(0, 10).replace(/-/g, '')}-${stamp.toISOString().slice(11, 19).replace(/:/g, '')}`,
  };
}

const scopedRows = (report) => {
  const rows = report.frozen.rows;
  if (report.scope === 'attention') return rows.filter((r) => r.flagged);
  if (report.scope === 'selected') return rows.filter((r) => report.indicators.includes(r.variable));
  return rows;
};

const cellAt = (row, horizon) => (row.forecasts || []).find((f) => f.horizon_hours === horizon) || null;

/* ── results ───────────────────────────────────────────────────────────── */
function resultRow(row) {
  const flag = FLAG[row.flag];
  const prev = row.previous;
  const change = !prev || prev.change === null ? '—'
    : prev.change === 0 ? 'unchanged' : `${prev.change > 0 ? '+' : '−'}${fmtNum(Math.abs(prev.change))}`;
  const abnormal = row.attention.kind === 'outside';
  // Each cell says what it is, and the stylesheet colours by that name rather than
  // by position. Only the measured value that is outside its interval, and the flag
  // beside it, are deep red: a previous value and a change are facts about
  // direction, and colouring them would read as a verdict on whether the patient is
  // getting better or worse - which this page does not make.
  return `<tr${abnormal ? ' class="abnormal"' : ''} data-flag="${esc(row.flag)}">
    <th scope="row" class="indicator-name">${esc(row.name)}<span class="d">${esc((row.organs || []).join(', ') || 'systemic')}</span></th>
    <td class="v current-result">${esc(fmtNum(row.state.value))} <small>${esc(row.state.unit)}</small></td>
    <td class="result-flag"><span class="flag flag-${esc(row.flag)}">${mark(flag.mark)}${flag.text}</span></td>
    <td class="reference-interval">${esc(rangeText(row) || 'Not supplied')}</td>
    <td class="v previous-result">${prev ? `${esc(fmtNum(prev.value))} <small>${esc(prev.unit)}</small>` : '—'}</td>
    <td class="v result-change">${esc(change)}</td>
    <td class="observed-at">${esc(fmtClinical(row.state.effective_time))}<span class="d">age ${esc(fmtAge(row.ageHours))}</span></td>
  </tr>`;
}

const RESULT_HEAD = `<thead><tr><th scope="col">Indicator</th><th scope="col">Result</th><th scope="col">Flag</th>
  <th scope="col">Reference interval</th><th scope="col">Previous</th><th scope="col">Change</th><th scope="col">Observed at</th></tr></thead>`;

/* ── forecast outlook ──────────────────────────────────────────────────── */
function forecastRow(row, horizon) {
  const cell = cellAt(row, horizon);
  const current = row.state ? row.state.value : NaN;
  if (!cell) {
    return `<tr><th scope="row" class="indicator-name">${esc(row.name)}</th><td class="v current-result">${esc(fmtNum(current))} <small>${esc(row.unit)}</small></td>
      <td class="v forecast-result">—</td><td>${horizon}h</td><td class="v result-change">—</td><td>${(row.catalogHorizons || []).length ? 'No cell at this horizon' : NO_FORECAST_MODEL}</td><td>—</td></tr>`;
  }
  const value = Number.isFinite(cell.forecast_point) ? `${fmtNum(cell.forecast_point)} ${cell.unit || row.unit}` : 'null';
  const d = cell.status === 'model' && Number.isFinite(cell.forecast_point) && Number.isFinite(current)
    ? cell.forecast_point - current : null;
  const diff = d === null ? (cell.status === 'persistence' ? '±0' : '—')
    : d === 0 ? 'unchanged' : `${d > 0 ? '+' : '−'}${fmtNum(Math.abs(d))}`;
  const state = isAbstained(cell) ? 'Carried forward — model abstained'
    : cell.status === 'model' ? 'Model forecast'
      : cell.status === 'persistence' ? 'Carried forward'
        : cell.status === 'missing' ? 'Missing' : 'Degraded';
  const projected = cell.projected
    ? `<span class="proj">${mark(PROJECTED[cell.projectedSide].mark)}${PROJECTED[cell.projectedSide].text}</span>`
    : (cell.status === 'model' && row.range ? 'Within supplied range' : '—');
  return `<tr data-status="${esc(cell.status)}"${cell.abstained ? ' data-abstained="true"' : ''}${cell.projected ? ` data-projected="${esc(cell.projectedSide)}"` : ''}>
    <th scope="row" class="indicator-name">${esc(row.name)}</th>
    <td class="v current-result">${esc(fmtNum(current))} <small>${esc(row.state.unit)}</small></td>
    <td class="v forecast-result${cell.status === 'model' ? ' model' : ''}${cell.projected ? ' projected' : ''}">${esc(value)}</td>
    <td>${cell.horizon_hours}h</td>
    <td class="v result-change">${esc(diff)}</td>
    <td>${esc(state)}${cell.status !== 'model' && cell.status !== 'persistence' ? `<span class="d">${esc(unavailableReason(cell.status, cell.fallback_reason))}</span>` : ''}</td>
    <td>${projected}</td>
  </tr>`;
}

/* ── trends ────────────────────────────────────────────────────────────── */
function trendChart(row) {
  // drawn two to a row on paper: wide enough to read the axis, short enough that
  // the trends and the body overview share a page instead of taking one each
  const chart = trajectoryChart({ traj: row.traj, unit: row.unit, label: row.name, axis: undefined, onset_ms: 0,
    window: 'full', width: 700, height: 186, idSuffix: `rep-${row.variable}`, range: row.range,
    headline: TRAJECTORY.liveHead });
  if (chart.empty) return '';
  return `<figure class="report-chart"><figcaption>${esc(row.name)} · ${esc(row.unit)}${row.range ? ` · supplied range ${esc(rangeText(row))}` : ' · no supplied range'}</figcaption>${chart.svg}${chartLegend(row.traj, row.range)}</figure>`;
}

/* ── anatomy overview ──────────────────────────────────────────────────── */
function anatomyList(report) {
  return report.frozen.organOrder.map(({ id, label: name }) => {
    const a = report.frozen.organs[id];
    if (!a) return '';
    const lines = a.abnormalIds.map((v) => {
      const row = report.frozen.rows.find((r) => r.variable === v);
      return `<li class="abnormal">${esc(row ? row.name : v)} — ${esc(row && row.attention.side === 'below' ? 'below supplied range' : 'above supplied range')}</li>`;
    }).concat(a.projectedIds.map((v) => {
      const row = report.frozen.rows.find((r) => r.variable === v);
      return `<li class="projected">${esc(row ? row.name : v)} — ${esc(row && row.projectedSide === 'below' ? 'projected below supplied range' : 'projected above supplied range')}</li>`;
    }));
    if (!lines.length) {
      lines.push(`<li class="quiet">${a.observed ? 'No current observed result outside supplied range' : 'No observation delivered'}</li>`);
    }
    return `<section class="organ-note" data-state="${esc(a.state)}"><h4>${esc(name)}</h4><ul>${lines.join('')}</ul></section>`;
  }).join('');
}

/* ── the builder ───────────────────────────────────────────────────────── */
function builder(report) {
  const { horizons } = indicatorView(label);
  const scopes = [['full', REPORT.scopes.full], ['attention', REPORT.scopes.attention], ['selected', REPORT.scopes.selected]];
  const rows = report.frozen.rows;
  return `<aside class="report-builder" aria-label="${BUILDER.title}">
    <div class="builder-head"><h2>${BUILDER.title}</h2>
      <div class="builder-actions">
        <button class="btn sm" type="button" data-act="report-close">${BUILDER.back}</button>
        <button class="btn sm" type="button" data-act="report-refresh">${BUILDER.refresh}</button>
        <button class="btn sm primary" type="button" data-act="report-print">${BUILDER.print}</button>
      </div></div>
    <div class="builder-grid">
      <fieldset><legend>${BUILDER.scope}</legend>
        ${scopes.map(([k, l]) => `<label><input type="radio" name="reportScope" value="${k}" data-act="report-scope" ${report.scope === k ? 'checked' : ''}> ${l}</label>`).join('')}
        ${report.scope === 'selected' ? `<div class="scope-picker">${rows.map((r) => `<label class="pick"><input type="checkbox" data-act="report-pick" data-variable="${esc(r.variable)}" ${report.indicators.includes(r.variable) ? 'checked' : ''}> ${esc(r.name)}</label>`).join('')}</div>` : ''}
      </fieldset>
      <fieldset><legend>${BUILDER.snapshot}</legend>
        <p class="why">${report.frozen.meta.head}${report.frozen.meta.cutoff ? ` · cutoff ${esc(report.frozen.meta.cutoff)}` : ''}</p>
        <p class="why">${BUILDER.frozen}</p>
        <p class="why">Snapshot ${esc(report.id)} · generated ${esc(report.generatedAt.replace('T', ' ').slice(0, 19))}Z</p>
      </fieldset>
      <fieldset><legend>${BUILDER.horizon}</legend>
        <span class="segmented" role="group" aria-label="${BUILDER.horizon}">${horizons.map((h) => `<button type="button" data-act="report-horizon" data-horizon="${h.hours}" aria-pressed="${report.horizon === h.hours}">${h.hours}h</button>`).join('') || '<span class="why">no horizon in this registry</span>'}</span>
      </fieldset>
      <fieldset><legend>${BUILDER.sections}</legend>
        <label><input type="checkbox" data-act="report-toggle" data-section="charts" ${report.charts ? 'checked' : ''}> ${BUILDER.charts}</label>
        <label><input type="checkbox" data-act="report-toggle" data-section="anatomy" ${report.anatomy ? 'checked' : ''}> ${BUILDER.anatomy}</label>
        <label><input type="checkbox" data-act="report-toggle" data-section="provenance" ${report.provenance ? 'checked' : ''}> ${BUILDER.provenance}</label>
      </fieldset>
    </div>
  </aside>`;
}

/* ── the document ──────────────────────────────────────────────────────── */
export function renderReport(host) {
  const report = S.report;
  const m = report.frozen.meta;
  const horizon = report.horizon;
  const rows = scopedRows(report);
  const attention = rows.filter((r) => r.attention.kind === 'outside');
  const other = rows.filter((r) => !attention.includes(r));
  const chartRows = report.charts
    ? rows.filter((r) => r.attention.kind === 'outside' || r.forecasts.some((f) => f.projected)
      || (report.scope === 'selected' && report.indicators.includes(r.variable)))
    : [];
  const scopeLabel = REPORT.scopes[report.scope];
  host.innerHTML = `${builder(report)}
    <article class="report" id="reportDoc" aria-label="${REPORT.title}">
      <header class="report-head">
        <h1>${REPORT.title}</h1>
        <p class="report-boundary">${REPORT.boundary}</p>
        <dl class="report-meta">
          <dt>Subject</dt><dd>${esc(m.patient)} · identifier <span class="mono">${esc(m.patientId)}</span></dd>
          <dt>Source</dt><dd>${esc(m.mode)}${m.sourceSystem ? ` · <span class="mono">${esc(m.sourceSystem)}</span>` : ''}${m.fixture ? ` · fixture ${esc(m.fixture)}` : ''}</dd>
          <dt>Snapshot</dt><dd>${esc(m.head)}${m.anchored ? '' : ' · live at the time of generation'}</dd>
          <dt>Clinical cutoff</dt><dd>${esc(m.cutoff || 'no model run yet')}${m.clinicalNow ? ` · latest observation ${esc(m.clinicalNow)}` : ''}</dd>
          <dt>Issued</dt><dd>${esc(report.generatedAt.replace('T', ' ').slice(0, 19))}Z</dd>
          <dt>Report snapshot ID</dt><dd class="mono">${esc(report.id)}</dd>
          <dt>Scope</dt><dd>${scopeLabel} · ${rows.length} of ${report.frozen.rows.length} observed indicators${horizon === null ? '' : ` · forecast comparison at ${horizon}h`}</dd>
          <dt>Report status</dt><dd>${REPORT.status}</dd>
          <dt>Reporting system</dt><dd>${REPORT.system}${m.runId ? ` · run <span class="mono">${esc(m.runId.slice(0, 8))}</span> · ${esc(m.runtimeMode)}` : ''}</dd>
        </dl>
      </header>

      <section class="report-section" aria-label="${REPORT.attentionHeading}">
        <h2>${REPORT.attentionHeading}</h2>
        ${attention.length
          ? `<table class="report-table results">${RESULT_HEAD}<tbody>${attention.map(resultRow).join('')}</tbody></table>`
          : `<p class="quiet">${REPORT.noAttention}</p>`}
      </section>

      <section class="report-section" aria-label="${REPORT.otherHeading}">
        <h2>${REPORT.otherHeading}</h2>
        ${other.length
          ? `<table class="report-table results compact">${RESULT_HEAD}<tbody>${other.map(resultRow).join('')}</tbody></table>`
          : '<p class="quiet">No other observed indicator is in scope.</p>'}
      </section>

      <section class="report-section" aria-label="${REPORT.forecastHeading}">
        <h2>${REPORT.forecastHeading}</h2>
        <p class="report-note">${REPORT.forecastNote}</p>
        ${horizon === null ? '<p class="quiet">No forecast horizon exists in the pinned registry.</p>' : `
        <table class="report-table forecast"><thead><tr><th scope="col">Indicator</th><th scope="col">Current observed</th>
          <th scope="col">Forecast value</th><th scope="col">Horizon</th><th scope="col">Difference</th>
          <th scope="col">Forecast status</th><th scope="col">Projected range status</th></tr></thead>
          <tbody>${rows.map((r) => forecastRow(r, horizon)).join('')}</tbody></table>`}
        <p class="report-note">Counted across the whole snapshot: ${report.frozen.counts.modelCells} model forecast cells ·
          ${report.frozen.counts.carriedCells} carried forward (${report.frozen.counts.abstainedCells} model abstained) ·
          ${report.frozen.counts.withheldCells} withheld. A carried-forward value is not a model forecast.</p>
      </section>

      ${chartRows.length ? `<section class="report-section" aria-label="${REPORT.chartsHeading}">
        <h2>${REPORT.chartsHeading}</h2>
        <p class="report-note">Drawn for indicators with a current result outside a supplied range, a forecast projected outside one, or an indicator you selected.
          Observed history is a solid ink line; the model input cutoff is the dashed rule; a model point forecast is a hollow blue diamond;
          a carried-forward value is a grey square; a point projected outside the supplied range is an amber dashed diamond; the supplied
          reference range is the pale band. ${POINT_ONLY_NOTE}</p>
        <div class="report-charts">${chartRows.map(trendChart).join('')}</div>
      </section>` : ''}

      ${report.anatomy ? `<section class="report-section anatomy-section" aria-label="${REPORT.anatomyHeading}">
        <h2>${REPORT.anatomyHeading}</h2>
        <div class="report-anatomy"><div class="report-figure" id="reportAnatomy"></div>
        <div class="organ-notes">${anatomyList(report)}</div></div>
        <p class="report-note">Colour marks where a current observed result sits against a supplied range, and nothing else. It is not a diagnosis and states nothing about an organ's condition.</p>
      </section>` : ''}

      <footer class="report-foot">
        <h2>${REPORT.limitsHeading}</h2>
        <ul>
          <li>Reference intervals are supplied by the observations themselves and are reproduced as received; this system writes no threshold of its own. Where none was supplied, the result is reported without a comparison.</li>
          <li>Forecasts are research-only point values. ${POINT_ONLY_NOTE} Carried-forward values repeat the last observation and are not predictions.</li>
          <li>Runtime: ${m.runtimeAttached ? 'attached' : 'not attached'}${m.runtimeMode ? ` · ${esc(m.runtimeMode)}` : ''}${m.modelCommit ? ` · model <span class="mono">${esc(m.modelCommit.slice(0, 7))}</span>` : ''}${m.registry ? ` · registry <span class="mono">${esc(m.registry.slice(0, 12))}</span>` : ''}.</li>
          <li>No diagnosis, causal claim, treatment recommendation or clinical alert is made. causal_claim_allowed: false.</li>
        </ul>
        ${report.provenance ? `<h2>${REPORT.provenanceHeading}</h2>
        <table class="report-table prov"><thead><tr><th scope="col">Indicator</th><th scope="col">Observation</th>
          <th scope="col">Stated value</th><th scope="col">Reference interval source</th></tr></thead><tbody>
          ${report.frozen.rows.map((r) => `<tr><th scope="row">${esc(r.name)}</th>
            <td class="mono">${esc(r.state.resource_id)} v${esc(r.state.version_id)}</td>
            <td class="mono">${esc(fmtNum(r.state.input_value))} ${esc(r.state.input_unit || r.state.unit)} · ${esc(r.state.derivation || 'identity')}</td>
            <td>${r.range && r.range.text ? esc(r.range.text) : 'not supplied'}</td></tr>`).join('')}
        </tbody></table>` : ''}
        <p class="report-id">${REPORT.title} · snapshot <span class="mono">${esc(report.id)}</span> · generated ${esc(report.generatedAt.replace('T', ' ').slice(0, 19))}Z · ${esc(m.patientId)} · ${REPORT.system}</p>
      </footer>
    </article>`;
}

/** The anatomy overview is the same figure the page draws, on the frozen states:
 *  one drawing rule, so the printed body and the screen cannot disagree. It is
 *  rebuilt only when the snapshot it draws changes, so a re-render of the report
 *  around it does not throw the figure away and build another. */
export function mountReportAnatomy() {
  const host = document.getElementById('reportAnatomy');
  if (!host || !S.anatomy || !S.organLayout || !S.report) return;
  if (host.dataset.snapshot === S.report.id && host.firstChild) return;
  const figure = buildAnatomy(host, S.anatomy.organs, S.organLayout, () => {});
  figure.update({ selected: null, byOrgan: {}, organEvents: {}, streamNow: null,
    runtimeProducing: false, frozen: true, paused: true, attention: S.report.frozen.organs });
  host.dataset.snapshot = S.report.id;
}

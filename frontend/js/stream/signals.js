/* Signals & forecasts: one patient, one row per indicator.
 *
 * A result list, read the way a result list is read. The indicator's name is the
 * first column and the entry point (the NHS App puts the test first and the
 * history behind it); the current result, its flag, the interval it was compared
 * against, the value before it and the forecast at one chosen horizon follow, in
 * the order a report states them (CLIA 42 CFR 493.1291(c): the test performed,
 * the result with its units and interpretation, and (d) the pertinent reference
 * interval). FHIR R4 puts the categorical assessment next to the value, so the
 * flag sits beside the number and never at the far left.
 *
 * Every number, comparison and order comes from `indicatorRows()` in store.js,
 * which the organ figure and the printed report read as well. Nothing is
 * computed twice and no threshold is written here.
 *
 * Attention is the observation's own statement. A current result outside the
 * interval it supplied is deep red, bold, marked ▲ or ▼ and sorted first; a
 * model forecast that leaves the same interval is amber, hollow and marked △ or
 * ▽, is counted apart and never called an alert (AHRQ: an interface that flags
 * everything is read as flagging nothing). A carried-forward value - a head that
 * abstained included - is never a model forecast and is never projected.
 */

import { S, activeHorizon, currentRun, indicatorView, sourceInfo, trajectoryFor } from './store.js';
import { chartFrame, chartLegend, trajectoryChart } from './chart.js';
import { esc, fmtNum, DATA_VIEW, FLAG, NO_FORECAST_MODEL, INTERVAL_NOTE, PROJECTED, RANGE, TRAJECTORY,
  UPDATING_FORECAST, forecastLabel, forecastState, flagOf, isAbstained, unavailableReason } from './format.js';
import { cellChip, fmtAge, fmtClinical, label } from './render.js';

const CRITICAL = ['HH', 'LL', 'AA'];
const mark = (glyph) => `<i class="mk" aria-hidden="true">${glyph}</i>`;

/** The reference interval as a report prints it, or why there is none. */
export function rangeText(row) {
  const r = row.range;
  if (!r) return null;
  const low = r.low === null ? '…' : fmtNum(r.low);
  const high = r.high === null ? '…' : fmtNum(r.high);
  return `${low}–${high}${r.unit ? ` ${r.unit}` : ''}`;
}

function rangeCell(row) {
  const a = row.attention;
  if (a.kind === 'unit') {
    return `<span class="rng na">Not applicable</span><span class="why">${esc(a.note)}</span>`;
  }
  const text = rangeText(row);
  if (!text) {
    return a.basis === 'interpretation'
      ? `<span class="rng na">Not supplied</span><span class="why">source-coded reading ${esc(a.codes.join(', '))}</span>`
      : '<span class="rng na">Not supplied</span>';
  }
  const qualified = a.range && a.range.qualified
    ? '<span class="why">the only range supplied is qualified by population or age</span>' : '';
  return `<span class="rng">${esc(text)}</span>${qualified}`;
}

/** The flag: a mark and a word, in the column beside the result. */
function flagCell(row) {
  const key = flagOf(row);
  const flag = FLAG[key];
  const critical = row.attention.critical
    ? `<span class="why">source-coded critical reading ${esc(row.attention.codes.filter((c) => CRITICAL.includes(c)).join(', '))}</span>` : '';
  return `<span class="flag flag-${key}">${mark(flag.mark)}${flag.text}</span>${critical}`;
}

function previousCell(row) {
  const p = row.previous;
  if (!p) return `<span class="prev na">${DATA_VIEW.noPrevious}</span>`;
  const change = p.change === null ? ''
    : p.change === 0 ? `<span class="why">${DATA_VIEW.unchanged}</span>`
      : `<span class="why">${p.change > 0 ? '+' : '−'}${esc(fmtNum(Math.abs(p.change)))} from previous</span>`;
  return `<span class="prev">${esc(fmtNum(p.value))} <small>${esc(p.unit)}</small></span>${change}<span class="why">${esc(fmtClinical(new Date(p.ms).toISOString()))}</span>`;
}

/** The forecast at the one horizon the page is comparing at. */
function forecastCell(row) {
  const h = row.horizon;
  if (h === null) return `<span class="fc na">${NO_FORECAST_MODEL}</span>`;
  if (!row.hasHorizonCell) return `<span class="fc na">No ${h}h forecast</span><span class="why">${row.catalogHorizons.length ? `this indicator has ${row.catalogHorizons.map((x) => `${x}h`).join(' · ')}` : NO_FORECAST_MODEL}</span>`;
  if (!row.run) return `<span class="fc na">${TRAJECTORY.noRun}</span>`;
  const cell = row.atHorizon;
  if (!cell) return `<span class="fc na">${UPDATING_FORECAST}</span>`;
  if (cell.status === 'model' && Number.isFinite(cell.forecast_point)) {
    const base = row.state ? row.state.value : NaN;
    const d = Number.isFinite(base) ? cell.forecast_point - base : null;
    const vs = d === null ? '' : d === 0 ? DATA_VIEW.unchanged
      : `${d > 0 ? '+' : '−'}${fmtNum(Math.abs(d))} vs current`;
    const projected = cell.projected
      ? `<span class="proj">${mark(PROJECTED[cell.projectedSide].mark)}${PROJECTED[cell.projectedSide].text}</span>` : '';
    return `<span class="fc model">${esc(fmtNum(cell.forecast_point))} <small>${esc(cell.unit || row.unit)}</small></span>`
      + `<span class="why">${esc(vs)}</span><span class="why">${h}h · model</span>${projected}`;
  }
  if (cell.status === 'persistence') {
    return `<span class="fc carried">${isAbstained(cell) ? forecastLabel(cell) : esc(fmtNum(cell.forecast_point))}${isAbstained(cell) ? '' : ` <small>${esc(cell.unit || row.unit)}</small>`}</span>`
      + `<span class="why">${isAbstained(cell) ? 'repeats the observation' : DATA_VIEW.unchanged}</span><span class="why">${h}h · carried forward</span>`;
  }
  return `<span class="fc na">${RANGE.unavailable}</span><span class="why">${h}h · ${esc(unavailableReason(cell.status, cell.fallback_reason))}</span>`;
}

/* ── the detail panel: one indicator, opened in place ───────────────────── */
function detailPanel(row) {
  const t = trajectoryFor(row.variable, null);
  // This chart is read next to the comparison and the forecast cells beside it
  // and the trajectory link under it, so it asks for the inline frame by name
  // rather than the analysis-sized default. The window width picks the rung; the
  // stylesheet stops the drawing widening past INLINE_MAX_WIDTH.
  const frame = chartFrame('inline-detail', typeof window === 'undefined' ? 1440 : window.innerWidth);
  const chart = trajectoryChart({ traj: t, unit: row.unit, label: label(row.variable), axis: undefined,
    onset_ms: 0, window: 'full', width: frame.width, height: frame.height, context: 'inline-detail',
    idSuffix: row.variable, range: row.range,
    headline: S.anchor ? TRAJECTORY.snapshot : TRAJECTORY.liveHead });
  const flag = FLAG[flagOf(row)];
  const cells = row.forecasts.length
    ? `<table class="detail-cells"><thead><tr><th scope="col">Horizon</th><th scope="col">Forecast</th><th scope="col">State</th><th scope="col">vs current</th></tr></thead><tbody>${row.forecasts.map((f) => {
      const empty = f.forecast_point === null || f.forecast_point === undefined;
      const base = row.state ? row.state.value : NaN;
      const d = f.status === 'model' && Number.isFinite(f.forecast_point) && Number.isFinite(base) ? f.forecast_point - base : null;
      const change = d === null ? (f.status === 'persistence' ? '±0 · repeats the observation' : '—')
        : d === 0 ? DATA_VIEW.unchanged : `${d > 0 ? '+' : '−'}${fmtNum(Math.abs(d))}`;
      const projected = f.projected ? `<span class="proj">${mark(PROJECTED[f.projectedSide].mark)}${PROJECTED[f.projectedSide].text}</span>` : '';
      const why = !f.projected && f.status !== 'model' && f.status !== 'persistence'
        ? `<span class="why">${esc(unavailableReason(f.status, f.fallback_reason))}</span>` : '';
      return `<tr data-status="${esc(f.status)}"${f.abstained ? ' data-abstained="true"' : ''}${f.projected ? ` data-projected="${esc(f.projectedSide)}"` : ''}${f.horizon_hours === row.horizon ? ' data-compared="true"' : ''}>`
        + `<td class="v">${f.horizon_hours}h</td><td class="v${f.status === 'model' ? ' model' : ''}">${empty ? 'null' : esc(fmtNum(f.forecast_point))}</td>`
        + `<td>${cellChip(f)}${projected}${why}</td><td class="v">${esc(change)}</td></tr>`;
    }).join('')}</tbody></table>`
    : `<p class="why">${row.catalogHorizons.length ? (row.run ? UPDATING_FORECAST : TRAJECTORY.noRun) : `${NO_FORECAST_MODEL} · observations only`}</p>`;
  const basis = row.attention.basis === 'referenceRange'
    ? `compared against ${esc(rangeText(row))} supplied with the observation${row.attention.derived ? ' · the reading was derived from that range by the data source, not a laboratory reading' : ''}${row.range && row.range.text ? ` · ${esc(row.range.text)}` : ''}`
    : row.attention.basis === 'interpretation' ? `source-coded reading ${esc(row.attention.codes.join(', '))} · no comparable range supplied`
      : row.attention.kind === 'unit' ? esc(row.attention.note)
        : 'the observation carries no reference range and no coded reading · nothing is inferred in its place';
  const run = row.run;
  return `<tr class="ind-detail" data-variable="${esc(row.variable)}"><td colspan="7"><div class="detail-head">
      <h3>${esc(label(row.variable))}</h3>
      <span class="detail-value">${esc(fmtNum(row.state.value))}<small>${esc(row.state.unit)}</small></span>
      <span class="flag flag-${flagOf(row)}">${mark(flag.mark)}${flag.text}</span>
      <span class="why">${esc(fmtClinical(row.state.effective_time))} · age ${esc(fmtAge(row.ageHours))} · ${esc(row.organs.join(', ') || 'systemic')}</span>
    </div>
    <div class="detail-grid">
      <section><h4>Comparison</h4>
        <p class="why">${basis}</p>
        <p class="why">Previous result: ${row.previous
          ? `${esc(fmtNum(row.previous.value))} ${esc(row.previous.unit)} at ${esc(fmtClinical(new Date(row.previous.ms).toISOString()))}${row.previous.change === null ? '' : row.previous.change === 0 ? ` · ${DATA_VIEW.unchanged}` : ` · ${row.previous.change > 0 ? '+' : '−'}${esc(fmtNum(Math.abs(row.previous.change)))}`}`
          : DATA_VIEW.noPrevious}</p>
        <h4>Data quality</h4>
        <p class="why">${row.series.length} valid observation${row.series.length === 1 ? '' : 's'} in view · ${row.state.count} delivered · resource <span class="mono">${esc(row.state.resource_id)} v${esc(row.state.version_id)}</span>${row.series.length < 2 ? ' · too few points for a trend' : ''}</p>
        <p class="why">A retracted or superseded version never reaches this series.</p>
      </section>
      <section><h4>Forecast cells · ${esc(run ? run.runtime_mode : 'no run')}</h4>${cells}
        <p class="why">${INTERVAL_NOTE} A carried-forward value repeats the observation and is not a model forecast.</p>
      </section>
    </div>
    <div class="detail-chart">${chart.empty ? '<p class="empty-note">Nothing to plot yet.</p>' : chart.svg}<div class="chart-tip" id="chartTip" role="status" hidden></div></div>
    ${chart.empty ? '' : chartLegend(t, row.range)}
    <details class="detail-prov"><summary>Provenance</summary><dl class="kv">
      <dt>Indicator</dt><dd class="mono">${esc(row.variable)}</dd>
      <dt>Observation</dt><dd class="mono">${esc(row.state.resource_id)} v${esc(row.state.version_id)}</dd>
      <dt>Stated value</dt><dd class="mono">${esc(fmtNum(row.state.input_value))} ${esc(row.state.input_unit || row.state.unit)} · ${esc(row.state.derivation || 'identity')}</dd>
      ${run ? `<dt>Run</dt><dd class="mono">${esc(run.run_id.slice(0, 8))} · cutoff ${esc(fmtClinical(run.input_cutoff))} · ${esc(run.runtime_mode)}</dd>` : ''}
    </dl></details>
    <div class="detail-actions"><button class="btn sm" type="button" data-act="open-metric" data-variable="${esc(row.variable)}">Open full trajectory →</button></div>
  </td></tr>`;
}

/* ── one row ────────────────────────────────────────────────────────────── */
function indicatorRow(row) {
  const open = S.dataExpanded === row.variable;
  const key = flagOf(row);
  const abnormal = row.attention.kind === 'outside';
  return `<tr class="ind-row tier-${row.tier}${abnormal ? ' abnormal' : ''}${row.projectedSide && !abnormal ? ' projected' : ''}${open ? ' open' : ''}"
      data-act="data-expand" data-variable="${esc(row.variable)}" data-flag="${key}" tabindex="0" role="button" aria-expanded="${open}"
      aria-label="${esc(label(row.variable))}, ${esc(FLAG[key].full)}, ${esc(fmtNum(row.state.value))} ${esc(row.state.unit)}. ${open ? 'Collapse' : 'Expand'} details">
    <td class="ind"><span class="t">${esc(label(row.variable))}</span><span class="d">${esc(row.organs.join(', ') || 'systemic')}</span></td>
    <td class="result"><span class="v">${esc(fmtNum(row.state.value))}</span><span class="u">${esc(row.state.unit)}</span></td>
    <td class="flag-cell">${flagCell(row)}</td>
    <td class="range-cell">${rangeCell(row)}</td>
    <td class="prev-cell">${previousCell(row)}</td>
    <td class="fc-cell">${forecastCell(row)}</td>
    <td class="updated">${esc(fmtClinical(row.state.effective_time))}<span class="why">age ${esc(fmtAge(row.ageHours))}</span>${row.fresh ? `<span class="new-dot" title="new observation"><i aria-hidden="true"></i><span class="sr-only">New observation</span></span>` : ''}</td>
  </tr>${open ? detailPanel(row) : ''}`;
}

const HEAD = `<thead><tr><th scope="col">Indicator</th><th scope="col">Current result</th><th scope="col">Flag</th>
  <th scope="col">Reference range</th><th scope="col">Previous result</th><th scope="col">Forecast</th><th scope="col">Updated</th></tr></thead>`;

function table(rows, caption) {
  if (!rows.length) return '';
  return `<div class="ind-scroll"><table class="ind-table" aria-label="${esc(caption)}">${HEAD}<tbody>${rows.map(indicatorRow).join('')}</tbody></table></div>`;
}

function unobservedSection(view) {
  const n = view.unobserved.length;
  if (!n) return '';
  const open = S.dataShowUnobserved;
  const list = open ? `<ul class="unobserved-list">${view.unobserved.map((r) => `<li>${mark(FLAG.none.mark)}<span class="t">${esc(label(r.variable))}</span><span class="d">${esc(r.organs.join(', ') || 'systemic')} · ${esc(r.unit)}</span></li>`).join('')}</ul>` : '';
  return `<section class="unobserved" aria-label="${DATA_VIEW.unobserved}"><button class="disc unobserved-toggle" type="button" data-act="data-unobserved" aria-expanded="${open}">${DATA_VIEW.unobserved} (${n}) · nothing delivered for these mapped indicators ${open ? '⌃' : '⌄'}</button>${list}</section>`;
}

function attentionEmpty(view) {
  if (!view.supplied) return DATA_VIEW.emptyNoRanges;
  if (!view.outside && view.modelCells && !view.projected) return `${DATA_VIEW.emptyNoOutside}, and ${DATA_VIEW.emptyNoProjected.toLowerCase()}`;
  return DATA_VIEW.emptyNoOutside;
}

export function renderDataView(host) {
  const view = indicatorView(label);
  const src = sourceInfo();
  const run = currentRun();
  const patient = src.display_name || (S.patient ? S.patient.display : 'Patient');
  const stay = String(S.mode).includes(':');
  const where = stay
    ? `retrospective stay · case ${esc(src.case_id || String(S.mode).split(':')[1])} · not a live feed`
    : (src.mode === 'retrospective' ? 'retrospective replay · not a live feed' : 'synthetic FHIR R4 stream · not a live feed');
  const head = S.anchor
    ? `HISTORICAL ANCHOR · run ${esc(S.anchor.run.run_id.slice(0, 8))} · frozen at ${esc(fmtClinical(S.anchor.run.input_cutoff))}`
    : TRAJECTORY.liveHead;
  const horizon = activeHorizon();
  // the phone folds the ordinary results by default; a desktop shows them. Either
  // way the reader's own choice, once made, is what is used.
  const narrow = typeof window !== 'undefined' && window.innerWidth <= 767;
  const otherOpen = S.dataShowOther === null ? !narrow : S.dataShowOther;
  const sorts = [['attention', 'attention'], ['organ', 'organ'], ['name', 'name'], ['latest', 'latest'], ['age', 'age']];
  const summary = `<ul class="att-summary" aria-label="Attention summary">
      <li>${mark(FLAG.above.mark)}${mark(FLAG.below.mark)}<b>${view.outside}</b> outside supplied range</li>
      <li>${mark(PROJECTED.above.mark)}<b>${view.projected}</b> projected outside</li>
      <li>${mark(FLAG.unsupplied.mark)}<b>${view.unsupplied}</b> range not supplied</li>
      <li><b>${view.observed}</b> of ${view.total} indicators observed</li></ul>`;
  const horizonBar = view.horizons.length ? `<div class="horizon-bar"><span class="note" id="horizonLabel">${DATA_VIEW.compareAt}:</span>
      <span class="segmented" role="group" aria-labelledby="horizonLabel">${view.horizons.map((h) => `<button type="button" data-act="forecast-horizon" data-horizon="${h.hours}" aria-pressed="${horizon === h.hours}" title="${h.count} forecast cell${h.count === 1 ? '' : 's'} at this horizon">${h.hours}h</button>`).join('')}</span>
      <span class="note">${horizon === null ? 'no horizon in this registry' : `${view.horizons.find((h) => h.hours === horizon).count} of ${view.total} indicators have a ${horizon}h cell`}</span></div>` : '';
  host.innerHTML = `
    <div class="ws-h"><h2>${DATA_VIEW.title}</h2><span class="sub">${esc(patient)} · ${where} · <span class="mono">${head}</span></span>
      <span class="fn data-controls">
        <input class="field sm" type="search" placeholder="Search indicator or organ" value="${esc(S.dataQuery)}" data-act="data-query" aria-label="Search indicators">
        <span class="segmented" role="group" aria-label="Sort indicators">${sorts.map(([k, l]) => `<button type="button" data-act="data-sort" data-sort="${k}" aria-pressed="${S.dataSort === k}">${l}</button>`).join('')}</span>
        <button class="btn sm" type="button" data-act="report-open">${DATA_VIEW.print}</button>
      </span></div>
    ${summary}
    ${horizonBar}
    <section class="ind-section attention-section" aria-label="${DATA_VIEW.attentionHeading}">
      <div class="sec-h"><h3>${DATA_VIEW.attentionHeading}</h3><span class="note">${view.attention.length
        ? `${view.attention.length} indicator${view.attention.length === 1 ? '' : 's'} · a current result outside its supplied range, or a model forecast projected outside it`
        : ''}</span></div>
      ${view.attention.length ? table(view.attention, DATA_VIEW.attentionHeading) : `<p class="empty-note quiet">${attentionEmpty(view)}.</p>`}
    </section>
    <section class="ind-section" data-open="${otherOpen}" aria-label="${DATA_VIEW.otherHeading}">
      <div class="sec-h"><h3>${DATA_VIEW.otherHeading}</h3><span class="note">${view.other.length} indicator${view.other.length === 1 ? '' : 's'}</span></div>
      ${view.other.length ? `<button class="other-toggle" type="button" data-act="data-other" aria-expanded="${otherOpen}">${otherOpen ? DATA_VIEW.hideOther : `${DATA_VIEW.showOther} (${view.other.length})`}<span class="chev" aria-hidden="true">${otherOpen ? '⌃' : '⌄'}</span></button>` : ''}
      ${view.other.length ? table(view.other, DATA_VIEW.otherHeading) : `<p class="empty-note quiet">${DATA_VIEW.emptySearch}.</p>`}
    </section>
    ${unobservedSection(view)}
    <p class="foot">A flag is the observation's own statement: a reference range it supplies, compared in the unit it was written in, or a reading the source coded. This page writes no threshold, and the multi-stay reading order's grouping is not applied here. ${INTERVAL_NOTE} A carried-forward value, a head that abstained included, is never a model forecast and is never projected. causal_claim_allowed: false · no clinical alert.</p>`;
}

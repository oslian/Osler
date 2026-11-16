/* Forecast scorecard: past forecasts of this stream against what was measured.
 *
 * One row per forecast cell: how many forecasts are resolved, the mean absolute
 * error of the forecast, and beside it the error of carrying the anchor's value
 * forward -- the baseline each cell had to beat when it was validated. Below, every
 * forecast with its anchor, its measurement and the source observation.
 *
 * This is a fact about one stream: one patient and a few anchors. The page says
 * so above the table and prints the counts in every row, so a reader cannot take
 * a mean of two forecasts for the model's accuracy. It names no winner, colours
 * nothing by which error is smaller and raises nothing: the numbers are the
 * whole statement.
 *
 * Everything comes from /api/patients/<id>/forecast-scores. Nothing is computed
 * here but formatting. */

import { S } from './store.js';
import { esc, fmtNum } from './format.js';

const SCOPE = 'Scored on this stream only: one patient and a few anchors. It is not the model\'s accuracy at a hospital.';
const STATE_TEXT = {
  pending: 'Pending — the stream has not passed the end of the measurement window',
  unmeasured: 'Not measured within the window',
};
const shortTime = (iso) => String(iso || '').replace('T', ' ').replace(/:00\.000Z$/, 'Z').replace(/^\d{4}-/, '');

function cellRow(c) {
  const mae = (v) => (v === null || v === undefined ? '—' : esc(fmtNum(v)));
  return `<tr><td class="cell">${esc(c.cell)}</td><td class="num">${c.resolved}</td>`
    + `<td class="num">${mae(c.mean_abs_error_forecast)} <span class="why">${esc(c.unit || '')}</span></td>`
    + `<td class="num">${mae(c.mean_abs_error_carried_forward)} <span class="why">${esc(c.unit || '')}</span></td>`
    + `<td class="num">${c.pending}</td><td class="num">${c.unmeasured}</td></tr>`;
}

function forecastRow(f) {
  const signed = (v) => (v === null || v === undefined ? '—' : `${v > 0 ? '+' : ''}${esc(fmtNum(v))}`);
  const measured = f.state === 'resolved'
    ? `<td class="num">${esc(fmtNum(f.measured_value))}<span class="why"> ${esc(shortTime(f.measured_time))}</span></td>`
      + `<td class="num">${signed(f.forecast_error)}</td><td class="num">${signed(f.carried_forward_error)}</td>`
      + `<td class="cell" title="${esc(f.reference)}">${esc(f.reference.split('/')[1])}</td>`
    : `<td colspan="4" class="why">${esc(STATE_TEXT[f.state] || f.state)}</td>`;
  return `<tr data-state="${esc(f.state)}"><td class="cell">${esc(f.cell)}</td><td>${esc(shortTime(f.anchor_time))}</td>`
    + `<td class="num">${esc(fmtNum(f.anchor_value))}</td><td class="num">${esc(fmtNum(f.forecast_point))} <span class="why">${esc(f.unit || '')}</span></td>${measured}</tr>`;
}

export function renderScoresWorkspace(host) {
  if (S.scoresError || !S.scores) {
    host.innerHTML = `<p class="empty-note">${S.scoresError ? esc(S.scoresError) : 'Loading the scorecard…'}</p>`;
    return;
  }
  const doc = S.scores;
  const head = `<div class="section-head"><h2 class="h-sec">Forecast scorecard</h2>`
    + `<span class="meta">as of ${esc(shortTime(doc.as_of))} · ${doc.anchors_scored} anchor${doc.anchors_scored === 1 ? '' : 's'} · `
    + `${doc.counts.resolved} resolved · ${doc.counts.pending} pending · ${doc.counts.unmeasured} not measured</span></div>`
    + `<p class="foot" data-scope="stream">${esc(SCOPE)}</p>`
    + `<p class="foot">Measured = ${esc(doc.label_rule)}. ${esc(doc.baseline.charAt(0).toUpperCase() + doc.baseline.slice(1))}.</p>`;
  if (!doc.forecasts.length) {
    host.innerHTML = `${head}<p class="empty-note">No model forecast has been issued on this stream yet. The first one comes 6 hours after the encounter starts, and only while the forecast API is attached.</p>`;
    return;
  }
  host.innerHTML = `${head}
    <div class="scroll-x"><table class="cells-table" data-table="score-cells"><thead><tr><th>Cell</th><th>Resolved</th>
      <th>Mean abs. error · forecast</th><th>Mean abs. error · carried forward</th><th>Pending</th><th>Not measured</th></tr></thead>
      <tbody>${doc.cells.map(cellRow).join('')}</tbody></table></div>
    <p class="foot">A mean over ${doc.counts.resolved} resolved forecast${doc.counts.resolved === 1 ? '' : 's'} in total. Read the count before the mean.</p>
    <h2 class="h-sec" style="margin-top:22px">Every forecast</h2>
    <div class="scroll-x"><table class="cells-table" data-table="score-forecasts"><thead><tr><th>Cell</th><th>Anchor</th><th>Value at anchor</th>
      <th>Forecast</th><th>Measured</th><th>Forecast − measured</th><th>Carried forward − measured</th><th>Source</th></tr></thead>
      <tbody>${doc.forecasts.map(forecastRow).join('')}</tbody></table></div>`;
}

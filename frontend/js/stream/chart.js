/* Shared time-series chart for the live stream and the retrospective replay.
 *
 * Every plotted observation is a timeline row; every forecast marker is a cell
 * of a persisted model run. Encoding (not colour alone):
 *   observed          solid ink line, filled circles
 *   model point       dashed blue path from the last observation, hollow diamonds at the due time
 *   persistence       dotted caution baseline from the cutoff to the due time, small square marker
 *   missing/degraded  no marker; a text note in the future band
 *   cutoff            dashed oxide vertical rule; the future band is shaded
 * An interval band is never drawn: this runtime bundles no patient-specific
 * conformal artifact. */

import { esc, fmtNum, FORECAST_LABEL, POINT_ONLY_NOTE } from './format.js';

const COLORS = { ink: '#171A1C', ink2: '#626762', blue: '#344F69', oxide: '#B64A32', caution: '#A0741C',
  cautionText: '#8A6316', divider: '#D1CCC2', soft: '#E7E3DA', surface: '#FCFAF5', bg: '#F1EFE9', teal: '#27645E', nul: '#8B8F89' };
const HOUR = 3600000;

/** Time-axis formatter: calendar (UTC) or hours since an onset. */
export function makeTimeFormatter(axis, onsetMs) {
  const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  const p = (n) => String(n).padStart(2, '0');
  if (axis === 'hours_since_admission') {
    return {
      tick: (ms) => `${((ms - onsetMs) / HOUR).toFixed(((ms - onsetMs) / HOUR) % 1 ? 1 : 0)} h`,
      full: (ms) => `${((ms - onsetMs) / HOUR).toFixed(2)} h since admission`,
      short: (ms) => `${((ms - onsetMs) / HOUR).toFixed(2)} h`,
    };
  }
  return {
    tick: (ms, first) => { const d = new Date(ms); const t = `${p(d.getUTCHours())}Z`; return first || d.getUTCHours() === 0 ? `${MONTHS[d.getUTCMonth()]} ${d.getUTCDate()} ${t}` : t; },
    full: (ms) => { const d = new Date(ms); return `${MONTHS[d.getUTCMonth()]} ${d.getUTCDate()} · ${p(d.getUTCHours())}:${p(d.getUTCMinutes())}Z`; },
    short: (ms) => { const d = new Date(ms); return `${MONTHS[d.getUTCMonth()]} ${d.getUTCDate()} ${p(d.getUTCHours())}:${p(d.getUTCMinutes())}Z`; },
  };
}

/** Collision avoidance for point labels: sort by x, push each label into the
 * first lane whose already-placed labels do not overlap it horizontally. Lanes
 * stack away from the marker; a label above lane 0 gets a thin leader line. */
function placeLabels(items) {
  const placed = [];
  items.slice().sort((a, b) => a.x - b.x).forEach((item) => {
    const w = item.text.length * 6.4 + 10;
    let lane = 0;
    while (placed.some((p) => p.lane === lane && Math.abs(p.x - item.x) < (p.w + w) / 2)) lane++;
    placed.push({ ...item, w, lane });
  });
  return placed;
}
function labelAnchor(x, w, left, right) {
  if (x - w / 2 < left) return { anchor: 'start', tx: left };
  if (x + w / 2 > right) return { anchor: 'end', tx: right };
  return { anchor: 'middle', tx: x };
}

function niceStepHours(spanHours, maxTicks) {
  const steps = [0.25, 0.5, 1, 2, 3, 6, 12, 24, 48, 96];
  for (const s of steps) if (spanHours / s <= maxTicks) return s;
  return 96;
}

/** Small inline sparkline: last N points, no axes. */
export function sparkline(points, width = 120, height = 32) {
  if (!points || points.length < 2) return '';
  const values = points.map((p) => p.value);
  const min = Math.min(...values), max = Math.max(...values), span = max - min || 1;
  const coords = points.map((p, i) => [(i / (points.length - 1)) * (width - 6) + 3, height - 4 - ((p.value - min) / span) * (height - 8)]);
  const last = coords[coords.length - 1];
  return `<svg viewBox="0 0 ${width} ${height}" class="spark" aria-hidden="true"><polyline points="${coords.map((c) => c.map((n) => n.toFixed(1)).join(',')).join(' ')}"/><circle cx="${last[0].toFixed(1)}" cy="${last[1].toFixed(1)}" r="2.6"/></svg>`;
}

/**
 * Big chart. opts:
 *   points: [{ms, value, resource_id, version_id, received_at, status}] sorted by ms
 *   forecasts: [{cell, horizon_hours, due_ms, forecast_point, status, source_model, run_id, fallback_reason}]
 *   cutoff_ms, unit, label, axis ('calendar'|'hours_since_admission'), onset_ms,
 *   window ('recent'|'full'), horizons (Set of numbers) or null, fresh_ms, frozen (bool), width, height
 */
export function bigChart(opts) {
  const W = opts.width || 980, H = opts.height || 360, PL = 64, PR = 20, PT = 34, PB = 40;
  const fmt = makeTimeFormatter(opts.axis, opts.onset_ms);
  const all = opts.points || [];
  const forecasts = (opts.forecasts || []).filter((f) => !opts.horizons || opts.horizons.has(f.horizon_hours));
  const drawn = forecasts.filter((f) => f.status === 'model' || f.status === 'persistence');
  const withheld = forecasts.filter((f) => f.status !== 'model' && f.status !== 'persistence');
  let points = all;
  if (opts.window === 'recent' && all.length) {
    const end = opts.cutoff_ms || all[all.length - 1].ms;
    points = all.filter((p) => p.ms >= end - 12 * HOUR);
    if (points.length < 2) points = all.slice(-2);
  }
  if (!points.length && !drawn.length) return { svg: '', empty: true };

  const xs = points.map((p) => p.ms);
  if (opts.cutoff_ms) xs.push(opts.cutoff_ms);
  // withheld cells still claim their due time on the axis, so the future band shows where a point would sit
  forecasts.forEach((f) => xs.push(f.due_ms));
  let xMin = Math.min(...xs), xMax = Math.max(...xs);
  if (xMax - xMin < HOUR) { xMin -= HOUR / 2; xMax += HOUR / 2; }
  const xPad = (xMax - xMin) * 0.03;
  xMin -= xPad; xMax += xPad;
  const ys = points.map((p) => p.value).concat(drawn.map((f) => f.forecast_point)).filter((v) => Number.isFinite(v));
  let yMin = Math.min(...ys), yMax = Math.max(...ys);
  if (yMax - yMin < 1e-6) { yMax = yMin + Math.max(1, Math.abs(yMin) * 0.1); }
  const rawMin = yMin;
  const pad = (yMax - yMin) * 0.18;
  yMin -= pad; yMax += pad;
  if (rawMin >= 0 && yMin < 0) yMin = 0;
  const X = (ms) => PL + ((ms - xMin) / (xMax - xMin)) * (W - PL - PR);
  const Y = (v) => PT + (1 - (v - yMin) / (yMax - yMin)) * (H - PT - PB);

  let s = `<svg viewBox="0 0 ${W} ${H}" class="bigchart" role="img" aria-label="${esc(opts.label)} observed series${opts.cutoff_ms ? ' with model input cutoff' : ''}${drawn.length ? ' and forecast points' : ''}">`;
  // future band
  if (opts.cutoff_ms) {
    const cx = X(opts.cutoff_ms);
    s += `<rect x="${cx.toFixed(1)}" y="${PT}" width="${Math.max(0, W - PR - cx).toFixed(1)}" height="${H - PT - PB}" fill="${COLORS.bg}" opacity="0.6"/>`;
  }
  // y grid
  const yTicks = [yMax, yMin + (yMax - yMin) * 0.75, (yMax + yMin) / 2, yMin + (yMax - yMin) * 0.25, yMin];
  yTicks.forEach((v) => {
    const y = Y(v);
    s += `<line x1="${PL}" y1="${y.toFixed(1)}" x2="${W - PR}" y2="${y.toFixed(1)}" stroke="${COLORS.soft}"/>`;
    s += `<text x="${PL - 10}" y="${(y + 4).toFixed(1)}" text-anchor="end" class="ct">${esc(fmtNum(v))}</text>`;
  });
  s += `<text x="${PL - 10}" y="${PT - 14}" text-anchor="end" class="ct">${esc(opts.unit || '')}</text>`;
  // x axis
  s += `<line x1="${PL}" y1="${H - PB}" x2="${W - PR}" y2="${H - PB}" stroke="${COLORS.divider}"/>`;
  const spanH = (xMax - xMin) / HOUR;
  const step = niceStepHours(spanH, Math.max(4, Math.floor((W - PL - PR) / 110)));
  const origin = opts.axis === 'hours_since_admission' ? opts.onset_ms : 0;
  let t = Math.ceil((xMin - origin) / (step * HOUR)) * step * HOUR + origin;
  let first = true;
  for (; t <= xMax; t += step * HOUR) {
    const x = X(t);
    s += `<line x1="${x.toFixed(1)}" y1="${H - PB}" x2="${x.toFixed(1)}" y2="${H - PB + 4}" stroke="${COLORS.divider}"/>`;
    s += `<text x="${x.toFixed(1)}" y="${H - PB + 18}" text-anchor="middle" class="ct">${esc(fmt.tick(t, first))}</text>`;
    first = false;
  }
  // cutoff rule
  if (opts.cutoff_ms) {
    const cx = X(opts.cutoff_ms);
    s += `<line x1="${cx.toFixed(1)}" y1="${PT}" x2="${cx.toFixed(1)}" y2="${H - PB}" stroke="${COLORS.oxide}" stroke-dasharray="4 3"/>`;
    const anchorLabel = `${opts.frozen ? 'historical anchor' : 'model input cutoff'} · ${fmt.short(opts.cutoff_ms)}`;
    const leftSide = cx > (W - PR) * 0.6;
    s += `<text x="${(leftSide ? cx - 6 : cx + 6).toFixed(1)}" y="${PT - 8}" text-anchor="${leftSide ? 'end' : 'start'}" class="cl" fill="${COLORS.oxide}">${esc(anchorLabel)}</text>`;
    if (drawn.length || withheld.length) {
      const noteX = Math.min(cx + 8, W - PR - 8);
      s += `<text x="${noteX.toFixed(1)}" y="${PT + 14}" class="cl" fill="${COLORS.ink2}">future · point forecasts only · no calibrated interval for this runtime cell</text>`;
    }
  }
  // forecast labels never straddle the cutoff rule: it is the left bound for anything due after it
  const labelLeft = opts.cutoff_ms ? Math.max(PL + 2, X(opts.cutoff_ms) + 4) : PL + 2;
  // persistence baselines
  const lastObs = points[points.length - 1];
  const persistence = drawn.filter((f) => f.status === 'persistence');
  const persistenceLabels = placeLabels(persistence.map((f) => ({ x: X(f.due_ms), y: Y(f.forecast_point), text: `${f.horizon_hours}h carried forward`, f })));
  persistenceLabels.forEach(({ x, y, text, f, w, lane }) => {
    const x1 = X(opts.cutoff_ms || (lastObs ? lastObs.ms : f.due_ms));
    s += `<line x1="${x1.toFixed(1)}" y1="${y.toFixed(1)}" x2="${x.toFixed(1)}" y2="${y.toFixed(1)}" stroke="${COLORS.caution}" stroke-dasharray="1 4"/>`;
    const ty = y + 16 + lane * 13;
    const { anchor, tx } = labelAnchor(x, w, labelLeft, W - PR - 2);
    const leader = lane > 0 ? `<line x1="${x.toFixed(1)}" y1="${(y + 5).toFixed(1)}" x2="${x.toFixed(1)}" y2="${(ty - 9).toFixed(1)}" stroke="${COLORS.divider}"/>` : '';
    s += `<g class="chart-pt" tabindex="0" role="button" data-kind="forecast" data-cell="${esc(f.cell)}" data-run="${esc(f.run_id)}" data-status="persistence" data-value="${f.forecast_point}" data-ms="${f.due_ms}" data-source="${esc(f.source_model)}" data-horizon="${f.horizon_hours}">${leader}<rect x="${(x - 4).toFixed(1)}" y="${(y - 4).toFixed(1)}" width="8" height="8" fill="${COLORS.surface}" stroke="${COLORS.caution}" stroke-width="1.5"/><text x="${tx.toFixed(1)}" y="${ty.toFixed(1)}" text-anchor="${anchor}" class="cl" fill="${COLORS.cautionText}">${esc(text)}</text></g>`;
  });
  // model forecast path
  const model = drawn.filter((f) => f.status === 'model').sort((a, b) => a.due_ms - b.due_ms);
  if (model.length) {
    const pts = [];
    if (lastObs) pts.push([X(lastObs.ms), Y(lastObs.value)]);
    model.forEach((f) => pts.push([X(f.due_ms), Y(f.forecast_point)]));
    s += `<polyline fill="none" stroke="${COLORS.blue}" stroke-width="1.5" stroke-dasharray="5 4" points="${pts.map((p) => p.map((n) => n.toFixed(1)).join(',')).join(' ')}"/>`;
    const modelLabels = placeLabels(model.map((f) => ({ x: X(f.due_ms), y: Y(f.forecast_point), text: `${f.horizon_hours}h · ${fmtNum(f.forecast_point)}`, f })));
    modelLabels.forEach(({ x, y, text, f, w, lane }) => {
      const r = 5;
      const ty = y - 10 - lane * 13;
      const { anchor, tx } = labelAnchor(x, w, labelLeft, W - PR - 2);
      const leader = lane > 0 ? `<line x1="${x.toFixed(1)}" y1="${(y - r - 1).toFixed(1)}" x2="${x.toFixed(1)}" y2="${(ty + 3).toFixed(1)}" stroke="${COLORS.divider}"/>` : '';
      s += `<g class="chart-pt" tabindex="0" role="button" data-kind="forecast" data-cell="${esc(f.cell)}" data-run="${esc(f.run_id)}" data-status="model" data-value="${f.forecast_point}" data-ms="${f.due_ms}" data-source="${esc(f.source_model)}" data-horizon="${f.horizon_hours}">${leader}<polygon points="${x.toFixed(1)},${(y - r).toFixed(1)} ${(x + r).toFixed(1)},${y.toFixed(1)} ${x.toFixed(1)},${(y + r).toFixed(1)} ${(x - r).toFixed(1)},${y.toFixed(1)}" fill="${COLORS.surface}" stroke="${COLORS.blue}" stroke-width="1.6"/><text x="${tx.toFixed(1)}" y="${ty.toFixed(1)}" text-anchor="${anchor}" class="cl" fill="${COLORS.blue}">${esc(text)}</text></g>`;
    });
  }
  // withheld note
  if (withheld.length && opts.cutoff_ms) {
    const reasons = Array.from(new Set(withheld.map((f) => `${f.horizon_hours}h ${f.status}${f.fallback_reason ? ' · ' + f.fallback_reason : ''}`)));
    const cx = X(opts.cutoff_ms);
    const flip = cx > (W - PR) * 0.55;
    reasons.slice(0, 3).forEach((r, i) => {
      s += `<text x="${(flip ? cx - 8 : cx + 8).toFixed(1)}" y="${(H - PB - 12 - i * 14).toFixed(1)}" text-anchor="${flip ? 'end' : 'start'}" class="cl" fill="${COLORS.cautionText}">withheld · ${esc(r)}</text>`;
    });
    placeLabels(withheld.map((f) => ({ x: X(f.due_ms), y: H - PB, text: `${f.horizon_hours}h`, f }))).forEach(({ x, text, w, lane }) => {
      const ty = H - PB - 9 - lane * 12;
      const { anchor, tx } = labelAnchor(x, w, labelLeft, W - PR - 2);
      s += `<line x1="${x.toFixed(1)}" y1="${(ty + 3).toFixed(1)}" x2="${x.toFixed(1)}" y2="${H - PB}" stroke="${COLORS.caution}"/><text x="${tx.toFixed(1)}" y="${ty.toFixed(1)}" text-anchor="${anchor}" class="cl" fill="${COLORS.cautionText}">${esc(text)}</text>`;
    });
  }
  // observed
  if (points.length) {
    s += `<polyline fill="none" stroke="${COLORS.ink}" stroke-width="1.8" stroke-linejoin="round" points="${points.map((p) => `${X(p.ms).toFixed(1)},${Y(p.value).toFixed(1)}`).join(' ')}"/>`;
    points.forEach((p) => {
      const x = X(p.ms), y = Y(p.value);
      const fresh = opts.fresh_ms && p.ms === opts.fresh_ms;
      s += `<g class="chart-pt${fresh ? ' fresh' : ''}" tabindex="0" role="button" data-kind="observed" data-resource="${esc(p.resource_id)}" data-version="${esc(p.version_id)}" data-value="${p.value}" data-ms="${p.ms}" data-received="${esc(p.received_at || '')}"><circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${fresh ? 5 : 3.8}" fill="${COLORS.ink}"/>${fresh ? `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="9" fill="none" stroke="${COLORS.teal}" stroke-width="1.5" class="fresh-ring"/>` : ''}</g>`;
    });
  } else if (opts.cutoff_ms) {
    s += `<text x="${PL + 8}" y="${(H - PB - 12).toFixed(1)}" class="cl" fill="${COLORS.nul}">not observed before the cutoff · nothing imputed</text>`;
  }
  s += '</svg>';
  return { svg: s, empty: false, window_points: points.length, total_points: all.length };
}

export function chartLegend(hasModel, hasPersistence) {
  return `<div class="chart-legend">
    <span><svg width="26" height="10" aria-hidden="true"><line x1="1" y1="5" x2="25" y2="5" stroke="${COLORS.ink}" stroke-width="1.8"/><circle cx="13" cy="5" r="3.5" fill="${COLORS.ink}"/></svg>Observed</span>
    ${hasModel ? `<span><svg width="26" height="12" aria-hidden="true"><line x1="1" y1="6" x2="25" y2="6" stroke="${COLORS.blue}" stroke-width="1.5" stroke-dasharray="5 4"/><polygon points="13,1 18,6 13,11 8,6" fill="${COLORS.surface}" stroke="${COLORS.blue}" stroke-width="1.5"/></svg>${FORECAST_LABEL.model}</span>` : ''}
    ${hasPersistence ? `<span><svg width="26" height="10" aria-hidden="true"><line x1="1" y1="5" x2="25" y2="5" stroke="${COLORS.caution}" stroke-dasharray="1 4"/><rect x="20" y="1" width="8" height="8" fill="${COLORS.surface}" stroke="${COLORS.caution}"/></svg>${FORECAST_LABEL.persistence} — not a model prediction</span>` : ''}
    <span><svg width="12" height="12" aria-hidden="true"><line x1="6" y1="0" x2="6" y2="12" stroke="${COLORS.oxide}" stroke-dasharray="4 3"/></svg>Model input cutoff</span>
    <span><i class="lg-band"></i>${POINT_ONLY_NOTE}</span>
  </div>`;
}

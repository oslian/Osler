/* Patient-trajectory chart for the live stream and the retrospective replay.
 *
 * One continuous time axis, split at the run's model input cutoff into two
 * regions a reader can tell apart without a legend:
 *   OBSERVED HISTORY  left of the cutoff, white ground   solid ink line, filled circles
 *   FUTURE FORECAST   right of the cutoff, pale blue     dashed blue path, hollow diamonds
 * Every plotted observation is a timeline row; every forecast marker is a cell
 * of one persisted model run, and the lead line starts at the observation that
 * run actually used as the cell's basis - no value is invented at the cutoff.
 * A carried-forward cell is a grey dashed baseline with hollow squares and says
 * "Baseline only · not a model forecast"; a withheld cell draws nothing and says
 * why; a variable with no runtime cell says so in the future region.
 * A 90% interval is drawn only around a model forecast whose cell the release
 * calibrated (the runtime refuses any other bound); it is a hatched band with
 * end caps, never the pale future region, which is time, not uncertainty.
 * Nothing here depends on colour alone: line style, marker shape, region
 * titles and the point labels carry the same distinction. */

import { esc, fmtNum, INTERVAL_NOTE, TRAJECTORY, unavailableReason } from './format.js';

const COLORS = { ink: '#171A1C', ink2: '#626762', blue: '#344F69', baseline: '#6F746F', divider: '#D1CCC2', soft: '#E7E3DA',
  surface: '#FCFAF5', history: '#FFFFFF', future: '#EDF2F7', cutoff: '#3D4449', teal: '#27645E', nul: '#8B8F89',
  // a current value outside its supplied interval, and a forecast that leaves it:
  // two different claims, so two different colours, each also carrying a shape
  danger: '#8C1D18', amber: '#8A5A0B', band: '#E7EDE4', bandEdge: '#C4D0BE' };
const HOUR = 3600000;
const CHAR = 6.4; // estimated glyph advance at 11px, for label-collision and wrapping decisions

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
    const w = item.text.length * CHAR + 10;
    let lane = 0;
    while (placed.some((p) => p.lane === lane && Math.abs(p.x - item.x) < (p.w + w) / 2)) lane++;
    placed.push({ ...item, w, lane });
  });
  return placed;
}
function labelAnchor(x, w, left, right) {
  // A label too wide for the room it has is centred in that room rather than
  // pinned to one edge: pinning it to the left ran the rightmost forecast label
  // off the right of the frame, where an SVG clips it away.
  if (w >= right - left) return { anchor: 'middle', tx: (left + right) / 2 };
  if (x - w / 2 < left) return { anchor: 'start', tx: left };
  if (x + w / 2 > right) return { anchor: 'end', tx: right };
  return { anchor: 'middle', tx: x };
}
/** Greedy word wrap on the glyph estimate, for captions inside a narrow region. */
function wrap(text, maxWidth) {
  const perLine = Math.max(8, Math.floor(maxWidth / CHAR));
  const lines = [];
  let line = '';
  text.split(' ').forEach((word) => {
    if (line && (line + ' ' + word).length > perLine) { lines.push(line); line = word; }
    else line = line ? line + ' ' + word : word;
  });
  if (line) lines.push(line);
  return lines;
}
function niceStepHours(spanHours, maxTicks) {
  const steps = [0.25, 0.5, 1, 2, 3, 6, 12, 24, 48, 96];
  for (const s of steps) if (spanHours / s <= maxTicks) return s;
  return 96;
}
const n1 = (n) => n.toFixed(1);

/** Small inline sparkline: last N points, no axes. */
export function sparkline(points, width = 120, height = 32) {
  if (!points || points.length < 2) return '';
  const values = points.map((p) => p.value);
  const min = Math.min(...values), max = Math.max(...values), span = max - min || 1;
  const coords = points.map((p, i) => [(i / (points.length - 1)) * (width - 6) + 3, height - 4 - ((p.value - min) / span) * (height - 8)]);
  const last = coords[coords.length - 1];
  return `<svg viewBox="0 0 ${width} ${height}" class="spark" aria-hidden="true"><polyline points="${coords.map((c) => c.map(n1).join(',')).join(' ')}"/><circle cx="${n1(last[0])}" cy="${n1(last[1])}" r="2.6"/></svg>`;
}

/** The accessible description: the same facts the picture shows, in order. */
function describe(t, points, fmt, unit, label, headline) {
  const model = t.forecasts.filter((f) => f.status === 'model' && Number.isFinite(f.forecast_point));
  const baseline = t.forecasts.filter((f) => f.status === 'persistence' && Number.isFinite(f.forecast_point));
  const withheld = t.forecasts.filter((f) => f.status !== 'model' && f.status !== 'persistence');
  const hs = (list) => list.map((f) => `${f.horizon_hours}h`).join(', ');
  const observationsOnly = t.kind === 'observations';
  const parts = [observationsOnly
    ? `${label} ${TRAJECTORY.observedSeries.toLowerCase()}, ${TRAJECTORY.observationsOnly.toLowerCase()}.`
    : `${label} patient trajectory, ${headline.toLowerCase()}.`];
  if (points.length) {
    const first = points[0], last = points[points.length - 1];
    parts.push(`Observed history: ${points.length} FHIR observation${points.length === 1 ? '' : 's'} from ${fmt.short(first.ms)} to ${fmt.short(last.ms)}; last observed ${fmtNum(last.value)} ${unit} at ${fmt.short(last.ms)}.`);
  } else parts.push('Observed history: no observation in view.');
  if (t.cutoff_ms) parts.push(`Forecast starts here: model input cutoff ${fmt.short(t.cutoff_ms)}.`);
  if (observationsOnly) { /* nothing was requested, so nothing is missing */ }
  else if (t.kind === 'no-run') parts.push(`${TRAJECTORY.noRun}.`);
  else if (t.kind === 'none') parts.push(`${TRAJECTORY.noModel}; observations only.`);
  else {
    if (model.length) parts.push(`Future forecast: model point forecasts ${model.map((f) => `${f.horizon_hours}h ${fmtNum(f.forecast_point)}`).join(', ')} ${unit}.`);
    if (baseline.length) parts.push(`${TRAJECTORY.baselineOnly} for ${hs(baseline)}: the last observed value carried forward.`);
    if (withheld.length) parts.push(`${TRAJECTORY.unavailable} for ${hs(withheld)}: ${Array.from(new Set(withheld.map((f) => unavailableReason(f.status, f.fallback_reason)))).join('; ')}.`);
  }
  const bounded = model.filter((f) => Number.isFinite(f.lower) && Number.isFinite(f.upper));
  if (bounded.length) parts.push(`Calibrated 90% intervals: ${bounded.map((f) => `${f.horizon_hours}h ${fmtNum(f.lower)} to ${fmtNum(f.upper)}`).join(', ')} ${unit}.`);
  else if (!observationsOnly) parts.push('No calibrated interval for these forecasts; points only.');
  return parts.join(' ');
}

/* Where a chart is drawn decides how tall it should land, and nothing else does:
 * no caller reads a global, and no frame depends on which chart was built first.
 *
 * A frame is a viewBox, so the drawing is never stretched - it scales uniformly
 * into whatever width its box gets, and the ratio alone fixes the rendered
 * height. The ladder gets taller as the window narrows because a chart that
 * keeps a wide ratio on a phone collapses to an unreadable strip; it stops
 * getting wider past INLINE_MAX_WIDTH because on a very wide monitor the same
 * ratio would grow the drawing towards a whole screen of chart.
 *
 * `inline-detail` is the panel that opens inside the results list. It is read
 * together with the comparison and the forecast cells above it and the
 * "Open full trajectory" link below, so it is deliberately shorter than the
 * standalone trajectory page, which is the analysis view and keeps its own
 * larger frame (see render.js). Each rung's viewBox is close to the width that
 * rung's typical window actually gives the panel, so the 11px labels render at
 * about their true size instead of being scaled up or down.
 */
export const INLINE_MAX_WIDTH = 1500;
export const CHART_FRAMES = {
  'inline-detail': [
    { minWidth: 1366, width: 1224, height: 335 },
    { minWidth: 1152, width: 1088, height: 320 },
    { minWidth: 900, width: 861, height: 287 },
    { minWidth: 768, width: 660, height: 275 },
    { minWidth: 600, width: 620, height: 310 },
    { minWidth: 480, width: 440, height: 275 },
    { minWidth: 0, width: 310, height: 267 },
  ],
};

/** The frame for a named context at a given window width. The context is named
 *  by the caller, never inferred. */
export function chartFrame(context, viewportWidth) {
  const ladder = CHART_FRAMES[context];
  if (!ladder) throw new Error(`chartFrame: no frame is defined for context "${context}"`);
  const vw = Number.isFinite(viewportWidth) ? viewportWidth : 1440;
  return ladder.find((f) => vw >= f.minWidth) || ladder[ladder.length - 1];
}

/**
 * The chart. opts:
 *   traj      from store.trajectoryFor(): { points, cutoff_ms, forecasts, kind, frozen }
 *   unit, label, axis ('calendar'|'hours_since_admission'), onset_ms
 *   window ('recent'|'full'), fresh_ms, width, height
 *   headline  'LIVE HEAD' | 'HISTORICAL SNAPSHOT' (for the description)
 *   idSuffix  distinguishes the title/desc ids when one page draws several charts
 *   context   where this chart is drawn, e.g. 'inline-detail'; recorded on the
 *             root so a stylesheet can size the one place without reaching the
 *             others. Sizes come from chartFrame(context, width), not from here.
 *   range     { low, high, unit } the observation supplied: drawn as a band behind
 *             the series, so a reader sees where the values sit without arithmetic
 * A traj of kind 'observations' (no run, no cutoff, no cells) is drawn as the
 * observed series alone and says so; it never mentions a model run.
 */
export function trajectoryChart(opts) {
  const t = opts.traj;
  const W = opts.width || 980, H = opts.height || 360;
  const narrow = W < 520;
  const hasCutoff = Number.isFinite(t.cutoff_ms);
  const PL = narrow ? 46 : 58, PR = narrow ? 10 : 16, PT = 46, PB = hasCutoff ? 66 : 40;
  const plotW = W - PL - PR, plotBottom = H - PB;
  const fmt = makeTimeFormatter(opts.axis, opts.onset_ms);
  const all = t.points || [];
  const forecasts = t.forecasts || [];
  const model = forecasts.filter((f) => f.status === 'model' && Number.isFinite(f.forecast_point));
  const baseline = forecasts.filter((f) => f.status === 'persistence' && Number.isFinite(f.forecast_point));
  const withheld = forecasts.filter((f) => f.status !== 'model' && f.status !== 'persistence');
  const drawn = model.concat(baseline);

  let points = all;
  if (opts.window === 'recent' && all.length) {
    const end = hasCutoff ? t.cutoff_ms : all[all.length - 1].ms;
    points = all.filter((p) => p.ms >= end - 12 * HOUR);
    if (points.length < 2) points = all.slice(-2);
  }
  // the observation each cell actually used stays in view whatever the window
  const bases = [];
  forecasts.forEach((f) => { if (f.basis && !bases.some((b) => b.ms === f.basis.ms)) bases.push(f.basis); });
  if (!points.length && !forecasts.length) return { svg: '', empty: true };

  // one linear time scale for both regions; the extent only ever grows
  const xs = points.map((p) => p.ms).concat(bases.map((b) => b.ms));
  if (hasCutoff) xs.push(t.cutoff_ms);
  forecasts.forEach((f) => xs.push(f.due_ms));   // withheld cells still claim their due time
  let xMin = Math.min(...xs), xMax = Math.max(...xs);
  if (xMax - xMin < HOUR) { xMin -= HOUR / 2; xMax += HOUR / 2; }
  const short = plotW < 250;
  const titles = { history: short ? 'HISTORY' : TRAJECTORY.historyTitle, future: short ? 'FORECAST' : TRAJECTORY.forecastTitle };
  if (hasCutoff) {
    // Neither region collapses: each keeps room for its own title (and the future
    // region for its caption when it has no points). Still the same scale.
    const captionWidth = (t.kind === 'none' || t.kind === 'unavailable') ? Math.min(plotW * 0.5, 34 * CHAR) : 0;
    const need = Math.max(0.22, (Math.max(titles.future.length * (CHAR + 0.6), captionWidth) + 16) / plotW);
    const MIN = Math.min(0.5, need);
    const past = t.cutoff_ms - xMin, future = xMax - t.cutoff_ms;
    if (future < MIN * (past + future)) xMax = t.cutoff_ms + (MIN / (1 - MIN)) * past;
    if (t.cutoff_ms - xMin < MIN * (xMax - xMin)) xMin = t.cutoff_ms - (MIN / (1 - MIN)) * (xMax - t.cutoff_ms);
  }
  const xPad = (xMax - xMin) * 0.03;
  xMin -= xPad; xMax += xPad;

  const band = opts.range && (Number.isFinite(opts.range.low) || Number.isFinite(opts.range.high)) ? opts.range : null;
  const ys = points.map((p) => p.value).concat(bases.map((b) => b.value), drawn.map((f) => f.forecast_point),
    model.flatMap((f) => [f.lower, f.upper])).filter((v) => Number.isFinite(v));
  if (!ys.length) return { svg: '', empty: true };
  // the band is kept in view, so "inside" and "outside" are visible and not implied
  if (band) [band.low, band.high].filter((v) => Number.isFinite(v)).forEach((v) => ys.push(v));
  let yMin = Math.min(...ys), yMax = Math.max(...ys);
  if (yMax - yMin < 1e-6) yMax = yMin + Math.max(1, Math.abs(yMin) * 0.1);
  const rawMin = yMin;
  const pad = (yMax - yMin) * 0.2;
  yMin -= pad; yMax += pad;
  if (rawMin >= 0 && yMin < 0) yMin = 0;
  const X = (ms) => PL + ((ms - xMin) / (xMax - xMin)) * plotW;
  const Y = (v) => PT + (1 - (v - yMin) / (yMax - yMin)) * (plotBottom - PT);
  const cx = hasCutoff ? X(t.cutoff_ms) : null;
  const unit = opts.unit || '';
  const headline = opts.headline || (t.frozen ? TRAJECTORY.snapshot : TRAJECTORY.liveHead);

  const sfx = opts.idSuffix ? `-${String(opts.idSuffix).replace(/[^A-Za-z0-9_-]+/g, '-')}` : '';
  const observationsOnly = t.kind === 'observations';
  const context = opts.context || 'standalone';
  let s = `<svg viewBox="0 0 ${W} ${H}" class="bigchart" data-context="${esc(context)}" role="group" aria-labelledby="trajTitle${sfx}" aria-describedby="trajDesc${sfx}" data-kind="${esc(t.kind)}" data-head="${observationsOnly ? 'observations' : t.frozen ? 'snapshot' : 'live'}">`;
  s += `<title id="trajTitle${sfx}">${esc(opts.label)} · ${observationsOnly ? TRAJECTORY.observedSeries : TRAJECTORY.title}</title><desc id="trajDesc${sfx}">${esc(describe(t, points, fmt, unit, opts.label, headline))}</desc>`;

  // the supplied reference interval, behind everything: a band, not a threshold
  if (band) {
    const top = Y(Number.isFinite(band.high) ? band.high : yMax);
    const bottom = Y(Number.isFinite(band.low) ? band.low : yMin);
    s += `<rect data-region="range" x="${PL}" y="${n1(Math.min(top, bottom))}" width="${n1(plotW)}" height="${n1(Math.abs(bottom - top))}" fill="${COLORS.band}"/>`;
    [band.low, band.high].filter((v) => Number.isFinite(v)).forEach((v) => {
      s += `<line data-range="limit" x1="${PL}" y1="${n1(Y(v))}" x2="${n1(W - PR)}" y2="${n1(Y(v))}" stroke="${COLORS.bandEdge}" stroke-dasharray="6 3"/>`;
    });
    s += `<text x="${n1(W - PR)}" y="${n1(Math.min(top, bottom) - 3)}" text-anchor="end" class="rg-s">${esc(`Supplied range ${fmtNum(band.low)}–${fmtNum(band.high)}${band.unit ? ` ${band.unit}` : ''}`)}</text>`;
  }
  // regions: history ground, then the future ground (the future, not a band of uncertainty)
  const historyRight = cx === null ? W - PR : cx;
  s += `<rect data-region="history" x="${PL}" y="0" width="${n1(Math.max(0, historyRight - PL))}" height="${n1(plotBottom)}" fill="${COLORS.history}" opacity="0.7"/>`;
  if (cx !== null) s += `<rect data-region="future" x="${n1(cx)}" y="0" width="${n1(Math.max(0, W - PR - cx))}" height="${n1(plotBottom)}" fill="${COLORS.future}"/>`;
  // region titles: outer edges, so they cannot overlap each other whatever the split
  s += `<text x="${PL}" y="16" class="rg-t">${esc(titles.history)}</text><text x="${PL}" y="30" class="rg-s">${esc(TRAJECTORY.historySub)}</text>`;
  if (cx !== null) s += `<text x="${W - PR}" y="16" text-anchor="end" class="rg-t future">${esc(titles.future)}</text><text x="${W - PR}" y="30" text-anchor="end" class="rg-s">${esc(TRAJECTORY.forecastSub)}</text>`;
  else s += `<text x="${W - PR}" y="16" text-anchor="end" class="rg-s">${esc(observationsOnly ? TRAJECTORY.observationsOnly : TRAJECTORY.noRun)}</text>`;
  // The unit belongs to the y axis and sits in the gutter above the ticks -- but
  // a unit wider than the gutter ("breaths/min", "beats/min") was drawn from a
  // negative x and clipped off the frame at every chart width. One that does not
  // fit the gutter is stacked under the region caption instead, where it costs no
  // horizontal room and stays inside the frame. Nothing else moves: a unit that
  // fits the gutter is drawn exactly where it always was.
  s += unit.length * CHAR <= PL - 12
    ? `<text x="${PL - 8}" y="30" text-anchor="end" class="ct">${esc(unit)}</text>`
    : `<text x="${PL}" y="42" class="ct" data-unit="axis-top">${esc(unit)}</text>`;

  // y grid; a near-flat series needs more decimals than the value format gives, or ticks repeat
  let need = 0;
  while (need < 4 && ((yMax - yMin) / 4) * 10 ** need < 1) need++;
  const tickLabel = (v) => { const base = fmtNum(v); return (base.split('.')[1] || '').length >= need ? base : v.toFixed(need); };
  [yMax, yMin + (yMax - yMin) * 0.75, (yMax + yMin) / 2, yMin + (yMax - yMin) * 0.25, yMin].forEach((v) => {
    const y = Y(v);
    s += `<line x1="${PL}" y1="${n1(y)}" x2="${W - PR}" y2="${n1(y)}" stroke="${COLORS.soft}"/>`;
    s += `<text x="${PL - 8}" y="${n1(y + 4)}" text-anchor="end" class="ct">${esc(tickLabel(v))}</text>`;
  });
  // x axis
  s += `<line x1="${PL}" y1="${plotBottom}" x2="${W - PR}" y2="${plotBottom}" stroke="${COLORS.divider}"/>`;
  const step = niceStepHours((xMax - xMin) / HOUR, Math.max(3, Math.floor(plotW / 110)));
  const origin = opts.axis === 'hours_since_admission' ? opts.onset_ms : 0;
  let tick = Math.ceil((xMin - origin) / (step * HOUR)) * step * HOUR + origin;
  let first = true;
  for (; tick <= xMax; tick += step * HOUR) {
    const x = X(tick);
    s += `<line x1="${n1(x)}" y1="${plotBottom}" x2="${n1(x)}" y2="${plotBottom + 4}" stroke="${COLORS.divider}"/>`;
    s += `<text x="${n1(x)}" y="${plotBottom + 17}" text-anchor="middle" class="ct">${esc(fmt.tick(tick, first))}</text>`;
    first = false;
  }

  // the cutoff rule spans the whole plot, drawn under the data so it never covers a label
  if (cx !== null) {
    s += `<line data-cutoff="rule" x1="${n1(cx)}" y1="0" x2="${n1(cx)}" y2="${plotBottom + 6}" stroke="${COLORS.cutoff}" stroke-dasharray="4 3"/>`;
    s += `<polygon points="${n1(cx - 4)},${plotBottom + 10} ${n1(cx + 4)},${plotBottom + 10} ${n1(cx)},${plotBottom + 5}" fill="${COLORS.cutoff}"/>`;
    const l1 = TRAJECTORY.cutoffLabel, l2 = `${TRAJECTORY.cutoffSub} · ${fmt.short(t.cutoff_ms)}`;
    const w = Math.max(l1.length, l2.length) * CHAR;
    const { anchor, tx } = labelAnchor(cx, w, PL, W - PR);
    s += `<text data-cutoff="label" x="${n1(tx)}" y="${plotBottom + 36}" text-anchor="${anchor}" class="cutoff-l">${esc(l1)}</text>`;
    s += `<text x="${n1(tx)}" y="${plotBottom + 50}" text-anchor="${anchor}" class="cutoff-s">${esc(l2)}</text>`;
  }

  // captions in the future region: what a reader sees when there is no forecast line
  const futureW = cx === null ? 0 : W - PR - cx;
  const caption = (lines, y, cls, fill) => {
    const inner = Math.max(40, futureW - 16);
    const rows = lines.flatMap((l) => wrap(l, inner));
    const fits = rows.every((r) => r.length * CHAR <= inner);
    const x = fits ? cx + 8 : W - PR - 6;
    return rows.map((r, i) => `<text x="${n1(x)}" y="${n1(y + i * 14)}" text-anchor="${fits ? 'start' : 'end'}" class="${cls}" fill="${fill}">${esc(r)}</text>`).join('');
  };
  const labelLeft = cx === null ? PL + 2 : Math.max(PL + 2, cx + 4);
  const withheldReasons = Array.from(new Set(withheld.map((f) => unavailableReason(f.status, f.fallback_reason))));
  if (cx !== null && t.kind === 'none') {
    s += `<g data-note="no-model">${caption([TRAJECTORY.noModel, 'observations only'], (PT + plotBottom) / 2 - 4, 'cl', COLORS.ink2)}</g>`;
  } else if (cx !== null && t.kind === 'unavailable') {
    s += `<g data-note="unavailable">${caption([TRAJECTORY.unavailable].concat(withheldReasons), (PT + plotBottom) / 2 - 4, 'cl', COLORS.ink2)}</g>`;
  } else if (cx !== null && withheld.length) {
    // some horizons drawn, others withheld: say which, at the foot of the future region
    const lines = [`${withheld.map((f) => `${f.horizon_hours}h`).join(' · ')} · ${TRAJECTORY.unavailable}`].concat(withheldReasons);
    s += `<g data-note="partial-unavailable">${caption(lines, plotBottom - 12 - (lines.length - 1) * 14, 'cl', COLORS.ink2)}</g>`;
    placeLabels(withheld.map((f) => ({ x: X(f.due_ms), text: `${f.horizon_hours}h`, f }))).forEach(({ x, text, w, lane }) => {
      const ty = plotBottom - 9 - lane * 12;
      const { anchor, tx } = labelAnchor(x, w, labelLeft, W - PR - 2);
      s += `<line x1="${n1(x)}" y1="${n1(ty + 3)}" x2="${n1(x)}" y2="${plotBottom}" stroke="${COLORS.nul}"/><text x="${n1(tx)}" y="${n1(ty)}" text-anchor="${anchor}" class="cl" fill="${COLORS.ink2}">${esc(text)}</text>`;
    });
  }

  // calibrated 90% interval: one band from the basis observation (zero width) through each bounded horizon
  const bounded = model.filter((f) => Number.isFinite(f.lower) && Number.isFinite(f.upper));
  if (bounded.length) {
    const lead = bounded.find((f) => f.basis);
    const start = lead ? [[X(lead.basis.ms), Y(lead.basis.value)]] : [];
    const top = start.concat(bounded.map((f) => [X(f.due_ms), Y(f.upper)]));
    const bottom = bounded.map((f) => [X(f.due_ms), Y(f.lower)]).reverse().concat(start);
    s += `<polygon data-series="interval" points="${top.concat(bottom).map((p) => p.map(n1).join(',')).join(' ')}" fill="${COLORS.blue}" fill-opacity="0.12" stroke="none"/>`;
    bounded.forEach((f) => {
      const x = X(f.due_ms);
      s += `<g data-interval="${esc(f.cell)}" data-lower="${f.lower}" data-upper="${f.upper}"><line x1="${n1(x)}" y1="${n1(Y(f.upper))}" x2="${n1(x)}" y2="${n1(Y(f.lower))}" stroke="${COLORS.blue}" stroke-width="1" stroke-opacity="0.55"/>`
        + `<line x1="${n1(x - 4)}" y1="${n1(Y(f.upper))}" x2="${n1(x + 4)}" y2="${n1(Y(f.upper))}" stroke="${COLORS.blue}" stroke-width="1.2"/>`
        + `<line x1="${n1(x - 4)}" y1="${n1(Y(f.lower))}" x2="${n1(x + 4)}" y2="${n1(Y(f.lower))}" stroke="${COLORS.blue}" stroke-width="1.2"/></g>`;
    });
  }

  // carried-forward baselines: grey, horizontal, dashed, hollow squares, and named as a baseline
  const point = (f) => `data-kind="forecast" data-cell="${esc(f.cell)}" data-run="${esc(f.run_id)}" data-status="${esc(f.status)}" data-value="${f.forecast_point}" data-ms="${f.due_ms}" data-source="${esc(f.source_model)}" data-horizon="${f.horizon_hours}"${f.basis ? ` data-basis-value="${f.basis.value}" data-basis-ms="${f.basis.ms}"` : ''}`;
  if (baseline.length && cx !== null) {
    s += `<g data-note="baseline">${caption([TRAJECTORY.baselineOnly], PT + 16, 'cl', COLORS.baseline)}</g>`;
    placeLabels(baseline.map((f) => ({ x: X(f.due_ms), y: Y(f.forecast_point), text: `${f.horizon_hours}h · ${fmtNum(f.forecast_point)}`, f }))).forEach(({ x, y, text, f, w, lane }) => {
      const x1 = f.basis ? X(f.basis.ms) : cx;
      s += `<line data-series="baseline" x1="${n1(x1)}" y1="${n1(y)}" x2="${n1(x)}" y2="${n1(y)}" stroke="${COLORS.baseline}" stroke-width="1.4" stroke-dasharray="3 4"/>`;
      const { anchor, tx } = labelAnchor(x, w, labelLeft, W - PR - 2);
      const ty = y + 17 + (lane + (anchor === 'middle' ? 0 : 1)) * 13;
      const leader = lane > 0 ? `<line x1="${n1(x)}" y1="${n1(y + 5)}" x2="${n1(x)}" y2="${n1(ty - 9)}" stroke="${COLORS.divider}"/>` : '';
      const aria = `${f.horizon_hours}h baseline only, last value carried forward, not a model forecast: ${fmtNum(f.forecast_point)} ${unit} at ${fmt.short(f.due_ms)}`;
      s += `<g class="chart-pt" tabindex="0" role="button" aria-label="${esc(aria)}" ${point(f)}>${leader}<rect x="${n1(x - 4.5)}" y="${n1(y - 4.5)}" width="9" height="9" fill="${COLORS.surface}" stroke="${COLORS.baseline}" stroke-width="1.6"/><text x="${n1(tx)}" y="${n1(ty)}" text-anchor="${anchor}" class="cl" fill="${COLORS.baseline}">${esc(text)}</text></g>`;
    });
  }

  // model forecast: one dashed path per run, from the cell's own basis observation to each horizon
  if (model.length) {
    const lead = model.find((f) => f.basis);
    const pts = (lead ? [[X(lead.basis.ms), Y(lead.basis.value)]] : []).concat(model.map((f) => [X(f.due_ms), Y(f.forecast_point)]));
    s += `<polyline data-series="model" fill="none" stroke="${COLORS.blue}" stroke-width="1.6" stroke-dasharray="5 4" points="${pts.map((p) => p.map(n1).join(',')).join(' ')}"/>`;
    placeLabels(model.map((f) => ({ x: X(f.due_ms), y: Y(f.forecast_point), text: `${f.horizon_hours}h · ${fmtNum(f.forecast_point)}`, f }))).forEach(({ x, y, text, f, w, lane }) => {
      const r = 5.5;
      const { anchor, tx } = labelAnchor(x, w, labelLeft, W - PR - 2);
      // a label pushed off centre by the chart edge would sit on its own marker; lift it one lane
      const ty = y - 11 - (lane + (anchor === 'middle' ? 0 : 1)) * 13;
      const leader = lane > 0 ? `<line x1="${n1(x)}" y1="${n1(y - r - 1)}" x2="${n1(x)}" y2="${n1(ty + 3)}" stroke="${COLORS.divider}"/>` : '';
      // a point that leaves the supplied interval is amber and hollow, never the
      // deep red a current measured abnormality carries: a projection is not a result
      const ink = f.projected ? COLORS.amber : COLORS.blue;
      const aria = `${f.horizon_hours}h model forecast: ${fmtNum(f.forecast_point)} ${unit} at ${fmt.short(f.due_ms)}`
        + (Number.isFinite(f.lower) && Number.isFinite(f.upper) ? `, 90% interval ${fmtNum(f.lower)} to ${fmtNum(f.upper)} ${unit}` : '')
        + (f.projected ? `, projected ${f.projectedSide} the supplied range` : '');
      s += `<g class="chart-pt" tabindex="0" role="button" aria-label="${esc(aria)}" ${point(f)}${f.projected ? ` data-projected="${esc(f.projectedSide || 'outside')}"` : ''}>${leader}<polygon points="${n1(x)},${n1(y - r)} ${n1(x + r)},${n1(y)} ${n1(x)},${n1(y + r)} ${n1(x - r)},${n1(y)}" fill="${COLORS.surface}" stroke="${ink}" stroke-width="${f.projected ? 2 : 1.7}"${f.projected ? ' stroke-dasharray="3 2"' : ''}/><text x="${n1(tx)}" y="${n1(ty)}" text-anchor="${anchor}" class="cl fc" fill="${ink}">${esc(text)}${f.projected ? ` ${f.projectedSide === 'below' ? '▽' : '△'}` : ''}</text></g>`;
    });
  }

  // observed history: solid ink line, filled circles; the run's basis observation is ringed.
  // One point is one point: no line is drawn through a single observation, because a
  // line between nothing and nothing is a trend the data does not have.
  if (points.length) {
    if (points.length > 1) {
      s += `<polyline data-series="observed" fill="none" stroke="${COLORS.ink}" stroke-width="1.8" stroke-linejoin="round" points="${points.map((p) => `${n1(X(p.ms))},${n1(Y(p.value))}`).join(' ')}"/>`;
    } else {
      s += `<g data-note="single-observation"><text x="${PL + 8}" y="${n1(PT + 14)}" class="cl" fill="${COLORS.ink2}">Single observation · no trend drawn</text></g>`;
    }
    const lastMs = points[points.length - 1].ms;
    points.forEach((p) => {
      const x = X(p.ms), y = Y(p.value);
      const fresh = opts.fresh_ms && p.ms === opts.fresh_ms;
      const isBasis = bases.some((b) => Math.abs(b.ms - p.ms) < 1000);
      const isCurrent = p.ms === lastMs;
      // the current result reads as a measured fact even in the band: deep red only
      // when the observation itself is outside the interval it was compared against
      const outside = isCurrent && band && ((Number.isFinite(band.low) && p.value < band.low)
        || (Number.isFinite(band.high) && p.value > band.high));
      const aria = `Observed ${fmt.short(p.ms)}: ${fmtNum(p.value)} ${unit}${isCurrent ? ', the current result' : ''}${outside ? ', outside the supplied range' : ''}${isBasis ? ', the observation this run used as its basis' : ''}`;
      s += `<g class="chart-pt${fresh ? ' fresh' : ''}${isBasis ? ' basis' : ''}${isCurrent ? ' current' : ''}" tabindex="0" role="button" aria-label="${esc(aria)}" data-kind="observed"${isCurrent ? ' data-current="true"' : ''}${outside ? ' data-outside="true"' : ''} data-resource="${esc(p.resource_id)}" data-version="${esc(p.version_id)}" data-value="${p.value}" data-ms="${p.ms}" data-received="${esc(p.received_at || '')}">`
        + (isBasis ? `<circle cx="${n1(x)}" cy="${n1(y)}" r="8" fill="none" stroke="${COLORS.blue}" stroke-width="1" stroke-dasharray="2 2"/>` : '')
        + `<circle cx="${n1(x)}" cy="${n1(y)}" r="${isCurrent ? 5.5 : fresh ? 5 : 4}" fill="${outside ? COLORS.danger : COLORS.ink}"/>`
        + (fresh ? `<circle cx="${n1(x)}" cy="${n1(y)}" r="9" fill="none" stroke="${COLORS.teal}" stroke-width="1.5" class="fresh-ring"/>` : '') + '</g>';
    });
  } else if (cx !== null) {
    s += `<text x="${PL + 8}" y="${n1(plotBottom - 12)}" class="cl" fill="${COLORS.nul}">not observed before the cutoff · nothing imputed</text>`;
  }
  s += '</svg>';
  return { svg: s, empty: false, window_points: points.length, total_points: all.length };
}

/** Legend: every entry pairs a line style and a marker shape with its words, so
 * the same distinction survives without colour. The pale future region is named
 * as the future - never as a confidence band; a calibrated interval has its own entry. */
export function chartLegend(t, range) {
  const hasModel = t.forecasts.some((f) => f.status === 'model');
  const hasBaseline = t.forecasts.some((f) => f.status === 'persistence');
  const hasProjected = t.forecasts.some((f) => f.projected);
  const hasInterval = t.forecasts.some((f) => f.status === 'model' && Number.isFinite(f.lower) && Number.isFinite(f.upper));
  return `<div class="chart-legend" aria-label="Chart legend">
    ${range ? `<span><i class="lg-range"></i>Supplied reference range</span>` : ''}
    ${hasProjected ? `<span><svg width="26" height="12" aria-hidden="true"><polygon points="13,1 18,6 13,11 8,6" fill="${COLORS.surface}" stroke="${COLORS.amber}" stroke-width="2" stroke-dasharray="3 2"/></svg>Projected outside supplied range</span>` : ''}
    <span><svg width="26" height="10" aria-hidden="true"><line x1="1" y1="5" x2="25" y2="5" stroke="${COLORS.ink}" stroke-width="1.8"/><circle cx="13" cy="5" r="3.5" fill="${COLORS.ink}"/></svg>Observed · FHIR observation</span>
    ${hasModel ? `<span><svg width="26" height="12" aria-hidden="true"><line x1="1" y1="6" x2="25" y2="6" stroke="${COLORS.blue}" stroke-width="1.5" stroke-dasharray="5 4"/><polygon points="13,1 18,6 13,11 8,6" fill="${COLORS.surface}" stroke="${COLORS.blue}" stroke-width="1.5"/></svg>Model forecast · point forecast</span>` : ''}
    ${hasInterval ? `<span><svg width="26" height="12" aria-hidden="true"><rect x="1" y="2" width="24" height="8" fill="${COLORS.blue}" fill-opacity="0.12"/><line x1="13" y1="1" x2="13" y2="11" stroke="${COLORS.blue}" stroke-width="1"/></svg>Calibrated 90% interval</span>` : ''}
    ${hasBaseline ? `<span><svg width="26" height="10" aria-hidden="true"><line x1="1" y1="5" x2="25" y2="5" stroke="${COLORS.baseline}" stroke-width="1.4" stroke-dasharray="3 4"/><rect x="9" y="1" width="8" height="8" fill="${COLORS.surface}" stroke="${COLORS.baseline}" stroke-width="1.4"/></svg>${TRAJECTORY.baselineOnly}</span>` : ''}
    ${t.cutoff_ms ? `<span><svg width="12" height="12" aria-hidden="true"><line x1="6" y1="0" x2="6" y2="12" stroke="${COLORS.cutoff}" stroke-dasharray="4 3"/></svg>${TRAJECTORY.cutoffLabel} · ${TRAJECTORY.cutoffSub.toLowerCase()}</span>
    <span><i class="lg-band"></i>${TRAJECTORY.futureRegion}</span>` : ''}
    <span class="lg-note">${INTERVAL_NOTE}</span>
  </div>`;
}

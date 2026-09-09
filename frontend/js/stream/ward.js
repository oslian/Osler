/* Ward attention order: the rail is the order, the workspace is one stay.
 *
 * Same spatial grammar as anatomy and functions -- pick on the left, read on
 * the right -- so the interaction is the one a reader already knows from this
 * page. The order stays on screen while a row is read, which is the point: the
 * list answers "who first" and has to keep answering it while you look at one.
 *
 * Bands, not a quota. A fixed number of rows per shift was tried first and
 * measured worse: on a quiet ward it fills the list with stays nobody needs to
 * read. A band is the furthest out of range any one measured value on the stay
 * is, read off published cutpoints per analyte -- so one value far out puts a
 * stay on the list whether or not everything else is normal, and several mildly
 * out shows as a count beside it. Low bands are collapsed rather than dropped,
 * so no stay is off the list, only further from the first glance. An empty top
 * band is an answer.
 *
 * Two different questions, two different bases, and the page keeps them apart.
 * The band says who belongs on the list and comes from published clinical
 * cutpoints. The order inside one band is the stay's last measured creatinine at
 * any age, which is the only part measured as an ordering. Reading the whole
 * board top to bottom is therefore not the creatinine order, and the page does
 * not claim it is.
 *
 * This view runs no model. The charts draw the observations the bands were read
 * off -- no forecast markers, no cells, no runtime. That is why it can be shown
 * while the runtime is unattached and why it carries no forecast vocabulary.
 *
 * Tags state arithmetic and never a judgement: the multiple of the reference
 * limit, and the age of the measurement. Colour never carries meaning on its
 * own -- the text says it, and the one tinted variant marks an old measurement,
 * which is a fact about the data rather than about the patient.
 *
 * Urine output is stated per kilogram and never ranked. A charted rate in
 * mL/hour compares to nothing on its own, so where the stay has a usable weight
 * the workspace also gives the rate per kg and where it sits against the range
 * quoted for adults. That range is a convention rather than an interval the
 * laboratory prints, unlike the creatinine one, and the page says so. Where the
 * weight is missing the comparison is simply absent and the row says why --
 * never a ward average standing in for a body. None of it touches the order:
 * the rank, the band and the position on the rail are the creatinine's alone.
 */

import { esc, fmtNum } from './format.js';
import { sparkline, trajectoryChart } from './chart.js';
import { S } from './store.js';

const HOUR_MS = 3600000;

const bands = () => (S.ward ? S.ward.bands : []);
const allRows = () => bands().flatMap((b) => b.rows);

export function wardRowFor(id) {
  if (!S.ward) return null;
  return allRows().concat(S.ward.unrankable).find((r) => r.patientId === id) || null;
}

/** The row the workspace should show: the chosen one, else the top of the order. */
export function wardSelection() {
  const chosen = S.wardPatient ? wardRowFor(S.wardPatient) : null;
  if (chosen) return chosen;
  const rows = allRows();
  return rows.length ? rows[0] : (S.ward && S.ward.unrankable[0]) || null;
}

const selectedId = () => { const row = wardSelection(); return row ? row.patientId : null; };
const toPoints = (series) => (series || []).map(([hours, value]) => ({ ms: hours * HOUR_MS, value }));

/** Open unless the reader has said otherwise; the payload decides the default. */
function bandOpen(band) {
  const override = S.wardOpenBands && Object.prototype.hasOwnProperty.call(S.wardOpenBands, band.key);
  return override ? S.wardOpenBands[band.key] : band.openByDefault;
}

/* The per-kg rate, or the reason there is none. Words only, no tag: a second
 * tag beside the band one would read as a second grade, and this is not one. */
function urineLine(row) {
  const board = S.ward;
  const range = `${fmtNum(board.urineReferenceLower, 1)}–${fmtNum(board.urineReferenceUpper, 1)} mL/kg/h`;
  if (row.urineRatePerKg === null || row.urineRatePerKg === undefined) {
    const why = row.weightKg ? 'nothing charted' : 'no weight on record for this stay';
    return `<span class="soft">rate per kg not stated — ${why}</span>`;
  }
  return `${fmtNum(row.urineRatePerKg, 2)} mL/kg/h at ${fmtNum(row.weightKg, 1)} kg ·
    <span class="soft">${esc(row.urineBandLabel)}, against ${range}</span>`;
}

function ageTag(row) {
  if (row.measurementAgeHours === null || row.measurementAgeHours === undefined) return '';
  const h = row.measurementAgeHours;
  const text = h < 1 ? `${Math.round(h * 60)}min` : `${Math.round(h)}h`;
  // The score deliberately has no staleness cap, so the age travels with it.
  return `<span class="tag${row.stale ? ' caution' : ''}">${text}</span>`;
}

/* ── rail: the bands, and the order inside them ─────────────────────────── */

export function renderWardRail(head, list) {
  if (S.wardError) {
    head.innerHTML = '<h2 class="h-sec">Ward attention order</h2>';
    // Never an empty list: an empty ward reads as "nobody needs looking at".
    list.innerHTML = `<li class="ward-note">The ward could not be loaded — ${esc(S.wardError)}. No order is shown rather than an empty one.</li>`;
    return;
  }
  if (!S.ward) {
    head.innerHTML = '<h2 class="h-sec">Ward attention order</h2><span class="meta">loading…</span>';
    list.innerHTML = '';
    return;
  }
  const board = S.ward;
  const pinned = (board.cutpoints || []).filter((c) => c.sourcePinned).length;
  const total = (board.cutpoints || []).length;
  head.innerHTML = `<h2 class="h-sec">Ward attention order</h2>
    <span class="meta">${esc(board.overview)}</span>
    <span class="meta ward-basis-note">a band is the furthest out of range any one measured
      value is, over ${total} analytes read off published cutpoints
      (${pinned} of ${total} from a source this project can cite) ·
      order inside a band is the last measured creatinine</span>`;

  const item = (row) => {
    const spark = sparkline(toPoints(row.series && row.series.creatinine).slice(-8), 54, 18);
    // The band came from one analyte, and the row names it. Without this the
    // reader sees a creatinine of 0.6 at the top of the list and no reason why.
    const lead = row.findings && row.findings[0];
    const also = row.outsideCount > 1 ? ` +${row.outsideCount - 1}` : '';
    const why = lead
      ? `<span class="wy">${esc(lead.label)} ${fmtNum(lead.value)}${esc(also)}</span>`
      : '';
    return `<li><button type="button" data-act="select-ward-patient" data-patient="${esc(row.patientId)}"
      aria-pressed="${selectedId() === row.patientId}" class="ward">
      <span class="rk">${row.rank}</span>
      <span class="t">${esc(row.displayLabel)}<span class="unit">${esc(row.careUnit || '')}</span>${why}</span>
      <span class="n">${spark}${fmtNum(row.score)}${ageTag(row)}</span></button></li>`;
  };

  const group = (band) => {
    if (!band.count) {
      // An empty band is reported, not hidden: on a quiet ward that is the answer.
      return `<li class="ward-band empty"><span class="bl">${esc(band.label)}</span><span class="bc">none</span></li>`;
    }
    const open = bandOpen(band);
    const headRow = `<li class="ward-band"><button type="button" data-act="ward-band" data-band="${esc(band.key)}"
        aria-expanded="${open}"><span class="bl">${esc(band.label)}</span>
        <span class="bc">${band.count} ${open ? '⌃' : '⌄'}</span></button></li>`;
    if (!open) return headRow;
    const held = band.truncated
      ? `<li class="ward-note">${band.truncated} more in this band are not drawn (display limit ${board.displayCap}). They are counted above, not excluded.</li>`
      : '';
    return headRow + band.rows.map(item).join('') + held;
  };

  const unscored = board.unrankable.length
    ? `<li class="ward-note"><b>Not scored — ${board.unrankable.length}</b><br>${board.unrankable.map((r) => `<button type="button" class="disc" data-act="select-ward-patient" data-patient="${esc(r.patientId)}" aria-pressed="${selectedId() === r.patientId}">${esc(r.displayLabel)}</button>`).join(' · ')}<br>${esc(board.unrankable[0].unrankableReason)}. Listed apart, not ranked low — this rule cannot place them at all.</li>`
    : '';
  list.innerHTML = bands().map(group).join('') + unscored;
}

/* ── workspace: one stay ────────────────────────────────────────────────── */

/* Every measured analyte that is out of its range, worst first, with the value,
 * the range it is being read against and how old it is. Split out per analyte
 * rather than summed into one number: a total would hide which value is the one
 * to look at, and this list is the whole reason the stay is on screen. */
function findingsTable(row) {
  if (!row.findings || !row.findings.length) {
    return '<p class="empty-note">Every measured value is inside its reference range. '
      + 'Nothing on this stay put it on the list.</p>';
  }
  const rows = row.findings.map((f) => {
    const age = f.ageHours < 1 ? `${Math.round(f.ageHours * 60)}min` : `${Math.round(f.ageHours)}h`;
    const range = `${fmtNum(f.referenceLower, 1)}–${fmtNum(f.referenceUpper, 1)}`;
    // The direction is stated in words; nothing here is coloured by how far out
    // it is, which would be a severity scale wearing a different name.
    const side = f.high ? 'above' : 'below';
    const unpinned = f.sourcePinned ? ''
      : '<span class="soft"> · cutpoint not yet from a citable source</span>';
    return `<tr><th scope="row">${esc(f.label)}</th>
      <td class="v">${fmtNum(f.value)} <span class="soft">${esc(f.unit)}</span></td>
      <td><span class="tag">${esc(f.tierKey)}</span></td>
      <td class="soft">${side} ${range}${unpinned}</td>
      <td class="soft">${age}</td></tr>`;
  }).join('');
  return `<table class="ward-findings"><tbody>${rows}</tbody></table>`;
}


function seriesBlock(row, variable, label, unit, note) {
  const points = toPoints(row.series && row.series[variable]);
  if (!points.length) {
    // No note here on purpose: with nothing charted there is no rate to restate,
    // and the empty line below already says the series is empty.
    return `<div class="sec">${esc(label)}</div><p class="empty-note">Nothing recorded for this stay. Nothing is invented.</p>`;
  }
  // An empty `forecasts` and no cutoff, both deliberately. This view draws
  // observations and nothing a model produced, so there are no forecast markers
  // and no cutoff rule -- and the rule's label would have to read either
  // "model input cutoff" or "historical anchor", and neither is true here.
  // The traj names itself observation-only, so the chart says that rather
  // than "no model run yet": no run was ever requested.
  const chart = trajectoryChart({
    traj: { points, forecasts: [], kind: 'observations' }, unit, label,
    axis: 'hours_since_admission', onset_ms: 0, window: 'full',
    width: 720, height: 190, idSuffix: variable,
  });
  const latest = points[points.length - 1];
  // A single observation is still drawn -- one dot on a padded axis is the honest
  // picture of what was recorded, and the count above it says so.
  return `<div class="sec">${esc(label)} · <span class="soft">${points.length} observation${points.length === 1 ? '' : 's'} · latest ${fmtNum(latest.value)} ${esc(unit)}</span></div>
    ${note ? `<p class="ward-rate">${note}</p>` : ''}
    <div class="chart-wrap ward-chart">${chart.svg}</div>`;
}

export function renderWardWorkspace(host) {
  if (S.wardError || !S.ward) {
    host.innerHTML = `<p class="empty-note">${S.wardError ? esc(S.wardError) : 'Loading the ward…'}</p>`;
    return;
  }
  const row = wardSelection();
  if (!row) { host.innerHTML = '<p class="empty-note">This ward document contains no stays.</p>'; return; }
  const source = S.ward.source || {};
  // There is no band heading above the workspace, so the band travels as a tag.
  const tags = row.rankable
    ? `<span class="tag">${esc(row.bandLabel)}</span>${ageTag(row)}`
    : '<span class="tag">not scored</span>';
  // Out of every stay this rule could score, not out of the rows currently drawn.
  const scored = S.ward.totalPatients - S.ward.unrankable.length;
  const rankLine = row.rankable
    ? `${row.rank} of ${scored} reading down the board · creatinine ${fmtNum(row.score)} measured ${fmtNum(row.measurementAgeHours, 0)}h ago`
    : esc(row.unrankableReason);
  // The order answers "who first"; this is how a reader then reads that stay the
  // way the rest of the page reads one -- same timeline, same organ figure, same
  // provenance. It opens a session on the stay's own recorded observations, so
  // what loads is the same data the row was placed on.
  const open = `<button type="button" class="ward-open" data-act="open-ward-stay"
      data-patient="${esc(row.patientId)}">Open this stay</button>`;
  host.innerHTML = `
    <div class="ws-h"><h2>${esc(row.displayLabel)}</h2>
      <span class="sub">${esc(row.careUnit || 'unknown unit')} · ${rankLine}</span>
      <span class="fn">${tags}${open}</span></div>
    <p class="note ward-note-inline">Observations only · this view runs no model and requests no forecast</p>
    <div class="sec">What put this stay on the list</div>
    ${findingsTable(row)}
    <div class="sec">Recent context</div>
    <ul class="ward-why">${row.reasons.length
      ? row.reasons.map((r) => `<li>${esc(r)}</li>`).join('')
      : '<li class="soft">No observation this rule reads has been recorded for this stay.</li>'}</ul>
    ${seriesBlock(row, 'creatinine', 'Creatinine', 'mg/dL')}
    ${seriesBlock(row, 'urine_output', 'Urine output', 'mL/hour', urineLine(row))}
    <p class="foot"><b>Two different bases, kept apart.</b> The <i>band</i> is the furthest out of
      range any single measured value on the stay is, read off published cutpoints per analyte —
      each analyte's own reference range, and above it the two further cuts published for that
      analyte, because one piece of arithmetic does not fit them all: twice the
      creatinine limit is a real value and twice the sodium limit is 290 mmol/L. One value far out
      is enough on its own, whatever the rest are doing. The <i>order inside a band</i> is each
      stay's last measured creatinine at any age
      (<span class="mono">${esc(S.ward.orderWithinBandBasis || S.ward.scoreBasis)}</span>), which is
      the only part of this measured as an ordering: on the deidentified public demo sets it
      separates the stays whose creatinine moves at least 0.3 mg/dL over the next twelve hours with
      AUROC 0.814 (MIMIC-IV) and 0.819 (eICU). <b>Reading the whole board top to bottom is
      therefore not that order</b>, and this page does not present it as one. No staleness cap on
      the creatinine: capping the input at six hours the way the model path does would drop half the
      ward to a fill value, so the age is tagged instead. It is a reading order for a round — not a
      diagnosis, not a prediction of any value, and not a clinical alert.</p>
    <p class="foot">Urine output is stated per kilogram where the stay has a weight on record, and
      compared to ${fmtNum(S.ward.urineReferenceLower, 1)}–${fmtNum(S.ward.urineReferenceUpper, 1)} mL/kg/h — a
      range conventionally quoted for adults, not an interval the laboratory prints the way the
      creatinine one is. It is one charted interval's rate against that range: no duration is
      required of it and it names no class. It orders nothing and it bands nothing: urine output has
      no cutpoints in the table above, so it can neither put a stay on this list nor move one down
      it. Where no usable weight is recorded the rate is left unstated rather than taken from the
      rest of the ward.
      ${esc(source.label || '')} · ${esc((source.metadata || {}).license || '')} · retrospective, not a live feed.</p>`;
}

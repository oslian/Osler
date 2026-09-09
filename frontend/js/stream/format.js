/* Text helpers every view module needs: HTML escaping and the one number format
 * the page uses. They live here rather than in store.js because they are view
 * concerns, and rather than in render.js because chart.js needs them too. */

export const esc = (s) => String(s == null ? '' : s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

/* One vocabulary for forecast state, used by every view.
 *
 * The audit words are still the truth underneath - a cell is `model`,
 * `persistence`, `missing` or `degraded` and nothing here changes that - but
 * the screen says what each one means to a reader. In particular, a
 * carried-forward value is never called a forecast, because it is not one: the
 * target x horizon did not pass its validation gate, so the cell repeats the
 * patient's last observed value instead. */
export const FORECAST_LABEL = {
  model: 'Model forecast',
  persistence: 'Last value carried forward',
  missing: 'No forecast available',
  degraded: 'Forecast withheld',
};

export const FORECAST_NOTE = {
  model: 'predicted by the model',
  persistence: 'not a model prediction — the last observed value, repeated',
  missing: 'the model returned no value for this target and horizon',
  degraded: 'the model runtime is not attached, so no value is shown',
};

/* A shared head that declined to move emits the observed value and the release
 * still names the head as the source. The runtime badges that cell `persistence`
 * with this reason, and every view reads the reason rather than the equality:
 * the frozen backbone can equal the observation by coincidence and still be a
 * real prediction. A carried-forward value is never a model forecast, whichever
 * way it came to be one. */
export const ABSTAINED_REASON = 'shared_jepa_head_abstained';
export const CARRIED_ABSTAINED = 'Carried forward — model abstained';
export const isAbstained = (cell) => !!cell && cell.status === 'persistence' && cell.fallback_reason === ABSTAINED_REASON;
export function forecastLabel(cell) {
  if (isAbstained(cell)) return CARRIED_ABSTAINED;
  return FORECAST_LABEL[cell.status] || String(cell.status);
}
export function forecastNote(cell) {
  if (isAbstained(cell)) return 'the shared head declined to move, so the observed value is repeated — not a model prediction';
  return FORECAST_NOTE[cell.status] || '';
}

/* Signals & forecasts: what a row can say about an observation's value. The
 * statement is the observation's own (its FHIR interpretation, or a reference
 * range it supplies); this page writes no threshold of its own, so with neither
 * the row says the range was not supplied and is never flagged. */
export const RANGE = {
  none: 'Not yet observed',
  unsupplied: 'Range not supplied',
  unit: 'Range unavailable for this unit',
  within: 'Within supplied range',
  above: 'Above supplied range',
  below: 'Below supplied range',
  outside: 'Outside supplied range',          // a source reading with no direction (A, AA)
  projectedAbove: 'Projected above supplied range',
  projectedBelow: 'Projected below supplied range',
  carried: 'Carried forward',
  unavailable: 'Forecast unavailable',
};

/* The flag beside a result, in the column a laboratory report puts it in. FHIR
 * R4 calls Observation.interpretation "a categorical assessment of an
 * observation value ... high, low, normal", and CLIA (42 CFR 493.1291(c)(6),
 * (d)) asks a report to carry the result with its units and interpretation and
 * to make the pertinent reference interval available. Each flag is a mark and a
 * word, never a colour alone (WCAG 2.2 SC 1.4.1). No flag is an alert: a model
 * forecast never appears in this column. */
export const FLAG = {
  above: { mark: '▲', text: 'HIGH', full: 'Above supplied range' },
  below: { mark: '▼', text: 'LOW', full: 'Below supplied range' },
  within: { mark: '●', text: 'Within range', full: 'Within supplied range' },
  outside: { mark: '◆', text: 'ABNORMAL', full: 'Outside supplied range' },
  unsupplied: { mark: '○', text: 'Range not supplied', full: 'No reference range or coded reading supplied' },
  unit: { mark: '—', text: 'Range unavailable for this unit', full: 'A range was supplied in another unit and is not compared' },
  none: { mark: '·', text: 'Not yet observed', full: 'Nothing delivered for this indicator' },
};
export const PROJECTED = {
  above: { mark: '△', text: 'Projected high', full: 'Projected above supplied range' },
  below: { mark: '▽', text: 'Projected low', full: 'Projected below supplied range' },
};

/** The flag key for one indicator row's observation. */
export function flagOf(row) {
  if (!row.state) return 'none';
  const a = row.attention;
  if (a.kind === 'outside') return a.side || 'outside';
  if (a.kind === 'within') return 'within';
  if (a.kind === 'unit') return 'unit';
  return 'unsupplied';
}

/* What one forecast cell says at the compared horizon, in words that never call
 * a repeated observation a prediction. */
export const FORECAST_STATE = {
  model: 'model',
  persistence: 'carried forward',
  abstained: 'carried forward — model abstained',
  missing: 'no value produced',
  degraded: 'withheld',
};
export function forecastState(cell) {
  if (!cell) return null;
  if (isAbstained(cell)) return FORECAST_STATE.abstained;
  return FORECAST_STATE[cell.status] || cell.status;
}

/* The printed report. Oslian is not a testing laboratory and issues no clinical
 * report, so the title says what it is; the header still carries what CLIA
 * 493.1291(c) asks any result report to carry (who the subject is, what was
 * measured, when, in what units, against what interval) and what FHIR R4
 * DiagnosticReport carries as context (status, issued, effective, performer). */
export const REPORT = {
  title: 'Oslian Monitoring Research Report',
  boundary: 'Research use only · synthetic or retrospective stream, not a live feed · not a clinical laboratory report · no diagnosis, causal claim, treatment recommendation or clinical alert',
  system: 'Reporting system: Oslian research demonstration',
  status: 'Report status: research demonstration output · not verified by a clinician',
  attentionHeading: 'Results requiring attention',
  otherHeading: 'Other observed results',
  forecastHeading: 'Oslian forecast outlook',
  forecastNote: 'Research-only point forecasts. No calibrated prediction interval is available for these cells.',
  chartsHeading: 'Trends',
  anatomyHeading: 'Anatomy overview',
  provenanceHeading: 'Provenance appendix',
  limitsHeading: 'Report limitations',
  noAttention: 'No observed result is outside a supplied reference interval.',
  scopes: { full: 'Full patient report', attention: 'Needs attention only', selected: 'Selected indicators' },
};
export const BUILDER = {
  title: 'Report preview',
  scope: 'Report scope',
  snapshot: 'Snapshot',
  horizon: 'Forecast comparison',
  sections: 'Optional sections',
  charts: 'Include trend charts',
  anatomy: 'Include anatomy overview',
  provenance: 'Include provenance appendix',
  back: 'Back to monitoring',
  print: 'Print / Save PDF',
  refresh: 'Refresh report',
  frozen: 'This report is frozen at the snapshot below. Later stream events do not change it.',
};
/* Every status has a shape as well as its words, so a black-and-white print and a
 * reader who does not see colour get the same distinction. */
export const STATUS_MARK = {
  above: '▲', below: '▼', within: '●', outside: '◆', unsupplied: '◌', unit: '◌', none: '·',
  projectedAbove: '△', projectedBelow: '▽', model: '◇', carried: '■', unavailable: '—', fresh: '✦',
};
export const DATA_VIEW = {
  title: 'Signals & forecasts',
  boundary: 'Research use only · synthetic or retrospective stream, not a live feed · no diagnosis, causal claim, treatment recommendation or clinical alert',
  attentionHeading: 'Needs attention',
  otherHeading: 'Other observed indicators',
  print: 'Print / PDF',
  printedFrom: 'Printed from current filtered view',
  emptyNoRanges: 'No reference ranges are supplied for the current data',
  emptyNoOutside: 'No observations are outside their supplied ranges',
  emptyNoProjected: 'No model forecasts are projected outside supplied ranges',
  emptySearch: 'No mapped variable matches this search',
  unobserved: 'Not yet observed',
  showOther: 'Show other observed indicators',
  hideOther: 'Hide other observed indicators',
  compareAt: 'Compare forecast at',
  noPrevious: 'No previous result',
  unchanged: 'unchanged',
  details: 'View details',
};

/* A signal with no target x horizon cell in the pinned registry: there is no
 * forecast model for it at all, and the page shows observations only. */
export const NO_FORECAST_MODEL = 'No forecast model';
export const OBSERVED_HISTORY = 'Observed history';
export const UPDATING_FORECAST = 'Updating forecast…';
export const POINT_ONLY_NOTE =
  'This release provides point forecasts only; no validated uncertainty interval is available.';

/* The patient-trajectory chart's own words. One continuous time axis is split at
 * the model input cutoff into two regions; the words below are what a reader
 * sees at each, and they never change with colour. */
export const TRAJECTORY = {
  title: 'Patient trajectory',
  historyTitle: 'OBSERVED HISTORY',
  historySub: 'FHIR observations',
  forecastTitle: 'FUTURE FORECAST',
  forecastSub: 'Point forecasts',
  cutoffLabel: 'Forecast starts here',
  cutoffSub: 'Model input cutoff',
  liveHead: 'LIVE HEAD',
  snapshot: 'HISTORICAL SNAPSHOT',
  baselineOnly: 'Baseline only · not a model forecast',
  noModel: 'No forecast model for this metric',
  unavailable: 'Forecast unavailable',
  noRun: 'No model run yet',
  futureRegion: 'Future forecast region · point forecasts only, no uncertainty interval',
  observedSeries: 'Observed series',
  observationsOnly: 'Observations only · no forecast requested',
};

/* What a withheld cell says to a reader. The audit reason stays in provenance;
 * this is the one-line explanation that goes next to the empty region. */
export function unavailableReason(status, fallbackReason) {
  if (fallbackReason === 'wholebody_runtime_not_attached') return 'The model runtime is not attached';
  if (fallbackReason === 'shared_jepa_head_not_external_validated') return 'This horizon is not externally validated, so no model value is shown';
  if (status === 'missing') return 'The model returned no value for this horizon';
  if (status === 'degraded') return 'The model runtime did not produce this horizon';
  return 'No value was produced for this horizon';
}

export function fmtNum(v, decimals) {
  if (v === null || v === undefined || Number.isNaN(Number(v))) return '—';
  const n = Number(v);
  if (decimals !== undefined) return n.toFixed(decimals);
  const abs = Math.abs(n);
  if (abs >= 100) return n.toFixed(0);
  if (abs >= 10) return n.toFixed(1);
  return n.toFixed(2);
}

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

/* A signal with no target x horizon cell in the pinned registry: there is no
 * forecast model for it at all, and the page shows observations only. */
export const NO_FORECAST_MODEL = 'No forecast model';
export const OBSERVED_HISTORY = 'Observed history';
export const UPDATING_FORECAST = 'Updating forecast…';
export const POINT_ONLY_NOTE =
  'This release provides point forecasts only; no validated uncertainty interval is available.';

export function fmtNum(v, decimals) {
  if (v === null || v === undefined || Number.isNaN(Number(v))) return '—';
  const n = Number(v);
  if (decimals !== undefined) return n.toFixed(decimals);
  const abs = Math.abs(n);
  if (abs >= 100) return n.toFixed(0);
  if (abs >= 10) return n.toFixed(1);
  return n.toFixed(2);
}

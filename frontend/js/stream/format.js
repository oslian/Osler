/* Text helpers every view module needs: HTML escaping and the one number format
 * the page uses. They live here rather than in store.js because they are view
 * concerns, and rather than in render.js because chart.js needs them too. */

export const esc = (s) => String(s == null ? '' : s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

export function fmtNum(v, decimals) {
  if (v === null || v === undefined || Number.isNaN(Number(v))) return '—';
  const n = Number(v);
  if (decimals !== undefined) return n.toFixed(decimals);
  const abs = Math.abs(n);
  if (abs >= 100) return n.toFixed(0);
  if (abs >= 10) return n.toFixed(1);
  return n.toFixed(2);
}

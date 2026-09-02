/* REST + SSE client for the synthetic stream. The SSE channel is one-way;
 * every control action is a REST call. Reconnects carry Last-Event-ID
 * automatically; a page reload passes the last known sequence explicitly. */

export async function getJSON(url) {
  const r = await fetch(url, { headers: { Accept: 'application/json' } });
  const ct = r.headers.get('content-type') || '';
  const body = ct.includes('application/json') ? await r.json() : null;
  if (!r.ok && !(r.status === 503 && body)) {
    throw new Error((body && body.error) || `${r.status} ${r.statusText} for ${url}`);
  }
  if (body === null) throw new Error(`Server did not return JSON (HTTP ${r.status}) for ${url}`);
  body.__status = r.status;
  return body;
}

export async function postJSON(url, payload) {
  const r = await fetch(url, {
    method: 'POST', headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
    body: payload === undefined ? undefined : JSON.stringify(payload),
  });
  const body = await r.json().catch(() => null);
  if (!r.ok) throw new Error((body && body.error) || `${r.status} ${r.statusText} for ${url}`);
  return body;
}

export const EVENT_TYPES = [
  'stream.state', 'fhir.resource.accepted', 'fhir.resource.rejected', 'observation.normalized',
  'observation.unmapped', 'model.run.started', 'model.run.completed', 'model.run.failed',
];

/** Open the SSE channel. `handlers.onEvent(type, data, sequence)` for every typed event. */
export function connectEvents(sessionId, lastSequence, handlers) {
  const url = `/api/demo-streams/${encodeURIComponent(sessionId)}/events?last_event_id=${lastSequence || 0}`;
  const es = new EventSource(url);
  es.onopen = () => handlers.onStatus('open');
  es.onerror = () => handlers.onStatus(es.readyState === EventSource.CLOSED ? 'closed' : 'reconnecting');
  EVENT_TYPES.forEach((type) => {
    es.addEventListener(type, (e) => {
      let data = null;
      try { data = JSON.parse(e.data); } catch (_) { return; }
      handlers.onEvent(type, data, Number(e.lastEventId) || 0);
    });
  });
  return { close: () => es.close(), source: es };
}

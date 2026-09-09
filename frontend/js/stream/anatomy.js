/* The anatomy figure: the LadyofHats hand-fix baseplate with the detailed organ
 * artwork placed on it, and one container per canonical organ.
 *
 * Geometry comes from /static/assets/anatomy-motion/organ-layout.json and from
 * nowhere else. This module never computes a position: it reads the sprite's
 * transform, its clip and its hit box straight out of the manifest. That is what
 * keeps an organ from drifting - a response never touches a transform, only the
 * overlay's opacity and filter.
 *
 * The body is drawn as two variants of the same baseplate file, so the organs
 * sit between the veins and the arteries and the arterial detail stays on top.
 * The heart is drawn once more above the arteries, because the descending aorta
 * passes behind it.
 *
 * Response layer. Each organ carries a second copy of its own artwork, clipped
 * to its own region, used only as a tint. At rest it is invisible; a state turns
 * it on through an SVG filter that recolours the organ's alpha, so the highlight
 * has the organ's shape and the artwork underneath is never altered.
 *
 * An organ id is drawn only if the server's organ map knows it. Organs with no
 * detailed asset (thyroid, bone marrow, skeletal muscle) get no shape at all:
 * their events are still carried and counted, and the figure says nothing it
 * cannot draw honestly.
 */

const NS = 'http://www.w3.org/2000/svg';
const BASE = '/static/assets/anatomy-motion/';

/* How long an organ stays lit after the observation that touched it, measured on
 * the stream's own clinical clock. Replay is therefore reproducible: the same
 * event sequence lights the same organs at the same stream position, whatever
 * the wall clock is doing. */
const ACTIVE_HOURS = 2;

/* Pulse length, matching the organ-pulse keyframes in stream.css. Cosmetic: it
 * decides how long the flourish lasts, never which organ is lit. */
const PULSE_MS = 900;

/* Each state is a halo drawn around the organ's own outline plus a light wash
 * inside it. The halo is the blurred alpha with the organ itself cut back out,
 * so the artwork underneath stays legible at every zoom: the response reads as
 * the organ lighting up, never as the organ being repainted.
 *
 * Every figure defines its own filters under its own ids and publishes them to
 * its own CSS as custom properties, so two figures on one page (the monitoring
 * rail and the report's overview) never share an id and neither depends on the
 * other having been built. */
const fx = (id, r, g, b, blur) =>
  `<filter id="${id}" x="-30%" y="-30%" width="160%" height="160%">` +
  `<feColorMatrix type="matrix" values="0 0 0 0 ${r}  0 0 0 0 ${g}  0 0 0 0 ${b}  0 0 0 1 0" result="t"/>` +
  `<feGaussianBlur in="t" stdDeviation="${blur}" result="b"/>` +
  '<feComposite in="b" in2="SourceAlpha" operator="out" result="halo"/>' +
  '<feComponentTransfer in="t" result="wash"><feFuncA type="linear" slope="0.10"/></feComponentTransfer>' +
  '<feMerge><feMergeNode in="halo"/><feMergeNode in="halo"/><feMergeNode in="wash"/></feMerge></filter>';

/* name -> (r, g, b, blur). `danger` is deep red: at least one current observed
 * result on this organ is outside the interval its own observation supplied.
 * `amber` is the other claim - a model forecast leaving that interval - and the
 * two never share a colour. The CSS custom property each one is published under
 * is `--fx-<name>`. */
const RESPONSES = {
  teal: [0.153, 0.392, 0.369, 3.2],
  oxide: [0.612, 0.231, 0.149, 3.2],
  amber: [0.706, 0.510, 0.180, 2.8],
  danger: [0.549, 0.114, 0.094, 3.4],
};

let figureSeq = 0;
let clipSeq = 0;

function svgEl(name, attrs) {
  const el = document.createElementNS(NS, name);
  Object.entries(attrs || {}).forEach(([k, v]) => el.setAttribute(k, v));
  return el;
}

function assetUrl(file) {
  return file.startsWith('body-') ? BASE + file : `${BASE}organs/${file}`;
}

/** One sprite: a group carrying the manifest's transform, and an untransformed
 *  <image> inside it, optionally clipped in the asset's own coordinates. */
function sprite(defs, sp, cls, prefix) {
  const g = svgEl('g', { transform: sp.transform, class: cls });
  const img = svgEl('image', { href: assetUrl(sp.asset), x: 0, y: 0, width: sp.vw, height: sp.vh });
  img.setAttributeNS('http://www.w3.org/1999/xlink', 'href', assetUrl(sp.asset));
  if (sp.clip_d || sp.clip_rect) {
    const id = `${prefix}-clip-${++clipSeq}`;
    const clip = svgEl('clipPath', { id, clipPathUnits: 'userSpaceOnUse' });
    clip.appendChild(sp.clip_d
      ? svgEl('path', { d: sp.clip_d })
      : svgEl('rect', { x: sp.clip_rect[0], y: sp.clip_rect[1],
                        width: sp.clip_rect[2] - sp.clip_rect[0], height: sp.clip_rect[3] - sp.clip_rect[1] }));
    defs.appendChild(clip);
    img.setAttribute('clip-path', `url(#${id})`);
  }
  g.appendChild(img);
  return g;
}

export function buildAnatomy(host, organs, layout, onSelect) {
  host.innerHTML = '';
  const known = new Map(organs.map((o) => [o.id, o]));
  // this figure's own namespace: every id it writes starts here, so a second
  // figure on the same page collides with nothing and needs nothing from the first
  const prefix = `anatomy-${++figureSeq}`;
  const root = svgEl('svg', {
    viewBox: layout.frame.viewBox, role: 'group', class: 'anatomy-figure', id: prefix,
    'aria-label': 'Interactive anatomy. Organs are buttons; arrow keys move between them, Enter selects.',
  });
  // this figure's own filters, and the custom properties its own CSS reads them
  // through: the stylesheet never names an id, so it cannot point at another figure
  const defs = svgEl('defs');
  defs.innerHTML = Object.entries(RESPONSES)
    .map(([name, [r, g, b, blur]]) => fx(`${prefix}-fx-${name}`, r, g, b, blur)).join('');
  root.setAttribute('style', Object.keys(RESPONSES)
    .map((name) => `--fx-${name}: url(#${prefix}-fx-${name})`).join('; '));
  root.appendChild(defs);

  const bp = layout.baseplate;
  const plate = (file, cls) => {
    const img = svgEl('image', { href: BASE + file, x: 0, y: 0,
      width: layout.frame.w, height: layout.frame.h, class: cls });
    img.setAttributeNS('http://www.w3.org/1999/xlink', 'href', BASE + file);
    return img;
  };

  // the manifest is the only source of depth: draw order is z, and `over_vessels`
  // decides which side of the arterial layer an organ falls on
  const placed = layout.organs.filter((o) => known.has(o.organ) && !o.visual_asset_missing);
  const byZ = placed.slice().sort((a, b) => a.z - b.z);

  root.appendChild(plate(bp.back, 'baseplate back'));
  const artBack = svgEl('g', { class: 'organ-art' });
  root.appendChild(artBack);
  root.appendChild(plate(bp.front, 'baseplate front'));
  const artFront = svgEl('g', { class: 'organ-art over' });
  root.appendChild(artFront);
  const fxLayer = svgEl('g', { class: 'organ-fx-layer', 'aria-hidden': 'true' });
  root.appendChild(fxLayer);
  const hitLayer = svgEl('g', { class: 'organ-hits' });
  root.appendChild(hitLayer);

  byZ.forEach((o) => {
    if (o.art) (o.over_vessels ? artFront : artBack).appendChild(sprite(defs, o.art, 'art', prefix));
  });

  // one container per canonical organ; a paired organ has one entry per side and
  // both entries answer to the same organ id
  const nodes = {};
  const order = [];
  byZ.forEach((o) => {
    if (!o.fx) return;
    const g = svgEl('g', { class: 'organ-fx', 'data-organ': o.organ, 'data-side': o.side });
    g.appendChild(sprite(defs, o.fx, 'fx', prefix));
    fxLayer.appendChild(g);
    (nodes[o.organ] || (nodes[o.organ] = { fx: [], hits: [], label: known.get(o.organ).label,
                                           anchor: o.anchor, missing: false })).fx.push(g);
    if (!order.includes(o.organ)) order.push(o.organ);
  });

  // hit areas last, smallest first, so a small organ stays reachable on top of a large one
  placed.filter((o) => o.hit).sort((a, b) => ((b.hit[2] - b.hit[0]) * (b.hit[3] - b.hit[1]))
                                           - ((a.hit[2] - a.hit[0]) * (a.hit[3] - a.hit[1])))
    .forEach((o) => {
      const h = o.hit;
      const r = svgEl('rect', { x: h[0], y: h[1], width: h[2] - h[0], height: h[3] - h[1],
        class: 'organ-hit', 'data-organ': o.organ, 'data-side': o.side,
        role: 'button', tabindex: '0', 'aria-pressed': 'false' });
      const title = svgEl('title', {});
      r.appendChild(title);
      r.addEventListener('click', () => onSelect(o.organ));
      r.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onSelect(o.organ); }
        else if (e.key === 'ArrowDown' || e.key === 'ArrowRight') { e.preventDefault(); step(o.organ, 1); }
        else if (e.key === 'ArrowUp' || e.key === 'ArrowLeft') { e.preventDefault(); step(o.organ, -1); }
      });
      r.addEventListener('mouseenter', () => showPill(o.organ));
      r.addEventListener('mouseleave', () => showPill(null));
      r.addEventListener('focus', () => showPill(o.organ));
      r.addEventListener('blur', () => showPill(null));
      hitLayer.appendChild(r);
      const node = nodes[o.organ] || (nodes[o.organ] = { fx: [], hits: [], label: known.get(o.organ).label,
                                                         anchor: o.anchor, missing: false });
      node.hits.push({ rect: r, title });
      if (!order.includes(o.organ)) order.push(o.organ);
    });

  // organs the manifest knows but cannot draw: recorded, never faked
  layout.organs.filter((o) => known.has(o.organ) && o.visual_asset_missing).forEach((o) => {
    nodes[o.organ] = { fx: [], hits: [], label: known.get(o.organ).label, anchor: o.anchor, missing: true };
  });

  const pill = svgEl('g', { class: 'organ-pill', 'aria-hidden': 'true' });
  pill.innerHTML = '<rect rx="3" ry="3" height="22"></rect><text y="15" text-anchor="middle"></text>';
  root.appendChild(pill);
  host.appendChild(root);

  let selected = null;
  let summary = {};
  const seen = {};

  function showPill(id) {
    const target = id || selected;
    const node = target && nodes[target];
    if (!node || !node.anchor) { pill.setAttribute('visibility', 'hidden'); return; }
    const info = summary[target] || {};
    const text = `${node.label}${info.attention && info.attention.abnormal ? ` · ${info.attention.abnormal} outside range` : (info.signals != null ? ` · ${info.signals} signal${info.signals === 1 ? '' : 's'}` : '')}`
      + (node.missing ? ' · no visual asset' : '');
    const [x, y] = node.anchor;
    const t = pill.querySelector('text');
    const r = pill.querySelector('rect');
    t.textContent = text;
    const w = Math.min(240, text.length * 6.4 + 18);
    r.setAttribute('width', w.toFixed(0));
    r.setAttribute('x', (x - w / 2).toFixed(0));
    t.setAttribute('x', x);
    pill.setAttribute('transform', `translate(0 ${y - 34})`);
    pill.setAttribute('visibility', 'visible');
  }
  function step(id, by) {
    const idx = order.indexOf(id);
    const next = nodes[order[(idx + by + order.length) % order.length]];
    if (next && next.hits.length) next.hits[0].rect.focus();
  }

  return {
    nodes,
    order,
    id: prefix,
    /** Every filter this figure defines, so a caller can prove they resolve here. */
    filterIds: Object.keys(RESPONSES).map((name) => `${prefix}-fx-${name}`),
    /** Take the figure out of the document; nothing else on the page depends on it. */
    destroy() { host.innerHTML = ''; },
    /** state: { selected, byOrgan, organEvents, streamNow, frozen, paused, attention }
     *  `attention` is organAttentionState() per organ id, computed once in the store.
     *  The figure draws it and decides nothing itself. */
    update(state) {
      selected = state.selected;
      summary = state.byOrgan || {};
      const events = state.organEvents || {};
      const nowMs = state.streamNow ? Date.parse(state.streamNow) : NaN;
      Object.entries(nodes).forEach(([id, node]) => {
        const info = summary[id] || { signals: 0, cells: 0, boundaryCells: 0 };
        const ev = events[id];
        const ageH = ev && !Number.isNaN(nowMs) ? (nowMs - Date.parse(ev.at)) / 3600000 : Infinity;
        const active = ev && ageH >= 0 && ageH <= ACTIVE_HOURS;
        // What this organ is showing is the indicators' own statement, aggregated once
        // in the store: a current result outside the range its observation supplied
        // (deep red), else a model forecast projected outside one (amber), else new
        // data (the existing teal pulse). One state at a time, worst first, and a
        // selected organ keeps whichever it has.
        const att = (state.attention || {})[id] || null;
        const isSelected = state.selected === id;
        const cls = ['organ-state'];
        if (att && att.state === 'abnormal') cls.push('state-abnormal');
        else if (att && att.state === 'projected') cls.push('state-projected');
        else if (active) cls.push('state-active');
        if (state.frozen) cls.push('frozen');
        const says = !att ? ''
          : att.abnormal ? `, ${att.abnormal} observed result${att.abnormal === 1 ? '' : 's'} outside supplied range`
            : att.projected ? `, ${att.projected} forecast${att.projected === 1 ? '' : 's'} projected outside supplied range`
              : att.fresh ? ', new observation' : '';
        const name = `${node.label}: ${info.signals} observed signal${info.signals === 1 ? '' : 's'}, `
          + (info.cells ? `${info.cells} forecast cell${info.cells === 1 ? '' : 's'}` : 'no runtime cell')
          + says + (node.missing ? ', no visual asset' : '');
        node.fx.forEach((g) => { g.setAttribute('class', `organ-fx ${cls.join(' ')}`); });
        node.hits.forEach(({ rect, title }) => {
          rect.setAttribute('class', `organ-hit ${cls.join(' ')}`);
          rect.setAttribute('aria-pressed', isSelected ? 'true' : 'false');
          rect.setAttribute('aria-label', name);
          title.textContent = name;
        });
        node.fx.concat(node.hits.map((h) => h.rect)).forEach((el) => {
          el.classList.toggle('selected', isSelected);
        });
        // the pulse is triggered by the event sequence, so replay repeats it exactly;
        // its duration is cosmetic and never decides which organ is lit
        // Streamed state, the live head's snapshot and a historical run are three
        // separate sequence spaces, so they are keyed apart; restored state lights the
        // organ without replaying a pulse that already happened.
        const key = ev ? `${ev.restored || 'stream'}:${ev.sequence}` : null;
        if (key && seen[id] !== key) {
          seen[id] = key;
          if (!state.paused && active && !ev.restored) {
            // one timer per organ: a faster event cancels the timer still running for
            // this organ instead of stacking another, so a high speed cannot end a new
            // pulse early or flicker between them
            clearTimeout(node.pulseTimer);
            node.fx.forEach((g) => {
              g.classList.remove('pulse');
              void g.getBoundingClientRect();
              g.classList.add('pulse');
            });
            node.pulseTimer = setTimeout(() => {
              node.pulseTimer = 0;
              node.fx.forEach((g) => g.classList.remove('pulse'));
            }, PULSE_MS);
          }
        }
      });
      showPill(null);
    },
  };
}

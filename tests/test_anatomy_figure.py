"""Two anatomy figures on one page, executed rather than grepped.

The monitoring rail and the printed report draw the same figure from the same
`buildAnatomy()`. Both are in one document at once, so each has to be entirely
self-contained: its own filter definitions under its own ids, its own clip
paths, and a stylesheet that reaches them without naming an id. A figure that
borrowed another's definitions, or that depended on a class on some ancestor
box, is what painted solid black rectangles over the head and torso of the
report's copy - an SVG ``<rect>`` fills black by default, and the rule making
the click targets transparent was scoped to the container rather than to the
figure.

``anatomy.js`` touches only DOM methods, so it runs under Node against the small
document shim below: enough of `createElementNS`, attributes, children and
selectors to build the figure and read it back. Node is not part of the demo's
runtime; without it these tests skip with a reason.
"""

import json
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from oslian_demo.config import CSS_DIR, JS_DIR

NODE = shutil.which("node")
STREAM_JS = JS_DIR / "stream"
STREAM_CSS = (CSS_DIR / "stream.css").read_text(encoding="utf-8")

SHIM = r"""
/* The smallest document that can hold an SVG figure: elements with attributes,
 * children, a class list and the three selector shapes anatomy.js and these
 * tests use (tag, #id, .class, and a descendant combinator between them). */
class El {
  constructor(tag) { this.tagName = tag; this.attrs = {}; this.children = []; this.parentNode = null;
    this.textContent = ''; this._html = ''; this.style = { setProperty() {} };
    this.classList = {
      el: this,
      add: (...c) => this.el0().setAttribute('class', [...new Set([...this.classes(), ...c])].join(' ')),
      remove: (...c) => this.el0().setAttribute('class', this.classes().filter((x) => !c.includes(x)).join(' ')),
      toggle: (c, on) => (on ? this.classList.add(c) : this.classList.remove(c)),
      contains: (c) => this.classes().includes(c),
    };
  }
  el0() { return this; }
  classes() { return (this.attrs.class || '').split(/\s+/).filter(Boolean); }
  setAttribute(k, v) { this.attrs[k] = String(v); }
  setAttributeNS(ns, k, v) { this.attrs[k.replace(/^.*:/, '')] = String(v); }
  getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; }
  get id() { return this.attrs.id || ''; }
  get className() { return { baseVal: this.attrs.class || '' }; }
  appendChild(c) { c.parentNode = this; this.children.push(c); return c; }
  set innerHTML(v) { this._html = v; this.children = []; if (v) parseInto(this, v); }
  get innerHTML() { return this._html; }
  addEventListener() {}
  focus() {}
  getBoundingClientRect() { return { x: 0, y: 0, width: 100, height: 100 }; }
  descendants() { return this.children.flatMap((c) => [c, ...c.descendants()]); }
  matches(sel) {
    return sel.split('').length && sel.split(/(?=[.#])/).every((part) => {
      if (part.startsWith('#')) return this.id === part.slice(1);
      if (part.startsWith('.')) return this.classes().includes(part.slice(1));
      return this.tagName === part;
    });
  }
  querySelectorAll(sel) {
    const steps = sel.trim().split(/\s+/);
    let pool = this.descendants();
    steps.forEach((step, i) => {
      const hit = pool.filter((e) => e.matches(step));
      pool = i === steps.length - 1 ? hit : hit.flatMap((e) => e.descendants());
    });
    return pool;
  }
  querySelector(sel) { return this.querySelectorAll(sel)[0] || null; }
}

/* Only what a defs block needs: <filter id="..."> ... </filter> repeated. */
function parseInto(parent, html) {
  const re = /<(\w+)([^>]*)>/g;
  let m;
  while ((m = re.exec(html))) {
    const el = new El(m[1]);
    const attrs = m[2];
    const ar = /([\w:-]+)="([^"]*)"/g;
    let a;
    while ((a = ar.exec(attrs))) el.setAttribute(a[1], a[2]);
    if (m[1] === 'filter' || m[1] === 'rect' || m[1] === 'text') parent.appendChild(el);
  }
}

const root = new El('body');
globalThis.document = {
  body: root,
  createElementNS: (ns, tag) => new El(tag),
  getElementById: (id) => root.querySelectorAll(`#${id}`)[0] || null,
  querySelectorAll: (s) => root.querySelectorAll(s),
  querySelector: (s) => root.querySelector(s),
};
globalThis.setTimeout = (fn) => 0;
globalThis.clearTimeout = () => {};
export { El, root };
"""

PROBE = r"""
import { root } from './shim.mjs';
import { buildAnatomy } from './anatomy.js';

const layout = {
  frame: { viewBox: '0 0 550 830', w: 550, h: 830 },
  baseplate: { back: 'body.back.svg', front: 'body.front.svg', vessels: 'body.vessels.svg' },
  organs: [
    { organ: 'kidneys', side: 'left', z: 1, anchor: [100, 100], over_vessels: false,
      art: { asset: 'kidney.svg', vw: 10, vh: 10, transform: 't', clip_rect: [0, 0, 5, 5] },
      fx: { asset: 'kidney.svg', vw: 10, vh: 10, transform: 't' }, hit: [0, 0, 10, 10] },
    { organ: 'heart', side: null, z: 2, anchor: [120, 90], over_vessels: true,
      art: { asset: 'heart.svg', vw: 10, vh: 10, transform: 't' },
      fx: { asset: 'heart.svg', vw: 10, vh: 10, transform: 't', clip_d: 'M0 0' }, hit: [20, 20, 30, 30] },
    { organ: 'lungs', side: null, z: 3, anchor: [140, 80], over_vessels: false,
      art: { asset: 'lung.svg', vw: 10, vh: 10, transform: 't' },
      fx: { asset: 'lung.svg', vw: 10, vh: 10, transform: 't' }, hit: [40, 40, 50, 50] },
    { organ: 'bone_marrow', side: null, z: 4, anchor: [10, 10], visual_asset_missing: true },
  ],
};
const organs = [
  { id: 'kidneys', label: 'Kidneys', variables: [] },
  { id: 'heart', label: 'Heart', variables: [] },
  { id: 'lungs', label: 'Lungs', variables: [] },
  { id: 'bone_marrow', label: 'Bone marrow', variables: [] },
];
const attention = {
  kidneys: { state: 'abnormal', abnormal: 2, projected: 1, fresh: false },
  heart: { state: 'projected', abnormal: 0, projected: 1, fresh: false },
  lungs: { state: 'new', abnormal: 0, projected: 0, fresh: true },
};

const hostA = root.appendChild(new (root.constructor)('div'));
const hostB = root.appendChild(new (root.constructor)('div'));
hostA.setAttribute('id', 'anatomyHost');
hostB.setAttribute('id', 'reportAnatomy');

const a = buildAnatomy(hostA, organs, layout, () => {});
const b = buildAnatomy(hostB, organs, layout, () => {});

const svg = (host) => host.querySelector('svg');
const describe = (figure, host) => {
  const s = svg(host);
  const filters = s.querySelectorAll('defs filter').map((f) => f.id);
  const clips = s.querySelectorAll('defs clipPath').map((c) => c.id);
  const vars = Object.fromEntries((s.getAttribute('style') || '').split(';').map((p) => p.trim())
    .filter(Boolean).map((p) => p.split(':').map((x) => x.trim())));
  return { id: figure.id, cls: s.getAttribute('class'), filters, clips, vars,
    declared: figure.filterIds,
    // every custom property points at a filter defined inside this same figure
    varsResolveHere: Object.values(vars).every((v) => {
      const id = (v.match(/url\(#([^)]+)\)/) || [])[1];
      return !!id && filters.includes(id);
    }),
    hits: s.querySelectorAll('rect.organ-hit').length,
    fxStates: Object.fromEntries(s.querySelectorAll('g.organ-fx')
      .map((g) => [g.getAttribute('data-organ'), g.getAttribute('class')])),
    ariaLabels: s.querySelectorAll('rect.organ-hit').map((r) => r.getAttribute('aria-label')),
  };
};

const out = {};
// teal is the existing event-driven pulse: it comes from the stream's own clock,
// not from the aggregate, so it is driven here the way the page drives it
const streamNow = '2026-03-15T12:00:00.000Z';
const organEvents = { lungs: { at: '2026-03-15T11:00:00.000Z', sequence: 4, restored: 'snapshot' } };
a.update({ selected: 'kidneys', byOrgan: {}, organEvents, streamNow, runtimeProducing: false,
  frozen: false, paused: true, attention });
b.update({ selected: null, byOrgan: {}, organEvents, streamNow, runtimeProducing: false,
  frozen: true, paused: true, attention });
out.a = describe(a, hostA);
out.b = describe(b, hostB);

// every id anywhere in the document, to prove none repeats
const ids = root.descendants().map((e) => e.id).filter(Boolean);
out.ids = { total: ids.length, unique: new Set(ids).size,
  duplicates: ids.filter((x, i) => ids.indexOf(x) !== i) };

// the second figure never reached into the first
out.disjoint = out.a.filters.every((f) => !out.b.filters.includes(f))
  && out.a.clips.every((c) => !out.b.clips.includes(c));

// take the first figure away: the second keeps its own definitions and redraws
a.destroy();
out.afterDestroy = { firstEmpty: hostA.children.length === 0, secondSvg: !!svg(hostB) };
b.update({ selected: 'heart', byOrgan: {}, organEvents, streamNow, runtimeProducing: false,
  frozen: false, paused: true, attention });
out.bAfterDestroy = describe(b, hostB);

// a third figure, built with neither of the others present
hostB.innerHTML = '';
const hostC = root.appendChild(new (root.constructor)('div'));
const c = buildAnatomy(hostC, organs, layout, () => {});
c.update({ selected: null, byOrgan: {}, organEvents, streamNow, runtimeProducing: false,
  frozen: false, paused: true, attention });
out.c = describe(c, hostC);

console.log(JSON.stringify(out));
"""


@unittest.skipUnless(NODE, "node is not on PATH; the anatomy figure cannot be executed here")
class TwoAnatomyFiguresTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="oslian-anatomy-")
        shutil.copy(STREAM_JS / "anatomy.js", Path(cls.tmp) / "anatomy.js")
        (Path(cls.tmp) / "package.json").write_text('{"type": "module"}', encoding="utf-8")
        (Path(cls.tmp) / "shim.mjs").write_text(SHIM, encoding="utf-8")
        (Path(cls.tmp) / "probe.mjs").write_text(PROBE, encoding="utf-8")
        run = subprocess.run([NODE, "probe.mjs"], cwd=cls.tmp, capture_output=True, text=True,
                             encoding="utf-8", timeout=60)
        assert run.returncode == 0, run.stderr
        cls.out = json.loads(run.stdout)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_each_figure_defines_its_own_filters_under_its_own_ids(self):
        a, b = self.out["a"], self.out["b"]
        self.assertEqual(a["id"], "anatomy-1")
        self.assertEqual(b["id"], "anatomy-2")
        for figure in (a, b):
            self.assertEqual(sorted(figure["filters"]), sorted(figure["declared"]))
            self.assertEqual(len(figure["filters"]), 4)
            for fid in figure["filters"]:
                self.assertTrue(fid.startswith(f"{figure['id']}-fx-"), fid)

    def test_no_id_in_the_document_is_used_twice(self):
        ids = self.out["ids"]
        self.assertEqual(ids["duplicates"], [])
        self.assertEqual(ids["total"], ids["unique"])
        self.assertGreater(ids["total"], 8)   # filters and clip paths from both figures

    def test_every_filter_reference_resolves_inside_its_own_figure(self):
        for key in ("a", "b"):
            figure = self.out[key]
            self.assertTrue(figure["varsResolveHere"], key)
            self.assertEqual(sorted(figure["vars"]), ["--fx-amber", "--fx-danger", "--fx-oxide", "--fx-teal"])
            for value in figure["vars"].values():
                self.assertRegex(value, r"^url\(#anatomy-\d+-fx-\w+\)$")

    def test_the_two_figures_share_nothing(self):
        self.assertTrue(self.out["disjoint"])
        self.assertEqual(set(self.out["a"]["clips"]) & set(self.out["b"]["clips"]), set())

    def test_removing_one_figure_leaves_the_other_working(self):
        self.assertTrue(self.out["afterDestroy"]["firstEmpty"])
        self.assertTrue(self.out["afterDestroy"]["secondSvg"])
        after = self.out["bAfterDestroy"]
        self.assertTrue(after["varsResolveHere"])
        self.assertIn("state-abnormal", after["fxStates"]["kidneys"])
        self.assertIn("state-projected", after["fxStates"]["heart"])

    def test_a_figure_built_with_no_other_figure_present_is_complete(self):
        c = self.out["c"]
        self.assertEqual(len(c["filters"]), 4)
        self.assertTrue(c["varsResolveHere"])
        self.assertIn("state-abnormal", c["fxStates"]["kidneys"])

    def test_each_state_reaches_the_overlay_and_the_label(self):
        a = self.out["a"]
        self.assertIn("state-abnormal", a["fxStates"]["kidneys"])
        self.assertIn("selected", a["fxStates"]["kidneys"])      # selected and abnormal at once
        self.assertIn("state-projected", a["fxStates"]["heart"])
        # teal stays what it was: the stream's own event clock, not the aggregate
        self.assertIn("state-active", a["fxStates"]["lungs"])
        self.assertNotIn("state-abnormal", a["fxStates"]["heart"])
        self.assertTrue(any("outside supplied range" in label for label in a["ariaLabels"]))
        self.assertTrue(any("projected outside supplied range" in label for label in a["ariaLabels"]))
        # a frozen figure keeps its states too
        self.assertIn("state-abnormal", self.out["b"]["fxStates"]["kidneys"])
        self.assertIn("frozen", self.out["b"]["fxStates"]["kidneys"])


class AnatomyStyleScopeTests(unittest.TestCase):
    """The figure carries its own appearance, so it looks the same wherever it is mounted."""

    ANATOMY = (STREAM_JS / "anatomy.js").read_text(encoding="utf-8")

    def test_the_stylesheet_never_names_a_filter_id(self):
        self.assertNotIn("url(#fx", STREAM_CSS)
        for name in ("--fx-teal", "--fx-oxide", "--fx-amber", "--fx-danger"):
            self.assertIn(f"var({name})", STREAM_CSS, name)

    def test_no_shared_global_filter_host_remains(self):
        for gone in ("anatomyFilters", "ensureFilters", "FILTER_HOST_ID"):
            self.assertNotIn(gone, self.ANATOMY, gone)
        self.assertNotIn("anatomyFilters", STREAM_CSS)
        self.assertNotIn("document.body.appendChild", self.ANATOMY)

    def test_every_rule_that_makes_the_figure_look_right_is_scoped_to_the_figure(self):
        # this is the black-rectangle regression: an SVG <rect> fills black by
        # default, so the rule that makes the click targets transparent must not
        # depend on a class on some ancestor box.
        declarations = re.sub(r"/\*.*?\*/", "", STREAM_CSS, flags=re.S)
        for selector in (".organ-hit", ".organ-fx", ".baseplate", ".organ-art", ".organ-pill"):
            for line in declarations.splitlines():
                if selector in line and "{" in line:
                    self.assertIn(".anatomy-figure", line,
                                  f"{selector} is styled without naming the figure: {line.strip()[:110]}")
        self.assertRegex(declarations, r"\.anatomy-figure \.organ-hit \{[^}]*fill: transparent")
        # the container keeps layout only
        host_rules = [l for l in declarations.splitlines() if ".anatomy-host" in l and "{" in l]
        for line in host_rules:
            self.assertNotRegex(line, r"\b(fill|filter|opacity)\s*:",
                                f"appearance left on the container: {line.strip()[:110]}")

    def test_the_figure_is_never_hidden_or_stripped_when_printed(self):
        printed = STREAM_CSS[STREAM_CSS.index("@media print"):]
        self.assertNotRegex(printed, r"\.anatomy-figure[^{]*\{[^}]*display:\s*none")
        self.assertNotRegex(printed, r"\.report-figure[^{]*\{[^}]*display:\s*none")
        # the report's own figure keeps a printable size
        self.assertRegex(printed, r"\.report-figure \.anatomy-figure \{[^}]*max-height")


if __name__ == "__main__":
    unittest.main()

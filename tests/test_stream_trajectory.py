"""The patient-trajectory chart, executed rather than grepped.

``chart.js`` is pure (state in, SVG out) and ``store.js`` touches no DOM, so
both run unchanged under Node. A probe script builds one variable's history and
forecast cells in every state the runtime can produce, draws each, and hands
the SVG back here. The assertions are the invariants a reader relies on:

- observed points and forecast points sit on one linear time axis;
- the cutoff splits the plot into a labelled history region and a labelled
  future region;
- a model cell, a carried-forward cell, a withheld cell and a variable with no
  runtime cell each draw only what they are - never a model-styled line for a
  baseline, never a marker for a value that does not exist;
- a historical anchor draws the frozen run's inputs while the live head has
  moved on;
- the future region is the future, never an uncertainty band;
- the chart carries an accurate accessible description and reachable points.

Node is not part of the demo's runtime and is not installed by any script here;
without it these tests skip with a reason rather than pass vacuously.
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

PROBE = r"""
import { trajectoryChart, chartLegend } from './chart.js';
import { S, trajectoryFor } from './store.js';
const H = 3600000, T0 = Date.UTC(2026, 2, 14, 6);
const iso = (ms) => new Date(ms).toISOString();
const obs = (hours, value, unit = 'mL/hour') => ({ effective_time: iso(T0 + hours * H), value, unit,
  resource_id: `obs-${hours}`, version_id: '1', received_at: iso(T0 + hours * H) });
const cell = (h, status, point, extra) => ({ cell: `urine_output@${h}h`, target: 'urine_output', unit: 'mL/hour',
  horizon_hours: h, observed_value: 58, measurement_age_hours: 0, observations: 4, forecast_point: point,
  source_model: status === 'model' ? 'frozen_whole_body_jepa' : status === 'persistence' ? 'patient_persistence_anchor' : 'none',
  status, fallback_reason: null, interval_status: 'none', lower: null, upper: null, causal_claim_allowed: false, ...(extra || {}) });
const run = (id, cutoffHours) => ({ run_id: id, input_cutoff: iso(T0 + cutoffHours * H), epoch: 1, runtime_mode: 'release', input_resource_count: 4 });
S.capabilities = { cells: [{ cell: 'urine_output@1h', target: 'urine_output', horizon_hours: 1 },
                           { cell: 'urine_output@3h', target: 'urine_output', horizon_hours: 3 }] };
const history = () => [obs(0, 40), obs(2, 50), obs(4, 55), obs(6, 58)];
S.series = { urine_output: history(), temperature: [obs(0, 37, 'degC'), obs(6, 37.4, 'degC')] };
S.live = { run: run('live-run', 6), cells: [cell(1, 'model', 68.5), cell(3, 'model', 56.4)] };
const draw = (v, o = {}) => {
  const t = trajectoryFor(v, o.horizons || null);
  const chart = trajectoryChart({ traj: t, unit: v === 'temperature' ? 'degC' : 'mL/hour', label: v, axis: 'calendar', onset_ms: 0,
    window: o.window || 'full', width: o.width || 980, height: 360 });
  return { kind: t.kind, forecasts: t.forecasts, svg: chart.svg, empty: chart.empty, legend: chartLegend(t) };
};
const out = {};
out.model = draw('urine_output');
out.narrow = draw('urine_output', { width: 320 });
out.recent = draw('urine_output', { window: 'recent' });
S.live.cells = [cell(1, 'persistence', 58), cell(3, 'persistence', 58)];
out.persistence = draw('urine_output');
S.live.cells = [cell(1, 'degraded', null, { fallback_reason: 'wholebody_runtime_not_attached' }), cell(3, 'missing', null)];
out.unavailable = draw('urine_output');
S.live.cells = [cell(1, 'model', 68.5), cell(3, 'degraded', null, { fallback_reason: 'shared_jepa_head_not_external_validated' })];
out.mixed = draw('urine_output');
out.none = draw('temperature');
S.live.cells = [cell(1, 'model', 68.5), cell(3, 'model', 56.4)];
out.filtered = draw('urine_output', { horizons: new Set([3]) });
S.live.cells = [cell(1, 'model', 68.5, { observed_value: null, measurement_age_hours: null })];
out.nobasis = draw('urine_output');
// the live head moves on; an open anchor keeps the old run's inputs and cells
S.live = { run: run('live-run-2', 10), cells: [cell(1, 'model', 99), cell(3, 'model', 98)] };
S.series.urine_output.push(obs(8, 70), obs(10, 75));
S.anchor = { run: run('old-run', 6), cells: [cell(1, 'model', 68.5), cell(3, 'model', 56.4)],
  series: { urine_output: history() }, variables: {}, organEvents: {} };
out.anchor = draw('urine_output');
S.anchor = null;
out.liveAgain = draw('urine_output');
S.live = { run: null, cells: [] };
out.norun = draw('urine_output');
console.log(JSON.stringify(out));
"""

POINT = re.compile(r'<g class="chart-pt[^"]*"[^>]*data-kind="(observed|forecast)"[^>]*data-ms="(\d+)"[^>]*>(.*?)</g>', re.S)
HOUR = 3600000


def _marker_x(inner):
    m = re.search(r'<circle cx="([\d.]+)"', inner) or re.search(r'<rect x="([\d.]+)"', inner)
    if m and m.group(0).startswith("<rect"):
        return float(m.group(1)) + 4.5          # the square is drawn from its top-left corner
    if m:
        return float(m.group(1))
    m = re.search(r'<polygon points="([\d.]+),', inner)
    return float(m.group(1))


def _points(svg):
    """{(kind, ms): x} for every interactive point in the SVG."""
    out = {}
    for kind, ms, inner in POINT.findall(svg):
        out[(kind, int(ms))] = _marker_x(inner)
    return out


@unittest.skipUnless(NODE, "node is not on PATH; the executed chart tests need it")
class TrajectoryChartTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("chart.js", "store.js", "format.js"):
                shutil.copy(STREAM_JS / name, root / name)
            (root / "package.json").write_text('{"type": "module"}', encoding="utf-8")
            (root / "probe.mjs").write_text(PROBE, encoding="utf-8")
            result = subprocess.run([NODE, "probe.mjs"], cwd=root, capture_output=True, text=True, encoding="utf-8", timeout=60)
        if result.returncode != 0:
            raise AssertionError(f"probe failed:\n{result.stderr}")
        cls.out = json.loads(result.stdout)
        cls.T0 = 1773468000000  # Date.UTC(2026, 2, 14, 6)

    def ms(self, hours):
        return self.T0 + hours * HOUR

    # ---- one axis --------------------------------------------------------
    def test_observed_and_forecast_points_share_one_linear_time_axis(self):
        pts = _points(self.out["model"]["svg"])
        x4, x6 = pts[("observed", self.ms(4))], pts[("observed", self.ms(6))]
        x7, x9 = pts[("forecast", self.ms(7))], pts[("forecast", self.ms(9))]
        per_hour = (x6 - x4) / 2
        self.assertGreater(per_hour, 0)
        self.assertAlmostEqual(x7 - x6, per_hour, delta=0.3, msg="the 1h forecast is not one hour right of the last observation")
        self.assertAlmostEqual(x9 - x7, 2 * per_hour, delta=0.3, msg="the 3h forecast is not two hours right of the 1h forecast")

    def test_cutoff_splits_the_plot_into_two_labelled_regions(self):
        svg = self.out["model"]["svg"]
        history = re.search(r'<rect data-region="history" x="([\d.]+)" y="0" width="([\d.]+)"', svg)
        future = re.search(r'<rect data-region="future" x="([\d.]+)"', svg)
        rule = re.search(r'<line data-cutoff="rule" x1="([\d.]+)"', svg)
        self.assertTrue(history and future and rule)
        edge = float(history.group(1)) + float(history.group(2))
        self.assertAlmostEqual(edge, float(future.group(1)), delta=0.15, msg="the regions must meet at the cutoff")
        self.assertAlmostEqual(float(rule.group(1)), float(future.group(1)), delta=0.15, msg="the cutoff rule must sit on the region edge")
        # the cutoff is the last observation's time, and no value is invented there
        pts = _points(svg)
        self.assertAlmostEqual(pts[("observed", self.ms(6))], float(rule.group(1)), delta=0.15)
        self.assertEqual(sum(1 for (kind, ms) in pts if ms == self.ms(6)), 1, "exactly one point at the cutoff: the real observation")
        for text in ("OBSERVED HISTORY", "FHIR observations", "FUTURE FORECAST", "Point forecasts",
                     "Forecast starts here", "Model input cutoff · Mar 14 12:00Z"):
            self.assertIn(text, svg, text)
        self.assertNotIn("historical anchor", svg.lower())     # the state word lives in the header badge, not on the rule

    # ---- the four states -------------------------------------------------
    def test_model_forecast_draws_a_dashed_lead_from_the_basis_with_hollow_diamonds_and_values(self):
        s = self.out["model"]
        self.assertEqual(s["kind"], "model")
        line = re.search(r'<polyline data-series="model"[^>]*stroke-dasharray="5 4"[^>]*points="([^"]+)"', s["svg"])
        self.assertTrue(line, "the model path is a dashed polyline")
        xs = [float(p.split(",")[0]) for p in line.group(1).split(" ")]
        pts = _points(s["svg"])
        self.assertAlmostEqual(xs[0], pts[("observed", self.ms(6))], delta=0.15, msg="the lead line starts at the basis observation")
        self.assertEqual(len(xs), 3)
        self.assertEqual(s["svg"].count('data-status="model"'), 2)
        self.assertEqual(len(re.findall(r'data-status="model"[^>]*>(?:<line[^>]*>)?<polygon', s["svg"])), 2, "hollow diamonds")
        for label in ("1h · 68.5", "3h · 56.4"):
            self.assertIn(label, s["svg"])
        self.assertNotIn('data-series="baseline"', s["svg"])
        self.assertIn("Model forecast", s["legend"])
        self.assertNotIn("Baseline only", s["legend"])

    def test_persistence_is_a_grey_baseline_and_never_wears_the_model_style(self):
        s = self.out["persistence"]
        self.assertEqual(s["kind"], "persistence")
        svg = s["svg"]
        self.assertNotIn('data-series="model"', svg)
        self.assertNotRegex(svg, r'data-status="persistence"[^>]*>(?:<line[^>]*>)?<polygon', "no diamonds for a baseline")
        self.assertEqual(len(re.findall(r'data-status="persistence"[^>]*>(?:<line[^>]*>)?<rect', svg)), 2, "hollow squares")
        self.assertEqual(svg.count('data-status="persistence"'), 2)
        lines = re.findall(r'<line data-series="baseline" x1="([\d.]+)" y1="([\d.]+)" x2="([\d.]+)" y2="([\d.]+)"[^>]*stroke="(#[0-9A-Fa-f]+)"[^>]*stroke-dasharray="([^"]+)"', svg)
        self.assertEqual(len(lines), 2)
        for x1, y1, x2, y2, colour, dash in lines:
            self.assertEqual(y1, y2, "a carried-forward value is horizontal")
            self.assertNotEqual(colour.upper(), "#344F69", "a baseline is never forecast blue")
            self.assertNotEqual(dash, "5 4", "a baseline never uses the model dash")
        self.assertIn("Baseline only · not a model forecast", svg)
        self.assertIn("Baseline only · not a model forecast", s["legend"])
        self.assertNotIn("Model forecast", s["legend"])
        self.assertNotIn("model forecast", " ".join(re.findall(r'aria-label="([^"]*)"', svg)).lower().replace("not a model forecast", ""))

    def test_no_runtime_cell_draws_no_forecast_and_says_so(self):
        s = self.out["none"]
        self.assertEqual(s["kind"], "none")
        self.assertIn("No forecast model for this metric", s["svg"])
        self.assertIn('data-region="future"', s["svg"], "the future region still exists; it is not blank")
        self.assertNotIn('data-kind="forecast"', s["svg"])
        self.assertNotIn("data-series=\"model\"", s["svg"])
        self.assertNotIn("data-series=\"baseline\"", s["svg"])

    def test_withheld_cells_draw_nothing_and_give_a_reader_facing_reason(self):
        s = self.out["unavailable"]
        self.assertEqual(s["kind"], "unavailable")
        self.assertIn("Forecast unavailable", s["svg"])
        self.assertIn("The model runtime is not attached", s["svg"])
        self.assertNotIn("wholebody_runtime_not_attached", s["svg"], "the audit reason stays in provenance")
        self.assertNotIn('data-kind="forecast"', s["svg"], "nothing is drawn for a value that does not exist")
        self.assertNotIn('data-series=', s["svg"].split('data-series="observed"')[0])
        # partially withheld: the drawn horizon is drawn, the withheld one is named
        m = self.out["mixed"]
        self.assertEqual(m["kind"], "model")
        self.assertIn("1h · 68.5", m["svg"])
        self.assertNotRegex(m["svg"], r"3h · \d", "no value is drawn or labelled for the withheld horizon")
        self.assertIn("3h · Forecast unavailable", m["svg"])
        self.assertIn("not externally validated", m["svg"])

    def test_no_run_means_observations_only_and_no_future_region(self):
        s = self.out["norun"]
        self.assertEqual(s["kind"], "no-run")
        self.assertNotIn('data-region="future"', s["svg"])
        self.assertIn("No model run yet", s["svg"])

    # ---- basis, filters, windows ----------------------------------------
    def test_no_basis_means_no_lead_line_and_no_delta(self):
        s = self.out["nobasis"]
        self.assertIsNone(s["forecasts"][0]["basis"])
        line = re.search(r'<polyline data-series="model"[^>]*points="([^"]+)"', s["svg"])
        self.assertEqual(len(line.group(1).split(" ")), 1, "no basis observation: the path starts at the forecast itself")

    def test_horizon_filter_keeps_the_run_and_drops_the_horizon(self):
        s = self.out["filtered"]
        self.assertEqual([f["horizon_hours"] for f in s["forecasts"]], [3])
        self.assertIn("3h · 56.4", s["svg"])
        self.assertNotIn("1h · 68.5", s["svg"])

    def test_recent_window_keeps_the_basis_observation_in_view(self):
        self.assertIn(("observed", self.ms(6)), _points(self.out["recent"]["svg"]))

    # ---- historical snapshot --------------------------------------------
    def test_a_historical_anchor_draws_the_frozen_run_not_the_live_head(self):
        anchor, live = self.out["anchor"], self.out["liveAgain"]
        self.assertIn('data-head="snapshot"', anchor["svg"])
        self.assertIn("historical snapshot", anchor["svg"])
        self.assertIn("Model input cutoff · Mar 14 12:00Z", anchor["svg"])
        self.assertIn("1h · 68.5", anchor["svg"])
        self.assertNotIn("1h · 99", anchor["svg"])
        self.assertEqual(sum(1 for k in _points(anchor["svg"]) if k[0] == "observed"), 4, "only the run's own inputs")
        # returning to the live head shows the newer run and every delivered observation
        self.assertIn('data-head="live"', live["svg"])
        self.assertIn("Model input cutoff · Mar 14 16:00Z", live["svg"])
        self.assertIn("1h · 99", live["svg"])
        self.assertEqual(sum(1 for k in _points(live["svg"]) if k[0] == "observed"), 6)

    # ---- no intervals ----------------------------------------------------
    def test_the_future_region_is_never_an_uncertainty_band(self):
        for name, s in self.out.items():
            if s["empty"]:
                continue
            text = (s["svg"] + s["legend"]).lower()
            for word in ("confidence", "uncertainty band", "interval band", "prediction interval"):
                self.assertNotIn(word, text, f"{name}: {word}")
            self.assertEqual(s["svg"].count('data-region="future"'), 1 if s["kind"] != "no-run" else 0, name)
            # the only filled areas are the two region grounds; nothing is shaded between forecast points
            fills = re.findall(r"<(?:rect|path|polygon)[^>]*fill=\"(#[0-9A-Fa-f]{6})\"", s["svg"])
            self.assertFalse([f for f in fills if f.upper() not in {"#FFFFFF", "#EDF2F7", "#FCFAF5", "#171A1C", "#3D4449"}], name)
        self.assertIn("no uncertainty interval", self.out["model"]["legend"])

    # ---- accessibility and responsiveness --------------------------------
    def test_the_chart_describes_itself_and_every_point_is_reachable(self):
        svg = self.out["model"]["svg"]
        self.assertIn('role="group" aria-labelledby="trajTitle" aria-describedby="trajDesc"', svg)
        desc = re.search(r'<desc id="trajDesc">(.*?)</desc>', svg).group(1)
        for phrase in ("live head", "Observed history: 4 FHIR observations", "last observed 58.0 mL/hour",
                       "Forecast starts here: model input cutoff Mar 14 12:00Z", "model point forecasts 1h 68.5, 3h 56.4 mL/hour",
                       "no uncertainty interval"):
            self.assertIn(phrase, desc, phrase)
        groups = re.findall(r'<g class="chart-pt[^"]*"([^>]*)>', svg)
        self.assertEqual(len(groups), 6)
        for attrs in groups:
            self.assertIn('tabindex="0"', attrs)
            self.assertIn('role="button"', attrs)
            self.assertRegex(attrs, r'aria-label="[^"]{10,}"')
        labels = re.findall(r'aria-label="([^"]*)"', svg)
        self.assertEqual(sum(1 for l in labels if l.startswith("Observed ")), 4)
        self.assertEqual(sum(1 for l in labels if "model forecast" in l), 2)
        self.assertTrue(any("basis" in l for l in labels))

    def test_region_titles_never_overlap_on_a_narrow_chart(self):
        svg = self.out["narrow"]["svg"]
        left = re.search(r'<text x="([\d.]+)" y="16" class="rg-t">([^<]+)</text>', svg)
        right = re.search(r'<text x="([\d.]+)" y="16" text-anchor="end" class="rg-t future">([^<]+)</text>', svg)
        self.assertTrue(left and right)
        per_char = 7.0
        left_end = float(left.group(1)) + len(left.group(2)) * per_char
        right_start = float(right.group(1)) - len(right.group(2)) * per_char
        self.assertLess(left_end + 8, right_start, "the two region titles collide at 320px")
        self.assertIn('viewBox="0 0 320 360"', svg)


class TrajectoryLayoutContractTests(unittest.TestCase):
    """Source-level: the page draws the chart at the column's real width and stays one column on a phone."""

    RENDER = (STREAM_JS / "render.js").read_text(encoding="utf-8")
    CSS = (CSS_DIR / "stream.css").read_text(encoding="utf-8")

    def test_the_chart_is_drawn_at_the_workspace_width(self):
        self.assertIn("function chartWidth(host)", self.RENDER)
        self.assertIn("host.clientWidth", self.RENDER)
        self.assertNotIn("min-width: 600px", self.CSS, "a fixed chart width would scroll sideways on a phone")
        self.assertNotIn("width: narrow ? 640 : 980", self.RENDER)

    def test_phone_layout_is_one_column(self):
        phone = self.CSS[self.CSS.index("@media (max-width: 767px)"):]
        phone = phone[:phone.index("@media", 10)] if "@media" in phone[10:] else phone
        self.assertIn(".traj-below { grid-template-columns: minmax(0, 1fr); }", phone)
        self.assertIn(".cmp-row { grid-template-columns: 96px minmax(0, 1fr); }", phone)
        self.assertNotIn("overflow-x: auto", phone)

    def test_reading_order_and_state_badges(self):
        render = self.RENDER
        order = [render.index(s) for s in ('<div class="metric-h">', '<section class="traj"', '${comparisonSummary(t, v)}',
                                            '<section class="traj-controls"', '<section class="run-details')]
        self.assertEqual(order, sorted(order), "name · trajectory · summary · controls · provenance")
        self.assertIn("TRAJECTORY.liveHead", render)
        self.assertIn("TRAJECTORY.snapshot", render)
        self.assertIn("Later observations are not included", render)
        self.assertIn("Return to live head", render)

    def test_deltas_use_the_cells_own_basis_and_carry_no_verdict(self):
        render = self.RENDER
        delta = render[render.index("function deltaText(f)"):render.index("function comparisonSummary")]
        self.assertIn("f.basis", delta)
        self.assertIn("return ''", delta)                           # no basis, no delta
        self.assertNotRegex(delta, r"good|bad|worse|better|risk")
        self.assertNotRegex(self.CSS, r"\.cmp-row[^{]*\{[^}]*(#(?:e|f)[0-9a-f]0000|red|green)")


if __name__ == "__main__":
    unittest.main()

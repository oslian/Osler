"""The chart inside an expanded indicator row, and the four charts it must not touch.

The list's detail panel is read as a whole: the indicator's name, what it is being
compared against, its forecast cells, the chart, and the link to the full
trajectory. Before this, the panel asked for one fixed 900x300 frame and the
stylesheet scaled it to whatever width the window gave it, so on a wide monitor
the chart alone grew past 790px and pushed everything under it off the screen -
while on a phone the same ratio collapsed it to a 103px strip.

A chart is a viewBox, so the drawing scales uniformly into its box and the ratio
alone decides the rendered height. That is what these tests pin: the ratio ladder
per context, the cap that stops a very wide window from making the panel taller,
that nothing forces a width and a height together (which would stretch the
drawing), and that everything the reader needs is still drawn and still inside
the frame at the narrowest size.

The standalone trajectory page is the analysis view and keeps its own larger
frame; the report, the printed PDF and the ward order keep theirs exactly. Those
are pinned here too, because they all call the same function.
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
SIGNALS = (STREAM_JS / "signals.js").read_text(encoding="utf-8")
CHART = (STREAM_JS / "chart.js").read_text(encoding="utf-8")
REPORT = (STREAM_JS / "report.js").read_text(encoding="utf-8")
WARD = (STREAM_JS / "ward.js").read_text(encoding="utf-8")
RENDER = (STREAM_JS / "render.js").read_text(encoding="utf-8")

# How much of the window the detail panel's chart actually gets, measured in the
# browser at each of these widths against the live release runtime (the workspace
# gutters, the card border and the panel's own 16px/10px padding). The frames are
# chosen against these numbers, so the test computes the same heights a reader
# sees rather than trusting the viewBox alone.
CHART_WIDTH_AT = {2477: 1500, 1920: 1500, 1600: 1500, 1440: 1345, 1280: 1193, 1024: 937, 375: 308}
# What each of those widths has to come out at, from the brief.
HEIGHT_BAND = {2477: (360, 420), 1920: (360, 420), 1600: (360, 420), 1440: (360, 420),
               1280: (340, 380), 1024: (300, 350), 375: (260, 300)}

PROBE = r"""
import { CHART_FRAMES, INLINE_MAX_WIDTH, chartFrame, trajectoryChart } from './chart.js';

const H = 3600000, T0 = Date.UTC(2026, 2, 14, 6);
const iso = (ms) => new Date(ms).toISOString();
const obs = (h, v) => ({ ms: T0 + h * H, value: v, effective_time: iso(T0 + h * H), resource_id: `o-${h}`, version_id: '1' });
const cell = (h, status, point) => ({ cell: `respiratory_rate@${h}h`, target: 'respiratory_rate', horizon_hours: h,
  status, forecast_point: point, unit: 'breaths/min', due_ms: T0 + (6 + h) * H,
  basis: { ms: T0 + 6 * H, value: 22 }, source_model: status === 'model' ? 'frozen_whole_body_jepa' : 'patient_persistence_anchor' });

const traj = {
  points: [obs(0, 18), obs(2, 19.5), obs(4, 21), obs(6, 22)],
  cutoff_ms: T0 + 6 * H, kind: 'model', frozen: false,
  forecasts: [cell(1, 'model', 23.4), cell(3, 'model', 24.1), cell(6, 'model', 25.8),
              cell(12, 'model', 27.2), cell(24, 'persistence', 22)],
};

const sweep = {};
for (const vw of [320, 375, 480, 600, 768, 900, 1024, 1152, 1280, 1366, 1440, 1600, 1920, 2477, 3840]) {
  const f = chartFrame('inline-detail', vw);
  sweep[vw] = { width: f.width, height: f.height, ratio: +(f.width / f.height).toFixed(5) };
}

const draw = (vw, unit) => {
  const f = chartFrame('inline-detail', vw);
  const c = trajectoryChart({ traj, unit, label: 'Respiratory rate', axis: 'calendar', onset_ms: 0,
    window: 'full', width: f.width, height: f.height, context: 'inline-detail', idSuffix: 'respiratory_rate',
    range: { low: 12, high: 20, unit } });
  return { svg: c.svg, frame: f };
};

let threw = null;
try { chartFrame('report-trend', 1440); } catch (e) { threw = e.message; }

console.log(JSON.stringify({
  sweep, ladder: CHART_FRAMES['inline-detail'], cap: INLINE_MAX_WIDTH,
  narrow: draw(375, 'breaths/min'), wide: draw(1440, 'breaths/min'),
  shortUnit: draw(1440, 'mmol/L'),
  unknownContext: threw,
  defaulted: chartFrame('inline-detail', undefined),
}));
"""


def _run_probe():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name in ("chart.js", "format.js"):
            shutil.copy(STREAM_JS / name, root / name)
        (root / "package.json").write_text('{"type": "module"}', encoding="utf-8")
        (root / "probe.mjs").write_text(PROBE, encoding="utf-8")
        result = subprocess.run([NODE, "probe.mjs"], cwd=root, capture_output=True, text=True,
                                encoding="utf-8", timeout=60)
    if result.returncode != 0:
        raise AssertionError(f"probe failed:\n{result.stderr}")
    return json.loads(result.stdout)


def _text_boxes(svg):
    """Every <text>'s x, its anchor and its glyph-estimated width, in viewBox units."""
    out = []
    for tag in re.findall(r"<text\b[^>]*>(?:(?!</text>).)*</text>", svg, re.S):
        x = re.search(r'\bx="(-?[\d.]+)"', tag)
        if not x:
            continue
        anchor = re.search(r'text-anchor="(\w+)"', tag)
        body = re.sub(r"<[^>]+>", "", tag)
        out.append((float(x.group(1)), (anchor.group(1) if anchor else "start"), len(body) * 6.4, body))
    return out


@unittest.skipUnless(NODE, "node is not on PATH; the executed chart tests need it")
class InlineDetailFrameTests(unittest.TestCase):
    """The frame ladder, run rather than read."""

    @classmethod
    def setUpClass(cls):
        cls.out = _run_probe()

    def test_every_required_width_lands_in_its_band(self):
        for vw, (lo, hi) in HEIGHT_BAND.items():
            f = self.out["sweep"][str(vw)]
            height = CHART_WIDTH_AT[vw] / f["ratio"]
            self.assertGreaterEqual(round(height), lo, f"{vw}px: the inline chart is only {height:.0f}px tall")
            self.assertLessEqual(round(height), hi, f"{vw}px: the inline chart is {height:.0f}px tall")

    def test_a_very_wide_window_stops_making_the_panel_taller(self):
        # past the cap the drawing does not widen, so the height is flat, not climbing
        self.assertEqual(self.out["cap"], 1500)
        heights = {vw: self.out["cap"] / self.out["sweep"][str(vw)]["ratio"] for vw in (1600, 1920, 2477, 3840)}
        self.assertEqual(len(set(round(h) for h in heights.values())), 1, f"the chart keeps growing: {heights}")
        self.assertLessEqual(max(heights.values()), 420)

    def test_the_panel_is_far_shorter_than_the_frame_it_replaced(self):
        # the old panel asked for one 900x300 frame at every width and let CSS
        # scale it: on this monitor that alone was 794px of chart
        before = 2414 / (900 / 300)
        after = self.out["cap"] / self.out["sweep"]["2477"]["ratio"]
        self.assertLess(after, before * 0.7, f"{before:.0f}px -> {after:.0f}px is less than a 30% cut")

    def test_the_ladder_gets_taller_as_the_window_narrows(self):
        ratios = [self.out["sweep"][str(vw)]["ratio"] for vw in (320, 375, 480, 600, 768, 900, 1024, 1280, 1440)]
        self.assertEqual(ratios, sorted(ratios), f"a narrower window must not get a wider ratio: {ratios}")
        self.assertLess(ratios[0], 2.0, "a phone needs a nearly square frame or the plot is a strip")

    def test_a_context_is_named_and_never_guessed(self):
        # an unnamed place is an error, not a silent default
        self.assertIn("no frame is defined for context", self.out["unknownContext"] or "")
        # and a caller with no window (a test, a server render) gets the desktop rung
        self.assertEqual((self.out["defaulted"]["width"], self.out["defaulted"]["height"]),
                         (self.out["sweep"]["1440"]["width"], self.out["sweep"]["1440"]["height"]))

    def test_the_drawing_is_scaled_and_never_stretched(self):
        for key in ("narrow", "wide"):
            drawn = self.out[key]
            vb = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', drawn["svg"])
            self.assertIsNotNone(vb, key)
            self.assertEqual((float(vb.group(1)), float(vb.group(2))),
                             (drawn["frame"]["width"], drawn["frame"]["height"]))
            self.assertNotRegex(drawn["svg"][:400], r'preserveAspectRatio="none"')
            self.assertIn('data-context="inline-detail"', drawn["svg"])

    # ---- nothing the reader needs is lost or clipped ---------------------
    def test_the_narrow_chart_still_says_everything(self):
        svg = self.out["narrow"]["svg"]
        for needed in ("OBSERVED HISTORY", "FUTURE FORECAST", "Forecast starts here", "Model input cutoff",
                       "breaths/min", "Supplied range", "trajTitle-respiratory_rate", "trajDesc-respiratory_rate"):
            self.assertIn(needed, svg, needed)
        self.assertRegex(svg, r'<title id="trajTitle-respiratory_rate">[^<]*Respiratory rate')
        self.assertRegex(svg, r'<desc id="trajDesc-respiratory_rate">.{40,}</desc>')
        self.assertRegex(svg, r'aria-labelledby="trajTitle-respiratory_rate"')
        self.assertRegex(svg, r'aria-describedby="trajDesc-respiratory_rate"')
        for horizon in ("1h", "3h", "6h", "12h", "24h"):
            self.assertRegex(svg, rf">{horizon} · [\d.]+", f"the {horizon} forecast label is missing")
        self.assertIn('data-region="range"', svg)
        self.assertIn('data-region="future"', svg)
        self.assertIn('data-cutoff="rule"', svg)

    def test_no_label_is_drawn_outside_the_frame(self):
        for key in ("narrow", "wide", "shortUnit"):
            drawn = self.out[key]
            frame_w = drawn["frame"]["width"]
            for x, anchor, w, body in _text_boxes(drawn["svg"]):
                left = x if anchor == "start" else x - w if anchor == "end" else x - w / 2
                right = left + w
                self.assertGreaterEqual(round(left), -1, f"{key}: {body!r} starts at {left:.0f}, off the left edge")
                self.assertLessEqual(round(right), frame_w + 1,
                                     f"{key}: {body!r} ends at {right:.0f}, past the {frame_w} frame")

    def test_a_unit_too_wide_for_the_gutter_moves_instead_of_being_clipped(self):
        # "breaths/min" used to be drawn from a negative x and cut off the frame
        self.assertIn('data-unit="axis-top"', self.out["narrow"]["svg"])
        self.assertIn('data-unit="axis-top"', self.out["wide"]["svg"])
        # a unit that fits the gutter is left exactly where it always was
        self.assertNotIn('data-unit="axis-top"', self.out["shortUnit"]["svg"])
        self.assertIn('<text x="50" y="30" text-anchor="end" class="ct">mmol/L</text>', self.out["shortUnit"]["svg"])


class InlineDetailWiringTests(unittest.TestCase):
    """Source-level: who asks for which frame, and what the stylesheet is allowed to do."""

    def test_the_list_asks_for_the_inline_frame_by_name(self):
        self.assertIn("chartFrame('inline-detail'", SIGNALS)
        self.assertIn("context: 'inline-detail'", SIGNALS)
        self.assertIn("width: frame.width, height: frame.height", SIGNALS)
        self.assertNotIn("width: 900, height: 300", SIGNALS)
        # the rung comes from the window width alone, never from what was drawn before
        self.assertIn("typeof window === 'undefined' ? 1440 : window.innerWidth", SIGNALS)

    def test_the_stylesheet_caps_the_width_and_never_forces_a_height(self):
        rule = re.search(r'\.detail-chart \.bigchart\[data-context="inline-detail"\] \{([^}]*)\}', STREAM_CSS)
        self.assertIsNotNone(rule, "the inline chart is sized by context, not by container alone")
        body = rule.group(1)
        self.assertIn("max-width: 1500px", body)
        self.assertIn("height: auto", body)
        self.assertIn("margin: 0 auto", body)
        self.assertNotRegex(body, r"height:\s*(?!auto)\S")
        self.assertNotRegex(body, r"aspect-ratio")
        # and the cap agrees with the one the module publishes
        self.assertIn("export const INLINE_MAX_WIDTH = 1500;", CHART)

    def test_nothing_scrolls_sideways_on_a_phone(self):
        phone = STREAM_CSS[STREAM_CSS.index("@media (max-width: 767px)"):]
        phone = phone[:phone.index("@media", 10)] if "@media" in phone[10:] else phone
        self.assertNotRegex(phone, r"\.detail-chart[^{]*\{[^}]*min-width")
        self.assertNotRegex(STREAM_CSS, r"\.bigchart[^{]*\{[^}]*min-width")
        self.assertIn(".detail-head, .detail-grid, .detail-chart, .detail-actions, .detail-prov "
                      "{ padding-left: 10px; padding-right: 10px; }", phone)


class OtherChartsUnchangedTests(unittest.TestCase):
    """The three charts this change was not allowed to touch, plus the analysis page."""

    def test_the_standalone_trajectory_page_keeps_its_large_frame(self):
        self.assertIn("const width = chartWidth(host);", RENDER)
        self.assertIn("const height = Math.round(Math.min(380, Math.max(300, width * 0.32)));", RENDER)
        self.assertNotIn("chartFrame(", RENDER, "the analysis view keeps its own sizing")
        self.assertNotIn("inline-detail", RENDER)

    def test_the_report_and_printed_pdf_charts_keep_their_frame(self):
        self.assertIn("width: 700, height: 186", REPORT)
        self.assertNotIn("chartFrame(", REPORT)
        self.assertNotIn("inline-detail", REPORT)
        self.assertRegex(STREAM_CSS, r"\.report-chart \.bigchart \{[^}]*width: 100%")
        self.assertNotRegex(STREAM_CSS, r"\.report-chart \.bigchart \{[^}]*max-width")

    def test_the_ward_observation_chart_keeps_its_frame(self):
        self.assertIn("width: 720, height: 190", WARD)
        self.assertNotIn("chartFrame(", WARD)
        self.assertNotIn("inline-detail", WARD)

    def test_only_the_list_names_a_context(self):
        named = [name for name, src in (("signals.js", SIGNALS), ("render.js", RENDER),
                                        ("report.js", REPORT), ("ward.js", WARD)) if "context:" in src]
        self.assertEqual(named, ["signals.js"])

    def test_the_anatomy_history_sparkline_is_not_a_trajectory_chart(self):
        self.assertIn("export function sparkline(points, width = 120, height = 32)", CHART)
        self.assertNotIn("chartFrame", (STREAM_JS / "anatomy.js").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()

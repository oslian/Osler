"""Source-level UI contract, accessibility hooks, safety wording, and a secret/PHI/path scan."""

import json
import re
import unittest

from oslian_demo.config import (ANATOMY_ASSETS_DIR, CSS_DIR, FIXTURE_DIR, FRONTEND_DIR, JS_DIR,
                                PACKAGE_DIR)

HTML = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")
MODULES = {name: (JS_DIR / "stream" / name).read_text(encoding="utf-8")
           for name in ("api.js", "store.js", "anatomy.js", "chart.js", "render.js", "main.js", "format.js")}
STREAM_JS = "\n".join(MODULES.values())
SHELL_JS = (JS_DIR / "shell.js").read_text(encoding="utf-8")
STREAM_CSS = (CSS_DIR / "stream.css").read_text(encoding="utf-8")
APP_CSS = (CSS_DIR / "app.css").read_text(encoding="utf-8")

# Every `.stream-main…{ grid-template-columns: … }` rule written outside a media
# query, in source order: (selector, value).
_GRID_RULES = re.findall(
    r"(\.stream-main[^{;@]*?)\{[^{}]*?grid-template-columns:\s*([^;}]+)",
    re.sub(r"/\*.*?\*/", "", re.split(r"@media", STREAM_CSS)[0], flags=re.S))


def _grid_columns(view, metric_open):
    """The grid-template-columns a browser would compute for `.stream-main`.

    Applies the two rules the cascade actually turns on here: higher specificity
    wins, and source order breaks a tie. Deliberately narrow — it understands
    only `.stream-main`, `[data-view="…"]` and `.metric-open`, which is the whole
    vocabulary of these selectors.
    """
    winner, best = None, (-1, -1)
    for order, (selector, value) in enumerate(_GRID_RULES):
        for one in (s.strip() for s in selector.split(",")):
            views = re.findall(r'\[data-view="([^"]+)"\]', one)
            if views and view not in views:
                continue
            if ".metric-open" in one and not metric_open:
                continue
            specificity = one.count(".") + one.count("[")
            if (specificity, order) >= best:
                winner, best = value.strip(), (specificity, order)
    return winner
ANATOMY_ASSETS = ANATOMY_ASSETS_DIR
ORGAN_LAYOUT = json.loads((ANATOMY_ASSETS / "organ-layout.json").read_text(encoding="utf-8"))
ORGAN_SOURCES = json.loads((ANATOMY_ASSETS / "organs" / "SOURCES.json").read_text(encoding="utf-8"))


class StreamUiContractTests(unittest.TestCase):
    def test_first_view_states_mode_status_clocks_and_output_source(self):
        self.assertIn("Synthetic FHIR R4", STREAM_JS)
        self.assertIn("Not a live Epic or hospital feed", STREAM_JS)
        self.assertIn("Retrospective replay", STREAM_JS)
        for word in ("Running", "Paused", "Reconnecting", "Completed"):
            self.assertIn(word, STREAM_JS)
        for clock in ("Latest clinical event", "Last received", "Model input watermark", "Causal anchor"):
            self.assertIn(clock, STREAM_JS)
        self.assertIn("runtime not attached", STREAM_JS)
        self.assertIn("runtime failure", STREAM_JS)
        self.assertIn("persistence baseline", STREAM_JS)

    def test_controls_speeds_and_transport(self):
        for act in ('data-act="start"', 'data-act="pause"', 'data-act="reset"', 'data-act="speed"', 'data-act="step"'):
            self.assertIn(act, STREAM_JS)
        self.assertIn("[1, 5, 20]", STREAM_JS)
        self.assertNotIn("setInterval(() => fetch", STREAM_JS)          # no polling of the ledger
        self.assertIn("new EventSource(", STREAM_JS)
        self.assertIn("last_event_id=", STREAM_JS)

    def test_three_level_navigation(self):
        for view in ("anatomy", "functions", "signals", "cells"):
            self.assertIn(f"['{view}',", MODULES["render.js"])
        self.assertIn('data-act="open-metric"', MODULES["render.js"])
        self.assertIn('data-act="close-metric"', MODULES["render.js"])
        self.assertIn("workspaceFor('organ'", MODULES["render.js"])
        self.assertIn("workspaceFor('function'", MODULES["render.js"])
        self.assertIn("Primary indicators", MODULES["render.js"])
        self.assertIn("Related indicators", MODULES["render.js"])
        self.assertIn("organ_primary_function", MODULES["store.js"])
        self.assertIn('data-act="mode"', MODULES["render.js"])

    def test_metric_chart_semantics(self):
        chart = MODULES["chart.js"]
        self.assertIn("no calibrated interval for this runtime cell", chart)
        self.assertIn("no interval band is ever drawn", chart)
        self.assertNotIn("f.lower", chart)
        self.assertNotIn("f.upper", chart)
        self.assertIn("persistence baseline", chart)
        self.assertIn('stroke-dasharray="5 4"', chart)         # model path is dashed
        self.assertIn('stroke-dasharray="1 4"', chart)         # persistence baseline is dotted
        self.assertIn("withheld", chart)
        self.assertIn("model input cutoff", chart)
        self.assertIn("historical anchor", chart)
        self.assertIn('tabindex="0"', chart)                   # chart points are keyboard reachable
        self.assertIn("focusin", MODULES["main.js"])           # tooltips on focus, not only hover
        for window in ("recent", "full"):
            self.assertIn(f"'{window}'", MODULES["render.js"])
        self.assertIn("scrub-run", MODULES["render.js"])

    def test_bounded_research_demo_wording_and_forbidden_claims(self):
        render = MODULES["render.js"]
        for phrase in ("bounded research demo", "research-only", "point forecasts only", "no clinical alert"):
            self.assertIn(phrase, render)
        self.assertNotIn("production runtime", STREAM_JS)
        self.assertNotIn("formal adjacent", STREAM_JS)
        self.assertNotIn("validated model", STREAM_JS.replace("not a formal or clinically validated model", ""))
        self.assertNotIn("formal adjacent", HTML)

    def test_horizon_labels_avoid_collisions(self):
        chart = MODULES["chart.js"]
        self.assertIn("function placeLabels(", chart)
        self.assertIn("lane", chart)
        self.assertNotIn("(i % 2) * 12", chart, "the old alternate-offset heuristic is gone")
        self.assertIn("labelAnchor(", chart)

    def test_anchor_mode_is_unmistakable(self):
        self.assertIn("Historical anchor · not the live head", STREAM_JS)
        self.assertIn("Return to live head", STREAM_JS)
        self.assertIn("frozen at this cutoff", STREAM_JS)
        self.assertIn("anchor-band", STREAM_CSS)
        self.assertIn("frozen", MODULES["anatomy.js"])

    def test_no_fabrication_and_no_intervals_or_alerts(self):
        self.assertIn("Nothing is fabricated", STREAM_JS)
        self.assertIn("Intervals are null", STREAM_JS)
        self.assertIn("causal_claim_allowed: false", STREAM_JS)
        self.assertNotIn("alert(", STREAM_JS)
        self.assertNotRegex(STREAM_JS, r"\b(deteriorat|sepsis risk|diagnostic|diagnosed)", "no diagnostic narrative")
        self.assertIn("never states that an organ or function is normal or abnormal", MODULES["render.js"])
        declarations = re.sub(r"/\*.*?\*/", "", STREAM_CSS, flags=re.S)
        self.assertNotRegex(declarations, r"#(?:e|f)[0-9a-f]0000|\bred\b|\bgreen\b", "no red/green status colours")

    def test_the_page_container_has_a_gutter_and_home_is_centred(self):
        """`.shell` wraps Home and Medication Safety and once had no rule at all,
        so their copy ran to the window edge while the bars above it kept the
        gutter. Both halves are asserted: the container's gutter, and the centring
        that stops the home block from being stranded left on a wide display.
        """
        self.assertIn('<div class="shell">', SHELL_JS + HTML)
        shell = re.search(r"^\.shell\s*\{([^}]*)\}", APP_CSS, re.M)
        self.assertIsNotNone(shell, ".shell must carry the page gutter")
        self.assertIn("var(--gutter)", shell.group(1))
        home = re.search(r"^\.home\s*\{([^}]*)\}", APP_CSS, re.M)
        self.assertIsNotNone(home)
        self.assertIn("margin-left: auto", home.group(1))
        self.assertIn("margin-right: auto", home.group(1))
        # Monitoring is deliberately full-bleed: it must not be wrapped in .shell
        self.assertNotRegex(HTML[HTML.index('id="view-stream"'):], r'^[^<]*<div class="shell"')

    def test_a_hidden_rail_never_reserves_its_column_when_a_chart_is_open(self):
        """Resolve the grid cascade the way a browser would, not by string search.

        ``All signals`` and ``All forecast cells`` hide the rail, so opening a
        chart from either must give the workspace the whole width. The rule that
        collapses the rail to 128px is written for Anatomy and Functions, and a
        plain string assertion would not notice it winning here again on order or
        specificity — which is exactly how this broke.
        """
        for view, expected in (("signals", "minmax(0, 1fr)"),
                               ("cells", "minmax(0, 1fr)"),
                               ("anatomy", "128px minmax(0, 1fr)"),
                               ("functions", "128px minmax(0, 1fr)")):
            self.assertEqual(_grid_columns(view, metric_open=True), expected,
                             f"{view} with a chart open")
        # and with no chart open the rail keeps its full width where it is shown
        self.assertEqual(_grid_columns("anatomy", metric_open=False), "372px minmax(0, 1fr)")
        self.assertEqual(_grid_columns("signals", metric_open=False), "minmax(0, 1fr)")

    def test_anatomy_is_anatomical_accessible_and_reduced_motion_safe(self):
        anatomy = MODULES["anatomy.js"]
        self.assertIn("role: 'button'", anatomy)
        self.assertIn("tabindex", anatomy)
        self.assertIn("aria-label", anatomy)
        self.assertIn("ArrowDown", anatomy)
        self.assertIn("mouseenter", anatomy)
        self.assertIn("organ-pill", anatomy)
        self.assertIn("prefers-reduced-motion", STREAM_CSS)
        self.assertIn("organ-pulse", STREAM_CSS)
        self.assertIn(".stream-main.paused .anatomy-host .organ-fx.pulse", STREAM_CSS)
        self.assertNotRegex(anatomy, r"perfusion|acid_base|coagulation|inflammatory", "no functional system drawn as organ")
        self.assertIn("max-width: 1023px", STREAM_CSS)          # organ selector on small screens
        self.assertIn("overflow-x: auto", STREAM_CSS)

    def test_organ_geometry_lives_only_in_the_layout_manifest(self):
        """The renderer must not carry organ coordinates of its own."""
        anatomy = MODULES["anatomy.js"]
        self.assertIn("organ-layout.json", anatomy)
        # no path data and no hard-coded organ id table in the renderer
        self.assertNotRegex(anatomy, r'<path d="M\d', "no organ geometry hard-coded in the renderer")
        for organ in ("kidneys", "liver", "lungs", "heart", "pancreas", "spleen", "brain", "intestines"):
            self.assertNotIn(f"  {organ}:", anatomy)
        self.assertIn("layout.frame.viewBox", anatomy)
        self.assertIn("sp.transform", anatomy)

    def test_layout_manifest_covers_every_canonical_organ_with_patient_laterality(self):
        self.assertEqual(ORGAN_LAYOUT["schema"], "oslian.organ-layout.v1")
        self.assertEqual(ORGAN_LAYOUT["frame"]["laterality"], "patient")
        entries = ORGAN_LAYOUT["organs"]
        ids = {o["organ"] for o in entries}
        canonical = {"heart", "blood_vessels", "lungs", "kidneys", "liver", "pancreas", "stomach",
                     "intestines", "thyroid", "adrenal_glands", "bone_marrow", "spleen", "brain",
                     "skeletal_muscle"}
        self.assertEqual(ids, canonical, "every canonical organ is present and nothing else is")
        for o in entries:
            self.assertIn(o["side"], {"patient_left", "patient_right", "midline", "bilateral"})
            self.assertNotRegex(o["organ"], r"perfusion|acid_base|coagulation|inflammatory|metabolic")
            if o["visual_asset_missing"]:
                self.assertIsNone(o["art"], f"{o['organ']} has no asset, so it draws nothing")
                self.assertIsNone(o["fx"], f"{o['organ']} has no asset, so it highlights nothing")
        # the paired organs each carry one container per side
        for organ in ("kidneys", "adrenal_glands"):
            sides = sorted(o["side"] for o in entries if o["organ"] == organ)
            self.assertEqual(sides, ["patient_left", "patient_right"], organ)

    def test_organ_placement_is_translation_and_uniform_scale_only(self):
        """No stretch, no shear, no warp: every sprite transform is translate + one scale."""
        allowed = re.compile(r"^(?:translate\(-?[\d.]+,0\) scale\(-1,1\) )?"
                             r"translate\(-?[\d.]+,-?[\d.]+\) scale\([\d.]+\)$")
        seen = 0
        for o in ORGAN_LAYOUT["organs"]:
            for sp in (o["art"], o["fx"]):
                if not sp:
                    continue
                self.assertRegex(sp["transform"], allowed, f"{o['organ']} {o['side']}")
                seen += 1
        self.assertGreaterEqual(seen, 12)

    def test_baseplate_display_variants_keep_every_source_path(self):
        """The native simplified organ outlines are hidden for display, never deleted."""
        source = (ANATOMY_ASSETS / "body-ladyofhats-handfix.svg").read_text(encoding="utf-8")
        suppressed = ORGAN_LAYOUT["baseplate"]["suppressed_native_shapes"]
        self.assertTrue(suppressed)
        source_paths = len(re.findall(r"<(?:\w+:)?path", source))
        for key in ("back", "front", "vessels"):
            text = (ANATOMY_ASSETS / ORGAN_LAYOUT["baseplate"][key]).read_text(encoding="utf-8")
            self.assertEqual(len(re.findall(r"<(?:\w+:)?path", text)), source_paths,
                             f"{key} variant must keep every source path")
            for element_id in suppressed:
                self.assertIn(f'id="{element_id}"', text, f"{element_id} must still exist in {key}")

    def test_organ_response_never_moves_an_organ(self):
        """A state may change opacity and filter; it may not touch a transform."""
        css = re.sub(r"/\*.*?\*/", "", STREAM_CSS, flags=re.S)
        for rule in re.findall(r"\.anatomy-host \.organ-fx[^{]*\{([^}]*)\}", css):
            self.assertNotIn("transform", rule, "an organ response must not move the organ")
        # the update pass may set classes and aria state; it may never set a transform
        anatomy = MODULES["anatomy.js"]
        update_body = anatomy[anatomy.index("    update(state) {"):]
        self.assertNotIn("transform", update_body, "the response pass must not touch any transform")

    def test_every_drawn_organ_asset_is_an_anterior_view(self):
        """The figure faces the viewer, so no organ may be drawn from a lateral view."""
        drawn = {sp["asset"] for o in ORGAN_LAYOUT["organs"] for sp in (o["art"], o["fx"])
                 if sp and not sp["asset"].startswith("body-")}
        self.assertIn("brain-anterior.svg", drawn)
        self.assertNotIn("brain.svg", drawn)
        views = {e["file"]: e["view"] for e in ORGAN_SOURCES}
        for asset in drawn:
            self.assertIn(asset, views, f"{asset} must carry a SOURCES entry")
            self.assertEqual(views[asset], "anterior",
                             f"{asset} is a {views[asset]} view and cannot sit in a front-facing figure")
            self.assertTrue((ANATOMY_ASSETS / "organs" / asset).exists(), asset)
        # an asset SOURCES marks unused must not be drawn anywhere
        for e in ORGAN_SOURCES:
            if not e["used"]:
                self.assertNotIn(e["file"], drawn, f"{e['file']} is marked unused")

    def test_a_reload_restores_the_organ_states_from_the_server_snapshot(self):
        """Refreshing mid-stream must not go dark: the snapshot carries organ routing."""
        self.assertIn("applySnapshotOrgans", MODULES["store.js"])
        self.assertIn("applySnapshotOrgans(state.organs)", MODULES["main.js"])
        # the browser reads the server's routing; it never derives an organ itself
        self.assertNotIn("organsForVariable", STREAM_JS)

    def test_live_and_historical_organ_state_are_kept_apart(self):
        """An open anchor shows its run's organs; SSE events only touch the live state."""
        store, render, main = MODULES["store.js"], MODULES["render.js"], MODULES["main.js"]
        self.assertIn("export const currentOrganEvents", store)
        # the anchor carries its own organ state, built from the run's own snapshot
        self.assertIn("organEvents: organStateFrom(forecasts.organs, 'anchor')", main)
        self.assertIn("if (forecasts.anchor === 'live_head') S.anchor = null;", main)
        # the figure and the rail both read through the selector, never S.organEvents
        self.assertIn("organEvents: currentOrganEvents()", render)
        self.assertIn("currentOrganEvents()[id]", render)
        self.assertNotIn("S.organEvents", render)
        # the reducer writes live state only, so an open anchor cannot be polluted
        normalized = store[store.index("case 'observation.normalized'"):store.index("case 'observation.unmapped'")]
        self.assertIn("S.organEvents[o] =", normalized)
        self.assertNotIn("S.anchor", normalized)
        # the snapshot reload runs on every amendment while the stream is live; it may
        # replace the live state and must never reach into an open anchor
        apply_snapshot = store[store.index("export function applySnapshotOrgans("):]
        self.assertNotIn("S.anchor", apply_snapshot[:apply_snapshot.index(chr(10) + "}")])

    def test_a_historical_anchor_is_aged_on_its_run_cutoff_and_never_pulses(self):
        store, anatomy = MODULES["store.js"], MODULES["anatomy.js"]
        # freshness in anchor mode is measured on the run's own cutoff
        self.assertIn("S.anchor ? S.anchor.run.input_cutoff", store)
        # restored state (the live snapshot and a historical run alike) lights without pulsing
        self.assertIn("restored: origin", store)
        self.assertIn("organStateFrom(organs, 'snapshot')", store)
        update_body = anatomy[anatomy.index("    update(state) {"):]
        self.assertIn("`${ev.restored || 'stream'}:${ev.sequence}`", update_body)
        self.assertIn("&& !ev.restored", update_body)

    def test_a_reset_clears_every_organ_state(self):
        store = MODULES["store.js"]
        self.assertIn("clearOrganState", store)
        reset_branch = store[store.index("if (reason === 'reset')"):store.index("return 'reset';")]
        self.assertIn("clearOrganState();", reset_branch)
        self.assertIn("anchor: null", reset_branch)          # the historical state goes with it
        # ... and clearing the live state must not reach into an open anchor: a snapshot
        # reload runs on any amendment, and it would blank the historical view
        clear = store[store.index("export function clearOrganState() {"):]
        self.assertNotIn("S.anchor", clear[:clear.index(chr(10) + "}")])
        session_reset = store[store.index("export function resetSessionState()"):]
        self.assertIn("organEvents: {}", session_reset[:session_reset.index("});")])
        self.assertIn("unassetedOrganEvents: {}", session_reset[:session_reset.index("});")])

    def test_one_pulse_timer_per_organ_so_high_speed_cannot_queue(self):
        anatomy = MODULES["anatomy.js"]
        update_body = anatomy[anatomy.index("    update(state) {"):]
        self.assertIn("clearTimeout(node.pulseTimer)", update_body)
        self.assertEqual(update_body.count("setTimeout("), 1, "one pulse timer, per organ")
        self.assertIn("node.pulseTimer = setTimeout(", update_body)

    def test_markup_wires_the_monitoring_view_and_drops_the_replay_contract(self):
        self.assertIn('id="view-stream"', HTML)
        self.assertIn('data-view="stream"', HTML)
        self.assertIn("/static/js/stream/main.js", HTML)
        # the frontend tree is served from frontend/ at /static; every asset the page
        # names must resolve there, or the page loads with no styling and no modules
        for url in ("/static/css/app.css", "/static/css/stream.css", "/static/js/shell.js",
                    "/static/js/stream/main.js"):
            self.assertIn(url, HTML, url)
            self.assertTrue((FRONTEND_DIR / url[len("/static/"):]).is_file(), url)
        self.assertIn('id="anatomyHost"', HTML)
        self.assertIn('id="workspace"', HTML)
        self.assertNotIn("forecast-monitoring", HTML)
        self.assertNotIn("/api/forecast", HTML)
        self.assertIn("Synthetic FHIR R4 stream, not a live feed", HTML)
        self.assertIn("clinical alert", HTML)
        self.assertIn("go-stream", SHELL_JS)
        self.assertIn("Retrospective replay of a deidentified public eICU demo case", SHELL_JS)


class HygieneScanTests(unittest.TestCase):
    FILES = [p for root in (PACKAGE_DIR, FRONTEND_DIR) for p in root.rglob("*")
             if p.suffix in {".py", ".js", ".css", ".html", ".json", ".md"} and "data" not in p.parts]

    def test_no_secrets_local_paths_or_phi_in_demo_tree(self):
        patterns = {
            "windows path": re.compile(r"[A-Za-z]:\\\\?Users|[A-Za-z]:/Users|\\\\?jingb|/jingb"),
            "conda path": re.compile(r"conda_envs"),
            "private key": re.compile(r"BEGIN (RSA |EC )?PRIVATE KEY"),
            "api key": re.compile(r"sk-[A-Za-z0-9]{20,}|AIza[0-9A-Za-z_-]{30,}"),
            "bearer": re.compile(r"Bearer [A-Za-z0-9._-]{20,}"),
        }
        for path in self.FILES:
            text = path.read_text(encoding="utf-8", errors="replace")
            for name, pattern in patterns.items():
                self.assertIsNone(pattern.search(text), f"{name} in {path}")

    def test_fixtures_are_synthetic_or_public_deidentified_only(self):
        for name in ("synthetic_patient_v1.json", "wholebody_release_contract_v1.json", "retrospective_eicu_demo_case.json"):
            text = (FIXTURE_DIR / name).read_text(encoding="utf-8")
            self.assertNotRegex(text, r"\b(MRN|SSN)\b")
            self.assertNotRegex(text, r"\b\d{3}-\d{2}-\d{4}\b")
        synthetic = (FIXTURE_DIR / "synthetic_patient_v1.json").read_text(encoding="utf-8")
        self.assertIn('"synthetic": true', synthetic)
        self.assertNotIn("birthDate", synthetic)
        retro = (FIXTURE_DIR / "retrospective_eicu_demo_case.json").read_text(encoding="utf-8")
        self.assertIn('"retrospective": true', retro)
        self.assertIn("HIPAA safe-harbor deidentified", retro)


if __name__ == "__main__":
    unittest.main()

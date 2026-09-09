"""Source-level UI contract, accessibility hooks, safety wording, and a secret/PHI/path scan."""

import json
import re
import unittest

from oslian_demo.config import (ANATOMY_ASSETS_DIR, CSS_DIR, FIXTURE_DIR, FRONTEND_DIR, JS_DIR,
                                PACKAGE_DIR)

HTML = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")
MODULES = {name: (JS_DIR / "stream" / name).read_text(encoding="utf-8")
           for name in ("api.js", "store.js", "anatomy.js", "chart.js", "render.js", "main.js", "format.js",
                        "ward.js", "signals.js", "report.js")}
STREAM_JS = "\n".join(MODULES.values())
SHELL_JS = (JS_DIR / "shell.js").read_text(encoding="utf-8")
STREAM_CSS = (CSS_DIR / "stream.css").read_text(encoding="utf-8")
APP_CSS = (CSS_DIR / "app.css").read_text(encoding="utf-8")

# Every `.stream-main…{ grid-template-columns: … }` rule written outside a media
# query, in source order: (selector, value).
_GRID_RULES = re.findall(
    r"(\.stream-main[^{;@]*?)\{[^{}]*?grid-template-columns:\s*([^;}]+)",
    re.sub(r"/\*.*?\*/", "", re.split(r"@media", STREAM_CSS)[0], flags=re.S))


def _grid_columns(view, metric_open, report_open=False):
    """The grid-template-columns a browser would compute for `.stream-main`.

    Applies the two rules the cascade actually turns on here: higher specificity
    wins, and source order breaks a tie. Deliberately narrow — it understands
    only `.stream-main`, `[data-view="…"]`, `.metric-open` and `.report-open`,
    which is the whole vocabulary of these selectors.
    """
    winner, best = None, (-1, -1)
    for order, (selector, value) in enumerate(_GRID_RULES):
        for one in (s.strip() for s in selector.split(",")):
            views = re.findall(r'\[data-view="([^"]+)"\]', one)
            if views and view not in views:
                continue
            if ".metric-open" in one and not metric_open:
                continue
            if ".report-open" in one and not report_open:
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
        # The page must not decide which speeds exist: it renders what the
        # source declares, so a real feed can declare none and get no selector.
        self.assertNotIn("[1, 5, 20]", STREAM_JS, "speeds are hard-coded into the page again")
        self.assertNotRegex(MODULES["render.js"], r"\b20\s*×|\b20x\b",
                            "a bare 20x label is back in the main transport")
        self.assertIn("src.playback", MODULES["render.js"])
        self.assertIn("playback.speeds", MODULES["render.js"])
        self.assertIn("playback.developer_speeds", MODULES["render.js"])
        self.assertNotIn("setInterval(() => fetch", STREAM_JS)          # no polling of the ledger
        self.assertIn("new EventSource(", STREAM_JS)
        self.assertIn("last_event_id=", STREAM_JS)

    def test_the_stress_speed_is_a_developer_tool_not_a_demo_speed(self):
        """20x stays available, but never as one of the speeds a reader is offered.

        A speed here is a simulated arrival clock over a fixture. Presenting
        1x / 5x / 20x side by side invited the reading that 20x is how fast a
        hospital sends data, so the two demo speeds are named and the stress
        speed sits behind a disclosure that says what it is for.
        """
        render = MODULES["render.js"]
        self.assertIn('<details class="dev-tools"', render)
        self.assertIn("Developer / QA", render)
        self.assertIn('aria-label="Stress test speed"', render)
        # the disclosure is a real one: default-closed unless it is the active speed
        self.assertIn("devOptions.some((o) => speed === Number(o.value)) ? ' open' : ''", render)
        self.assertIn(".dev-tools", STREAM_CSS)

    def test_three_level_navigation(self):
        for view in ("anatomy", "functions", "data", "ward"):
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
        """The trajectory chart's safety invariants, at source level; the executed
        checks live in test_stream_trajectory.py."""
        chart = MODULES["chart.js"]
        self.assertIn("An interval band is never drawn", chart)
        self.assertNotIn("f.lower", chart)
        self.assertNotIn("f.upper", chart)
        without_comment = re.sub(r"/\*.*?\*/", "", chart + MODULES["format.js"], flags=re.S)
        self.assertNotIn("confidence", without_comment.lower())
        self.assertIn("POINT_ONLY_NOTE", chart)                # the legend says it in plain language
        self.assertIn("no uncertainty interval is drawn", chart)  # and so does the accessible description
        self.assertIn("carried forward", chart)
        self.assertIn('stroke-dasharray="5 4"', chart)         # model path is dashed
        self.assertIn('stroke-dasharray="3 4"', chart)         # a baseline is dashed differently, and grey
        self.assertNotIn("COLORS.blue", chart[chart.index("carried-forward baselines"):chart.index("// model forecast:")])
        self.assertIn("withheld", chart)
        for word in ("historyTitle", "forecastTitle", "cutoffLabel", "cutoffSub", "baselineOnly", "noModel", "unavailable"):
            self.assertIn(f"TRAJECTORY.{word}", chart)
        fmt = MODULES["format.js"]
        self.assertIn("'Forecast starts here'", fmt)
        self.assertIn("'Model input cutoff'", fmt)
        self.assertIn("'Baseline only · not a model forecast'", fmt)
        self.assertIn("'No forecast model for this metric'", fmt)
        self.assertIn("'Forecast unavailable'", fmt)
        self.assertIn('tabindex="0"', chart)                   # chart points are keyboard reachable
        self.assertIn('aria-label="${esc(aria)}"', chart)      # and each one says what it is
        self.assertIn('<desc id="trajDesc${sfx}">', chart)   # one description per chart, even two on a page
        self.assertIn("focusin", MODULES["main.js"])           # tooltips on focus, not only hover
        for window in ("recent", "full"):
            self.assertIn(f"'{window}'", MODULES["render.js"])
        self.assertIn("scrub-run", MODULES["render.js"])
        # the chart never reads the live state directly: every input comes through the mode selectors
        store = MODULES["store.js"]
        traj = store[store.index("export function trajectoryFor("):store.index("/** Workspace split")]
        for selector in ("currentRun()", "currentSeries()", "currentCells()"):
            self.assertIn(selector, traj)
        self.assertNotIn("S.series", traj)
        self.assertNotIn("S.live", traj)

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
        # a historical snapshot is named as one, above the page and beside the chart title
        self.assertIn("snapshot: 'HISTORICAL SNAPSHOT'", MODULES["format.js"])
        self.assertIn("liveHead: 'LIVE HEAD'", MODULES["format.js"])
        self.assertIn("Later observations are not included", MODULES["render.js"])
        self.assertIn("head-badge", MODULES["render.js"])

    def test_no_fabrication_and_no_intervals_or_alerts(self):
        self.assertIn("Nothing is fabricated", STREAM_JS)
        self.assertIn("Intervals are null", STREAM_JS)
        self.assertIn("causal_claim_allowed: false", STREAM_JS)
        self.assertNotIn("alert(", STREAM_JS)
        self.assertNotRegex(STREAM_JS, r"\b(deteriorat|sepsis risk|diagnostic|diagnosed)", "no diagnostic narrative")
        self.assertIn("never states that an organ or function is normal or abnormal", MODULES["render.js"])
        declarations = re.sub(r"/\*.*?\*/", "", STREAM_CSS, flags=re.S)
        self.assertNotRegex(declarations, r"#(?:e|f)[0-9a-f]0000|\bred\b|\bgreen\b", "no red/green status colours")

    def test_stream_events_are_throttled_but_user_actions_are_not(self):
        """A 20x burst must not repaint the page once per event.

        Rendering is the browser's half of the same problem the backend had: at
        20x the SSE channel delivers far faster than anyone can read, and a full
        pass per event is what made tab switching feel stuck. Events go through a
        floored path; a click goes straight to the next frame, because a person
        waiting on their own action is exactly what must not be throttled.
        """
        main = MODULES["main.js"]
        floor = re.search(r"const STREAM_RENDER_MS = (\d+)", main)
        self.assertIsNotNone(floor, "the stream render floor is gone")
        self.assertGreater(int(floor.group(1)), 0)

        # the SSE handler must use the throttled path, and only that one
        handler = re.search(r"onEvent\(type, data, sequence\) \{(.*?)\n    \},", main, re.S)
        self.assertIsNotNone(handler)
        body = handler.group(1)
        self.assertIn("scheduleStreamRender()", body)
        self.assertNotIn("scheduleRender();", body)

        # user-driven changes keep the immediate path
        for action in ("selectOrgan", "selectFunction", "async function control"):
            block = main[main.index(action):main.index(action) + 400]
            self.assertIn("scheduleRender()", block, action)

        # the throttle must always leave a trailing pass, or the last event of a
        # burst would never be drawn
        throttle = re.search(r"function scheduleStreamRender\(\) \{(.*?)\n\}", main, re.S)
        self.assertIsNotNone(throttle)
        self.assertIn("setTimeout", throttle.group(1))

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

        ``Signals & forecasts`` hides the rail, so opening a chart from it must
        give the workspace the whole width. The rule that
        collapses the rail to 128px is written for Anatomy and Functions, and a
        plain string assertion would not notice it winning here again on order or
        specificity — which is exactly how this broke.
        """
        for view, expected in (("data", "minmax(0, 1fr)"),
                               ("anatomy", "128px minmax(0, 1fr)"),
                               ("functions", "128px minmax(0, 1fr)")):
            self.assertEqual(_grid_columns(view, metric_open=True), expected,
                             f"{view} with a chart open")
        # and with no chart open the rail keeps its full width where it is shown
        self.assertEqual(_grid_columns("anatomy", metric_open=False), "372px minmax(0, 1fr)")
        self.assertEqual(_grid_columns("data", metric_open=False), "minmax(0, 1fr)")
        # a report is a document: it takes the width from every view, chart open or not
        for view in ("anatomy", "functions", "data", "ward"):
            self.assertEqual(_grid_columns(view, metric_open=False, report_open=True), "minmax(0, 1fr)", view)

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
        self.assertIn(".stream-main.paused .anatomy-figure .organ-fx.pulse", STREAM_CSS)
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


class WardViewContractTests(unittest.TestCase):
    """The ward order is a reading order over many stays, and never a forecast."""

    WARD = MODULES["ward.js"]

    def test_the_ward_is_a_view_of_this_page_not_a_second_page(self):
        # It reuses the rail/workspace grammar, so it must ride the same view
        # switch as anatomy and functions rather than bolt on its own shell.
        self.assertIn("['ward', 'Ward order']", MODULES["render.js"])
        self.assertIn("S.view === 'ward'", MODULES["render.js"])
        for key in ("ward:", "wardPatient:", "wardOpenBands:", "wardError:"):
            self.assertIn(key, MODULES["store.js"], key)
        self.assertNotIn('id="view-ward"', HTML, "the ward must not be a separate top-level view")
        self.assertNotIn('data-view="ward"', HTML, "the ward must not get its own nav button")

    @staticmethod
    def _code(source):
        """Source with comments removed, so a rule reads the code and not the prose
        that explains the rule. Both scans below kept tripping over their own
        explanation before this."""
        source = re.sub(r"/\*.*?\*/", "", source, flags=re.S)
        return re.sub(r"^\s*//.*$", "", source, flags=re.M)

    def test_the_ward_view_draws_no_forecast(self):
        # The order is computed from observations alone; that is why it survives
        # an unattached runtime. A forecast marker here would be a claim the
        # ward never made. Options passed, not words used: the module explains in
        # prose why it withholds `cutoff_ms`, and that sentence is not a breach.
        code = self._code(self.WARD)
        self.assertIn("forecasts: []", code)
        for forbidden in ("forecast_point", "chartLegend", "FORECAST_LABEL", "cutoff_ms:", "frozen:"):
            self.assertNotIn(forbidden, code, forbidden)

    def test_the_ward_view_never_speaks_in_alarm_words(self):
        # The page has to name what it denies being, so the denials are removed
        # before the scan rather than exempted from it.
        body = self._code(self.WARD).lower()
        for denial in ("not a clinical alert", "no clinical alert", "isclinicalalert"):
            body = body.replace(denial, "")
        for word in ("alert", "alarm", "critical", "urgent", "severity", "escalate"):
            self.assertNotIn(word, body, f"{word!r} belongs to alerting, not to a reading order")

    def test_the_denial_is_still_there_after_the_scan_strips_it(self):
        # Guard the guard: the strip above must not be able to hide a missing denial.
        self.assertIn("not a clinical alert", self.WARD)

    def test_the_ward_states_what_orders_it_and_what_it_is_not(self):
        for phrase in ("last measured creatinine", "runs no model", "not a clinical alert",
                       "retrospective, not a live feed", "published cutpoints"):
            self.assertIn(phrase, self.WARD, phrase)

    def test_the_page_keeps_the_band_basis_and_the_order_basis_apart(self):
        # The band comes from published cutpoints across every analyte; only the
        # order inside a band is the measured one. A page that let a reader take
        # the whole top-to-bottom order for the validated claim would be
        # overstating what was measured, so it says so outright.
        for phrase in ("Two different bases", "order inside a band",
                       "not that order"):
            self.assertIn(phrase, self.WARD, phrase)
        self.assertIn("AUROC 0.814", self.WARD)

    def test_the_page_says_one_value_far_out_is_enough_on_its_own(self):
        self.assertIn("furthest out of range any", self.WARD)
        self.assertIn("whatever the rest are doing", self.WARD)

    def test_a_row_names_the_analyte_that_put_it_on_the_list(self):
        # Without this a reader sees a creatinine of 0.6 at the top of the list
        # and no reason why it is there.
        code = self._code(self.WARD)
        self.assertIn("findings", code)
        self.assertIn("outsideCount", code)
        self.assertIn("sourcePinned", code)
        self.assertIn("cutpoint not yet from a citable source", self.WARD)

    def test_bands_group_the_rail_and_an_empty_band_is_still_reported(self):
        # A quiet ward is answered with an empty top band, not with filler rows.
        self.assertIn("ward-band", self.WARD)
        self.assertIn("none", self.WARD)
        self.assertIn("openByDefault", self.WARD)
        self.assertNotIn("quota", self._code(self.WARD).lower(), "the quota was replaced by bands")

    def test_a_tag_states_arithmetic_and_never_a_judgement(self):
        # Tags carry which published cut a value passed and the age of the
        # measurement. The published gradings have severity words for these
        # tiers; those words are judgements about a patient, and naming the cut
        # instead keeps the same ordering without the claim.
        code = self._code(self.WARD)
        self.assertIn("ageTag", code)
        self.assertIn('class="tag', code)
        for judgement in ("severe", "critical", "high risk", "stage ", "deteriorat",
                          "mild", "moderate"):
            self.assertNotIn(judgement, code.lower(), judgement)

    def test_urine_output_is_stated_per_kg_and_against_the_range_it_is_compared_to(self):
        # The bare mL/hour compares to nothing; the page must give the reader the
        # weight it divided by and the range it read the position off.
        code = self._code(self.WARD)
        self.assertIn("mL/kg/h", code)
        self.assertIn("urineRatePerKg", code)
        self.assertIn("urineBandLabel", code)
        self.assertIn("weightKg", code)
        self.assertIn("urineReferenceLower", code)

    def test_the_page_says_the_urine_range_is_a_convention_not_a_printed_interval(self):
        # Creatinine's range is on the laboratory's own report; this one is not,
        # and a reader weighing the two comparisons has to be told the difference.
        for phrase in ("range conventionally quoted for adults",
                       "not an interval the laboratory prints"):
            self.assertIn(phrase, self.WARD, phrase)

    def test_the_page_says_the_rate_per_kg_orders_nothing(self):
        # Only the creatinine signal was measured. If the page ever implies the
        # urine rate placed a row, it is claiming something nobody validated.
        self.assertIn("It orders nothing", self.WARD)
        self.assertIn("the creatinine's alone", self.WARD)

    def test_a_missing_weight_is_said_rather_than_filled_in(self):
        for phrase in ("rate per kg not stated", "no weight on record"):
            self.assertIn(phrase, self.WARD, phrase)
        self.assertNotIn("average weight", self.WARD.lower())

    def test_the_page_never_states_the_criterion_that_pairs_a_rate_with_a_duration(self):
        # Comparing one charted rate to a range is arithmetic. Requiring it to
        # hold for a span in order to name a class is a diagnosis the page does
        # not make -- see the matching scan over ranking.py.
        body = self._code(self.WARD).lower()
        for word in ("kdigo", "oliguri", "anuri", "aki", "stage 1", "stage 2", "stage 3"):
            self.assertNotIn(word, body, f"{word!r} classifies; the page only compares")

    def test_a_row_leads_into_the_stay_it_names(self):
        # The order answers "who first"; it has to be possible to then read that
        # stay, or the answer stops at a name.
        self.assertIn('data-act="open-ward-stay"', self.WARD)
        self.assertIn("Open this stay", self.WARD)
        main = MODULES["main.js"]
        self.assertIn("open-ward-stay", main)
        self.assertIn("openWardStay", main)
        # The stay is opened on its own recorded observations, through the same
        # session path every other source uses.
        self.assertIn("retrospective:${patientId}", main)

    def test_a_per_stay_source_key_is_a_legal_deep_link(self):
        # `#stream?mode=retrospective:<case id>` has to survive the sources
        # check, which only lists the two base keys.
        main = MODULES["main.js"]
        self.assertIn("knownMode", main)
        self.assertIn("split(':')[0]", main)

    def test_a_stay_that_will_not_open_is_reported_not_swallowed(self):
        self.assertIn("Could not open this stay", MODULES["main.js"])

    def test_colour_is_never_the_only_carrier_of_a_band(self):
        # Bands are grouped and labelled in text; no CSS rule may tint a row by band.
        declarations = re.sub(r"/\*.*?\*/", "", STREAM_CSS, flags=re.S)
        for key in ('.ward-band.x3', '.ward-band.x2', '[data-band="x3"]',
                    '.u_lowest', '.u_third', '[data-urine-band'):
            self.assertNotIn(key, declarations, key)

    def test_an_empty_ward_is_never_drawn_as_an_empty_list(self):
        self.assertIn("No order is shown rather than an empty one", self.WARD)

    def test_the_unscored_rows_are_listed_apart_not_ranked_low(self):
        self.assertIn("Listed apart, not ranked low", self.WARD)

    def test_the_anatomy_figure_is_hidden_on_the_ward_view(self):
        # The rail belongs to the order here; the body figure is about one patient.
        self.assertRegex(STREAM_CSS, r'\.stream-main\[data-view="ward"\] \.anatomy-host')


class IndicatorViewContractTests(unittest.TestCase):
    """Signals & forecasts is a result list: the indicator is the entry point, the
    flag sits beside the result, and one row's numbers replace the old per-row plot."""

    DATA = MODULES["signals.js"]
    STORE = MODULES["store.js"]

    @staticmethod
    def _code(source):
        source = re.sub(r"/\*.*?\*/", "", source, flags=re.S)
        return re.sub(r"^\s*//.*$", "", source, flags=re.M)

    def test_the_indicator_is_the_first_column_and_the_largest_text_in_the_row(self):
        head = self.DATA[self.DATA.index("const HEAD ="):self.DATA.index("function table(")]
        columns = re.findall(r'<th scope="col">([^<]+)</th>', head)
        self.assertEqual(columns, ["Indicator", "Current result", "Flag", "Reference range",
                                   "Previous result", "Forecast", "Updated"])
        # the name is bigger than the row's other text, and the row's own label leads with it
        name_rule = re.search(r"\.ind-row \.ind \.t \{([^}]+)\}", STREAM_CSS).group(1)
        self.assertIn("font-size: 14px", name_rule)
        self.assertIn("font-weight: 600", name_rule)
        self.assertIn('aria-label="${esc(label(row.variable))}, ${esc(FLAG[key].full)}', self.DATA)
        # a canonical id, a run id or a source model never appears in a row
        row = self.DATA[self.DATA.index("function indicatorRow("):self.DATA.index("const HEAD =")]
        for forbidden in ("run_id", "source_model", "registry"):
            self.assertNotIn(forbidden, row, forbidden)

    def test_a_row_states_its_numbers_and_draws_no_miniature_plot(self):
        self.assertNotIn("miniTrajectory", MODULES["chart.js"])
        for name, module in MODULES.items():
            self.assertNotIn("miniTrajectory", module, name)
        row = self.DATA[self.DATA.index("function indicatorRow("):self.DATA.index("const HEAD =")]
        self.assertNotIn("<svg", row)
        self.assertNotIn("sparkline", self.DATA)
        # and the vague "+N more" is gone: the forecast column names one horizon
        self.assertNotIn("more</span>", self.DATA)
        self.assertNotIn("+${row.forecasts.length - 1}", self.DATA)

    def test_current_previous_and_forecast_are_three_stated_comparisons(self):
        self.assertIn("function previousCell(", self.DATA)
        self.assertIn("from previous", self.DATA)
        self.assertIn("DATA_VIEW.noPrevious", self.DATA)
        self.assertIn("noPrevious: 'No previous result'", MODULES["format.js"])
        self.assertIn("vs current", self.DATA)
        self.assertIn("unchanged: 'unchanged'", MODULES["format.js"])
        # a previous result is the previous valid observation, never a fabricated zero
        prev = self.STORE[self.STORE.index("function previousResult("):self.STORE.index("export function indicatorRows(")]
        self.assertIn("points.length < 2", prev)
        self.assertIn("return null", prev)

    def test_one_horizon_control_drives_every_forecast_column(self):
        self.assertIn('data-act="forecast-horizon"', self.DATA)
        self.assertIn("compareAt: 'Compare forecast at'", MODULES["format.js"])
        self.assertIn("export function activeHorizon()", self.STORE)
        self.assertIn("export function horizonCoverage()", self.STORE)
        # the default is the horizon that covers the most indicators
        coverage = self.STORE[self.STORE.index("export function horizonCoverage()"):self.STORE.index("export function activeHorizon()")]
        self.assertIn("h.count > top.count", coverage)
        self.assertIn("forecastHorizon: null", self.STORE)
        self.assertIn("No ${h}h forecast", self.DATA)   # said outright, never borrowed from another horizon

    def test_the_flag_column_carries_only_the_observation(self):
        fmt = MODULES["format.js"]
        for word in ("mark: '▲', text: 'HIGH'", "mark: '▼', text: 'LOW'", "text: 'Within range'",
                     "text: 'Range not supplied'", "text: 'Range unavailable for this unit'"):
            self.assertIn(word, fmt, word)
        flag_cell = self.DATA[self.DATA.index("function flagCell("):self.DATA.index("function previousCell(")]
        for forbidden in ("projected", "forecast", "atHorizon"):
            self.assertNotIn(forbidden, flag_cell, forbidden)

    def test_the_page_uses_no_alerting_or_diagnostic_word(self):
        body = self._code(self.DATA).lower().replace("no clinical alert", "")
        for word in ("patient critical", "high risk", "clinical alert", "diagnos", "deteriorat", "alarm", "urgent"):
            self.assertNotIn(word, body, word)

    def test_no_threshold_is_written_into_the_page(self):
        interval = r"\d+(\.\d+)?\s*[–-]\s*\d+(\.\d+)?\s*(mg/dL|mmol/L|mEq/L|mL|/min|%|Cel|g/dL|kg)"
        self.assertNotRegex(self._code(self.DATA), interval)
        selectors = self.STORE[self.STORE.index("Indicators: one view-model"):self.STORE.index("/** Workspace split")]
        self.assertNotRegex(self._code(selectors), interval)
        for forbidden in ("CUTPOINTS", "cutpoint", "./ranking", "wardRowFor"):
            self.assertNotIn(forbidden, self._code(self.DATA), forbidden)
            self.assertNotIn(forbidden, self._code(selectors), forbidden)

    def test_one_indicator_detail_is_open_at_a_time_and_it_carries_the_chart(self):
        self.assertIn("S.dataExpanded = S.dataExpanded === v ? null : v", MODULES["main.js"])
        self.assertIn("dataExpanded: null", self.STORE)
        self.assertIn("${open ? detailPanel(row) : ''}", self.DATA)
        detail = self.DATA[self.DATA.index("function detailPanel("):self.DATA.index("/* ── one row")]
        for piece in ("trajectoryChart(", "range: row.range", "detail-cells", "Data quality",
                      "detail-prov", "Previous result", "chartLegend("):
            self.assertIn(piece, detail, piece)
        self.assertIn("dataShowUnobserved: false", self.STORE)   # the never-observed stay folded

    def test_the_ward_order_and_the_indicator_list_never_share_a_rule(self):
        ward = MODULES["ward.js"]
        self.assertNotIn("./ward.js", self.DATA)
        self.assertNotIn("./signals.js", ward)
        for shared in ("observationAttention", "forecastAttention", "indicatorView", "organAttentionState", "FLAG"):
            self.assertNotIn(shared, ward, shared)


class AttentionColourContractTests(unittest.TestCase):
    """Two claims, two colours, and neither one carried by colour alone."""

    TOKENS = ("--observed-abnormal", "--observed-abnormal-bg", "--projected-attention", "--projected-attention-bg")

    @staticmethod
    def _hex(name):
        return re.search(rf"{name}:\s*(#[0-9A-Fa-f]{{6}})", STREAM_CSS).group(1)

    @staticmethod
    def _luminance(value):
        def channel(c):
            c /= 255
            return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
        r, g, b = (int(value[i:i + 2], 16) for i in (1, 3, 5))
        return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)

    def _contrast(self, fg, bg):
        a, b = self._luminance(fg), self._luminance(bg)
        hi, lo = max(a, b), min(a, b)
        return (hi + 0.05) / (lo + 0.05)

    def test_the_two_attention_tokens_exist_and_are_different_colours(self):
        for token in self.TOKENS:
            self.assertRegex(STREAM_CSS, rf"{token}:\s*#[0-9A-Fa-f]{{6}}", token)
        self.assertNotEqual(self._hex("--observed-abnormal"), self._hex("--projected-attention"))

    def test_both_pass_wcag_aa_on_their_own_tint_and_on_the_page(self):
        page = "#FCFAF5"
        for fg, bg in (("--observed-abnormal", "--observed-abnormal-bg"),
                       ("--projected-attention", "--projected-attention-bg")):
            on_tint = self._contrast(self._hex(fg), self._hex(bg))
            on_page = self._contrast(self._hex(fg), page)
            self.assertGreaterEqual(round(on_tint, 2), 4.5, f"{fg} on {bg} is {on_tint:.2f}:1")
            self.assertGreaterEqual(round(on_page, 2), 4.5, f"{fg} on the page is {on_page:.2f}:1")

    def test_a_current_abnormality_is_deep_red_bold_marked_and_ruled(self):
        rule = re.search(r"\.ind-row\.abnormal td:first-child \{([^}]+)\}", STREAM_CSS).group(1)
        self.assertIn("inset 4px 0 0 var(--observed-abnormal)", rule)
        self.assertRegex(STREAM_CSS, r"\.ind-row\.abnormal \{ background: var\(--observed-abnormal-bg\)")
        self.assertRegex(STREAM_CSS, r"\.ind-row\.abnormal \.ind \.t \{[^}]*color: var\(--observed-abnormal\)")
        self.assertRegex(STREAM_CSS, r"\.flag-above, \.flag-below, \.flag-outside \{[^}]*color: var\(--observed-abnormal\)")

    def test_a_projection_never_borrows_the_current_abnormal_colour(self):
        projected = re.search(r"\.ind-row\.projected \{([^}]+)\}", STREAM_CSS).group(1)
        self.assertIn("--projected-attention-bg", projected)
        self.assertNotIn("--observed-abnormal", projected)
        proj_text = re.search(r"^\.proj \{([^}]+)\}", STREAM_CSS, flags=re.M).group(1)
        self.assertIn("var(--projected-attention)", proj_text)
        self.assertNotIn("--observed-abnormal", proj_text)
        # in the chart too: amber, dashed and hollow, never the measured-result red
        chart = MODULES["chart.js"]
        self.assertIn("danger: '#8C1D18', amber: '#8A5A0B'", chart)
        self.assertIn("const ink = f.projected ? COLORS.amber : COLORS.blue;", chart)

    def test_every_state_carries_a_mark_and_a_word_as_well_as_a_colour(self):
        fmt = MODULES["format.js"]
        marks = dict(re.findall(r"(\w+): \{ mark: '(.)'", fmt))
        for key in ("above", "below", "within", "unsupplied", "unit", "none"):
            self.assertIn(key, marks, key)
        self.assertEqual(len(set(marks.values())), len(marks), "two states share a mark")
        self.assertIn("PROJECTED = {", fmt)
        self.assertIn("text: 'Projected high'", fmt)
        self.assertIn("text: 'Projected low'", fmt)


class ReportContractTests(unittest.TestCase):
    """The printed report: a structured research report, frozen when it is opened."""

    REPORT = MODULES["report.js"]

    def test_only_the_report_preview_prints(self):
        main = MODULES["main.js"]
        self.assertIn("else if (act === 'report-print') window.print();", main)
        self.assertEqual(main.count("window.print()"), 1)
        self.assertNotIn("window.print", MODULES["signals.js"])
        self.assertIn('data-act="report-open"', MODULES["signals.js"])

    def test_the_title_says_what_this_is_and_never_claims_a_laboratory(self):
        fmt = MODULES["format.js"]
        self.assertIn("title: 'Oslian Monitoring Research Report'", fmt)
        self.assertIn("Reporting system: Oslian research demonstration", fmt)
        body = self.REPORT + fmt
        for forbidden in ("Laboratory Report", "laboratory report</", "performed by", "Performing laboratory"):
            self.assertNotIn(forbidden, body, forbidden)
        self.assertIn("not a clinical laboratory report", fmt)

    def test_the_header_carries_what_a_result_report_has_to_carry(self):
        for field in ("<dt>Subject</dt>", "<dt>Source</dt>", "<dt>Snapshot</dt>", "<dt>Clinical cutoff</dt>",
                      "<dt>Issued</dt>", "<dt>Report snapshot ID</dt>", "<dt>Scope</dt>",
                      "<dt>Report status</dt>", "<dt>Reporting system</dt>"):
            self.assertIn(field, self.REPORT, field)
        self.assertIn("REPORT.boundary", self.REPORT)
        self.assertIn("identifier", self.REPORT)

    def test_the_snapshot_is_a_deep_copy_and_the_builder_can_refresh_it(self):
        self.assertIn("const plain = (value) => JSON.parse(JSON.stringify(", self.REPORT)
        self.assertIn("export function freezeReport()", self.REPORT)
        self.assertIn("frozen: freezeReport()", self.REPORT)
        main = MODULES["main.js"]
        self.assertIn("else if (act === 'report-open') { S.report = openReport(null); scheduleRender(); }", main)
        self.assertIn("else if (act === 'report-refresh') { S.report = openReport(S.report); scheduleRender(); }", main)
        self.assertIn("frozen: 'This report is frozen at the snapshot below.", MODULES["format.js"])

    def test_the_builder_offers_the_three_scopes_and_the_optional_sections(self):
        for piece in ("scopes: { full:", "'Needs attention only'", "'Selected indicators'",
                      'data-act="report-scope"', 'data-act="report-pick"', 'data-act="report-toggle"',
                      'data-act="report-horizon"', 'data-section="charts"', 'data-section="anatomy"',
                      'data-section="provenance"'):
            self.assertIn(piece, self.REPORT + MODULES["format.js"], piece)
        # the defaults: charts and anatomy on, provenance off
        self.assertIn("charts: previous ? previous.charts : true", self.REPORT)
        self.assertIn("anatomy: previous ? previous.anatomy : true", self.REPORT)
        self.assertIn("provenance: previous ? previous.provenance : false", self.REPORT)

    def test_the_results_sections_never_repeat_an_indicator(self):
        self.assertIn("const attention = rows.filter((r) => r.attention.kind === 'outside');", self.REPORT)
        self.assertIn("const other = rows.filter((r) => !attention.includes(r));", self.REPORT)

    def test_the_forecast_section_separates_a_model_from_a_repeated_observation(self):
        forecast = self.REPORT[self.REPORT.index("function forecastRow("):self.REPORT.index("/* ── trends")]
        for word in ("'Carried forward — model abstained'", "'Model forecast'", "'Carried forward'",
                     "'Missing'", "'Degraded'"):
            self.assertIn(word, forecast, word)
        self.assertIn("forecastNote: 'Research-only point forecasts.", MODULES["format.js"])
        self.assertIn("A carried-forward value is not a model forecast.", self.REPORT)

    def test_trend_charts_are_drawn_only_where_they_earn_the_paper(self):
        self.assertIn("r.attention.kind === 'outside' || r.forecasts.some((f) => f.projected)", self.REPORT)
        self.assertIn("report.scope === 'selected' && report.indicators.includes(r.variable)", self.REPORT)

    def test_the_anatomy_overview_is_the_same_figure_on_the_frozen_states(self):
        self.assertIn("buildAnatomy(host, S.anatomy.organs, S.organLayout", self.REPORT)
        self.assertIn("attention: S.report.frozen.organs", self.REPORT)
        self.assertIn("No current observed result outside supplied range", self.REPORT)
        self.assertIn("It is not a diagnosis and states nothing about an organ's condition.", self.REPORT)

    def test_the_footer_states_the_limits_and_the_identifiers(self):
        for piece in ("limitsHeading", "report-id", "causal_claim_allowed: false",
                      "Reference intervals are supplied by the observations themselves"):
            self.assertIn(piece, self.REPORT + MODULES["format.js"], piece)


class AnatomyAttentionContractTests(unittest.TestCase):
    """One rule decides what an organ shows, and selection never hides it."""

    def test_one_pure_function_aggregates_an_organ(self):
        store = MODULES["store.js"]
        self.assertIn("export function organAttentionState(organ, rows)", store)
        self.assertIn("export function organAttentionMap(rows)", store)
        self.assertIn("export function organsByAttention(rows)", store)
        # nobody else aggregates an organ: the other modules read the result
        for name in ("anatomy.js", "signals.js", "ward.js", "report.js", "render.js"):
            self.assertNotIn("function organAttentionState", MODULES[name], name)
            self.assertNotIn("abnormalIds:", MODULES[name], name)
        self.assertIn("abnormalIds: abnormal.map", store)
        self.assertIn("organAttentionMap()", MODULES["render.js"])
        self.assertIn("attention: S.report.frozen.organs", MODULES["report.js"])

    def test_the_figure_draws_three_different_states(self):
        anatomy = MODULES["anatomy.js"]
        # the four responses are named once and built per figure under its own ids
        self.assertIn("const RESPONSES = {", anatomy)
        for name in ("teal:", "oxide:", "amber:", "danger:"):
            self.assertIn(name, anatomy, name)
        self.assertIn("fx(`${prefix}-fx-${name}`", anatomy)
        self.assertIn("cls.push('state-abnormal')", anatomy)
        self.assertIn("cls.push('state-projected')", anatomy)
        self.assertIn("cls.push('state-active')", anatomy)
        # each state names its own custom property; the id behind it belongs to the
        # figure that defined it, so the stylesheet can never point at another figure
        for selector, prop in (("state-abnormal", "--fx-danger"), ("state-projected", "--fx-amber"),
                               ("state-active", "--fx-teal")):
            self.assertRegex(STREAM_CSS, rf"\.organ-fx\.{selector}[^{{]*\{{[^}}]*var\({prop}\)", selector)

    def test_a_selected_organ_keeps_the_state_it_has(self):
        # the selection ring is drawn on the hit area; the abnormal wash is not replaced by it
        declarations = re.sub(r"/\*.*?\*/", "", STREAM_CSS, flags=re.S)
        selected = declarations.index(".anatomy-figure .organ-fx.selected")
        abnormal = declarations.index(".anatomy-figure .organ-fx.state-abnormal")
        self.assertGreater(abnormal, selected, "the abnormal wash must win over the selection wash")
        self.assertIn(".anatomy-figure .organ-hit.state-abnormal.selected", declarations)

    def test_the_rail_puts_an_abnormal_organ_first_and_says_why(self):
        render = MODULES["render.js"]
        self.assertIn("organsByAttention(rows)", render)
        self.assertIn("Needs attention", render)
        self.assertIn("observed result${a.abnormal === 1 ? '' : 's'} outside range", render)
        self.assertIn("projected outside", render)
        self.assertIn('data-attention="${esc(a.state)}"', render)

    def test_an_organ_workspace_and_its_tiles_use_the_indicator_words(self):
        render = MODULES["render.js"]
        self.assertIn("variablesByAttention(spec.primary, rows)", render)
        self.assertIn("variablesByAttention(spec.secondary, rows)", render)
        self.assertIn("variablesByAttention(spec.related, rows)", render)
        self.assertIn("tile-flag flag-${key}", render)
        self.assertRegex(STREAM_CSS, r"\.tile\.abnormal \{[^}]*var\(--observed-abnormal\)")

    def test_the_ward_figure_and_ranking_are_untouched(self):
        ward = MODULES["ward.js"]
        for forbidden in ("observed-abnormal", "organAttention", "state-abnormal", "FLAG", "indicatorRows"):
            self.assertNotIn(forbidden, ward, forbidden)
        self.assertIn("data-act=\"open-ward-stay\"", ward)
        self.assertIn("last measured creatinine", ward)


class ReportColourSemanticsTests(unittest.TestCase):
    """Deep red marks one thing: a measured value outside its own supplied interval.

    A previous value and a change are facts about direction, and a change of
    +1.00 back towards the interval looks exactly like one away from it. Colouring
    either would read as a verdict on whether the patient is improving, which this
    page does not make - so both stay neutral ink, and the rule names the cell it
    colours instead of counting columns.
    """

    REPORT = MODULES["report.js"]

    @staticmethod
    def _rule(selector):
        match = re.search(re.escape(selector) + r"\s*\{([^}]*)\}", STREAM_CSS)
        return match.group(1) if match else None

    def test_every_result_cell_is_named(self):
        for cls in ("indicator-name", "current-result", "result-flag", "reference-interval",
                    "previous-result", "result-change", "observed-at", "forecast-result"):
            self.assertIn(cls, self.REPORT, cls)

    def test_only_the_current_result_and_its_flag_are_deep_red(self):
        current = self._rule(".report-table tr.abnormal .current-result")
        self.assertIsNotNone(current)
        self.assertIn("var(--observed-abnormal)", current)
        name = self._rule(".report-table tr.abnormal .indicator-name")
        self.assertIn("var(--observed-abnormal)", name)
        self.assertRegex(STREAM_CSS, r"\.flag-above, \.flag-below, \.flag-outside \{[^}]*var\(--observed-abnormal\)")

    def test_previous_and_change_stay_neutral(self):
        neutral = self._rule(".report-table .previous-result, .report-table .result-change")
        self.assertIsNotNone(neutral, "previous and change must be pinned to neutral ink")
        self.assertIn("var(--ink)", neutral)
        self.assertNotIn("observed-abnormal", neutral)
        # and nothing else may colour them
        for line in STREAM_CSS.splitlines():
            if ("previous-result" in line or "result-change" in line) and "observed-abnormal" in line:
                self.fail(f"a rule colours a previous value or a change: {line.strip()[:110]}")

    def test_the_positional_rule_that_reddened_whole_rows_is_gone(self):
        # `.report-table tr.abnormal .v` caught the result, the previous value and
        # the change alike; naming the cells replaced it. It has to be gone from
        # every block, the print one included - where it also carried !important,
        # which is how it survived a first pass at this.
        declarations = re.sub(r"/\*.*?\*/", "", STREAM_CSS, flags=re.S)
        # a positional selector may still carry typography (the monospace figures);
        # what it may not carry is a colour, which is the thing that reads as a verdict
        for selector, body in re.findall(r"([^{}]+)\{([^}]*)\}", declarations):
            if "color:" not in body:
                continue
            for one in selector.split(","):
                one = one.strip()
                if "report-table" in one and one.endswith(".v"):
                    self.fail(f"a report cell is coloured by position: {one} {{{body.strip()[:60]}}}")
        self.assertNotIn(".report-table .v.model", STREAM_CSS)
        self.assertNotRegex(STREAM_CSS, r"\.report-table[^{\n]*nth-child")

    def test_previous_and_change_stay_neutral_on_paper_too(self):
        printed = STREAM_CSS[STREAM_CSS.index("@media print"):]
        reddened = [l for l in printed.splitlines() if "7A1512" in l]
        self.assertTrue(reddened, "the printed report still needs its one red")
        for line in reddened:
            self.assertNotIn(".v,", line, f"a positional cell is reddened on paper: {line.strip()[:110]}")
            self.assertNotIn(".previous-result", line)
            self.assertNotIn(".result-change", line)
        self.assertRegex(printed, r"\.report-table \.previous-result, \.report-table \.result-change \{[^}]*#171A1C")

    def test_a_model_forecast_keeps_blue_and_a_projection_keeps_amber(self):
        model = self._rule(".report-table .forecast-result.model")
        projected = self._rule(".report-table .forecast-result.projected")
        self.assertIn("var(--blue)", model)
        self.assertIn("var(--projected-attention)", projected)
        self.assertNotIn("observed-abnormal", projected)
        # a projection is never carried into the current-result colour
        self.assertNotIn("forecast", self._rule(".report-table tr.abnormal .current-result"))

    def test_a_normal_current_result_is_never_reddened_by_its_forecast(self):
        # the row is only marked abnormal by the observation's own kind
        self.assertIn("const abnormal = row.attention.kind === 'outside';", self.REPORT)
        self.assertIn("""<tr${abnormal ? ' class="abnormal"' : ''}""", self.REPORT)

    def test_black_and_white_keeps_every_distinction(self):
        # each state prints its mark and its word; the report reads them from the
        # same table the screen uses, so a greyscale copy loses only the colour
        self.assertIn("${mark(flag.mark)}${flag.text}", self.REPORT)
        self.assertIn("${mark(PROJECTED[cell.projectedSide].mark)}${PROJECTED[cell.projectedSide].text}", self.REPORT)
        for state in ("'Model forecast'", "'Carried forward'", "'Carried forward — model abstained'",
                      "'Missing'", "'Degraded'"):
            self.assertIn(state, self.REPORT, state)


class MobileDensityTests(unittest.TestCase):
    """A phone opens on what is worth reading first; nothing flagged is ever folded away."""

    DATA = MODULES["signals.js"]

    def test_the_ordinary_results_fold_on_a_phone_and_not_on_a_desktop(self):
        self.assertIn("const narrow = typeof window !== 'undefined' && window.innerWidth <= 767;", self.DATA)
        self.assertIn("const otherOpen = S.dataShowOther === null ? !narrow : S.dataShowOther;", self.DATA)
        self.assertIn("dataShowOther: null", MODULES["store.js"])
        self.assertIn("showOther: 'Show other observed indicators'", MODULES["format.js"])
        self.assertIn('data-act="data-other"', self.DATA)
        phone = STREAM_CSS[STREAM_CSS.index("@media (max-width: 767px)"):]
        self.assertIn(".other-toggle { display: flex; }", phone)
        self.assertIn('.ind-section[data-open="false"] .ind-scroll { display: none; }', phone)
        self.assertRegex(STREAM_CSS, r"\.other-toggle \{[^}]*display: none")   # hidden on a desktop

    def test_the_readers_choice_survives_a_stream_event(self):
        # the answer lives in the store, so a re-render cannot reset it
        main = MODULES["main.js"]
        self.assertIn("else if (act === 'data-other') { S.dataShowOther = !(t.getAttribute('aria-expanded') === 'true'); scheduleRender(); }", main)
        reset = MODULES["store.js"][MODULES["store.js"].index("export function resetSessionState"):
                                    MODULES["store.js"].index("export const sourceInfo")]
        self.assertNotIn("dataShowOther", reset)

    def test_needs_attention_is_never_folded(self):
        attention = self.DATA[self.DATA.index('class="ind-section attention-section"'):
                              self.DATA.index('<section class="ind-section" data-open=')]
        self.assertNotIn("data-open", attention)
        self.assertNotIn("other-toggle", attention)


class PrintContractTests(unittest.TestCase):
    """What reaches paper: the report, in landscape, whole rows, repeated heads."""

    PRINT = STREAM_CSS[STREAM_CSS.index("@media print"):]

    def test_the_page_is_landscape_and_the_chrome_is_gone(self):
        self.assertIn("size: A4 landscape", self.PRINT)
        for hidden in (".app-bar", ".stream-strip", ".view-bar", ".rail", ".report-builder",
                       ".data-controls", ".att-summary", ".horizon-bar", ".drawer", ".dev-tools"):
            self.assertIn(hidden, self.PRINT, hidden)

    def test_a_result_never_splits_across_pages_and_the_head_repeats(self):
        self.assertIn("display: table-header-group", self.PRINT)
        self.assertRegex(self.PRINT, r"\.report-table tr \{[^}]*break-inside: avoid")
        self.assertRegex(self.PRINT, r"\.report-chart, \.organ-note, \.report-foot \{[^}]*break-inside: avoid")
        # the body overview is one unit down to the sentence under it
        self.assertRegex(self.PRINT, r"\.anatomy-section \{[^}]*break-inside: avoid")
        self.assertIn('class="report-section anatomy-section"', MODULES["report.js"])

    def test_paper_is_filled_before_it_is_broken(self):
        # no section claims a page of its own any more: a section continues on the
        # page below whenever the room is there, and only a unit of meaning is
        # kept whole. A heading always travels with the block it introduces.
        self.assertNotIn("break-before: page", STREAM_CSS)
        self.assertNotIn("page-break-before: always", STREAM_CSS)
        self.assertIn(".page-break { break-before: auto; page-break-before: auto; }", self.PRINT)
        self.assertNotIn('class="report-section page-break"', MODULES["report.js"])
        self.assertNotIn('class="report-foot page-break"', MODULES["report.js"])
        self.assertRegex(self.PRINT, r"\.report-section h2, \.report-foot h2 \{[^}]*break-after: avoid")
        self.assertRegex(self.PRINT, r"\.report-table thead \{[^}]*break-after: avoid")

    def test_colour_only_repeats_what_the_marks_and_words_already_say(self):
        # the printed abnormal row keeps its rule and its flag text; the marks come from
        # the same FLAG table the screen uses, so a black-and-white copy loses nothing
        self.assertIn("print-color-adjust: exact", self.PRINT)
        self.assertIn("flag flag-${esc(row.flag)}", MODULES["report.js"])
        self.assertIn("${mark(flag.mark)}${flag.text}", MODULES["report.js"])

    def test_the_phone_layout_becomes_cards_without_sideways_scroll(self):
        phone = STREAM_CSS[STREAM_CSS.index("@media (max-width: 767px)"):]
        self.assertIn(".ind-table, .ind-table tbody, .ind-table tr, .ind-table td { display: block", phone)
        self.assertIn(".ind-scroll { overflow: visible; }", phone)
        self.assertIn('.ind-row .range-cell::before { content: "Reference range · "', phone)


if __name__ == "__main__":
    unittest.main()

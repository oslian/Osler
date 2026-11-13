"""The indicator view-model, executed rather than grepped.

``store.js`` touches no DOM and ``chart.js`` is pure, so the selectors the
indicator list, the organ figure and the printed report all read run unchanged
under Node. A probe script sets up one patient with every case the runtime can
produce and hands the results back here. The assertions are the invariants each
screen relies on:

- a flag is the observation's own statement: a supplied ``referenceRange``
  compared in the unit it was written in (inclusive limits), or a coded
  ``interpretation``; a fixture-derived reading never outranks the range it came
  from and is never critical; a range in another unit fails closed;
- current, previous and forecast are three separate comparisons, and the
  forecast column compares at one horizon the reader chooses;
- a current result outside its range sorts before a forecast projected outside
  one, and a carried-forward value - a head that abstained included - is never a
  model forecast and is never projected;
- one organ-attention rule feeds the rail, the figure and the report;
- a report is frozen when it is opened: later stream events do not reach it.

Node is not part of the demo's runtime; without it these tests skip with a reason.
"""

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from oslian_demo.config import JS_DIR

NODE = shutil.which("node")
STREAM_JS = JS_DIR / "stream"

PROBE = r"""
import { S, TIER, activeHorizon, applyEvent, horizonCoverage, indicatorRows, indicatorView,
  observationAttention, forecastAttention, organAttentionState, organAttentionMap, organsByAttention,
  trajectoryFor, variablesByAttention } from './store.js';
import { trajectoryChart } from './chart.js';
import { isAbstained, flagOf, forecastLabel } from './format.js';
import { freezeReport, openReport } from './report.js';
const H = 3600000, T0 = Date.UTC(2026, 2, 14, 6);
const iso = (ms) => new Date(ms).toISOString();
const DERIVED = 'derived_from_supplied_reference_range';
const V3 = 'http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation';
S.anatomy = {
  variable_units: { bun: 'mg/dL', bicarbonate: 'mEq/L', sodium: 'mEq/L', chloride: 'mEq/L', creatinine: 'mg/dL',
    glucose: 'mg/dL', temperature: 'degC', potassium: 'mEq/L', lactate: 'mmol/L', o2sat: '%', troponin_i: 'ng/mL',
    hemoglobin: 'g/dL' },
  organs: [
    { id: 'kidneys', label: 'Kidneys', variables: ['bun', 'bicarbonate', 'sodium', 'chloride', 'creatinine', 'potassium'] },
    { id: 'pancreas', label: 'Pancreas', variables: ['glucose'] },
    { id: 'lungs', label: 'Lungs', variables: ['o2sat'] },
    { id: 'heart', label: 'Heart', variables: ['troponin_i'] },
    { id: 'blood_vessels', label: 'Blood vessels', variables: ['lactate'] },
    { id: 'spleen', label: 'Spleen', variables: ['hemoglobin'] },
    { id: 'thyroid', label: 'Thyroid', variables: ['temperature'] },
  ],
};
S.capabilities = { cells: [
  { cell: 'bun@24h', target: 'bun', horizon_hours: 24 },
  { cell: 'sodium@12h', target: 'sodium', horizon_hours: 12 },
  { cell: 'sodium@24h', target: 'sodium', horizon_hours: 24 },
  { cell: 'chloride@12h', target: 'chloride', horizon_hours: 12 },
  { cell: 'creatinine@6h', target: 'creatinine', horizon_hours: 6 },
  { cell: 'creatinine@12h', target: 'creatinine', horizon_hours: 12 },
  { cell: 'glucose@12h', target: 'glucose', horizon_hours: 12 },
  { cell: 'potassium@12h', target: 'potassium', horizon_hours: 12 },
  { cell: 'lactate@6h', target: 'lactate', horizon_hours: 6 },
  { cell: 'hypoxemia_event@1h', target: 'hypoxemia_event', horizon_hours: 1 },
] };
const at = (hours) => iso(T0 + hours * H);
const state = (variable, value, unit, hours, extra) => ({ variable, value, unit, effective_time: at(hours),
  received_at: at(hours), resource_id: `obs-${variable}`, version_id: '1', count: 3, interpretation: [],
  reference_range: [], input_value: value, input_unit: unit, derivation: 'identity', ...(extra || {}) });
const range = (low, high, unit, extra) => ({ low, low_unit: unit, high, high_unit: unit, text: null,
  applies_to: false, age: false, ...(extra || {}) });
const derived = (code) => ({ system: V3, code, display: code, provenance: DERIVED });
const sourced = (code) => ({ system: V3, code, display: code, provenance: null });
const pt = (variable, hours, value, unit) => ({ effective_time: at(hours), value, unit,
  resource_id: `${variable}-${hours}`, version_id: '1' });
S.variables = {
  // above its supplied range; the reading was derived by the fixture from that range
  bun: state('bun', 26, 'mg/dL', 30, { reference_range: [range(6, 20, 'mg/dL')], interpretation: [derived('H')] }),
  // below its range; stated in mmol/L, canonical mEq/L: compared in the stated unit
  bicarbonate: state('bicarbonate', 21, 'mEq/L', 30, { input_value: 21, input_unit: 'mmol/L',
    reference_range: [range(23, 29, 'mmol/L')], interpretation: [derived('L')] }),
  // exactly on the upper limit: inside
  chloride: state('chloride', 107, 'mEq/L', 12, { input_value: 107, input_unit: 'mmol/L',
    reference_range: [range(98, 107, 'mmol/L')], interpretation: [derived('N')] }),
  // within, and a 12h model point that leaves the same range
  sodium: state('sodium', 139, 'mEq/L', 30, { input_value: 139, input_unit: 'mmol/L',
    reference_range: [range(135, 145, 'mmol/L')], interpretation: [derived('N')] }),
  creatinine: state('creatinine', 1.2, 'mg/dL', 30),
  glucose: state('glucose', 171, 'mg/dL', 30),
  // a range in a unit that matches neither value: fail closed
  potassium: state('potassium', 4.1, 'mEq/L', 30, { input_value: 4.1, input_unit: 'mmol/L',
    reference_range: [range(3.7, 5.2, 'mg/dL')], interpretation: [derived('N')] }),
  troponin_i: state('troponin_i', 2.4, 'ng/mL', 29, { interpretation: [sourced('HH')] }),
  lactate: state('lactate', 1.1, 'mmol/L', 30, { reference_range: [range(0.5, 2.2, 'mmol/L')], interpretation: [derived('N')] }),
  hemoglobin: state('hemoglobin', 18.9, 'g/dL', 28, { interpretation: [sourced('H')] }),
  o2sat: state('o2sat', 96, '%', 30),
};
S.series = {
  bun: [pt('bun', 0, 32, 'mg/dL'), pt('bun', 12, 29, 'mg/dL'), pt('bun', 30, 26, 'mg/dL')],
  bicarbonate: [pt('bicarbonate', 12, 22, 'mEq/L'), pt('bicarbonate', 30, 21, 'mEq/L')],
  sodium: [pt('sodium', 30, 139, 'mEq/L')],                       // one point only: no trend to draw
  chloride: [pt('chloride', 0, 104, 'mEq/L'), pt('chloride', 12, 107, 'mEq/L')],
};
S.clock.latest_clinical_event_time = at(30);
const cell = (name, target, h, status, point, observed, source, reason, unit) => ({ cell: name, target, unit,
  horizon_hours: h, observed_value: observed, measurement_age_hours: 0, observations: 3, forecast_point: point,
  source_model: source, status, fallback_reason: reason, interval_status: 'none', lower: null, upper: null,
  causal_claim_allowed: false });
S.live = { run: { run_id: 'run-1', input_cutoff: at(30), epoch: 1, runtime_mode: 'release', input_resource_count: 9,
                  summary: { model_cells: 5, persistence_cells: 2, missing_cells: 1, degraded_cells: 0 } },
  cells: [
    cell('bun@24h', 'bun', 24, 'model', 24.5, 26, 'unified_shared_jepa_target_head', null, 'mg/dL'),
    cell('sodium@12h', 'sodium', 12, 'model', 146.2, 139, 'unified_shared_jepa_target_head', null, 'mEq/L'),
    cell('sodium@24h', 'sodium', 24, 'model', 141.0, 139, 'unified_shared_jepa_target_head', null, 'mEq/L'),
    cell('chloride@12h', 'chloride', 12, 'persistence', 107, 107, 'patient_persistence_anchor', 'continuous_head_abstained', 'mEq/L'),
    cell('creatinine@6h', 'creatinine', 6, 'model', 1.25, 1.2, 'frozen_whole_body_jepa', null, 'mg/dL'),
    cell('creatinine@12h', 'creatinine', 12, 'persistence', 1.2, 1.2, 'patient_persistence_anchor', 'shared_jepa_head_not_external_validated', 'mg/dL'),
    cell('glucose@12h', 'glucose', 12, 'model', 171, 171, 'frozen_whole_body_jepa', null, 'mg/dL'),
    cell('potassium@12h', 'potassium', 12, 'model', 9.9, 4.1, 'unified_shared_jepa_target_head', null, 'mEq/L'),
    cell('lactate@6h', 'lactate', 6, 'degraded', null, 1.1, 'none', 'wholebody_runtime_not_attached', 'mmol/L'),
  ] };
S.freshVariables = { o2sat: Date.now() };
const out = {};
const slim = (r) => ({ variable: r.variable, tier: r.tier, flag: flagOf(r), flagged: r.flagged, fresh: r.fresh,
  attention: { kind: r.attention.kind, side: r.attention.side, basis: r.attention.basis,
    derived: r.attention.derived, critical: r.attention.critical, unit: r.attention.unit || null, note: r.attention.note || null },
  range: r.range ? { low: r.range.low, high: r.range.high, unit: r.range.unit } : null,
  previous: r.previous, horizon: r.horizon, hasHorizonCell: r.hasHorizonCell,
  atHorizon: r.atHorizon ? { h: r.atHorizon.horizon_hours, status: r.atHorizon.status, point: r.atHorizon.forecast_point,
    abstained: r.atHorizon.abstained, projected: r.atHorizon.projected, side: r.atHorizon.projectedSide,
    label: forecastLabel(r.atHorizon) } : null,
  points: r.series.length, projectedSide: r.projectedSide });
out.coverage = horizonCoverage();
out.defaultHorizon = activeHorizon();
const view = indicatorView((v) => v);
out.view = { attention: view.attention.map((r) => r.variable), other: view.other.map((r) => r.variable),
  unobserved: view.unobserved.map((r) => r.variable), outside: view.outside, projected: view.projected,
  fresh: view.fresh, unsupplied: view.unsupplied, observed: view.observed, total: view.total,
  modelCells: view.modelCells, carriedCells: view.carriedCells, abstainedCells: view.abstainedCells,
  horizon: view.horizon, horizons: view.horizons };
out.rows = Object.fromEntries(indicatorRows().map((r) => [r.variable, slim(r)]));
// switching the compared horizon changes only the forecast column
S.forecastHorizon = 24;
out.at24 = Object.fromEntries(indicatorRows().map((r) => [r.variable, { hasCell: r.hasHorizonCell,
  atHorizon: r.atHorizon ? { h: r.atHorizon.horizon_hours, point: r.atHorizon.forecast_point } : null,
  tier: r.tier }]));
out.view24 = { attention: indicatorView((v) => v).attention.map((r) => r.variable) };
S.forecastHorizon = 6;
out.at6 = Object.fromEntries(indicatorRows().map((r) => [r.variable, { hasCell: r.hasHorizonCell,
  atHorizon: r.atHorizon ? { h: r.atHorizon.horizon_hours, status: r.atHorizon.status } : null }]));
S.forecastHorizon = null;
// organs: one rule, and the rail order it produces
const rows = indicatorRows();
out.organs = organAttentionMap(rows);
out.organOrder = organsByAttention(rows).map((o) => `${o.organ.id}:${o.attention.state}`);
out.kidneys = organAttentionState(S.anatomy.organs[0], rows);
out.workspaceOrder = variablesByAttention(['creatinine', 'potassium', 'bun', 'sodium', 'chloride'], rows);
// charts: the range band, the amber projection, and no line through one point
const draw = (variable, band) => trajectoryChart({ traj: trajectoryFor(variable, null), unit: 'u', label: variable,
  axis: 'calendar', onset_ms: 0, window: 'full', width: 900, height: 300, idSuffix: variable, range: band });
out.charts = {
  bun: draw('bun', { low: 6, high: 20, unit: 'mg/dL' }).svg,
  sodium: draw('sodium', { low: 135, high: 145, unit: 'mmol/L' }).svg,
  chloride: draw('chloride', null).svg,
};

// the report: frozen at the moment it is opened
const report = openReport(null);
out.report = { scope: report.scope, horizon: report.horizon, charts: report.charts, anatomy: report.anatomy,
  provenance: report.provenance, id: report.id, rows: report.frozen.rows.map((r) => r.variable),
  bun: report.frozen.rows.find((r) => r.variable === 'bun').state.value,
  organs: Object.fromEntries(Object.entries(report.frozen.organs).map(([k, v]) => [k, v.state])),
  counts: report.frozen.counts, hasTraj: !!report.frozen.rows[0].traj };
// a later observation moves the live page and must not move the open report
applyEvent('observation.normalized', { resource_id: 'obs-bun-2', version_id: '1', received_at: at(36),
  effective_time: at(36), organ_ids: ['kidneys'],
  values: [{ variable: 'bun', value: 12, unit: 'mg/dL', effective_time: at(36), derivation: 'identity',
             input_value: 12, input_unit: 'mg/dL' }],
  interpretation: [derived('N')], reference_range: [range(6, 20, 'mg/dL')] }, 900);
out.afterEvent = { live: indicatorRows().find((r) => r.variable === 'bun').state.value,
  liveFlag: flagOf(indicatorRows().find((r) => r.variable === 'bun')),
  reportStill: report.frozen.rows.find((r) => r.variable === 'bun').state.value,
  reportFlagStill: report.frozen.rows.find((r) => r.variable === 'bun').flag,
  reportOrgansStill: report.frozen.organs.kidneys.state };
const refreshed = openReport(report);
out.refreshed = { bun: refreshed.frozen.rows.find((r) => r.variable === 'bun').state.value,
  scope: refreshed.scope, charts: refreshed.charts, horizon: refreshed.horizon };
// the anchor freezes the organs too
S.anchor = { run: { run_id: 'old', input_cutoff: at(12), epoch: 1, runtime_mode: 'release', input_resource_count: 4 },
  cells: [cell('bun@24h', 'bun', 24, 'model', 30.5, 32, 'unified_shared_jepa_target_head', null, 'mg/dL')],
  series: { bun: [pt('bun', 0, 32, 'mg/dL')] },
  variables: { bun: state('bun', 32, 'mg/dL', 12, { reference_range: [range(6, 20, 'mg/dL')], interpretation: [derived('H')] }),
               glucose: state('glucose', 90, 'mg/dL', 12) },
  organEvents: {} };
const anchorRows = indicatorRows();
out.anchor = { rows: anchorRows.filter((r) => r.state).map((r) => [r.variable, r.tier, r.attention.side, r.state.value, r.fresh]),
  organs: Object.fromEntries(Object.entries(organAttentionMap(anchorRows)).map(([k, v]) => [k, v.state])) };
S.anchor = null;
out.unitless = observationAttention(state('x', 5, 'u', 1, { reference_range: [{ low: 0, low_unit: null, high: 2, high_unit: null, applies_to: false, age: false }] }));
out.ambiguous = observationAttention(state('x', 1, 'u', 1, { reference_range: [range(0, 1, 'u', { applies_to: true }), range(0, 2, 'u', { applies_to: true })] }));
out.qualifiedAlone = observationAttention(state('x', 3, 'u', 1, { reference_range: [range(0, 2, 'u', { age: true })] }));
out.derivedNeverCritical = observationAttention(state('x', 99, 'u', 1, { reference_range: [range(0, 2, 'u')], interpretation: [derived('HH')] })).critical;
out.projectedPersistence = forecastAttention({ status: 'persistence', forecast_point: 99, unit: 'mg/dL', fallback_reason: 'continuous_head_abstained' }, S.variables.bun).projected;
out.projectedNull = forecastAttention({ status: 'model', forecast_point: null, unit: 'mg/dL' }, S.variables.bun).projected;
out.projectedDegraded = forecastAttention({ status: 'degraded', forecast_point: 99, unit: 'mg/dL' }, S.variables.bun).projected;
out.tiers = TIER;
console.log(JSON.stringify(out));
"""


@unittest.skipUnless(NODE, "node is not on PATH; the indicator view-model cannot be executed here")
class IndicatorViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="oslian-indicators-")
        for name in ("store.js", "chart.js", "format.js", "render.js", "signals.js", "anatomy.js", "report.js", "ward.js"):
            shutil.copy(STREAM_JS / name, Path(cls.tmp) / name)
        (Path(cls.tmp) / "package.json").write_text('{"type": "module"}', encoding="utf-8")
        (Path(cls.tmp) / "probe.mjs").write_text(PROBE, encoding="utf-8")
        run = subprocess.run([NODE, "probe.mjs"], cwd=cls.tmp, capture_output=True, text=True, encoding="utf-8", timeout=60)
        assert run.returncode == 0, run.stderr
        cls.out = json.loads(run.stdout)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    # ---- the flag is the observation's own statement ------------------------
    def test_high_and_low_are_read_from_the_supplied_range(self):
        bun, bic = self.out["rows"]["bun"], self.out["rows"]["bicarbonate"]
        self.assertEqual((bun["flag"], bun["attention"]["basis"]), ("above", "referenceRange"))
        self.assertEqual(bic["flag"], "below")
        self.assertEqual(bic["attention"]["unit"], "mmol/L")   # compared in the unit it was written in
        self.assertTrue(bun["attention"]["derived"])
        self.assertFalse(bun["attention"]["critical"])
        self.assertEqual(bun["range"], {"low": 6, "high": 20, "unit": "mg/dL"})

    def test_a_value_on_the_limit_is_inside(self):
        self.assertEqual(self.out["rows"]["chloride"]["flag"], "within")

    def test_no_range_says_so_and_is_never_flagged(self):
        for variable in ("creatinine", "glucose", "o2sat"):
            self.assertEqual(self.out["rows"][variable]["flag"], "unsupplied", variable)
            self.assertFalse(self.out["rows"][variable]["flagged"], variable)

    def test_a_range_in_an_unusable_unit_fails_closed(self):
        potassium = self.out["rows"]["potassium"]
        self.assertEqual(potassium["flag"], "unit")
        self.assertIn("mg/dL", potassium["attention"]["note"])
        self.assertFalse(potassium["flagged"])
        # and the model point far outside that range is never projected
        self.assertFalse(potassium["atHorizon"]["projected"])

    def test_source_readings_and_qualified_ranges(self):
        self.assertTrue(self.out["rows"]["troponin_i"]["attention"]["critical"])
        self.assertEqual(self.out["rows"]["hemoglobin"]["flag"], "above")
        self.assertEqual(self.out["rows"]["hemoglobin"]["attention"]["basis"], "interpretation")
        self.assertFalse(self.out["derivedNeverCritical"])
        self.assertEqual(self.out["ambiguous"]["kind"], "unsupplied")
        self.assertEqual(self.out["qualifiedAlone"]["kind"], "outside")
        self.assertEqual(self.out["unitless"]["kind"], "outside")

    # ---- current, previous, forecast ---------------------------------------
    def test_previous_result_is_the_previous_valid_observation_with_its_change(self):
        bun = self.out["rows"]["bun"]["previous"]
        self.assertEqual((bun["value"], bun["unit"]), (29, "mg/dL"))
        self.assertEqual(bun["change"], 26 - 29)
        # one observation only: no previous result, and never a fabricated zero change
        self.assertIsNone(self.out["rows"]["sodium"]["previous"])
        self.assertIsNone(self.out["rows"]["glucose"]["previous"])

    def test_the_forecast_column_compares_at_one_horizon(self):
        # 12h covers the most indicators in this registry, so it is the default
        self.assertEqual(self.out["defaultHorizon"], 12)
        self.assertEqual(sorted(h["hours"] for h in self.out["coverage"]["horizons"]), [1, 6, 12, 24])
        sodium = self.out["rows"]["sodium"]
        self.assertEqual((sodium["horizon"], sodium["atHorizon"]["h"], sodium["atHorizon"]["point"]), (12, 12, 146.2))
        # an indicator with no cell at the compared horizon says so instead of borrowing one
        bun = self.out["rows"]["bun"]
        self.assertFalse(bun["hasHorizonCell"])
        self.assertIsNone(bun["atHorizon"])

    def test_changing_the_horizon_changes_only_the_forecast_column(self):
        self.assertEqual(self.out["at24"]["bun"]["atHorizon"], {"h": 24, "point": 24.5})
        self.assertEqual(self.out["at24"]["sodium"]["atHorizon"], {"h": 24, "point": 141.0})
        self.assertFalse(self.out["at24"]["chloride"]["hasCell"])
        self.assertEqual(self.out["at6"]["creatinine"]["atHorizon"], {"h": 6, "status": "model"})
        self.assertEqual(self.out["at6"]["lactate"]["atHorizon"], {"h": 6, "status": "degraded"})
        # The column follows the chosen horizon; the ordering does not. Sodium leaves its
        # range at 12h, so it stays in Needs attention while the reader compares at 24h -
        # hiding a projection because another horizon is on screen would under-report it.
        self.assertEqual(self.out["rows"]["sodium"]["tier"], self.out["tiers"]["projected"])
        self.assertEqual(self.out["at24"]["sodium"]["tier"], self.out["tiers"]["projected"])
        self.assertIn("sodium", self.out["view24"]["attention"])

    # ---- forecast states ----------------------------------------------------
    def test_a_head_that_abstained_is_carried_forward_and_never_projected(self):
        chloride = self.out["rows"]["chloride"]["atHorizon"]
        self.assertEqual((chloride["status"], chloride["abstained"], chloride["projected"]), ("persistence", True, False))
        self.assertEqual(chloride["label"], "Carried forward — model abstained")
        self.assertEqual(self.out["view"]["abstainedCells"], 1)
        self.assertFalse(self.out["projectedPersistence"])
        self.assertFalse(self.out["projectedNull"])
        self.assertFalse(self.out["projectedDegraded"])

    def test_a_frozen_backbone_forecast_equal_to_the_observation_stays_a_model_forecast(self):
        glucose = self.out["rows"]["glucose"]["atHorizon"]
        self.assertEqual((glucose["status"], glucose["abstained"]), ("model", False))
        self.assertEqual(glucose["label"], "Model forecast")

    # ---- the reading order --------------------------------------------------
    def test_a_current_abnormality_sorts_before_a_projected_one(self):
        order = self.out["view"]["attention"]
        self.assertEqual(order, ["troponin_i", "bicarbonate", "bun", "hemoglobin", "sodium"])
        tiers = [self.out["rows"][v]["tier"] for v in order]
        self.assertEqual(tiers, sorted(tiers))
        self.assertLess(self.out["rows"]["bun"]["tier"], self.out["rows"]["sodium"]["tier"])
        self.assertEqual(self.out["view"]["outside"], 4)
        # bun is above its range now and its 24h point stays above it; sodium is inside
        # now and leaves at 12h. Both count as projected, and only sodium is sorted there.
        self.assertEqual(self.out["view"]["projected"], 2)
        self.assertEqual(self.out["rows"]["bun"]["tier"], self.out["tiers"]["outside"])

    def test_the_three_sections_never_share_an_indicator(self):
        v = self.out["view"]
        self.assertEqual(set(v["attention"]) & set(v["other"]), set())
        self.assertEqual(set(v["attention"]) & set(v["unobserved"]), set())
        self.assertEqual(set(v["other"]) & set(v["unobserved"]), set())
        self.assertEqual(len(v["attention"]) + len(v["other"]) + len(v["unobserved"]), v["total"])
        self.assertEqual(v["unobserved"], ["temperature"])

    # ---- organs -------------------------------------------------------------
    def test_one_rule_decides_what_an_organ_is_showing(self):
        kidneys = self.out["kidneys"]
        self.assertEqual(kidneys["state"], "abnormal")
        self.assertEqual(sorted(kidneys["abnormalIds"]), ["bicarbonate", "bun"])
        self.assertEqual(kidneys["above"], ["bun"])
        self.assertEqual(kidneys["below"], ["bicarbonate"])
        self.assertEqual(kidneys["projectedIds"], ["sodium"])   # counted, but the organ is already abnormal
        self.assertEqual(self.out["organs"]["heart"]["state"], "abnormal")     # troponin, source-coded
        self.assertEqual(self.out["organs"]["spleen"]["state"], "abnormal")    # hemoglobin, source-coded
        self.assertEqual(self.out["organs"]["pancreas"]["state"], "observed")
        self.assertEqual(self.out["organs"]["lungs"]["state"], "new")
        self.assertEqual(self.out["organs"]["thyroid"]["state"], "none")

    def test_an_organ_with_no_abnormality_is_not_marked_as_one(self):
        # blood vessels carry lactate, which is inside its range and has a withheld cell
        self.assertEqual(self.out["organs"]["blood_vessels"]["state"], "observed")
        states = {o.split(":")[0]: o.split(":")[1] for o in self.out["organOrder"]}
        self.assertNotIn("abnormal", [states["pancreas"], states["lungs"], states["thyroid"]])

    def test_the_organ_rail_puts_abnormal_organs_first(self):
        order = [o.split(":")[0] for o in self.out["organOrder"]]
        states = [o.split(":")[1] for o in self.out["organOrder"]]
        rank = {"abnormal": 0, "projected": 1, "new": 2, "observed": 3, "none": 4}
        self.assertEqual([rank[s] for s in states], sorted(rank[s] for s in states))
        self.assertEqual(set(order[:3]), {"kidneys", "heart", "spleen"})
        self.assertEqual(order[-1], "thyroid")

    def test_an_organ_workspace_puts_its_abnormal_indicators_first(self):
        # given creatinine, potassium, bun, sodium, chloride: bun is outside its range,
        # sodium is projected outside, chloride is inside, the rest have no range
        self.assertEqual(self.out["workspaceOrder"], ["bun", "sodium", "chloride", "creatinine", "potassium"])

    # ---- the report ---------------------------------------------------------
    def test_a_report_is_frozen_when_it_is_opened(self):
        after = self.out["afterEvent"]
        self.assertEqual(after["live"], 12)          # the page moved
        self.assertEqual(after["liveFlag"], "within")
        self.assertEqual(after["reportStill"], 26)   # the open report did not
        self.assertEqual(after["reportFlagStill"], "above")
        self.assertEqual(after["reportOrgansStill"], "abnormal")

    def test_refreshing_takes_a_new_snapshot_and_keeps_the_builder_choices(self):
        self.assertEqual(self.out["refreshed"]["bun"], 12)
        self.assertEqual(self.out["refreshed"]["scope"], self.out["report"]["scope"])
        self.assertEqual(self.out["refreshed"]["charts"], self.out["report"]["charts"])
        self.assertEqual(self.out["refreshed"]["horizon"], self.out["report"]["horizon"])

    def test_the_snapshot_carries_the_results_the_organs_and_the_counts(self):
        report = self.out["report"]
        self.assertNotIn("temperature", report["rows"])    # never observed, never a result
        self.assertEqual(report["organs"]["kidneys"], "abnormal")
        self.assertEqual(report["counts"]["outside"], 4)
        self.assertEqual(report["counts"]["abstainedCells"], 1)
        self.assertTrue(report["hasTraj"])                 # its own chart data, frozen with it
        self.assertRegex(report["id"], r"^OSL-\d{8}-\d{6}$")
        self.assertEqual((report["charts"], report["anatomy"], report["provenance"]), (True, True, False))

    # ---- a historical anchor ------------------------------------------------
    def test_a_historical_anchor_freezes_the_indicators_and_the_organs(self):
        anchor = self.out["anchor"]
        self.assertEqual(anchor["rows"], [["bun", 2, "above", 32, False], ["glucose", 7, None, 90, False]])
        self.assertEqual(anchor["organs"]["kidneys"], "abnormal")
        self.assertEqual(anchor["organs"]["lungs"], "none")   # the live fresh o2sat is not in the frozen state

    # ---- the charts ---------------------------------------------------------
    def test_the_chart_draws_the_supplied_range_as_a_band(self):
        bun = self.out["charts"]["bun"]
        self.assertIn('data-region="range"', bun)
        self.assertIn('data-range="limit"', bun)
        self.assertIn("Supplied range", bun)
        self.assertNotIn('data-region="range"', self.out["charts"]["chloride"])

    def test_a_projected_point_is_amber_and_dashed_never_the_current_abnormal_red(self):
        sodium = self.out["charts"]["sodium"]
        self.assertIn('data-projected="above"', sodium)
        self.assertIn("#8A5A0B", sodium)          # the projected amber
        self.assertNotIn("#8C1D18", sodium)       # never the deep red a measured result carries
        self.assertIn("#8C1D18", self.out["charts"]["bun"])   # bun's own current point is outside its range
        self.assertIn('data-outside="true"', self.out["charts"]["bun"])

    def test_one_observation_draws_no_trend_line(self):
        sodium = self.out["charts"]["sodium"]
        self.assertNotIn('data-series="observed"', sodium)
        self.assertIn('data-note="single-observation"', sodium)
        self.assertIn("no trend drawn", sodium.lower())
        self.assertIn('data-series="observed"', self.out["charts"]["bun"])   # three points: a line
        self.assertEqual(self.out["rows"]["sodium"]["points"], 1)


if __name__ == "__main__":
    unittest.main()

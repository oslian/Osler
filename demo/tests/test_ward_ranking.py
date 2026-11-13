"""Ward attention order: the design rules, pinned.

1. bands decide who is on the list -- over every analyte with cutpoints, not
   creatinine alone -- and the score decides the order inside one
2. an unscored stay is listed apart, never sorted to the bottom
3. every shown row carries a reason
4. it is an order, never an alarm
5. urine output per kg is reported and never ranked, and is absent when the
   weight is
"""

import inspect
import unittest

from oslian_demo.monitoring import ranking
from oslian_demo.monitoring.ranking import (
    BAND_DISPLAY_CAP, CUTPOINTS, PLAUSIBLE_WEIGHT_KG, REFERENCE_LOWER_CREATININE,
    REFERENCE_LOWER_URINE_RATE, REFERENCE_UPPER_CREATININE, REFERENCE_UPPER_URINE_RATE,
    STALE_AFTER_HOURS, TIER_FIRST_CUT, TIER_OUTSIDE, TIER_SECOND_CUT, TIER_WITHIN,
    PatientSnapshot, assess, band_of, board_to_dict, build_reasons, cutpoints_for,
    rank_ward, score_patient, urine_band_bounds, urine_band_of, urine_rate_per_kg, usable_weight,
)


def patient(pid, creatinine=(), now=24.0, urine=(), weight=None, labs=None, **treatments):
    return PatientSnapshot(
        patient_id=pid, display_label=f"Bed {pid}", now_hours=now,
        creatinine=tuple(creatinine), urine_output=tuple(urine), treatments=treatments,
        weight_kg=weight,
        labs={k: tuple(v) for k, v in (labs or {}).items()},
    )


def band(board, key):
    return next(b for b in board.bands if b.key == key)


def only_row(board):
    """The single row of a one-patient board, whichever band it landed in."""
    rows = [r for b in board.bands for r in b.rows]
    assert len(rows) == 1, rows
    return rows[0]


class ScoreTests(unittest.TestCase):
    def test_the_score_is_the_last_creatinine_at_any_age(self):
        # 14h old: the release's model path would drop this input; the order must not.
        score, age = score_patient(patient("a", [(2.0, 1.1), (10.0, 2.4)], now=24.0))
        self.assertEqual(score, 2.4)
        self.assertEqual(age, 14.0)

    def test_a_patient_with_no_creatinine_has_no_score(self):
        self.assertEqual(score_patient(patient("a")), (None, None))

    def test_a_snapshot_cannot_carry_an_observation_from_after_now(self):
        with self.assertRaises(ValueError):
            patient("a", [(30.0, 1.0)], now=24.0)

    def test_a_snapshot_cannot_carry_out_of_order_observations(self):
        with self.assertRaises(ValueError):
            patient("a", [(10.0, 1.0), (2.0, 1.2)], now=24.0)


class BandTests(unittest.TestCase):
    def test_a_creatinine_alone_lands_on_the_tier_its_own_cuts_give_it(self):
        hi = REFERENCE_UPPER_CREATININE
        self.assertEqual(band_of(hi * 3.5)[0], "second_cut")
        self.assertEqual(band_of(hi * 2.5)[0], "first_cut")
        self.assertEqual(band_of(hi * 1.5)[0], "outside")
        self.assertEqual(band_of(1.0)[0], "within")

    def test_a_value_under_the_lower_limit_is_outside_the_range_not_inside_it(self):
        # "within" for a value under the lower limit would be false, and it is
        # one stay in six on the demo sets.
        self.assertEqual(band_of(REFERENCE_LOWER_CREATININE - 0.01)[0], "outside")
        self.assertEqual(band_of(REFERENCE_LOWER_CREATININE)[0], "within")

    def test_the_boundaries_belong_to_the_higher_tier(self):
        # Read off the table rather than recomputed: 1.2 * 3.0 is
        # 3.5999999999999996, and a boundary test that lands a hair below the cut
        # is testing float arithmetic instead of the rule.
        cut = CUTPOINTS["creatinine"]
        self.assertEqual(band_of(cut.second_cut[1])[0], "second_cut")
        self.assertEqual(band_of(cut.first_cut[1])[0], "first_cut")
        self.assertEqual(band_of(cut.reference[1])[0], "within")
        self.assertEqual(band_of(cut.reference[0])[0], "within")

    def test_a_lab_with_a_different_reference_range_passes_its_own(self):
        board = rank_ward([patient("a", [(1.0, 0.8)])], reference_lower=0.9, reference_upper=1.4)
        self.assertEqual(band(board, "outside").rows[0].patient_id, "a")

    def test_reference_limits_must_be_ordered(self):
        with self.assertRaises(ValueError):
            rank_ward([patient("a", [(1.0, 1.0)])], reference_lower=2.0, reference_upper=1.0)

    def test_one_analyte_far_out_puts_a_stay_on_the_list_on_its_own(self):
        """The whole point of reading more than one analyte: a normal creatinine
        does not keep a stay off the list when something else is far out."""
        board = rank_ward([
            patient("quiet", [(1.0, 0.9)]),
            patient("low_sodium", [(1.0, 0.9)], labs={"sodium": [(1.0, 110.0)]}),
        ])
        self.assertEqual([r.patient_id for r in band(board, "second_cut").rows], ["low_sodium"])
        self.assertEqual([r.patient_id for r in band(board, "within").rows], ["quiet"])

    def test_a_band_can_outrank_a_higher_creatinine(self):
        # A stay the creatinine ordering would have put lower still comes first
        # when another analyte is further out. That is the rule, stated.
        board = rank_ward([
            patient("high_creat", [(1.0, 2.9)]),
            patient("low_sodium", [(1.0, 0.9)], labs={"sodium": [(1.0, 110.0)]}),
        ])
        order = [r.patient_id for b in board.bands for r in b.rows]
        self.assertEqual(order, ["low_sodium", "high_creat"])

    def test_the_score_still_orders_the_stays_inside_one_band(self):
        board = rank_ward([patient("lo", [(1.0, 2.5)]), patient("hi", [(1.0, 3.4)]),
                           patient("mid", [(1.0, 3.0)])])
        self.assertEqual([r.patient_id for r in band(board, "first_cut").rows],
                         ["hi", "mid", "lo"])

    def test_an_analyte_with_no_published_cuts_only_reaches_outside(self):
        # chloride has a reference range and nothing beyond it, so no chloride
        # value can put a stay in either upper band.
        self.assertEqual(CUTPOINTS["chloride"].first_cut, (None, None))
        board = rank_ward([patient("a", [(1.0, 0.9)], labs={"chloride": [(1.0, 140.0)]})])
        self.assertEqual(band(board, "outside").rows[0].patient_id, "a")

    def test_a_stay_with_no_extra_labs_is_banded_on_creatinine_alone(self):
        # Every caller that predates the other analytes keeps working.
        board = rank_ward([patient("a", [(1.0, 5.0)])])
        self.assertEqual(band(board, "second_cut").rows[0].patient_id, "a")

    def test_only_the_two_upper_bands_open_by_default(self):
        board = rank_ward([patient("a", [(1.0, 5.0)])])
        self.assertTrue(band(board, "second_cut").open_by_default)
        self.assertTrue(band(board, "first_cut").open_by_default)
        self.assertFalse(band(board, "outside").open_by_default)
        self.assertFalse(band(board, "within").open_by_default)

    def test_an_empty_band_is_reported_rather_than_dropped(self):
        board = rank_ward([patient("a", [(1.0, 0.8)])])
        self.assertEqual(band(board, "second_cut").rows, ())
        self.assertIn("second_cut", [b.key for b in board.bands])
        self.assertIn("in range", board.overview)

    def test_a_long_band_is_truncated_for_display_and_says_how_many(self):
        ward = [patient(f"p{i}", [(1.0, 0.8)]) for i in range(BAND_DISPLAY_CAP + 7)]
        held = band(rank_ward(ward), "within")
        self.assertEqual(len(held.rows), BAND_DISPLAY_CAP)
        self.assertEqual(held.truncated, 7)
        self.assertEqual(len(held.rows) + held.truncated, len(ward))


class FindingsTests(unittest.TestCase):
    """Every row says which analytes put it where, and how far out each is."""

    def test_findings_are_worst_first(self):
        row = only_row(rank_ward([patient("a", [(1.0, 1.5)], labs={
            "sodium": [(1.0, 110.0)], "potassium": [(1.0, 6.2)]})]))
        self.assertEqual([f["variable"] for f in row.findings],
                         ["sodium", "potassium", "creatinine"])
        self.assertEqual(row.findings[0]["tier"], TIER_SECOND_CUT)
        self.assertEqual(row.findings[1]["tier"], TIER_FIRST_CUT)
        self.assertEqual(row.findings[2]["tier"], TIER_OUTSIDE)

    def test_a_value_inside_its_range_is_not_a_finding(self):
        row = only_row(rank_ward([patient("a", [(1.0, 0.9)], labs={
            "sodium": [(1.0, 140.0)], "potassium": [(1.0, 6.6)]})]))
        self.assertEqual([f["variable"] for f in row.findings], ["potassium"])
        self.assertEqual(row.outside_count, 1)

    def test_the_count_covers_several_mildly_out_as_well_as_one_far_out(self):
        several = only_row(rank_ward([patient("a", [(1.0, 1.3)], labs={
            "sodium": [(1.0, 133.0)], "chloride": [(1.0, 110.0)],
            "bicarbonate": [(1.0, 20.0)]})]))
        self.assertEqual(several.band, "outside")
        self.assertEqual(several.outside_count, 4)

    def test_a_finding_carries_the_range_the_side_and_the_age(self):
        row = only_row(rank_ward([patient("a", [(1.0, 0.9)], now=13.0,
                                          labs={"sodium": [(1.0, 110.0)]})]))
        finding = row.findings[0]
        self.assertEqual((finding["referenceLower"], finding["referenceUpper"]), (135.0, 145.0))
        self.assertFalse(finding["high"])
        self.assertEqual(finding["ageHours"], 12.0)
        self.assertEqual(finding["unit"], "mmol/L")

    def test_every_cutpoint_says_whether_its_source_can_be_cited(self):
        # An unpinned cut is still used -- it is a real clinical number -- but the
        # board must be able to say which ones are not yet from a citable source.
        for name, cut in CUTPOINTS.items():
            self.assertIsInstance(cut.source_pinned, bool, name)
            self.assertTrue(cut.source, name)
        self.assertFalse(CUTPOINTS["bun"].source_pinned)
        self.assertTrue(CUTPOINTS["creatinine"].source_pinned)

    def test_the_cuts_of_a_setpoint_analyte_are_not_multiples_of_its_limit(self):
        """Twice the sodium limit is 290 mmol/L, which no one has ever measured.
        The upper tiers of these analytes have to be published numbers."""
        for name in ("sodium", "potassium", "bicarbonate"):
            cut = CUTPOINTS[name]
            self.assertLess(cut.second_cut[1], 2.0 * cut.reference[1], name)

    def test_a_ratio_analyte_uses_multiples_of_its_limit(self):
        for name in ("creatinine", "bun"):
            cut = CUTPOINTS[name]
            self.assertEqual(cut.first_cut[1], 2.0 * cut.reference[1], name)

    def test_calcium_is_not_in_the_table(self):
        # Dropped 2026-09-03: in the ICU raw total calcium tracks a low albumin,
        # and every stay it put in the top tier left once corrected.
        self.assertNotIn("calcium", CUTPOINTS)

    def test_assess_never_consults_the_score(self):
        # Two stays with the same analytes and different creatinine must land in
        # the same band; only their order inside it may differ.
        common = {"sodium": [(1.0, 110.0)]}
        board = rank_ward([patient("a", [(1.0, 0.7)], labs=common),
                           patient("b", [(1.0, 3.9)], labs=common)])
        bands = {r.patient_id: r.band for b in board.bands for r in b.rows}
        self.assertEqual(bands["a"], bands["b"])


class UnscoredTests(unittest.TestCase):
    def test_an_unscored_stay_is_separated_not_placed_in_the_lowest_band(self):
        board = rank_ward([patient("low", [(1.0, 0.8)]), patient("none")])
        self.assertEqual([r.patient_id for r in band(board, "within").rows], ["low"])
        self.assertEqual([r.patient_id for r in board.unrankable], ["none"])
        row = board.unrankable[0]
        self.assertFalse(row.rankable)
        self.assertTrue(row.unrankable_reason)
        self.assertIsNone(row.rank)
        self.assertIsNone(row.score)
        self.assertIsNone(row.band)

    def test_the_same_patient_twice_is_rejected_rather_than_deduplicated(self):
        with self.assertRaises(ValueError) as caught:
            rank_ward([patient("a", [(1.0, 2.0)]), patient("a", [(9.0, 3.0)])])
        self.assertIn("one row per patient", str(caught.exception))

    def test_a_display_cap_below_one_is_rejected(self):
        with self.assertRaises(ValueError):
            rank_ward([patient("a", [(1.0, 1.0)])], display_cap=0)


class StalenessTests(unittest.TestCase):
    def test_an_old_measurement_is_marked_but_still_ranks(self):
        # The score has no staleness cap; the age is reported instead.
        board = rank_ward([patient("old", [(0.0, 5.0)], now=STALE_AFTER_HOURS + 10),
                           patient("fresh", [(1.0, 0.9)], now=1.0)])
        old = band(board, "second_cut").rows[0]
        self.assertTrue(old.stale)
        self.assertEqual(old.rank, 1)
        self.assertFalse(band(board, "within").rows[0].stale)


class ReasonTests(unittest.TestCase):
    def test_every_shown_row_carries_at_least_the_number_it_was_ordered_by(self):
        board = rank_ward([patient(f"p{i}", [(1.0, 3.0)]) for i in range(5)])
        for row in band(board, "first_cut").rows:
            self.assertTrue(row.reasons)
            self.assertIn("creatinine", row.reasons[0])

    def test_the_creatinine_reason_carries_the_age_and_the_previous_value(self):
        reasons = build_reasons(patient("a", [(2.0, 1.9), (12.0, 2.4)], now=24.0))
        self.assertIn("creatinine 2.4 (12h ago)", reasons[0])
        self.assertIn("previous 1.9", reasons[0])
        self.assertIn("+0.5", reasons[0])

    def test_a_single_creatinine_reads_without_a_fabricated_previous_value(self):
        reasons = build_reasons(patient("a", [(12.0, 2.4)], now=24.0))
        self.assertIn("creatinine 2.4", reasons[0])
        self.assertNotIn("previous", reasons[0])

    def test_a_recent_observation_reads_in_minutes(self):
        self.assertIn("30min ago", build_reasons(patient("a", [(23.5, 2.4)], now=24.0))[0])

    def test_urine_output_and_treatments_are_added_when_present(self):
        joined = " | ".join(build_reasons(patient(
            "a", [(12.0, 2.4)], now=24.0, urine=((23.0, 22.0),), hist_diuretics=1, hist_fluids=1)))
        self.assertIn("urine output 22 mL/h (1h ago)", joined)
        self.assertIn("diuretics and fluids in the last 6h", joined)
        # The interval-derived rate carries more decimals than the measurement does.
        self.assertIn("urine output 17 mL/h", " | ".join(build_reasons(patient(
            "b", [(1.0, 2.0)], now=24.0, urine=((23.0, 16.6667),)))))

    def test_missing_urine_output_is_omitted_not_invented(self):
        # 47% of the eICU demo anchors have no urine output at all.
        self.assertNotIn("urine", " | ".join(build_reasons(patient("a", [(12.0, 2.4)], now=24.0))))

    def test_treatment_reasons_state_the_window_they_actually_cover(self):
        joined = " | ".join(build_reasons(patient("a", [(1.0, 2.0)], hist_nephrotoxin=1)))
        # hist_* is a 6h window flag, not a timestamp; the text must not imply one.
        self.assertIn("in the last 6h", joined)
        self.assertNotIn("ago", joined.split("nephrotoxic")[1])


class UrineRateTests(unittest.TestCase):
    """The rate per kg is an annotation: stated when it can be, absent when not,
    and never able to move a row."""

    def test_the_rate_is_the_last_charted_rate_over_the_weight(self):
        rate = urine_rate_per_kg(patient("a", urine=((20.0, 40.0), (23.0, 38.0)), weight=76.0))
        self.assertAlmostEqual(rate, 38.0 / 76.0)

    def test_no_weight_means_no_rate_rather_than_a_ward_average(self):
        # Fail closed. A stand-in weight would make a fabricated rate look measured.
        self.assertIsNone(urine_rate_per_kg(patient("a", urine=((23.0, 38.0),))))

    def test_no_urine_means_no_rate(self):
        self.assertIsNone(urine_rate_per_kg(patient("a", weight=76.0)))

    def test_an_impossible_weight_is_refused_the_same_way_a_missing_one_is(self):
        # eICU's demo table carries a 0.5 kg and a 515 kg; dividing by either
        # produces a rate that reads as measured and is not.
        low, high = PLAUSIBLE_WEIGHT_KG
        self.assertIsNone(usable_weight(low - 0.1))
        self.assertIsNone(usable_weight(high + 0.1))
        self.assertIsNone(usable_weight(None))
        self.assertEqual(usable_weight(low), low)
        self.assertEqual(usable_weight(high), high)
        self.assertIsNone(urine_rate_per_kg(patient("a", urine=((23.0, 38.0),), weight=0.5)))

    def test_the_urine_bands_are_read_off_the_quoted_range(self):
        lo, hi = REFERENCE_LOWER_URINE_RATE, REFERENCE_UPPER_URINE_RATE
        self.assertEqual(urine_band_of(hi + 0.1)[0], "u_above")
        self.assertEqual(urine_band_of((lo + hi) / 2)[0], "u_within")
        self.assertEqual(urine_band_of(lo * 0.8)[0], "u_two_thirds")
        self.assertEqual(urine_band_of(lo * 0.5)[0], "u_third")
        self.assertEqual(urine_band_of(lo * 0.2)[0], "u_lowest")
        self.assertEqual(urine_band_of(0.0)[0], "u_lowest")

    def test_the_urine_bands_tile_the_range_with_no_gap_and_no_overlap(self):
        ordered = sorted(urine_band_bounds(), key=lambda b: b[2])
        self.assertEqual(ordered[0][2], 0.0)
        for lower, upper in zip(ordered, ordered[1:]):
            self.assertEqual(lower[3], upper[2], f"{lower[0]} -> {upper[0]}")
        self.assertIsNone(ordered[-1][3])

    def test_a_unit_quoting_its_own_range_passes_its_own(self):
        self.assertEqual(urine_band_of(0.9, reference_lower=1.0, reference_upper=2.0)[0],
                         "u_two_thirds")

    def test_the_reason_gives_the_raw_rate_the_per_kg_rate_and_the_range(self):
        joined = " | ".join(build_reasons(
            patient("a", [(12.0, 2.4)], now=24.0, urine=((23.0, 16.6667),), weight=76.3)))
        self.assertIn("urine output 17 mL/h", joined)
        self.assertIn("0.22 mL/kg/h at 76.3 kg", joined)
        self.assertIn("0.5-1.5 mL/kg/h", joined)

    def test_a_row_with_no_weight_says_so_instead_of_going_quiet(self):
        joined = " | ".join(build_reasons(
            patient("a", [(12.0, 2.4)], now=24.0, urine=((23.0, 16.6667),))))
        self.assertIn("urine output 17 mL/h", joined)
        self.assertIn("weight not recorded", joined)
        self.assertNotIn("mL/kg/h", joined)

    def test_the_urine_rate_changes_no_score_no_rank_and_no_band(self):
        """The measured signal is the creatinine. Two stays with the same
        creatinine and opposite urine rates must come out identical."""
        common = dict(creatinine=[(1.0, 2.5)], now=24.0, weight=80.0)
        board = rank_ward([patient("dry", urine=((23.0, 4.0),), **common),
                           patient("wet", urine=((23.0, 400.0),), **common)])
        rows = {r.patient_id: r for b in board.bands for r in b.rows}
        self.assertEqual(rows["dry"].band, rows["wet"].band)
        self.assertEqual(rows["dry"].score, rows["wet"].score)
        # Only the tie-break on patient_id separates them, exactly as before.
        self.assertEqual(rows["dry"].rank, 1)
        self.assertEqual(rows["wet"].rank, 2)
        self.assertEqual(rows["dry"].urine_band, "u_lowest")
        self.assertEqual(rows["wet"].urine_band, "u_above")

    def test_an_unscored_stay_still_carries_its_rate(self):
        # No creatinine is a statement about the creatinine, not about the urine.
        board = rank_ward([patient("none", urine=((23.0, 40.0),), weight=80.0)])
        row = board.unrankable[0]
        self.assertIsNone(row.score)
        self.assertAlmostEqual(row.urine_rate, 0.5)
        self.assertEqual(row.urine_band, "u_within")

    def test_an_out_of_order_urine_range_is_rejected(self):
        with self.assertRaises(ValueError):
            rank_ward([patient("a", [(1.0, 1.0)])],
                      urine_reference_lower=2.0, urine_reference_upper=1.0)

    def test_the_payload_carries_the_rate_and_says_it_orders_nothing(self):
        payload = board_to_dict(rank_ward([
            patient("a", [(1.0, 2.5)], urine=((23.0, 20.0),), weight=80.0),
            patient("b", [(1.0, 2.5)], urine=((23.0, 20.0),))]))
        self.assertIs(payload["urineRateOrdersNothing"], True)
        self.assertEqual(payload["urineReferenceLower"], REFERENCE_LOWER_URINE_RATE)
        self.assertEqual(payload["urineReferenceUpper"], REFERENCE_UPPER_URINE_RATE)
        # The payload states that this range is a convention, not a printed interval.
        self.assertIn("not an interval printed by the laboratory",
                      payload["urineReferenceSource"])
        rows = {r["patientId"]: r for b in payload["bands"] for r in b["rows"]}
        self.assertAlmostEqual(rows["a"]["urineRatePerKg"], 0.25)
        self.assertEqual(rows["a"]["weightKg"], 80.0)
        self.assertIsNone(rows["b"]["urineRatePerKg"])
        self.assertIsNone(rows["b"]["weightKg"])
        self.assertIsNone(rows["b"]["urineBand"])

    def test_the_module_never_states_the_criterion_that_pairs_a_rate_with_a_duration(self):
        """Where a rate sits against a range is arithmetic. Pairing it with a
        duration to name a class is a diagnosis, and this module makes none."""
        source = inspect.getsource(ranking).lower()
        for word in ("kdigo", "oliguri", "anuri", "aki", "stage 1", "stage 2", "stage 3"):
            self.assertNotIn(word, source, f"{word!r} classifies; this module only compares")


class NotAnAlarmTests(unittest.TestCase):
    FORBIDDEN = ("alert", "alarm", "critical", "urgent", "warning", "severity",
                 "danger", "emergency", "escalate", "severe", "deteriorat")
    # The payload has to name the thing it denies being, so the key that carries
    # the denial is removed before the scan rather than exempted from it.
    DECLARATIONS = ("isclinicalalert", "is_clinical_alert")

    def _scannable(self, text):
        text = text.lower()
        for token in self.DECLARATIONS:
            text = text.replace(token, "")
        return text

    def test_the_module_never_speaks_in_alarm_words(self):
        body = inspect.getsource(ranking).split('"""', 2)[2]
        body = self._scannable(body)
        for word in self.FORBIDDEN:
            self.assertNotIn(word, body, f"{word!r} belongs to alerting, not to a reading order")

    def test_no_rendered_string_carries_an_alarm_word(self):
        board = rank_ward([patient("a", [(1.0, 9.9)], urine=((1.0, 5.0),), hist_diuretics=1),
                           patient("b")])
        rendered = self._scannable(str(board_to_dict(board)))
        for word in self.FORBIDDEN:
            self.assertNotIn(word, rendered)

    def test_a_band_label_states_arithmetic_and_never_a_judgement(self):
        # The published gradings have words for these tiers, and those words are
        # judgements about a patient. The labels name the line that was crossed.
        for b in rank_ward([patient("a", [(1.0, 5.0)])]).bands:
            self.assertRegex(b.label, r"published cut|reference range|in range")

    def test_the_denial_key_itself_is_still_present_and_false(self):
        payload = board_to_dict(rank_ward([patient("a", [(1.0, 1.0)])]))
        self.assertIn("isClinicalAlert", payload)
        self.assertIs(payload["isClinicalAlert"], False)
        self.assertIs(payload["causal_claim_allowed"], False)

    def test_the_payload_names_the_basis_of_the_order_on_every_row(self):
        payload = board_to_dict(rank_ward([patient("a", [(1.0, 1.0)]), patient("b")]))
        self.assertEqual(payload["scoreBasis"], "last_creatinine_any_age")
        self.assertEqual(payload["schema"], "oslian.ward-attention-order.v3")
        # The two bases are named apart, so nobody reads the whole board order
        # as the measured one.
        self.assertEqual(payload["bandBasis"], "highest_tier_of_any_measured_analyte")
        self.assertEqual(payload["orderWithinBandBasis"], "last_creatinine_any_age")
        rows = [r for b in payload["bands"] for r in b["rows"]] + payload["unrankable"]
        for row in rows:
            self.assertEqual(row["scoreBasis"], "last_creatinine_any_age")


if __name__ == "__main__":
    unittest.main()

"""Ward attention order: who to look at first, not what their number will be.

This is deliberately not a forecast. Re-scoring the 2026-08-22 release on the
eICU-CRD and MIMIC-IV demo sets (2026-09-02) found that a point forecast of
creatinine cannot beat carrying the last measured value forward, because three
quarters of ICU patients move less in twelve hours than the assay resolves. The
question that *is* answerable on the same data is which quarter is about to
move, and the answer needs no model:

    score = the patient's last measured creatinine, at any age

    AUROC 0.814 [0.743, 0.872] for |creatinine(t+12h) - creatinine(t)| >= 0.3
    on the MIMIC-IV demo set (453 anchors / 78 patients, 24.1% positive).
    A 53-variable gradient-boosting gate trained on eICU scored 0.703 on the
    same rows -- the free rule wins, so the free rule is what ships.

Two contract points follow from that measurement and are enforced here.

**No staleness cap on the score.** Capping the input at six hours the way the
release's model path does drops the AUROC to 0.615, because only 51.9% of
anchors have a creatinine that fresh and the rest fall back to a fill value. The
age is reported next to the number instead, as a reason a reader can weigh.

**A band, not a quota.** A fixed number of rows per shift was tried first, on
the argument that a threshold swings with how sick the ward is. Measured on the
demo sets, that argument does not survive: a quota fills a quiet ward with stays
nobody needs to read, and at a comparable list length a band cut caught more
movers (eICU 46.1% vs 37.7% recall, MIMIC 50.0% vs 47.7% with better precision).
The bands are multiples of the published reference upper limit -- arithmetic on
a published number, not a clinical threshold this view invented. A band may come
back empty, and on a ward where nothing is happening that is the answer, not a
failure.

**Urine output is reported, never ranked.** A urine rate is charted in mL/hour,
which cannot be compared to anything without a body weight, so the board used to
print the bare number. With a weight it also states the rate per kilogram and
where that rate sits against the range conventionally quoted for adults. That
comparison is an annotation on a row the creatinine already placed: it does not
enter the score, the order or the band a stay is listed under. The reason is
measured. The per-kg band's gradient over "creatinine moves at least 0.3 mg/dL
in twelve hours" reproduces on both demo sets (eICU 36.4% to 19.8%, MIMIC-IV
41.1% to 16.5%) but it is weak next to creatinine's own bands (83.8% to 4.1%),
and on MIMIC-IV it travels with creatinine -- median creatinine runs 2.20 down
to 0.90 across the same bands, against a flat 0.93 to 0.90 on eICU -- so it is
not established as information the order does not already have.

The per-kg comparison states where one charted interval rate sits against a
range, and stops there. It is not the criterion that pairs a rate with a
duration to classify a stay, which is a diagnosis this view is not allowed to
make and does not make: no duration is required, no class is named, and the
grouping a reader sees is still the creatinine one.

Nothing here is an alert. A band says how many multiples of the reference limit
a measured value sits at and nothing more: no severity word, no colour that
carries meaning on its own, no threshold crossing that fires. The output is a
grouping, an order inside it, and a reason per row.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

# ---- the cutpoint table ----------------------------------------------------
# Which band a stay is listed under is read off published numbers, one set per
# analyte. There is no single formula, because these are two kinds of analyte and
# one rule does not fit both -- measured on the demo sets, the largest value ever
# seen divided by the reference upper limit:
#
#     creatinine 7.5x / 12.7x     bun 6.1x / 6.5x        <- floors at zero, rises
#     bicarbonate 1.55x / 1.59x   potassium 1.48x / 1.28x   many-fold
#     chloride 1.18x / 1.27x      sodium 1.11x / 1.12x   <- held near a setpoint,
#     (eICU / MIMIC-IV)                                     never doubles
#
# So "twice the reference limit" is a real cut for creatinine and BUN and an
# impossible one for the rest: twice the sodium limit is 290 mmol/L. The second
# group takes its two upper tiers from the gradings published for that analyte
# and from the outer cut laboratories telephone on. This is why NEWS2 lists
# explicit bands per parameter rather than applying one arithmetic rule to all.
#
# `source` records where each number came from and `source_pinned` says whether
# it is one this project can cite. An unpinned number is still used -- it is a
# real clinical figure -- but the board reports that it is unpinned rather than
# presenting every cut as equally settled.

# Every analyte carries three published lines: its reference range, and two cuts
# beyond it. A tier names which line a value has passed and nothing else -- the
# published gradings do have words for these, and those words are judgements
# about a patient, which a reading order is not allowed to make. Naming the
# arithmetic keeps the same ordering without the claim.
TIER_WITHIN, TIER_OUTSIDE, TIER_FIRST_CUT, TIER_SECOND_CUT = 0, 1, 2, 3

TIER_LABELS = {
    TIER_SECOND_CUT: "a value past the second published cut",
    TIER_FIRST_CUT: "a value past the first published cut",
    TIER_OUTSIDE: "a value outside its reference range",
    TIER_WITHIN: "every measured value in range",
}

TIER_KEYS = {TIER_SECOND_CUT: "second_cut", TIER_FIRST_CUT: "first_cut",
             TIER_OUTSIDE: "outside", TIER_WITHIN: "within"}


@dataclass(frozen=True)
class Cutpoints:
    """One analyte's four tiers, and where the numbers came from.

    `first_cut` and `second_cut` are (low, high); `None` on a side means no
    published cut exists in that direction, and nothing is invented to fill it.
    """

    variable: str
    label: str
    unit: str
    reference: tuple[float, float]
    first_cut: tuple[float | None, float | None]
    second_cut: tuple[float | None, float | None]
    source: str
    source_pinned: bool
    caveat: str = ""

    def tier(self, value: float) -> int:
        low, high = self.second_cut
        if (low is not None and value <= low) or (high is not None and value >= high):
            return TIER_SECOND_CUT
        low, high = self.first_cut
        if (low is not None and value <= low) or (high is not None and value >= high):
            return TIER_FIRST_CUT
        return TIER_WITHIN if self.reference[0] <= value <= self.reference[1] else TIER_OUTSIDE


CUTPOINTS: dict[str, Cutpoints] = {
    "creatinine": Cutpoints(
        "creatinine", "creatinine", "mg/dL", (0.6, 1.2), (None, 2.4), (None, 3.6),
        source="2x and 3x the reference upper limit; reproduced on both demo sets "
               "(AUROC 0.814 / 0.819) and close to the 2.0 / 3.5 of the published organ score",
        source_pinned=True),
    "bun": Cutpoints(
        "bun", "BUN", "mg/dL", (7.0, 20.0), (None, 40.0), (None, 100.0),
        source="40 = 2x the reference upper limit; 100 is the outer cut commonly published "
               "for telephoned results",
        source_pinned=False,
        caveat="also moved by gastrointestinal bleeding, catabolism, protein intake and "
               "dehydration, so a raised value is not by itself a renal finding"),
    "potassium": Cutpoints(
        "potassium", "potassium", "mmol/L", (3.5, 5.0), (3.0, 6.0), (2.5, 6.5),
        source="published grading; 2.5 and 6.5 are also the outer cut carried by "
               "essentially every laboratory for telephoned results",
        source_pinned=True),
    "sodium": Cutpoints(
        "sodium", "sodium", "mmol/L", (135.0, 145.0), (130.0, 150.0), (125.0, 160.0),
        source="published grading; 125 and 160 are also the outer cut commonly "
               "published for telephoned results",
        source_pinned=True),
    "bicarbonate": Cutpoints(
        "bicarbonate", "bicarbonate", "mmol/L", (22.0, 29.0), (15.0, 35.0), (10.0, 40.0),
        source="commonly quoted grading, less standardised than the potassium and "
               "sodium ones",
        source_pinned=False),
    "chloride": Cutpoints(
        "chloride", "chloride", "mmol/L", (98.0, 107.0), (None, None), (None, None),
        source="reference range only -- no published grading or outer cut this project "
               "can cite, so it has two tiers and not four",
        source_pinned=True),
}

# Total calcium was measured and dropped (2026-09-03). In the ICU it tracks a low
# albumin rather than a calcium problem: 55% (eICU) and 63% (MIMIC-IV) of stays
# read below the reference lower limit, falling to 18% and 14% once corrected for
# albumin, and every stay the raw value put in the top tier left it. Correcting
# needs an albumin, which 21% of eICU and 62% of MIMIC-IV stays do not have.

# The band a stay is listed under is the highest tier any one of its measured
# values reaches -- one value far out puts the stay on the list whether or not
# every other one is inside its range. The score still orders the stays inside
# a band.
BANDS: tuple[tuple[str, str], ...] = tuple(
    (TIER_KEYS[t], TIER_LABELS[t])
    for t in (TIER_SECOND_CUT, TIER_FIRST_CUT, TIER_OUTSIDE, TIER_WITHIN)
)

# Kept as names because callers and a lab's own range still address creatinine
# directly; they are the same two numbers the table above carries.
REFERENCE_LOWER_CREATININE, REFERENCE_UPPER_CREATININE = CUTPOINTS["creatinine"].reference

# Opened by default. The rest stay collapsed with their count on screen, so no
# stay is hidden -- only further from the first glance. These two together are
# 27.7% of the eICU ward and 23.2% of the MIMIC-IV one, which is the working
# point the ordering was measured at (top 20-30%).
BANDS_OPEN_BY_DEFAULT = ("second_cut", "first_cut")

# A band longer than this is truncated on screen. It is a display limit, never a
# judgement: the number truncated is always reported.
BAND_DISPLAY_CAP = 15

# Beyond this, the age of the measurement is worth stating next to it. The score
# deliberately has no staleness cap (see the module docstring); this only marks
# the number so a reader can weigh it.
STALE_AFTER_HOURS = 12.0

# `hist_*` flags record whether an action was active in the six hours before
# `now_hours`, which is the window the anchor builder used. Reason text states
# that window rather than implying a timestamp the data does not carry.
TREATMENT_WINDOW_HOURS = 6.0

TREATMENT_LABELS = {
    "hist_diuretics": "diuretics",
    "hist_vasopressor": "vasopressors",
    "hist_nephrotoxin": "a nephrotoxic agent",
    "hist_renal_replacement": "renal replacement",
    "hist_fluids": "fluids",
}

# Ordered by how much they change what a reader would do next.
TREATMENT_ORDER = (
    "hist_renal_replacement", "hist_nephrotoxin", "hist_diuretics",
    "hist_vasopressor", "hist_fluids",
)

UNRANKABLE_NO_CREATININE = "no creatinine on record for this stay"

# ---- urine output per kilogram --------------------------------------------
# A charted urine rate is mL/hour and means nothing without a body weight, so
# the board printed the bare number and left the comparison to the reader. These
# two ends are the range conventionally quoted for an adult in mL/kg/h. Unlike
# the creatinine range, which the laboratory prints on its own report, this one
# is a convention rather than an assay's own interval -- the difference is stated
# wherever the comparison is shown, and a unit that quotes its own range passes
# its own values in.
REFERENCE_LOWER_URINE_RATE = 0.5
REFERENCE_UPPER_URINE_RATE = 1.5

# The mirror of the creatinine bands. Creatinine is read upward, in multiples of
# the range's upper end; a urine rate is read downward, in fractions of its lower
# end. (key, label, low, high) as fractions of the lower limit; the top two are
# the range's own ends. `None` is open-ended.
URINE_BANDS: tuple[tuple[str, str, float | None, float | None], ...] = (
    ("u_above", "above the quoted range", None, None),        # at or over the upper end
    ("u_within", "within the quoted range", None, None),      # the range's own two ends
    ("u_two_thirds", "two thirds of the lower end up to it", 2 / 3, 1.0),
    ("u_third", "one third to two thirds of the lower end", 1 / 3, 2 / 3),
    ("u_lowest", "one third of the lower end or less", 0.0, 1 / 3),
)

# A weight the board will not divide by. eICU's demo `admissionweight` carries a
# 0.5 kg and a 515 kg among 2322 values; dividing by either produces a rate that
# looks measured and is not. Out of range means the rate is withheld, exactly as
# a missing weight does -- never replaced with a population figure, because the
# whole point of the number is that it is this patient's.
PLAUSIBLE_WEIGHT_KG = (20.0, 400.0)

NO_WEIGHT_NOTE = "weight not recorded, so the rate per kg is not stated"


@dataclass(frozen=True)
class PatientSnapshot:
    """One patient's observed history up to `now_hours`. Nothing after it."""

    patient_id: str
    display_label: str
    now_hours: float
    # (hours_since_onset, value), ascending. May be empty.
    creatinine: tuple[tuple[float, float], ...] = ()
    urine_output: tuple[tuple[float, float], ...] = ()
    treatments: dict[str, Any] = field(default_factory=dict)
    # Optional, and stays optional: every caller that has no weight column keeps
    # working and simply gets no rate per kg.
    weight_kg: float | None = None
    # The other banded analytes, variable -> ((hours, value), ...). Creatinine
    # keeps its own field because it alone is the score; these only place a stay
    # in a band. A caller that passes none gets a board banded on creatinine
    # alone, which is what this module did before they existed.
    labs: dict[str, tuple[tuple[float, float], ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, series in self.each_series():
            hours = [h for h, _ in series]
            if hours != sorted(hours):
                raise ValueError(f"{name} must be ascending in time")
            if any(h > self.now_hours + 1e-9 for h in hours):
                raise ValueError(f"{name} contains an observation after now_hours")

    def each_series(self):
        """Every observed series on this stay, creatinine first."""
        yield "creatinine", self.creatinine
        yield "urine_output", self.urine_output
        for name, series in self.labs.items():
            yield name, tuple(series)

    def banded_values(self) -> dict[str, tuple[float, float]]:
        """variable -> (hour, value) of the last measurement the bands can read.

        Creatinine comes from its own field; the rest from `labs`. A variable
        with no cutpoints is skipped rather than banded by a guessed range.
        """
        out: dict[str, tuple[float, float]] = {}
        for name, series in self.each_series():
            if name in CUTPOINTS and series:
                out[name] = series[-1]
        return out


@dataclass(frozen=True)
class Row:
    patient_id: str
    display_label: str
    rank: int | None
    score: float | None
    score_basis: str
    measurement_age_hours: float | None
    reasons: tuple[str, ...]
    band: str | None
    band_label: str | None
    stale: bool
    rankable: bool
    unrankable_reason: str | None
    # Reported, never ranked: none of these four reaches `score`, `rank` or
    # `band`. `urine_rate` is None whenever the weight or the urine is missing.
    weight_kg: float | None = None
    urine_rate: float | None = None
    urine_band: str | None = None
    urine_band_label: str | None = None
    # What put the stay in its band: one entry per measured analyte that is at
    # least outside its reference range, worst first. `outside_count` counts them
    # all, so "several mildly abnormal" is visible next to "one far out".
    findings: tuple[dict[str, Any], ...] = ()
    outside_count: int = 0


@dataclass(frozen=True)
class Band:
    key: str
    label: str
    rows: tuple[Row, ...]
    open_by_default: bool
    truncated: int          # rows the display cap held back; 0 when none


@dataclass(frozen=True)
class Board:
    bands: tuple[Band, ...]
    unrankable: tuple[Row, ...]
    total_patients: int
    reference_upper: float
    reference_lower: float
    display_cap: int
    score_basis: str
    overview: str
    urine_reference_lower: float
    urine_reference_upper: float
    cutpoints: dict[str, Cutpoints]


def _age(now_hours: float, hour: float) -> float:
    return max(0.0, now_hours - hour)


def _format_age(hours: float) -> str:
    if hours < 1.0:
        return f"{round(hours * 60)}min ago"
    return f"{hours:.0f}h ago"


def _creatinine_reason(snapshot: PatientSnapshot) -> str:
    hour, value = snapshot.creatinine[-1]
    text = f"creatinine {value:g} ({_format_age(_age(snapshot.now_hours, hour))})"
    if len(snapshot.creatinine) >= 2:
        prev_hour, prev_value = snapshot.creatinine[-2]
        delta = value - prev_value
        arrow = "+" if delta > 0 else ""
        text += (f"; previous {prev_value:g} {_format_age(_age(snapshot.now_hours, prev_hour))}"
                 f", {arrow}{delta:.1f}")
    return text


def usable_weight(weight_kg: float | None) -> float | None:
    """The weight the board is willing to divide by, or None.

    Fail closed on both edges. A missing weight and an impossible one are the
    same answer -- the rate per kg is not stated -- because a number derived from
    a 0.5 kg adult reads exactly like a measured one and is not.
    """
    if weight_kg is None:
        return None
    low, high = PLAUSIBLE_WEIGHT_KG
    return float(weight_kg) if low <= float(weight_kg) <= high else None


def urine_rate_per_kg(snapshot: PatientSnapshot) -> float | None:
    """The last charted urine rate divided by the stay's weight, in mL/kg/h.

    One charted interval, not a window: the extractor already derived this value
    as a rate over the interval it was charted for, so nothing here accumulates
    across intervals or asks how long a rate has held.
    """
    weight = usable_weight(snapshot.weight_kg)
    if weight is None or not snapshot.urine_output:
        return None
    return snapshot.urine_output[-1][1] / weight


def urine_band_bounds(reference_lower: float = REFERENCE_LOWER_URINE_RATE,
                      reference_upper: float = REFERENCE_UPPER_URINE_RATE
                      ) -> list[tuple[str, str, float, float | None]]:
    """Each urine band as an absolute mL/kg/h rate, read off the quoted range."""
    out = []
    for key, label, low, high in URINE_BANDS:
        if key == "u_above":
            out.append((key, label, reference_upper, None))
        elif key == "u_within":
            out.append((key, label, reference_lower, reference_upper))
        else:
            out.append((key, label, low * reference_lower, high * reference_lower))
    return out


def urine_band_of(rate: float, reference_lower: float = REFERENCE_LOWER_URINE_RATE,
                  reference_upper: float = REFERENCE_UPPER_URINE_RATE) -> tuple[str, str]:
    """Where a measured rate sits against the quoted range. Arithmetic, not a class."""
    for key, label, low, high in urine_band_bounds(reference_lower, reference_upper):
        if rate >= low and (high is None or rate < high):
            return key, label
    return URINE_BANDS[-1][0], URINE_BANDS[-1][1]


def _urine_reason(snapshot: PatientSnapshot,
                  reference_lower: float = REFERENCE_LOWER_URINE_RATE,
                  reference_upper: float = REFERENCE_UPPER_URINE_RATE) -> str | None:
    if not snapshot.urine_output:
        return None
    hour, value = snapshot.urine_output[-1]
    # Charted to the whole mL/h; the derived interval rate carries more decimals
    # than the measurement does, and a board is read at a glance.
    text = f"urine output {round(value)} mL/h ({_format_age(_age(snapshot.now_hours, hour))})"
    rate = urine_rate_per_kg(snapshot)
    if rate is None:
        # Say why the comparison is absent rather than leaving the reader to
        # wonder whether it was withheld or never possible.
        return f"{text}; {NO_WEIGHT_NOTE}"
    _key, label = urine_band_of(rate, reference_lower, reference_upper)
    weight = usable_weight(snapshot.weight_kg)
    return (f"{text}; {rate:.2f} mL/kg/h at {weight:g} kg, "
            f"{label} ({reference_lower:g}-{reference_upper:g} mL/kg/h)")


def _treatment_reason(snapshot: PatientSnapshot) -> str | None:
    active = [TREATMENT_LABELS[k] for k in TREATMENT_ORDER if snapshot.treatments.get(k)]
    if not active:
        return None
    window = f"in the last {TREATMENT_WINDOW_HOURS:g}h"
    if len(active) == 1:
        return f"{active[0]} {window}"
    return f"{', '.join(active[:-1])} and {active[-1]} {window}"


def build_reasons(snapshot: PatientSnapshot,
                  urine_reference_lower: float = REFERENCE_LOWER_URINE_RATE,
                  urine_reference_upper: float = REFERENCE_UPPER_URINE_RATE) -> tuple[str, ...]:
    """Observed facts only, each with the age of the observation it came from.

    A row without a reason gets skipped by the reader, so every rankable row
    carries at least the number it was ordered by. No reason states or implies a
    forecast.
    """
    reasons = [_creatinine_reason(snapshot)] if snapshot.creatinine else []
    urine = _urine_reason(snapshot, urine_reference_lower, urine_reference_upper)
    for candidate in (urine, _treatment_reason(snapshot)):
        if candidate:
            reasons.append(candidate)
    return tuple(reasons)


def cutpoints_for(reference_upper: float = REFERENCE_UPPER_CREATININE,
                  reference_lower: float = REFERENCE_LOWER_CREATININE
                  ) -> dict[str, Cutpoints]:
    """The table, with creatinine's range replaced when a lab passes its own."""
    creatinine = CUTPOINTS["creatinine"]
    if (reference_lower, reference_upper) == creatinine.reference:
        return CUTPOINTS
    return {**CUTPOINTS, "creatinine": replace(
        creatinine, reference=(reference_lower, reference_upper),
        first_cut=(None, 2.0 * reference_upper), second_cut=(None, 3.0 * reference_upper))}


def band_of(score: float, reference_upper: float = REFERENCE_UPPER_CREATININE,
            reference_lower: float = REFERENCE_LOWER_CREATININE) -> tuple[str, str]:
    """The band a creatinine on its own would place a stay in.

    Kept because creatinine is still the score, and because a stay whose only
    measured analyte is creatinine lands exactly here.
    """
    tier = cutpoints_for(reference_upper, reference_lower)["creatinine"].tier(score)
    return TIER_KEYS[tier], TIER_LABELS[tier]


def assess(snapshot: PatientSnapshot, table: dict[str, Cutpoints]
           ) -> tuple[int, tuple[dict[str, Any], ...], int]:
    """The stay's band, what put it there, and how many analytes are out of range.

    The band is the highest tier any single measured value reaches -- one value
    far out is enough on its own, which is the whole point of reading more than
    one analyte. Nothing here consults the score: this decides who is on the
    list, and the score decides the order inside a band.
    """
    findings, worst, outside = [], TIER_WITHIN, 0
    for variable, (hour, value) in snapshot.banded_values().items():
        cut = table[variable]
        tier = cut.tier(value)
        worst = max(worst, tier)
        if tier == TIER_WITHIN:
            continue
        outside += 1
        findings.append({
            "variable": variable, "label": cut.label, "value": value, "unit": cut.unit,
            "tier": tier, "tierLabel": TIER_LABELS[tier], "tierKey": TIER_KEYS[tier],
            "referenceLower": cut.reference[0], "referenceUpper": cut.reference[1],
            "high": value > cut.reference[1],
            "ageHours": _age(snapshot.now_hours, hour),
            "sourcePinned": cut.source_pinned,
        })
    # Worst first, then the older reading before the fresher one at equal tier so
    # the order is reproducible; the variable name breaks any remaining tie.
    findings.sort(key=lambda f: (-f["tier"], -f["ageHours"], f["variable"]))
    return worst, tuple(findings), outside


def score_patient(snapshot: PatientSnapshot) -> tuple[float | None, float | None]:
    """The last measured creatinine and its age. No staleness cap -- see module docstring."""
    if not snapshot.creatinine:
        return None, None
    hour, value = snapshot.creatinine[-1]
    return float(value), _age(snapshot.now_hours, hour)


def _urine_fields(snapshot: PatientSnapshot, urine_lower: float, urine_upper: float
                  ) -> dict[str, Any]:
    """The per-kg annotation for one row. Never consulted for score, order or band."""
    rate = urine_rate_per_kg(snapshot)
    if rate is None:
        return {"weight_kg": usable_weight(snapshot.weight_kg), "urine_rate": None,
                "urine_band": None, "urine_band_label": None}
    key, label = urine_band_of(rate, urine_lower, urine_upper)
    return {"weight_kg": usable_weight(snapshot.weight_kg), "urine_rate": rate,
            "urine_band": key, "urine_band_label": label}


def rank_ward(snapshots: list[PatientSnapshot],
              reference_upper: float = REFERENCE_UPPER_CREATININE,
              reference_lower: float = REFERENCE_LOWER_CREATININE,
              display_cap: int = BAND_DISPLAY_CAP,
              urine_reference_lower: float = REFERENCE_LOWER_URINE_RATE,
              urine_reference_upper: float = REFERENCE_UPPER_URINE_RATE) -> Board:
    """Group a ward into bands, order inside each band, and say what was held back.

    Every stay that can be scored appears in a band -- the low bands are simply
    collapsed by default rather than dropped, so no stay is ever off the list,
    only further from the first glance.

    Patients with no creatinine cannot be scored by this rule. They are returned
    in their own list rather than placed in the lowest band, because an unscored
    stay and a low-scoring stay are different claims and a reader must not have
    to tell them apart by position.
    """
    if display_cap < 1:
        raise ValueError("display_cap must be at least 1")
    if reference_upper <= 0 or reference_lower <= 0:
        raise ValueError("reference limits must be positive")
    if reference_lower >= reference_upper:
        raise ValueError("reference_lower must be below reference_upper")
    if urine_reference_lower <= 0 or urine_reference_lower >= urine_reference_upper:
        raise ValueError("urine_reference_lower must be positive and below urine_reference_upper")
    # A ward is a set of patients at one wall-clock moment. The same patient
    # twice means the caller has passed two anchors and has not decided which
    # one is "now"; deduplicating here would hide that, so it fails instead.
    seen = [s.patient_id for s in snapshots]
    if len(set(seen)) != len(seen):
        duplicated = sorted({p for p in seen if seen.count(p) > 1})
        raise ValueError(f"one row per patient: {', '.join(duplicated)} appears more than once")

    table = cutpoints_for(reference_upper, reference_lower)
    scored, unrankable = [], []
    for snapshot in snapshots:
        score, age = score_patient(snapshot)
        reasons = build_reasons(snapshot, urine_reference_lower, urine_reference_upper)
        urine = _urine_fields(snapshot, urine_reference_lower, urine_reference_upper)
        tier, findings, outside = assess(snapshot, table)
        if score is None:
            # An unscored stay still carries its findings and its urine
            # annotation: those are facts about the stay, and withholding them
            # here would make a missing creatinine look like a missing everything.
            unrankable.append(Row(
                patient_id=snapshot.patient_id, display_label=snapshot.display_label,
                rank=None, score=None, score_basis="last_creatinine_any_age",
                measurement_age_hours=None, reasons=reasons, band=None, band_label=None,
                stale=False, rankable=False, unrankable_reason=UNRANKABLE_NO_CREATININE,
                findings=findings, outside_count=outside, **urine,
            ))
            continue
        scored.append((tier, score, age, snapshot, reasons, urine, findings, outside))

    # Band first, then descending score inside it, then patient_id so the order is
    # reproducible. The two keys answer different questions and are kept apart on
    # purpose: the band says who belongs on the list, and it is read off published
    # cutpoints; the score says who to read first within one, and it is the only
    # part of this that was measured as an ordering (AUROC 0.814 / 0.819).
    scored.sort(key=lambda item: (-item[0], -item[1], item[3].patient_id))
    rows = tuple(
        Row(
            patient_id=snapshot.patient_id, display_label=snapshot.display_label,
            rank=index, score=score, score_basis="last_creatinine_any_age",
            measurement_age_hours=age, reasons=reasons,
            band=TIER_KEYS[tier], band_label=TIER_LABELS[tier],
            stale=age is not None and age > STALE_AFTER_HOURS,
            rankable=True, unrankable_reason=None,
            findings=findings, outside_count=outside, **urine,
        )
        for index, (tier, score, age, snapshot, reasons, urine, findings, outside)
        in enumerate(scored, start=1)
    )

    bands = []
    for key, label in BANDS:
        members = tuple(r for r in rows if r.band == key)
        bands.append(Band(
            key=key, label=label, rows=members[:display_cap],
            open_by_default=key in BANDS_OPEN_BY_DEFAULT,
            truncated=max(0, len(members) - display_cap),
        ))

    # The overview names every band that has anyone in it, so a ward where
    # everybody is sick still reads as one, and an empty top band reads as empty.
    filled = [b for b in bands if b.rows or b.truncated]
    counts = ", ".join(f"{len(b.rows) + b.truncated} {b.label}" for b in filled) or "no stay scored"
    overview = f"{len(snapshots)} stays - {counts}"
    if unrankable:
        overview += f", {len(unrankable)} not scored"
    return Board(
        bands=tuple(bands), unrankable=tuple(unrankable), total_patients=len(snapshots),
        reference_upper=reference_upper, reference_lower=reference_lower,
        display_cap=display_cap,
        score_basis="last_creatinine_any_age", overview=overview,
        urine_reference_lower=urine_reference_lower,
        urine_reference_upper=urine_reference_upper,
        cutpoints=table,
    )


def board_to_dict(board: Board) -> dict[str, Any]:
    def row(item: Row) -> dict[str, Any]:
        return {
            "patientId": item.patient_id, "displayLabel": item.display_label,
            "rank": item.rank, "score": item.score, "scoreBasis": item.score_basis,
            "measurementAgeHours": item.measurement_age_hours, "reasons": list(item.reasons),
            "band": item.band, "bandLabel": item.band_label, "stale": item.stale,
            "rankable": item.rankable, "unrankableReason": item.unrankable_reason,
            # Reported alongside the row, and named so no reader of this payload
            # can mistake it for part of the ordering.
            "weightKg": item.weight_kg, "urineRatePerKg": item.urine_rate,
            "urineBand": item.urine_band, "urineBandLabel": item.urine_band_label,
            # What put the row in its band, worst first, and how many analytes
            # are out of range in total.
            "findings": [dict(f) for f in item.findings],
            "outsideCount": item.outside_count,
        }

    return {
        "schema": "oslian.ward-attention-order.v3",
        "overview": board.overview,
        "totalPatients": board.total_patients,
        "referenceUpper": board.reference_upper,
        "referenceLower": board.reference_lower,
        # The table the bands were read off, so a reader can check any cut and
        # see which ones this project can cite.
        "cutpoints": [{
            "variable": c.variable, "label": c.label, "unit": c.unit,
            "referenceLower": c.reference[0], "referenceUpper": c.reference[1],
            "firstCutLow": c.first_cut[0], "firstCutHigh": c.first_cut[1],
            "secondCutLow": c.second_cut[0], "secondCutHigh": c.second_cut[1],
            "source": c.source, "sourcePinned": c.source_pinned, "caveat": c.caveat,
        } for c in board.cutpoints.values()],
        # The band is read off published cutpoints across every measured analyte;
        # the order inside one is the creatinine ordering that was measured. The
        # payload says so rather than leaving a consumer to assume the whole
        # top-to-bottom order carries the validated claim.
        "bandBasis": "highest_tier_of_any_measured_analyte",
        "orderWithinBandBasis": "last_creatinine_any_age",
        "displayCap": board.display_cap,
        "staleAfterHours": STALE_AFTER_HOURS,
        "scoreBasis": board.score_basis,
        "urineReferenceLower": board.urine_reference_lower,
        "urineReferenceUpper": board.urine_reference_upper,
        # The urine annotation is not part of the order, and the payload says so
        # rather than leaving a consumer to infer it from the field names.
        "urineRateOrdersNothing": True,
        "urineReferenceSource": ("a range conventionally quoted for adults, not an "
                                 "interval printed by the laboratory the way the "
                                 "creatinine range is"),
        "bands": [{
            "key": b.key, "label": b.label,
            "openByDefault": b.open_by_default,
            "count": len(b.rows) + b.truncated, "truncated": b.truncated,
            "rows": [row(r) for r in b.rows],
        } for b in board.bands],
        "unrankable": [row(r) for r in board.unrankable],
        "causal_claim_allowed": False,
        "isClinicalAlert": False,
    }

"""MIMIC-IV target-trial emulation: first ICU crystalloid balanced (lactated Ringer's) versus 0.9% saline.

Active-comparator new-user design (Lund, Richardson & Stürmer 2015) emulating a
target trial (Hernán & Robins 2016).  Eligibility, assignment and the start of
follow-up coincide at time zero, and nothing recorded after time zero defines
who is in the trial, which arm they are in or their baseline.

- time zero: start of the first fluid-therapy crystalloid administration of
  the ICU stay: any bolus (category "03-IV Fluid Bolus", whatever its charted
  volume: that volume may be charted after the bolus was given), or an infusion
  whose row rate is >= 50 mL/h (an inputevents row has one rate, set at its
  start; mL/min and uL/kg/min rates are converted with the charted weight).
  No delivered amount is used: an infusion's amount depends on later rate
  changes and stops.  Time zero must fall in the first 24 h after ICU admission;
- new user: first recorded fluid-therapy crystalloid of the ICU stay (a stay
  whose first one started before ICU admission is excluded).  It is not
  crystalloid-naive: infusions below 50 mL/h (mostly keep-vein-open) started
  before time zero are allowed and adjusted for by kind; fluids given before
  ICU admission (ED, ward, operating room) are not recorded in inputevents.  A
  stay with an infusion of unconvertible rate at or before time zero is
  excluded;
- eligibility at time zero: adult, first ICU stay of the admission, a
  chemistry chloride sampled in the 24 h before time zero and resulted before
  time zero, no administration (of any rate) of the other fluid starting at
  time zero, and an admitting unit where
  both strategies are routine options.  Neuro units (saline by practice;
  lactated Ringer's is mildly hypotonic, and PLUS excluded traumatic brain
  injury or risk of cerebral oedema), the cardiac surgery unit (CVICU,
  protocolized lactated Ringer's, 89% balanced) and the coronary care unit
  (CCU, cardiology practice, 11% balanced) are excluded.  These restrictions
  were chosen from outcome-blind design diagnostics, in two rounds: neuro and
  CVICU under the first draft's time zero (overlap 0.87 -> 0.97, weighted
  |SMD| 0.22 -> 0.06), CCU after time zero was redefined (overlap 0.93 ->
  0.96; 35% of CCU stays had propensity < 0.05).  Crump trimming and a
  calendar restriction (estimated year >= 2014: overlap 0.99, half the stays)
  were also examined and not used;
- strategies: initiate with lactated Ringer's versus initiate with 0.9% saline
  (intention-to-treat analogue; later fluids are not constrained);
- baseline: low-rate infusions of each fluid started before time zero (started,
  not necessarily still running); labs sampled in the 24 h before time zero and resulted
  (storetime) before it; vitals charted in the 6 h before; vasopressor and
  invasive ventilation running at time zero; weight charted on the time-zero
  administration; estimated admission calendar year;
- outcomes (``--part outcomes``, only under a frozen protocol, once): chemistry
  chloride and bicarbonate nearest time zero + 24 h (18-30 h); the
  negative-control outcome, the last chloride sampled 24-48 h before time zero
  (disjoint from the baseline window, so never the same specimen); death and
  hospital discharge before time zero + 24 h, the intercurrent events of the
  hypothetical estimand (an outcome counts as observed only for a stay alive
  and in hospital at time zero + 24 h; a sample within 18-30 h stands for the
  24 h value).

Fluids: inputevents order categories "02-Fluids (Crystalloids)" and "03-IV Fluid
Bolus" (drug carriers and drip diluents are other categories).  These categories
also hold other items (dextrose, D5LR 225827, electrolyte boluses and additives,
piggybacks; the design report lists all of them); the comparators are NaCl 0.9%
(225158) and lactated Ringer's (225828) only, so time zero is the first
qualifying recorded LR or saline administration, whatever other fluids came first.
Ties are broken deterministically: baseline and negative-control values take
the latest charttime, then the largest labevent_id (labs) or value (vitals);
outcomes take the nearest charttime to 24 h, then the earlier charttime, then
the smaller labevent_id.  Every selected value keeps its provenance for audit
(``*_time``; labs also ``*_labevent_id``, ``*_storetime``, ``*_lab_hadm_id``,
``*_uom``), and each lab extraction reports, per item, the rows in its window,
the rows removed by the plausibility filter, the units seen and whether the
selected rows belong to the stay's admission.  Factual data; research-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd

from osler_jepa.frozen_protocol import claim_step, load_frozen, record_step, write_once


MIMIC = Path("/Users/chunyouchang/Downloads/mimic-iv-3.1")
FLUIDS = {225158: "saline", 225828: "balanced"}
FLUID_CATEGORIES = ("02-Fluids (Crystalloids)", "03-IV Fluid Bolus")
EXCLUDED_CAREUNITS = r"Neuro|CVICU|\(CCU\)"
MIN_INFUSION_RATE_ML_H = 50.0
INITIATION_WINDOW_H = 24
LAB_LOOKBACK_H = 24
VITAL_LOOKBACK_H = 6
OUTCOME_WINDOW_H = (18, 30)
NEGATIVE_CONTROL_WINDOW_H = (-48, -24)
INTERCURRENT_BOUNDARY_H = 24
# itemid: (name, plausible low, plausible high); implausible values are skipped before selection
BASELINE_LABS = {50902: ("chloride", 60, 150), 50983: ("sodium", 100, 190), 50882: ("bicarbonate", 2, 60),
                 50912: ("creatinine", 0.1, 30), 51006: ("bun", 1, 300), 50931: ("glucose", 10, 2000),
                 50971: ("potassium", 1, 12), 51265: ("platelets", 1, 2000), 51222: ("hemoglobin", 2, 25),
                 50813: ("lactate", 0.1, 40)}
VITALS = {220045: ("heart_rate", 20, 250), 220052: ("map", 20, 200), 220181: ("map", 20, 200), 220277: ("spo2", 50, 100),
          220210: ("resp_rate", 4, 70), 223761: ("temperature_f", 90, 110)}
VASOPRESSORS = (221906, 221289, 222315, 221749, 221662)
INVASIVE_VENTILATION = 225792
OUTCOME_LABS = {50902: BASELINE_LABS[50902], 50882: BASELINE_LABS[50882]}


def stays() -> pd.DataFrame:
    icu = pd.read_csv(MIMIC / "icu" / "icustays.csv.gz", parse_dates=["intime"])
    icu = icu.sort_values("intime").drop_duplicates("hadm_id")              # first ICU stay of the admission
    patients = pd.read_csv(MIMIC / "hosp" / "patients.csv.gz")
    admissions = pd.read_csv(MIMIC / "hosp" / "admissions.csv.gz", usecols=["hadm_id", "admission_type", "admission_location"])
    frame = icu.merge(patients, on="subject_id").merge(admissions, on="hadm_id", how="left")
    offset = frame["intime"].dt.year - frame["anchor_year"]
    frame["age"] = frame["anchor_age"] + offset
    # anchor_year_group is the patient's anchor year, not this admission's; shift by the admission's offset (about +-1 y)
    frame["estimated_year"] = frame["anchor_year_group"].str[:4].astype(int) + 1 + offset
    eligible = (frame["age"] >= 18) & ~frame["first_careunit"].str.contains(EXCLUDED_CAREUNITS)
    return frame[eligible].reset_index(drop=True)


def crystalloid_starts(c: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Start, kind, category, start rate (mL/h) and charted weight of every crystalloid administration."""

    categories = ",".join(f"'{name}'" for name in FLUID_CATEGORIES)
    frame = c.sql(f"""
        SELECT stay_id, orderid, starttime AS start, itemid, patientweight AS weight, ordercategoryname = '03-IV Fluid Bolus' AS bolus,
               CASE rateuom WHEN 'mL/hour' THEN rate WHEN 'mL/min' THEN rate * 60
                            WHEN 'µL/kg/min' THEN rate * patientweight * 60 / 1000 END AS rate_ml_h
        FROM read_csv_auto('{MIMIC / "icu" / "inputevents.csv.gz"}')
        WHERE itemid IN ({",".join(str(i) for i in FLUIDS)}) AND ordercategoryname IN ({categories})
          AND ordercomponenttypedescription = 'Main order parameter' AND statusdescription <> 'Rewritten'
    """).df()
    frame["kind"] = frame["itemid"].map(FLUIDS)
    return frame.drop(columns="itemid")


def fluid_audit(c: duckdb.DuckDBPyConnection) -> list[dict]:
    """Every item and rate unit in the crystalloid categories, so the comparator definition and rate conversion can be checked."""

    categories = ",".join(f"'{name}'" for name in FLUID_CATEGORIES)
    rows = c.sql(f"""
        SELECT e.itemid, d.label, e.ordercategoryname, e.ordercomponenttypedescription, e.rateuom, count(*) AS rows
        FROM read_csv_auto('{MIMIC / "icu" / "inputevents.csv.gz"}') e
        LEFT JOIN read_csv_auto('{MIMIC / "icu" / "d_items.csv.gz"}') d ON e.itemid = d.itemid
        WHERE e.ordercategoryname IN ({categories})
        GROUP BY ALL ORDER BY rows DESC
    """).df()
    return rows.to_dict("records")


def time_zero(cohort: pd.DataFrame, starts: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """First fluid-therapy start of the stay, its kind, and the low-rate infusions of each kind started before it.

    Excluded: any administration of the other kind starting at time zero, an
    infusion of unknown rate starting at or before time zero, and a first
    fluid-therapy start outside [ICU admission, + 24 h).
    """

    starts = starts.merge(cohort[["stay_id", "intime"]], on="stay_id")
    starts["fluid_therapy"] = starts["bolus"] | (starts["rate_ml_h"] >= MIN_INFUSION_RATE_ML_H)
    therapy = starts[starts["fluid_therapy"]].sort_values(["stay_id", "start", "orderid"])
    first = therapy.drop_duplicates("stay_id")[["stay_id", "start", "kind", "weight", "intime"]].rename(columns={"start": "t0"})
    upto = starts.merge(first[["stay_id", "t0"]], on="stay_id")
    upto = upto[upto["start"] <= upto["t0"]]
    mixed = upto[upto["start"] == upto["t0"]].groupby("stay_id")["kind"].nunique() > 1
    unknown = upto.loc[~upto["bolus"] & upto["rate_ml_h"].isna(), "stay_id"].unique()
    first = first[~first["stay_id"].isin(mixed.index[mixed]) & ~first["stay_id"].isin(unknown)]
    first = first.assign(hours_icu_to_t0=(first["t0"] - first["intime"]).dt.total_seconds() / 3600)
    in_window = first["hours_icu_to_t0"].between(0, INITIATION_WINDOW_H, inclusive="left")
    low_rate = upto[(upto["start"] < upto["t0"]) & ~upto["bolus"] & (upto["rate_ml_h"] < MIN_INFUSION_RATE_ML_H)]
    frame = first[in_window].drop(columns="intime")
    for kind in ("balanced", "saline"):
        frame[f"prior_low_rate_{kind}_infusion"] = frame["stay_id"].isin(low_rate.loc[low_rate["kind"] == kind, "stay_id"]).astype(int)
    counts = {"stays_with_fluid_therapy": int(therapy["stay_id"].nunique()), "mixed_start_excluded": int(mixed.sum()),
              "unknown_rate_excluded": int(len(unknown)), "first_start_outside_0_24h_excluded": int((~in_window).sum())}
    return frame.reset_index(drop=True), counts


def with_provenance(rows: pd.DataFrame, items: dict, stays: pd.Series) -> pd.DataFrame:
    """One row per stay: ``<name>`` values and ``<name>_<field>`` provenance of the selected row.

    The schema is fixed by ``items`` and ``stays``: an item with no selected row
    still gets its (all-missing) columns, so a downstream analysis sees an
    unavailable endpoint as missing values, never as an absent column.
    """

    names = sorted({name for name, _, _ in items.values()})
    fields = [column for column in rows.columns if column not in ("stay_id", "name", "value")]
    index = pd.Index(stays.unique(), name="stay_id")
    wide = rows.pivot(index="stay_id", columns="name", values="value").reindex(index=index, columns=names)
    for field in fields:
        wide = wide.join(rows.pivot(index="stay_id", columns="name", values=field).reindex(index=index, columns=names).add_suffix(f"_{field}"))
    return wide


def _names(c: duckdb.DuckDBPyConnection, items: dict[int, tuple[str, float, float]]) -> None:
    c.register("names", pd.DataFrame([(i, n, lo, hi) for i, (n, lo, hi) in items.items()], columns=["itemid", "name", "low", "high"]))


LAB_FIELDS = "e.charttime AS time, e.labevent_id, e.storetime, e.hadm_id AS lab_hadm_id, e.valueuom AS uom"


def lab_audit(c: duckdb.DuckDBPyConnection, frame: pd.DataFrame, items: dict, window_h: tuple[int, int], *,
              resulted_before_t0: bool, selected: pd.DataFrame, closed: bool = False) -> dict:
    """Per item: rows in the window, rows removed as implausible, units seen, admission match of the selected rows."""

    c.register("anchors", frame[["stay_id", "subject_id", "t0"]])
    _names(c, items)
    resulted = "AND e.storetime < a.t0" if resulted_before_t0 else ""
    rows = c.sql(f"""
        SELECT n.name, count(*) AS rows_in_window, count(*) FILTER (WHERE NOT (e.valuenum BETWEEN n.low AND n.high)) AS implausible,
               count(*) FILTER (WHERE e.valuenum IS NULL) AS non_numeric, list(DISTINCT e.valueuom) AS units
        FROM read_csv_auto('{MIMIC / "hosp" / "labevents.csv.gz"}') e
        JOIN names n ON e.itemid = n.itemid
        JOIN anchors a ON e.subject_id = a.subject_id
        WHERE e.charttime >= a.t0 + INTERVAL ({int(window_h[0])}) HOUR
          AND e.charttime {"<=" if closed else "<"} a.t0 + INTERVAL ({int(window_h[1])}) HOUR
          {resulted}
        GROUP BY n.name
    """).df().set_index("name")
    admission = frame.set_index("stay_id")["hadm_id"]
    audit = {}
    for name in rows.index:
        chosen = selected[f"{name}_lab_hadm_id"].dropna() if f"{name}_lab_hadm_id" in selected else pd.Series(dtype=float)
        has_value = selected[name].notna() if name in selected else pd.Series(dtype=bool)
        audit[name] = {**{k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in rows.loc[name].items()},
                       "selected": int(has_value.sum()),
                       "selected_same_admission": int((chosen == admission.reindex(chosen.index)).sum()),
                       "selected_without_admission": int(has_value.sum() - len(chosen))}
    return audit


def last_before(c: duckdb.DuckDBPyConnection, frame: pd.DataFrame, items: dict, window_h: tuple[int, int], *,
                vitals: bool, resulted_before_t0: bool) -> pd.DataFrame:
    """Last plausible value with ``t0 + window_h[0] <= charttime < t0 + window_h[1]``, with provenance.

    Labs are matched by subject (emergency labs lack hadm_id) and, when
    ``resulted_before_t0``, must also have been resulted (storetime) before time zero.
    """

    c.register("anchors", frame[["stay_id", "subject_id", "t0"]])
    _names(c, items)
    source, key, order, fields = ((MIMIC / "icu" / "chartevents.csv.gz", "stay_id", "e.valuenum DESC", "e.charttime AS time") if vitals
                                  else (MIMIC / "hosp" / "labevents.csv.gz", "subject_id", "e.labevent_id DESC", LAB_FIELDS))
    resulted = "AND e.storetime < a.t0" if resulted_before_t0 else ""
    rows = c.sql(f"""
        SELECT a.stay_id, n.name, e.valuenum AS value, {fields}
        FROM read_csv_auto('{source}') e
        JOIN names n ON e.itemid = n.itemid
        JOIN anchors a ON e.{key} = a.{key}
        WHERE e.valuenum BETWEEN n.low AND n.high
          AND e.charttime >= a.t0 + INTERVAL ({int(window_h[0])}) HOUR AND e.charttime < a.t0 + INTERVAL ({int(window_h[1])}) HOUR
          {resulted}
        QUALIFY row_number() OVER (PARTITION BY a.stay_id, n.name ORDER BY e.charttime DESC, {order}) = 1
    """).df()
    return with_provenance(rows, items, frame["stay_id"])


def nearest(c: duckdb.DuckDBPyConnection, frame: pd.DataFrame, items: dict, window_h: tuple[int, int], target_h: int) -> pd.DataFrame:
    """Plausible lab value nearest ``t0 + target_h`` inside the closed window (ties: earlier sample), with provenance."""

    c.register("anchors", frame[["stay_id", "subject_id", "t0"]])
    _names(c, items)
    rows = c.sql(f"""
        SELECT a.stay_id, n.name, e.valuenum AS value, {LAB_FIELDS}
        FROM read_csv_auto('{MIMIC / "hosp" / "labevents.csv.gz"}') e
        JOIN names n ON e.itemid = n.itemid
        JOIN anchors a ON e.subject_id = a.subject_id
        WHERE e.valuenum BETWEEN n.low AND n.high
          AND e.charttime >= a.t0 + INTERVAL ({int(window_h[0])}) HOUR AND e.charttime <= a.t0 + INTERVAL ({int(window_h[1])}) HOUR
        QUALIFY row_number() OVER (PARTITION BY a.stay_id, n.name
                                   ORDER BY abs(epoch(e.charttime) - epoch(a.t0 + INTERVAL ({int(target_h)}) HOUR)), e.charttime, e.labevent_id) = 1
    """).df()
    return with_provenance(rows, items, frame["stay_id"])


def running_at_t0(c: duckdb.DuckDBPyConnection, frame: pd.DataFrame, table: str, items: tuple[int, ...]) -> pd.Series:
    c.register("anchors", frame[["stay_id", "t0"]])
    rows = c.sql(f"""
        SELECT DISTINCT a.stay_id FROM read_csv_auto('{MIMIC / "icu" / table}') e
        JOIN anchors a ON e.stay_id = a.stay_id
        WHERE e.itemid IN ({",".join(str(i) for i in items)}) AND e.starttime < a.t0 AND e.endtime >= a.t0
    """).df()
    return frame["stay_id"].isin(rows["stay_id"]).astype(int)


def design(c: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, dict]:
    cohort = stays()
    initiated, counts = time_zero(cohort, crystalloid_starts(c))
    frame = cohort.merge(initiated, on="stay_id")
    labs = last_before(c, frame, BASELINE_LABS, (-LAB_LOOKBACK_H, 0), vitals=False, resulted_before_t0=True)
    baseline_audit = lab_audit(c, frame, BASELINE_LABS, (-LAB_LOOKBACK_H, 0), resulted_before_t0=True, selected=labs)
    vitals = last_before(c, frame, VITALS, (-VITAL_LOOKBACK_H, 0), vitals=True, resulted_before_t0=False)
    frame = frame.merge(labs.add_prefix("baseline_").reset_index(), on="stay_id", how="left")
    frame = frame.merge(vitals.add_prefix("baseline_").reset_index(), on="stay_id", how="left")
    frame["vasopressor_at_t0"] = running_at_t0(c, frame, "inputevents.csv.gz", VASOPRESSORS)
    frame["invasive_ventilation_at_t0"] = running_at_t0(c, frame, "procedureevents.csv.gz", (INVASIVE_VENTILATION,))
    counts["initiators_adult_first_stay_eligible_unit"] = int(len(frame))
    frame = frame[frame["baseline_chloride"].notna()].copy()
    frame["treated_balanced"] = (frame["kind"] == "balanced").astype(int)
    frame = frame.rename(columns={"stay_id": "stay_key", "first_careunit": "careunit", "weight": "t0_weight"})
    report = {**counts, "stays": int(len(frame)), "subjects": int(frame["subject_id"].nunique()),
              "treated_balanced_share": round(float(frame["treated_balanced"].mean()), 4),
              "baseline_lab_audit": baseline_audit, "fluid_category_audit": fluid_audit(c), "outcomes_included": False}
    keep = ["stay_key", "subject_id", "hadm_id", "t0", "treated_balanced", "hours_icu_to_t0", "prior_low_rate_balanced_infusion",
            "prior_low_rate_saline_infusion", "age", "gender", "careunit",
            "admission_type", "admission_location", "estimated_year", "t0_weight", "vasopressor_at_t0",
            "invasive_ventilation_at_t0", *sorted(col for col in frame.columns if col.startswith("baseline_"))]
    return frame[keep].reset_index(drop=True), report


def outcomes(c: duckdb.DuckDBPyConnection, design_frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    frame = design_frame.rename(columns={"stay_key": "stay_id"})
    later = nearest(c, frame, OUTCOME_LABS, OUTCOME_WINDOW_H, 24)
    earlier = last_before(c, frame, {50902: BASELINE_LABS[50902]}, NEGATIVE_CONTROL_WINDOW_H, vitals=False, resulted_before_t0=False)
    audit = {"day1": lab_audit(c, frame, OUTCOME_LABS, OUTCOME_WINDOW_H, resulted_before_t0=False, selected=later, closed=True),
             "negative_control": lab_audit(c, frame, {50902: BASELINE_LABS[50902]}, NEGATIVE_CONTROL_WINDOW_H,
                                           resulted_before_t0=False, selected=earlier)}
    admissions = pd.read_csv(MIMIC / "hosp" / "admissions.csv.gz", usecols=["hadm_id", "dischtime", "deathtime"],
                             parse_dates=["dischtime", "deathtime"])
    ends = frame[["stay_id", "hadm_id", "t0"]].merge(admissions, on="hadm_id", how="left").set_index("stay_id")
    boundary = ends["t0"] + pd.Timedelta(hours=INTERCURRENT_BOUNDARY_H)
    events = pd.DataFrame({"fu_died_before_24h": (ends["deathtime"] < boundary).astype(int),
                           "fu_discharged_alive_before_24h": ((ends["dischtime"] < boundary) & ends["deathtime"].isna()).astype(int)})
    table = events.join(later.add_prefix("day1_")).join(earlier.add_prefix("nc_prior_"))
    return table.reset_index().rename(columns={"stay_id": "stay_key"}), audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", choices=("design", "outcomes"), required=True)
    parser.add_argument("--design", type=Path, default=Path("mimic_crystalloid_tte_design.parquet"), help="design part only")
    parser.add_argument("--protocol", type=Path, help="frozen protocol; the outcome part reads its spec and design table from it")
    args = parser.parse_args()
    c = duckdb.connect(config={"memory_limit": "8GB", "threads": 4})
    if args.part == "design":
        frame, report = design(c)
        frame.to_parquet(args.design, index=False)
        args.design.with_suffix(".json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(json.dumps({k: v for k, v in report.items() if k != "fluid_category_audit"}))
        return
    if args.protocol is None:
        raise ValueError("outcomes are extracted only under a frozen protocol")
    protocol, spec = load_frozen(args.protocol)
    output = write_once(Path(spec["outcome_table"]))
    audit_path = write_once(output.with_suffix(".audit.json"))
    claim_step(args.protocol, "extract_outcomes")
    table, audit = outcomes(c, pd.read_parquet(spec["design_table"]))
    table.to_parquet(output, index=False)
    audit_path.write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")
    record = record_step(args.protocol, "extract_outcomes", [protocol["spec"], spec["design_table"]], [str(output), str(audit_path)])
    print(json.dumps(record))


if __name__ == "__main__":
    main()

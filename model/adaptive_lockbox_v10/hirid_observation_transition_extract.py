"""Build HiRID 1.1.1 observation transitions in the MIMIC-IV v5 contract (lockbox, protocol v10).

Source: HiRID 1.1.1 (PhysioNet, credentialed; Bern University Hospital, one
department of intensive care medicine): ``reference/general_table.csv``, the
raw-stage observation tables (``raw_stage/observation_tables/parquet/part-*.parquet``)
and the dictionary ``reference/hirid_variable_reference.csv``.  Anchors, as-of
state, labels (with their offsets) and plausibility cleaning reuse
``mimiciv_observation_transition_extract``; simultaneous-value averaging and
calculated bicarbonate reuse ``sicdb_observation_transition_extract``, so the
code path matches the other lockbox.

Variable mapping is declared from the data dictionary only (Variable Name, Unit
in ``hirid_variable_reference.csv``), following the harmonized definitions of
protocol v10:

- ICU stay = one ``patientid``; HiRID gives one id per admission and no link
  between admissions, so subject = stay: patient clustering and "patient-equal"
  coverage operate on admissions, and repeated admissions of one person cannot
  be accounted for; in-time = ``admissiontime``; HiRID has no discharge time, so
  out-time = the last valid heart rate (selected and validated as every other
  measurement), a monitored-window proxy that is not shown to equal discharge;
  admissions without a valid heart rate are excluded and counted;
- care unit = the single department (``"ICU"``): the care-unit axis of the
  interval gate has one group, so its coverage equals overall coverage and
  says nothing about robustness across units;
- observations at their ``datetime`` (when the observation was made; MIMIC uses
  chart time the same way; ``entertime`` only orders versions).  Of several
  entries for one item and time the last entered is kept (``entertime``, then
  preliminary < final < correction); if that version is flagged invalidated
  (status 2), notified-not-measured (32) or censored (64 "bigger than", 128
  "smaller than") or has no value, the measurement is withdrawn rather than
  falling back to an older entry.  Equally ranked versions of which any is
  withdrawn withdraw the measurement; otherwise the smaller value is kept.  This is a final-value,
  observation-time reconstruction: a value entered or corrected after an anchor
  can be part of that anchor's state, and the replay deadline anchor + h +
  |offset| does not wait for later entries, so prospective availability is not
  established;
- laboratory items from the central laboratory and blood-gas analyzers are
  pooled per analyte as MIMIC-IV pools chemistry and blood-gas items; arterial
  and venous lactate are pooled as MIMIC's lactate items are; pH, PaCO2, PaO2
  and bicarbonate are the arterial items;
- glucose is NOT mapped: the dictionary names all three glucose items
  "Glucose [Moles/volume] in Serum or Plasma" and does not distinguish
  chemistry, point-of-care and blood-gas measurement, so under protocol v10
  ``glucose`` and ``glucose_poc`` cells are not evaluable (their columns stay
  empty) and ``glucose_bg`` is not mapped;
- units to MIMIC-IV: creatinine umol/L / 88.42; urea mmol/L x 2.8013 -> BUN
  mg/dL; bilirubin umol/L / 17.104; calcium mmol/L x 4.008; magnesium x 2.431;
  phosphate x 3.097; haemoglobin and albumin g/L / 10; fibrinogen g/L x 100;
  troponin T ng/L x 0.001; cortisol nmol/L / 27.59; other mapped units already
  match (G/l = 10^9/L = K/uL; CRP mg/L; aPTT s);
- MAP = the measured invasive or non-invasive mean only (protocol v10); urine
  output = "Hourly urine volume" (mL/h) at its charted time (which end of the
  collection hour that time marks is not documented); temperature pools core,
  rectal and axillary (degrees C) as MIMIC's temperature items do; PEEP pools
  the setting and the measured value as MIMIC's PEEP items do (sites and set
  versus measured PEEP are not shown to be interchangeable);
- measured values are validated (``clean_measurements``) first; simultaneous
  values of one variable from several items are averaged (row-order invariant;
  they need not come from one specimen); then calculated bicarbonate from
  same-time validated pH and PaCO2 where no validated bicarbonate exists
  (same-time pH and PaCO2 are not proven to come from one sample), validated
  again;
- memory: the raw stage has about one observation per two minutes, so stays are
  processed in groups of observation-table parts (``observation_tables_index.csv``;
  every stay's history lies in its indexed part) and each group keeps only the
  rows ``assemble_state`` can select (for every anchor and label time, the last
  measurement at or before it and the first at or after it, per stay and
  variable, after collapsing): the transitions are identical to assembling from
  every row; ``measurement_rows_by_var`` counts rows before this pruning;
- not mapped: CK-MB (U/L activity, MIMIC ng/mL mass), NT-proBNP (MIMIC maps
  BNP), haematocrit and minute volume (no dictionary item), pharma records
  (treatment context is not extracted; ``osler_jepa.recipe_availability``
  falls treatment-defined recipes back to measurement-pure ones).

Disclosed differences: a nearest label up to ``min(2, h/2)`` h before anchor + h
is resolved at anchor + h + |offset| (a conservative deadline, not its
measurement time); the frozen replay's random stay arrivals, label-selected
stream and forecast-order feedback (protocol v10, ``replay_properties_disclosed``)
apply unchanged.

Before the protocol-v10 addendum (``lockbox_addendum.py``) only the support
counts in the report may be inspected, never target values.  Factual
observation data only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd

import mimiciv_observation_transition_extract as contract
from sicdb_observation_transition_extract import add_calculated_bicarbonate, collapse


# variableid -> (MIMIC variable, multiplicative factor to MIMIC-IV units); names and units from the dictionary
ITEMS: dict[int, tuple[str, float]] = {
    200: ("heart_rate", 1.0),                                                   # Heart rate /min
    410: ("temperature", 1.0), 7100: ("temperature", 1.0), 400: ("temperature", 1.0),   # core, rectal, axillary, degrees C
    110: ("map", 1.0), 610: ("map", 1.0),                                       # invasive, non-invasive mean arterial pressure mmHg
    4000: ("o2sat", 1.0), 8280: ("o2sat", 1.0),                                 # Peripheral oxygen saturation %
    300: ("respiratory_rate", 1.0), 310: ("respiratory_rate", 1.0), 5685: ("respiratory_rate", 1.0),
    2010: ("fio2", 1.0),                                                        # Inspired oxygen concentration %
    2600: ("peep", 1.0), 2610: ("peep", 1.0),                                   # PEEP setting, PEEP cmH2O
    10020000: ("urine_output", 1.0),                                            # Hourly urine volume ml/h
    20000300: ("ph", 1.0), 20001200: ("paco2", 1.0), 20000200: ("pao2", 1.0),   # arterial blood
    20004200: ("bicarbonate", 1.0),                                             # Bicarbonate in arterial blood mmol/l
    24000524: ("lactate", 1.0), 24000732: ("lactate", 1.0), 24000485: ("lactate", 1.0),   # arterial, venous mmol/l
    20000500: ("potassium", 1.0), 24000520: ("potassium", 1.0), 24000833: ("potassium", 1.0), 24000867: ("potassium", 1.0),
    20000400: ("sodium", 1.0), 24000519: ("sodium", 1.0), 24000658: ("sodium", 1.0), 24000835: ("sodium", 1.0),
    24000866: ("sodium", 1.0),
    24000439: ("chloride", 1.0), 24000521: ("chloride", 1.0),
    24000522: ("ionized_calcium", 1.0),                                         # mmol/l
    20005100: ("calcium", 4.008),                                               # Calcium in blood mmol/l -> mg/dL
    20002500: ("phosphate", 3.097), 24000230: ("magnesium", 2.431),             # mmol/l -> mg/dL
    20004100: ("bun", 2.8013),                                                  # Urea mmol/l -> urea nitrogen mg/dL
    20000600: ("creatinine", 1.0 / 88.42),                                      # umol/l -> mg/dL
    20000900: ("hemoglobin", 0.1), 24000836: ("hemoglobin", 0.1), 24000548: ("hemoglobin", 0.1),   # g/l -> g/dL
    20000700: ("wbc", 1.0), 20000110: ("platelets", 1.0),                       # G/l = K/uL
    20004300: ("bilirubin", 1.0 / 17.104), 24000560: ("bilirubin_direct", 1.0 / 17.104),           # umol/l -> mg/dL
    24000330: ("ast", 1.0), 20002600: ("alt", 1.0), 20002700: ("alkaline_phos", 1.0),            # U/l
    24000427: ("amylase", 1.0), 24000555: ("lipase", 1.0), 24000210: ("cpk", 1.0),               # U/l
    20004410: ("ptt", 1.0), 24000567: ("inr", 1.0),                             # aPTT s; INR
    24000536: ("fibrinogen", 100.0), 24000605: ("albumin", 0.1),                # g/L -> mg/dL, g/dL
    24000538: ("troponin_t", 0.001), 24000806: ("troponin_t", 0.001),           # ng/l -> ng/mL
    20002200: ("crp", 1.0), 24000678: ("ferritin", 1.0), 24000664: ("tsh", 1.0),   # mg/l, ug/l, mU/l
    24000665: ("cortisol", 1.0 / 27.59), 24000668: ("esr", 1.0),                # nmol/l -> ug/dL; mm/h
}
HEART_RATE = 200
PARTS_PER_GROUP = 10
DROPPED_STATUS = 2 | 32 | 64 | 128         # invalidated, notified-not-measured, censored above / below
TYPE_ORDER = "CASE type WHEN 'C' THEN 2 WHEN 'F' THEN 1 ELSE 0 END"
MEASUREMENT_COLUMNS = ["subject_id", "stay_id", "charttime", "var", "valuenum"]
CAREUNIT = "ICU"


def con() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(config={"memory_limit": "6GB", "threads": 4})


def observations(root: Path, parts: list[int] | None = None) -> str:
    folder = root / "raw_stage" / "observation_tables" / "parquet"
    files = [str(folder / "part-*.parquet")] if parts is None else [str(folder / f"part-{part}.parquet") for part in parts]
    return f"read_parquet({files})"


def current_values(root: Path, variables: str, parts: list[int] | None = None) -> str:
    """SQL for the last entered version per patient, item and time, withdrawn if that version is flagged or empty
    (an equally ranked flagged or empty version withdraws it too)."""

    return f"""
        SELECT patientid, variableid, datetime, value FROM (
            SELECT patientid, variableid, datetime, value,
                   value IS NULL OR (status & {DROPPED_STATUS}) <> 0 AS withdrawn
            FROM {observations(root, parts)}
            WHERE variableid IN ({variables})
            QUALIFY row_number() OVER (PARTITION BY patientid, variableid, datetime
                                       ORDER BY entertime DESC, {TYPE_ORDER} DESC, withdrawn DESC, value) = 1)
        WHERE NOT withdrawn"""


def load_stays(c: duckdb.DuckDBPyConnection, root: Path, horizon_hours: float) -> tuple[pd.DataFrame, dict]:
    """One stay per ``patientid``: admission time to the last valid heart rate (selected and validated as any measurement)."""

    general = pd.read_csv(root / "reference" / "general_table.csv", usecols=["patientid", "admissiontime"],
                          parse_dates=["admissiontime"])
    c.register("general", general)
    heart = c.sql(f"""
        SELECT v.patientid AS stay_id, CAST(v.patientid AS VARCHAR) AS subject_id, v.datetime AS charttime,
               'heart_rate' AS var, v.value AS valuenum
        FROM ({current_values(root, str(HEART_RATE))}) v JOIN general g USING (patientid)
        WHERE v.datetime >= g.admissiontime""").df()
    last = contract.clean_measurements(heart).groupby("stay_id")["charttime"].max().rename("outtime")
    stays = general.merge(last, left_on="patientid", right_index=True)
    counts = {"patients": int(len(general)), "patients_without_valid_heart_rate": int(len(general) - len(stays))}
    stays = stays[stays["outtime"] > stays["admissiontime"] + pd.Timedelta(hours=horizon_hours)]
    stays = stays.sort_values("patientid").reset_index(drop=True)
    return pd.DataFrame({
        "subject_id": stays["patientid"].astype(str),
        "stay_id": stays["patientid"].astype("int64"),
        "first_careunit": CAREUNIT,
        "last_careunit": CAREUNIT,
        "intime": stays["admissiontime"].astype("datetime64[ns]"),
        "outtime": stays["outtime"].astype("datetime64[ns]"),
    }), counts


def load_measurements(c: duckdb.DuckDBPyConnection, root: Path, stays: pd.DataFrame, parts: list[int] | None = None) -> pd.DataFrame:
    """Mapped current values inside the stay, converted to MIMIC-IV units."""

    c.register("items", pd.DataFrame([(k, v[0], v[1]) for k, v in ITEMS.items()], columns=["variableid", "var", "factor"]))
    c.register("stays", stays[["subject_id", "stay_id", "intime", "outtime"]])
    rows = c.sql(f"""
        SELECT s.subject_id, s.stay_id, v.datetime AS charttime, i.var, v.value * i.factor AS valuenum
        FROM ({current_values(root, "SELECT variableid FROM items", parts)}) v
        JOIN stays s ON v.patientid = s.stay_id JOIN items i USING (variableid)
        WHERE v.datetime >= s.intime AND v.datetime <= s.outtime
    """).df()
    return rows[MEASUREMENT_COLUMNS]


def measurements_for(c: duckdb.DuckDBPyConnection, root: Path, stays: pd.DataFrame, parts: list[int] | None = None) -> pd.DataFrame:
    """Validate measured inputs, collapse simultaneous values, then derive (validated) bicarbonate."""

    measured = collapse(contract.clean_measurements(load_measurements(c, root, stays, parts)))
    return collapse(add_calculated_bicarbonate(measured)).sort_values(["stay_id", "charttime"]).reset_index(drop=True)


def selectable(measurements: pd.DataFrame, anchors: pd.DataFrame) -> pd.DataFrame:
    """The rows ``assemble_state`` can select: per stay and variable, the last at or before and the first at or after
    every anchor and label time (one row per stay, variable and time after ``collapse``)."""

    if measurements.empty or anchors.empty:
        return measurements.iloc[0:0]
    queries = pd.concat([anchors[["stay_id", "t"]].rename(columns={"t": "q"}),
                         anchors[["stay_id", "t_plus"]].rename(columns={"t_plus": "q"})], ignore_index=True)
    queries = queries.astype({"q": "datetime64[ns]"}).sort_values("q")
    keys = []
    for var, values in measurements.groupby("var", sort=True):
        times = values[["stay_id", "charttime"]].astype({"charttime": "datetime64[ns]"}).sort_values("charttime")
        times = times.assign(chosen=times["charttime"])
        for direction in ("backward", "forward"):
            hit = pd.merge_asof(queries, times, left_on="q", right_on="charttime", by="stay_id", direction=direction)
            keys.append(hit.dropna(subset=["chosen"])[["stay_id", "chosen"]].assign(var=var))
    keys = pd.concat(keys, ignore_index=True).drop_duplicates().rename(columns={"chosen": "charttime"})
    return measurements.astype({"charttime": "datetime64[ns]"}).merge(keys, on=["stay_id", "var", "charttime"])


def build(root: Path, horizon_hours: float, anchor_step_hours: float) -> tuple[pd.DataFrame, dict]:
    contract.DELTA_H = float(horizon_hours)
    contract.ANCHOR_STEP_H = float(anchor_step_hours)
    contract.TARGET_TOL_H = float(min(2.0, horizon_hours / 2.0))
    suffix = f"_tp{int(horizon_hours)}" if float(horizon_hours).is_integer() else f"_tp{horizon_hours:g}"
    c = con()
    stays, stay_counts = load_stays(c, root, horizon_hours)
    anchors = contract.build_anchors(stays)
    part_of = pd.read_csv(root / "raw_stage" / "observation_tables" / "observation_tables_index.csv").set_index("patientid")["part"]
    stay_parts = stays["stay_id"].map(part_of)
    if stay_parts.isna().any():
        raise ValueError("stays missing from observation_tables_index.csv")
    parts = sorted(int(part) for part in stay_parts.unique())
    pieces, rows_by_var = [], pd.Series(dtype="int64")
    for start in range(0, len(parts), PARTS_PER_GROUP):
        group = parts[start:start + PARTS_PER_GROUP]
        batch = stays[stay_parts.isin(group).to_numpy()]
        measured = measurements_for(c, root, batch, group)
        rows_by_var = rows_by_var.add(measured["var"].value_counts(), fill_value=0)
        pieces.append(selectable(measured, anchors[anchors["stay_id"].isin(batch["stay_id"])]))
    pieces = [piece for piece in pieces if not piece.empty]
    measurements = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=MEASUREMENT_COLUMNS)
    state_t = contract.assemble_state(anchors, measurements, "_t", "t", "backward", contract.LOOKBACK_H)
    state_f = contract.assemble_state(anchors, measurements, suffix, "t_plus", "nearest", contract.TARGET_TOL_H)
    frame = pd.concat(
        [anchors[["subject_id", "stay_id", "first_careunit", "last_careunit", "t", "t_plus"]],
         state_t.drop(columns=["stay_id", "t"]), state_f.drop(columns=["stay_id", "t_plus"])],
        axis=1,
    )
    # Every anchor is kept, labelled or not (see mimiciv_observation_transition_extract).
    frame = frame.reset_index(drop=True)
    frame["hospitalid"] = "HiRID"
    report = {
        "artifact": "HiRID 1.1.1 ICU observation transitions (MIMIC-IV v5 contract; lockbox under protocol v10)",
        "delta_h": float(horizon_hours),
        "anchor_step_h": float(anchor_step_hours),
        "lookback_h": contract.LOOKBACK_H,
        "target_tolerance_h": contract.TARGET_TOL_H,
        "episode_max_h": contract.EPISODE_MAX_H,
        "stay_counts": stay_counts,
        "selected_stays": int(len(stays)),
        "subjects": int(frame["subject_id"].nunique()),
        "stays": int(frame["stay_id"].nunique()),
        "rows": int(len(frame)),
        "careunits": sorted(frame["first_careunit"].unique().tolist()),
        "careunit_definition": "one department of intensive care medicine (single group)",
        "stay_end_definition": "last valid heart rate, a monitored-window proxy (HiRID has no discharge time)",
        "subject_definition": "admission (patientid); no readmission link",
        "measurement_rows_by_var": {str(k): int(v) for k, v in rows_by_var.sort_values(ascending=False).items()},
        "pair_counts": contract.pair_counts(frame, suffix),
        "map_semantics": "validated measured invasive or non-invasive mean only",
        "dropped_status_bits": DROPPED_STATUS,
        "measurement_rows_by_var_note": "before pruning to selectable rows; includes selected stays without anchors",
        "simultaneous_values": "last entered version per item and time (entertime, then C > F > P), withdrawn if it or an "
                               "equally ranked version is flagged or empty, else the smaller value; then averaged per "
                               "stay, variable and time after validation",
        "availability": "final-value, observation-time reconstruction; prospective availability not established",
        "glucose_mapped": False,
        "glucose_not_mapped_reason": "the dictionary does not distinguish chemistry, point-of-care and blood-gas glucose (protocol v10)",
        "glucose_poc_available": False,
        "treatment_context_available": False,
        "patient_ids_included_in_report": False,
        "causal_claim_allowed": False,
    }
    return frame, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="HiRID 1.1.1 directory (reference/, raw_stage/)")
    parser.add_argument("--horizon-hours", type=float, required=True)
    parser.add_argument("--anchor-step-hours", type=float, default=6.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    frame, report = build(args.root, args.horizon_hours, args.anchor_step_hours)
    frame.to_parquet(args.output, index=False)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("rows", "subjects", "stays")}))


if __name__ == "__main__":
    main()

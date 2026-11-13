"""Explicit, versioned FHIR -> WholeBody variable mapping registry.

Rules are keyed on (code system, code, resource shape, input UCUM unit). Display
text is never used for matching. Every conversion is a linear
``canonical = value * factor + offset`` so it can be unit-tested exactly.

LOINC long common names were checked against the NLM Clinical Table Search
Service (``clinicaltables.nlm.nih.gov/api/loinc_items``) and the FHIR R4
vital-signs profile on 2026-09-01; they are recorded for humans only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

MAPPING_VERSION = "oslian.fhir-mapping.v2"
LOINC = "http://loinc.org"
UCUM = "http://unitsofmeasure.org"

SHAPE_QUANTITY = "valueQuantity"
SHAPE_PERIOD_VOLUME = "valueQuantity+effectivePeriod"
SHAPE_BP_PANEL = "component:8480-6+8462-4"

# Canonical WholeBody units (must match the release contract VARIABLE_UNITS).
CANONICAL_UNITS: dict[str, str] = {
    "albumin": "g/dL", "alkaline_phos": "U/L", "alt": "U/L", "amylase": "U/L",
    "anion_gap": "mEq/L", "ast": "U/L", "bicarbonate": "mEq/L", "bilirubin": "mg/dL",
    "bilirubin_direct": "mg/dL", "bnp": "pg/mL", "bun": "mg/dL", "calcium": "mg/dL",
    "chloride": "mEq/L", "ck_mb": "ng/mL", "cortisol": "ug/dL", "cpk": "U/L",
    "creatinine": "mg/dL", "crp": "mg/L", "crp_hs": "mg/L", "esr": "mm/hour",
    "ferritin": "ng/mL", "fibrinogen": "mg/dL", "free_t4": "ng/dL", "glucose": "mg/dL",
    "heart_rate": "beats/min", "hematocrit": "%", "hemoglobin": "g/dL", "inr": "ratio",
    "ionized_calcium": "mmol/L", "lactate": "mmol/L", "ldh": "U/L", "lipase": "U/L",
    "magnesium": "mg/dL", "map": "mmHg", "myoglobin": "ng/mL", "o2sat": "%", "ph": "pH",
    "phosphate": "mg/dL", "platelets": "K/uL", "potassium": "mEq/L", "prealbumin": "mg/dL",
    "ptt": "seconds", "respiratory_rate": "breaths/min", "serum_osmolality": "mOsm/kg",
    "sodium": "mEq/L", "temperature": "degC", "total_protein": "g/dL",
    "triglycerides": "mg/dL", "troponin_i": "ng/mL", "tsh": "uIU/mL", "uric_acid": "mg/dL",
    "urine_output": "mL/hour", "wbc": "K/uL",
}


@dataclass(frozen=True)
class MappingRule:
    system: str
    code: str
    shape: str
    input_unit: str          # UCUM code as it appears in valueQuantity.code
    variable: str
    factor: float = 1.0
    offset: float = 0.0
    display: str = ""        # verified LOINC long common name; informational only

    @property
    def canonical_unit(self) -> str:
        return CANONICAL_UNITS[self.variable]

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (self.system, self.code, self.shape, self.input_unit)

    def convert(self, value: float) -> float:
        return value * self.factor + self.offset


def _q(code: str, unit: str, variable: str, display: str, factor: float = 1.0, offset: float = 0.0) -> MappingRule:
    return MappingRule(LOINC, code, SHAPE_QUANTITY, unit, variable, factor, offset, display)


# Molar-mass based factors (g/mol): glucose 180.156, creatinine 113.12,
# urea nitrogen 28.014 (2 N per urea), calcium 40.08, bilirubin 584.66,
# lactate 90.08.
RULES: tuple[MappingRule, ...] = (
    # vital signs (FHIR R4 vital-signs profile codes)
    _q("8867-4", "/min", "heart_rate", "Heart rate"),
    _q("8478-0", "mm[Hg]", "map", "Mean blood pressure"),
    _q("59408-5", "%", "o2sat", "Oxygen saturation in Arterial blood by Pulse oximetry"),
    _q("2708-6", "%", "o2sat", "Oxygen saturation in Arterial blood"),
    _q("9279-1", "/min", "respiratory_rate", "Respiratory rate"),
    _q("8310-5", "Cel", "temperature", "Body temperature"),
    _q("8310-5", "[degF]", "temperature", "Body temperature", 5.0 / 9.0, -32.0 * 5.0 / 9.0),
    MappingRule(LOINC, "85354-9", SHAPE_BP_PANEL, "mm[Hg]", "map", 1.0, 0.0,
                "Blood pressure panel with all children optional"),
    # renal / urinary
    _q("2160-0", "mg/dL", "creatinine", "Creatinine [Mass/volume] in Serum or Plasma"),
    _q("14682-9", "umol/L", "creatinine", "Creatinine [Moles/volume] in Serum or Plasma", 1.0 / 88.42),
    _q("3094-0", "mg/dL", "bun", "Urea nitrogen [Mass/volume] in Serum or Plasma"),
    _q("3094-0", "mmol/L", "bun", "Urea nitrogen [Mass/volume] in Serum or Plasma", 2.8014),
    MappingRule(LOINC, "9187-6", SHAPE_PERIOD_VOLUME, "mL", "urine_output", 1.0, 0.0, "Urine output"),
    _q("3084-1", "mg/dL", "uric_acid", "Urate [Mass/volume] in Serum or Plasma"),
    # electrolytes / acid-base
    _q("2823-3", "mmol/L", "potassium", "Potassium [Moles/volume] in Serum or Plasma"),
    _q("2951-2", "mmol/L", "sodium", "Sodium [Moles/volume] in Serum or Plasma"),
    _q("2075-0", "mmol/L", "chloride", "Chloride [Moles/volume] in Serum or Plasma"),
    _q("1963-8", "mmol/L", "bicarbonate", "Bicarbonate [Moles/volume] in Serum or Plasma"),
    _q("33037-3", "mmol/L", "anion_gap", "Anion gap in Serum or Plasma by calculation"),
    _q("17861-6", "mg/dL", "calcium", "Calcium [Mass/volume] in Serum or Plasma"),
    _q("17861-6", "mmol/L", "calcium", "Calcium [Mass/volume] in Serum or Plasma", 4.008),
    _q("1994-3", "mmol/L", "ionized_calcium", "Calcium.ionized [Moles/volume] in Blood"),
    _q("19123-9", "mg/dL", "magnesium", "Magnesium [Mass/volume] in Serum or Plasma"),
    _q("2777-1", "mg/dL", "phosphate", "Phosphate [Mass/volume] in Serum or Plasma"),
    _q("2744-1", "[pH]", "ph", "pH of Arterial blood"),
    _q("2692-2", "mosm/kg", "serum_osmolality", "Osmolality of Serum or Plasma"),
    # endocrine / metabolic
    _q("2345-7", "mg/dL", "glucose", "Glucose [Mass/volume] in Serum or Plasma"),
    _q("2339-0", "mg/dL", "glucose", "Glucose [Mass/volume] in Blood"),
    _q("14749-6", "mmol/L", "glucose", "Glucose [Moles/volume] in Serum or Plasma", 18.0156),
    _q("2143-6", "ug/dL", "cortisol", "Cortisol [Mass/volume] in Serum or Plasma"),
    _q("3024-7", "ng/dL", "free_t4", "Thyroxine (T4) free [Mass/volume] in Serum or Plasma"),
    _q("3016-3", "m[IU]/L", "tsh", "Thyrotropin [Units/volume] in Serum or Plasma"),
    _q("3016-3", "u[IU]/mL", "tsh", "Thyrotropin [Units/volume] in Serum or Plasma"),
    _q("2571-8", "mg/dL", "triglycerides", "Triglyceride [Mass/volume] in Serum or Plasma"),
    # hepatic / synthetic
    _q("1751-7", "g/dL", "albumin", "Albumin [Mass/volume] in Serum or Plasma"),
    _q("1751-7", "g/L", "albumin", "Albumin [Mass/volume] in Serum or Plasma", 0.1),
    _q("6768-6", "U/L", "alkaline_phos", "Alkaline phosphatase [Enzymatic activity/volume] in Serum or Plasma"),
    _q("1742-6", "U/L", "alt", "Alanine aminotransferase [Enzymatic activity/volume] in Serum or Plasma"),
    _q("1920-8", "U/L", "ast", "Aspartate aminotransferase [Enzymatic activity/volume] in Serum or Plasma"),
    _q("1975-2", "mg/dL", "bilirubin", "Bilirubin.total [Mass/volume] in Serum or Plasma"),
    _q("1975-2", "umol/L", "bilirubin", "Bilirubin.total [Mass/volume] in Serum or Plasma", 1.0 / 17.104),
    _q("1968-7", "mg/dL", "bilirubin_direct", "Bilirubin.direct [Mass/volume] in Serum or Plasma"),
    _q("14338-8", "mg/dL", "prealbumin", "Prealbumin [Mass/volume] in Serum or Plasma"),
    _q("2885-2", "g/dL", "total_protein", "Protein [Mass/volume] in Serum or Plasma"),
    # gastrointestinal / pancreatic
    _q("1798-8", "U/L", "amylase", "Amylase [Enzymatic activity/volume] in Serum or Plasma"),
    _q("3040-3", "U/L", "lipase", "Lipase [Enzymatic activity/volume] in Serum or Plasma"),
    # hematologic / coagulation
    _q("718-7", "g/dL", "hemoglobin", "Hemoglobin [Mass/volume] in Blood"),
    _q("718-7", "g/L", "hemoglobin", "Hemoglobin [Mass/volume] in Blood", 0.1),
    _q("4544-3", "%", "hematocrit", "Hematocrit [Volume Fraction] of Blood by Automated count"),
    _q("777-3", "10*3/uL", "platelets", "Platelets [#/volume] in Blood by Automated count"),
    _q("6301-6", "{INR}", "inr", "INR in Platelet poor plasma by Coagulation assay"),
    _q("3173-2", "s", "ptt", "aPTT in Blood by Coagulation assay"),
    _q("3255-7", "mg/dL", "fibrinogen", "Fibrinogen [Mass/volume] in Platelet poor plasma by Coagulation assay"),
    # immune / inflammatory
    _q("6690-2", "10*3/uL", "wbc", "Leukocytes [#/volume] in Blood by Automated count"),
    _q("1988-5", "mg/L", "crp", "C reactive protein [Mass/volume] in Serum or Plasma"),
    _q("30522-7", "mg/L", "crp_hs", "C reactive protein [Mass/volume] in Serum or Plasma by High sensitivity method"),
    _q("30341-2", "mm/h", "esr", "Erythrocyte [Sedimentation Rate] in Blood"),
    _q("2276-4", "ng/mL", "ferritin", "Ferritin [Mass/volume] in Serum or Plasma"),
    # cardiovascular perfusion / cardiac injury
    _q("2524-7", "mmol/L", "lactate", "Lactate [Moles/volume] in Serum or Plasma"),
    _q("2524-7", "mg/dL", "lactate", "Lactate [Moles/volume] in Serum or Plasma", 1.0 / 9.008),
    _q("30934-4", "pg/mL", "bnp", "Natriuretic peptide B [Mass/volume] in Serum or Plasma"),
    _q("10839-9", "ng/mL", "troponin_i", "Troponin I.cardiac [Mass/volume] in Serum or Plasma"),
    _q("13969-1", "ng/mL", "ck_mb", "Creatine kinase.MB [Mass/volume] in Serum or Plasma"),
    # musculoskeletal injury
    _q("2157-6", "U/L", "cpk", "Creatine kinase [Enzymatic activity/volume] in Serum or Plasma"),
    _q("2532-0", "U/L", "ldh", "Lactate dehydrogenase [Enzymatic activity/volume] in Serum or Plasma"),
    _q("2639-3", "ng/mL", "myoglobin", "Myoglobin [Mass/volume] in Serum or Plasma"),
)

_CODES: dict[tuple[str, str], list[MappingRule]] = {}
for _rule in RULES:
    _CODES.setdefault((_rule.system, _rule.code), []).append(_rule)

FAIL_UNMAPPED_CODE = "unmapped_code"
FAIL_UNMAPPED_UNIT = "unmapped_unit"
FAIL_MISSING_UNIT = "missing_unit"
FAIL_DATA_ABSENT = "data_absent"
FAIL_NO_VALUE = "no_value"
FAIL_SHAPE = "unsupported_shape"
FAIL_STATUS = "excluded_status"


@dataclass(frozen=True)
class CanonicalValue:
    variable: str
    value: float
    unit: str
    effective_time: datetime
    rule_key: tuple[str, str, str, str]
    input_value: float
    input_unit: str
    derivation: str  # "identity" | "unit_conversion" | "map_from_sbp_dbp" | "volume/period (...)"


@dataclass(frozen=True)
class MappingOutcome:
    values: tuple[CanonicalValue, ...] = ()
    failure: str | None = None
    detail: str = ""
    matched_codes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def mapped(self) -> bool:
        return self.failure is None and bool(self.values)


def known_variables() -> tuple[str, ...]:
    return tuple(sorted({rule.variable for rule in RULES}))


def registry_document() -> dict[str, Any]:
    return {
        "mapping_version": MAPPING_VERSION,
        "rule_count": len(RULES),
        "variables": known_variables(),
        "rules": [
            {
                "system": r.system, "code": r.code, "shape": r.shape, "input_unit": r.input_unit,
                "variable": r.variable, "canonical_unit": r.canonical_unit,
                "factor": r.factor, "offset": r.offset, "display": r.display,
            }
            for r in RULES
        ],
    }


def _codings(concept: dict[str, Any] | None) -> list[tuple[str, str]]:
    out = []
    for coding in (concept or {}).get("coding", []) or []:
        system, code = coding.get("system"), coding.get("code")
        if isinstance(system, str) and isinstance(code, str):
            out.append((system, code))
    return out


def _quantity(quantity: dict[str, Any] | None) -> tuple[float | None, str | None, str | None]:
    if not isinstance(quantity, dict):
        return None, None, None
    value = quantity.get("value")
    number = float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None
    return number, quantity.get("system"), quantity.get("code")


def map_observation(resource: dict[str, Any], effective_time: datetime, period_hours: float | None) -> MappingOutcome:
    """Map one FHIR R4 Observation to zero or more canonical values.

    ``effective_time`` is the parsed clinical time (period end for a Period);
    ``period_hours`` is the Period length in hours, or None for a dateTime.
    """
    if resource.get("status") in {"entered-in-error", "cancelled", "unknown"}:
        return MappingOutcome(failure=FAIL_STATUS, detail=str(resource.get("status")))
    codes = _codings(resource.get("code"))
    candidates = [rule for key in codes for rule in _CODES.get(key, [])]
    if not candidates:
        return MappingOutcome(failure=FAIL_UNMAPPED_CODE,
                              detail=", ".join(f"{s}|{c}" for s, c in codes) or "no coding")
    matched = tuple(f"{r.system}|{r.code}" for r in candidates[:1])
    if resource.get("dataAbsentReason") is not None and "valueQuantity" not in resource:
        reason = ", ".join(c for _, c in _codings(resource.get("dataAbsentReason"))) or "unspecified"
        return MappingOutcome(failure=FAIL_DATA_ABSENT, detail=reason, matched_codes=matched)

    shapes = {rule.shape for rule in candidates}
    if SHAPE_BP_PANEL in shapes:
        return _map_bp_panel(resource, candidates, effective_time, matched)

    value, unit_system, unit_code = _quantity(resource.get("valueQuantity"))
    if value is None:
        return MappingOutcome(failure=FAIL_NO_VALUE, detail="valueQuantity.value missing", matched_codes=matched)
    if unit_system != UCUM or not isinstance(unit_code, str):
        return MappingOutcome(failure=FAIL_MISSING_UNIT, detail="valueQuantity has no UCUM code", matched_codes=matched)

    if SHAPE_PERIOD_VOLUME in shapes:
        if period_hours is None or period_hours <= 0:
            return MappingOutcome(failure=FAIL_SHAPE, detail="volume observation requires effectivePeriod",
                                  matched_codes=matched)
        rule = next(r for r in candidates if r.shape == SHAPE_PERIOD_VOLUME)
        if unit_code != rule.input_unit:
            return MappingOutcome(failure=FAIL_UNMAPPED_UNIT, detail=unit_code, matched_codes=matched)
        rate = rule.convert(value) / period_hours
        return MappingOutcome(values=(CanonicalValue(
            rule.variable, rate, rule.canonical_unit, effective_time, rule.key, value, unit_code,
            f"volume/period ({period_hours:g} h)"),), matched_codes=matched)

    for rule in candidates:
        if rule.shape == SHAPE_QUANTITY and rule.input_unit == unit_code:
            derivation = "identity" if rule.factor == 1.0 and rule.offset == 0.0 else "unit_conversion"
            return MappingOutcome(values=(CanonicalValue(
                rule.variable, rule.convert(value), rule.canonical_unit, effective_time, rule.key,
                value, unit_code, derivation),), matched_codes=matched)
    return MappingOutcome(failure=FAIL_UNMAPPED_UNIT, detail=unit_code, matched_codes=matched)


def _map_bp_panel(resource: dict[str, Any], candidates: list[MappingRule], effective_time: datetime,
                  matched: tuple[str, ...]) -> MappingOutcome:
    rule = next(r for r in candidates if r.shape == SHAPE_BP_PANEL)
    systolic = diastolic = None
    for component in resource.get("component", []) or []:
        codes = dict(_codings(component.get("code")))
        value, unit_system, unit_code = _quantity(component.get("valueQuantity"))
        if value is None or unit_system != UCUM or unit_code != rule.input_unit:
            continue
        if codes.get(LOINC) == "8480-6":
            systolic = value
        elif codes.get(LOINC) == "8462-4":
            diastolic = value
    if systolic is None or diastolic is None:
        return MappingOutcome(failure=FAIL_SHAPE,
                              detail="blood pressure panel needs 8480-6 and 8462-4 mm[Hg] components",
                              matched_codes=matched)
    mean = (systolic + 2.0 * diastolic) / 3.0
    return MappingOutcome(values=(CanonicalValue(
        rule.variable, rule.convert(mean), rule.canonical_unit, effective_time, rule.key, mean,
        rule.input_unit, "map_from_sbp_dbp"),), matched_codes=matched)

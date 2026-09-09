"""The demo's reference ranges, with where every number came from.

This is the single place a reference interval is written down. The synthetic
fixture builder reads it and writes each interval into the FHIR
``Observation.referenceRange`` it applies to, together with an
``Observation.interpretation`` derived from that same interval; the stream,
the ledger and the API then pass both through untouched, and the front end
reads only what the observation carries. No other module may know a number.

Every entry cites one public, citable source (MedlinePlus Medical Encyclopedia,
U.S. National Library of Medicine / National Institutes of Health) and records
the unit the source states, the unit the observation is written in, and why the
two are the same number when they differ in name. Nothing here is a diagnostic
or treatment threshold: it is the interval a report would print beside a value.

Ranges that differ by sex, by age or by sampling context are deliberately not
adopted: the synthetic patient carries no sex or date of birth, and no fasting or
specimen context, so no single interval applies. Those variables stay
"Range not supplied", and the reason is written down here too.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

V3_INTERPRETATION = "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation"
UCUM = "http://unitsofmeasure.org"

# The interpretation the builder writes is arithmetic on the supplied interval,
# not a laboratory's or a clinician's reading. It carries this marker in the
# concept's ``text`` so nothing downstream can mistake it for a source reading.
INTERPRETATION_PROVENANCE = "derived_from_supplied_reference_range"
RETRIEVAL_DATE = "2026-09-08"
SOURCE_ORGANIZATION = "MedlinePlus Medical Encyclopedia (A.D.A.M.), U.S. National Library of Medicine, National Institutes of Health"


@dataclass(frozen=True)
class ReferenceRange:
    variable: str            # canonical variable id
    low: float
    high: float
    unit: str                # the unit written on the observation and on the range (display)
    unit_code: str           # UCUM code for `unit`
    source_low: float        # the interval exactly as the source states it
    source_high: float
    source_unit: str
    unit_note: str           # why `unit` and `source_unit` are the same number, or "same unit"
    applicable_age: str
    applicable_sex: str
    context: str             # specimen / sampling context the source describes
    source_title: str
    source_url: str
    source_review_date: str  # the "Review Date" printed on the page
    applicability: str       # how this demo uses it
    origin: str = "external_reference_cited_by_demo_fixture"   # not from the data source itself
    citable: bool = True

    def interpret(self, value: float) -> str:
        """v3-ObservationInterpretation code for `value` against this interval. Inclusive limits."""
        if value < self.low:
            return "L"
        if value > self.high:
            return "H"
        return "N"

    def fhir_reference_range(self) -> dict[str, Any]:
        return {
            "low": {"value": self.low, "unit": self.unit, "system": UCUM, "code": self.unit_code},
            "high": {"value": self.high, "unit": self.unit, "system": UCUM, "code": self.unit_code},
            "type": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/referencerange-meaning",
                                 "code": "normal", "display": "Normal Range"}]},
            "text": f"{self.source_low:g}–{self.source_high:g} {self.source_unit} · {self.source_title}, "
                    f"{SOURCE_ORGANIZATION.split(' (')[0]} · review date {self.source_review_date} · "
                    f"demo fixture reference, not a clinical threshold",
        }

    def fhir_interpretation(self, value: float) -> dict[str, Any]:
        code = self.interpret(value)
        display = {"L": "Low", "N": "Normal", "H": "High"}[code]
        return {"coding": [{"system": V3_INTERPRETATION, "code": code, "display": display}],
                "text": INTERPRETATION_PROVENANCE}

    def provenance(self) -> dict[str, Any]:
        return {
            "variable": self.variable, "low": self.low, "high": self.high, "unit": self.unit, "unit_code": self.unit_code,
            "source_low": self.source_low, "source_high": self.source_high, "source_unit": self.source_unit,
            "unit_note": self.unit_note, "applicable_age": self.applicable_age, "applicable_sex": self.applicable_sex,
            "context": self.context, "source_title": self.source_title, "source_organization": SOURCE_ORGANIZATION,
            "source_url": self.source_url, "source_review_date": self.source_review_date,
            "retrieval_date": RETRIEVAL_DATE, "applicability": self.applicability, "origin": self.origin,
            "citable": self.citable,
        }


_ADULT = "adult (general adult interval as published; the synthetic patient is declared an adult inpatient and carries no date of birth)"
_NOT_SEX_SPECIFIC = "not sex-specific in the source"
_APPLICABILITY = ("written into the synthetic fixture's Observation.referenceRange as the interval a report would print "
                  "beside the value; used by the demo to say whether an observed value lies inside it; not a "
                  "diagnostic or treatment threshold")
_MEQ_NOTE = "source states mEq/L; for a monovalent ion 1 mEq/L equals 1 mmol/L by definition (factor 1)"
_PER_L_NOTE = "source states ×10⁹/L (cells per microlitre ×10³); 10⁹/L equals 10³/µL by definition (factor 1)"

DEMO_REFERENCE_RANGES: dict[str, ReferenceRange] = {
    "sodium": ReferenceRange(
        "sodium", 135.0, 145.0, "mmol/L", "mmol/L", 135.0, 145.0, "mEq/L", _MEQ_NOTE, _ADULT, _NOT_SEX_SPECIFIC,
        "blood (serum)", "Sodium blood test", "https://medlineplus.gov/ency/article/003481.htm", "2025-11-06",
        _APPLICABILITY),
    "potassium": ReferenceRange(
        "potassium", 3.7, 5.2, "mmol/L", "mmol/L", 3.7, 5.2, "mEq/L (3.70 to 5.20 mmol/L)",
        "source states both mEq/L and mmol/L; same number", _ADULT, _NOT_SEX_SPECIFIC,
        "blood (serum)", "Potassium test", "https://medlineplus.gov/ency/article/003484.htm", "2025-05-19",
        _APPLICABILITY),
    "chloride": ReferenceRange(
        "chloride", 98.0, 107.0, "mmol/L", "mmol/L", 98.0, 107.0, "mEq/L (98 to 107 mmol/L)",
        "source states both mEq/L and mmol/L; same number", _ADULT, _NOT_SEX_SPECIFIC,
        "blood (serum)", "Chloride test - blood", "https://medlineplus.gov/ency/article/003485.htm", "2025-05-19",
        _APPLICABILITY),
    "bicarbonate": ReferenceRange(
        "bicarbonate", 23.0, 29.0, "mmol/L", "mmol/L", 23.0, 29.0, "mEq/L (23 to 29 mmol/L)",
        "source states both mEq/L and mmol/L; same number. The source page describes the CO2 blood test, "
        "which measures bicarbonate; the observation is LOINC 1963-8 bicarbonate", _ADULT, _NOT_SEX_SPECIFIC,
        "blood (serum)", "CO2 blood test", "https://medlineplus.gov/ency/article/003469.htm", "2025-05-19",
        _APPLICABILITY),
    "bun": ReferenceRange(
        "bun", 6.0, 20.0, "mg/dL", "mg/dL", 6.0, 20.0, "mg/dL", "same unit", _ADULT, _NOT_SEX_SPECIFIC,
        "blood (serum)", "BUN - blood test", "https://medlineplus.gov/ency/article/003474.htm", "2025-05-19",
        _APPLICABILITY),
    "lactate": ReferenceRange(
        "lactate", 0.5, 2.2, "mmol/L", "mmol/L", 0.5, 2.2, "mmol/L (4.5 to 19.8 mg/dL)",
        "source states both mg/dL and mmol/L; the mmol/L figure is used as stated", _ADULT, _NOT_SEX_SPECIFIC,
        "blood", "Lactic acid test", "https://medlineplus.gov/ency/article/003507.htm", "2025-05-19",
        _APPLICABILITY),
    "wbc": ReferenceRange(
        "wbc", 4.5, 11.0, "10*3/uL", "10*3/uL", 4.5, 11.0, "×10⁹/L (4,500 to 11,000 per microliter)", _PER_L_NOTE,
        _ADULT, _NOT_SEX_SPECIFIC, "blood", "WBC count", "https://medlineplus.gov/ency/article/003643.htm",
        "2025-02-03", _APPLICABILITY),
    "platelets": ReferenceRange(
        "platelets", 150.0, 400.0, "10*3/uL", "10*3/uL", 150.0, 400.0, "×10⁹/L (150,000 to 400,000 per microliter)",
        _PER_L_NOTE, _ADULT, _NOT_SEX_SPECIFIC, "blood", "Platelet count",
        "https://medlineplus.gov/ency/article/003647.htm", "2025-02-03", _APPLICABILITY),
}

# Variables the synthetic stream carries for which no single interval applies.
# Each is a decision, written down, not an omission.
NOT_SUPPLIED: dict[str, str] = {
    "creatinine": "sex-specific in the source (0.7–1.3 mg/dL men, 0.5–0.95 mg/dL women; "
                  "https://medlineplus.gov/ency/article/003475.htm, review date 2025-07-13); the synthetic patient "
                  "carries no sex, so no interval applies",
    "hemoglobin": "sex- and age-specific in the source (13.8–17.2 g/dL men, 12.1–15.1 g/dL women; "
                  "https://medlineplus.gov/ency/article/003645.htm, review date 2026-02-27); no sex or age is carried",
    "hematocrit": "sex- and age-specific in the source (37–48 % men, 34–43 % women; "
                  "https://medlineplus.gov/ency/article/003646.htm, review date 2026-02-27); no sex or age is carried",
    "glucose": "depends on sampling context in the source (fasting 70–99 mg/dL; a random result depends on the last "
               "meal; https://medlineplus.gov/ency/article/003482.htm, review date 2025-01-10); the synthetic "
               "observations carry no fasting context",
    "heart_rate": "a vital sign; its interpretation is contextual and no laboratory reference interval is adopted",
    "respiratory_rate": "a vital sign; its interpretation is contextual and no laboratory reference interval is adopted",
    "o2sat": "a vital sign; its interpretation is contextual and no laboratory reference interval is adopted",
    "temperature": "a vital sign; its interpretation is contextual and no laboratory reference interval is adopted",
    "map": "a derived vital sign; its interpretation is contextual and no laboratory reference interval is adopted",
    "urine_output": "a charted rate, not a laboratory analyte; no reference interval is adopted",
}


def provenance_document() -> dict[str, Any]:
    """The block the fixture document carries, so a reader of the JSON can check every number."""
    return {
        "interpretation_provenance": INTERPRETATION_PROVENANCE,
        "retrieval_date": RETRIEVAL_DATE,
        "source_organization": SOURCE_ORGANIZATION,
        "supplied": [r.provenance() for r in DEMO_REFERENCE_RANGES.values()],
        "not_supplied": dict(NOT_SUPPLIED),
    }

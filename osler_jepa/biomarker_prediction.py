"""Canonical non-causal prediction policy for extended blood biomarkers."""

from __future__ import annotations

from dataclasses import asdict, dataclass


# Numeric MIMIC-IV mappings were verified against hosp/d_labitems.csv.gz v3.1.
MIMIC_BIOMARKER_ITEMIDS: dict[str, tuple[int, ...]] = {
    "total_cholesterol": (50907,),
    "hdl_cholesterol": (50904,),
    "ldl_cholesterol": (50905, 50906),
    "hba1c": (50852,),
    "iron": (50952,),
    "tibc": (50953,),
    "transferrin": (50998,),
    "transferrin_saturation": (51746,),
    "vitamin_d_25oh": (50853,),
    "vitamin_b12": (51010,),
    "folate": (50925, 51730, 52562),
    "estradiol": (50921,),
    "progesterone": (51707,),
    "testosterone_total": (50988,),
    "testosterone_free": (50894, 50989),
    "afp": (50864,),
    "ca_125": (50892,),
    "ca_19_9": (51579,),
    "psa": (50974,),
    "rheumatoid_factor": (50980,),
}


PLAUSIBLE_BIOMARKER_BOUNDS: dict[str, tuple[float, float]] = {
    "total_cholesterol": (20.0, 1500.0),
    "hdl_cholesterol": (1.0, 300.0),
    "ldl_cholesterol": (1.0, 1200.0),
    "hba1c": (2.0, 25.0),
    "iron": (0.0, 1000.0),
    "tibc": (1.0, 1500.0),
    "transferrin": (1.0, 1500.0),
    "transferrin_saturation": (0.0, 100.0),
    "vitamin_d_25oh": (0.0, 500.0),
    "vitamin_b12": (0.0, 10000.0),
    "folate": (0.0, 200.0),
    "estradiol": (0.0, 10000.0),
    "progesterone": (0.0, 1000.0),
    "testosterone_total": (0.0, 5000.0),
    "testosterone_free": (0.0, 1000.0),
    "afp": (0.0, 1000000.0),
    "ca_125": (0.0, 1000000.0),
    "ca_19_9": (0.0, 1000000.0),
    "psa": (0.0, 100000.0),
    "rheumatoid_factor": (0.0, 100000.0),
}


ROUTINE_VALUE_TARGETS = (
    "total_cholesterol",
    "hdl_cholesterol",
    "ldl_cholesterol",
    "triglycerides",
    "hba1c",
    "ast",
    "alt",
    "alkaline_phos",
    "bilirubin",
    "albumin",
    "hemoglobin",
    "hematocrit",
    "wbc",
    "platelets",
    "iron",
    "tibc",
    "transferrin",
    "transferrin_saturation",
    "ferritin",
    "tsh",
    "free_t4",
)

SPARSE_VALUE_OR_RISK_TARGETS = (
    "crp",
    "vitamin_d_25oh",
    "vitamin_b12",
    "folate",
    "inr",
    "ptt",
    "fibrinogen",
    "cortisol",
    "estradiol",
    "progesterone",
    "testosterone_total",
    "testosterone_free",
)

SPECIALTY_RISK_TARGETS = (
    "troponin_i",
    "troponin_t",
    "afp",
    "ca_125",
    "ca_19_9",
    "psa",
    "rheumatoid_factor",
)

OBSERVED_ONLY_TARGETS = (
    "germline_genetics",
    "hereditary_cancer_panel",
    "pharmacogenetics",
    "molecular_residual_disease",
)


@dataclass(frozen=True)
class BiomarkerPredictionContract:
    target: str
    tier: str
    current_estimate_allowed: bool
    numeric_future_candidate: bool
    future_abnormal_risk_candidate: bool
    stable_probability_candidate: bool
    next_measurement_time_allowed: bool = False
    causal_claim_allowed: bool = False
    status: str = "candidate_only"


def prediction_contract(target: str) -> dict[str, object]:
    """Return a fail-closed task contract; it never promotes a model."""

    normalized = str(target).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in ROUTINE_VALUE_TARGETS:
        contract = BiomarkerPredictionContract(
            target=normalized,
            tier="routine_value",
            current_estimate_allowed=True,
            numeric_future_candidate=True,
            future_abnormal_risk_candidate=True,
            stable_probability_candidate=True,
        )
    elif normalized in SPARSE_VALUE_OR_RISK_TARGETS:
        contract = BiomarkerPredictionContract(
            target=normalized,
            tier="sparse_value_or_risk",
            current_estimate_allowed=True,
            numeric_future_candidate=True,
            future_abnormal_risk_candidate=True,
            stable_probability_candidate=True,
        )
    elif normalized in SPECIALTY_RISK_TARGETS:
        contract = BiomarkerPredictionContract(
            target=normalized,
            tier="specialty_risk",
            current_estimate_allowed=True,
            numeric_future_candidate=False,
            future_abnormal_risk_candidate=True,
            stable_probability_candidate=False,
        )
    elif normalized in OBSERVED_ONLY_TARGETS:
        contract = BiomarkerPredictionContract(
            target=normalized,
            tier="observed_only",
            current_estimate_allowed=False,
            numeric_future_candidate=False,
            future_abnormal_risk_candidate=False,
            stable_probability_candidate=False,
            status="observed_only",
        )
    else:
        contract = BiomarkerPredictionContract(
            target=normalized,
            tier="unregistered",
            current_estimate_allowed=False,
            numeric_future_candidate=False,
            future_abnormal_risk_candidate=False,
            stable_probability_candidate=False,
            status="missing",
        )
    return asdict(contract)


def registered_numeric_targets() -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            (*ROUTINE_VALUE_TARGETS, *SPARSE_VALUE_OR_RISK_TARGETS, *SPECIALTY_RISK_TARGETS)
        )
    )


__all__ = [
    "MIMIC_BIOMARKER_ITEMIDS",
    "OBSERVED_ONLY_TARGETS",
    "PLAUSIBLE_BIOMARKER_BOUNDS",
    "ROUTINE_VALUE_TARGETS",
    "SPARSE_VALUE_OR_RISK_TARGETS",
    "SPECIALTY_RISK_TARGETS",
    "prediction_contract",
    "registered_numeric_targets",
]

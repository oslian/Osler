"""Public API for the Whole-Body JEPA research package."""

from __future__ import annotations


_RUNTIME_EXPORTS = {
    "DEFAULT_HORIZONS",
    "load_release",
    "parse_patient_csv",
    "predict_patient",
}
_SUBTYPE_EXPORTS = {
    "SoftPhysiologySubtypeModel",
    "SubtypeCandidatePrediction",
    "causal_history_covariates",
    "causal_history_covariates_at_anchors",
    "causal_trajectory_features",
    "subtype_features",
}


def __getattr__(name: str):
    if name in _RUNTIME_EXPORTS:
        from . import runtime

        return getattr(runtime, name)
    if name in _SUBTYPE_EXPORTS:
        from . import subtypes

        return getattr(subtypes, name)
    raise AttributeError(name)


__all__ = sorted(_RUNTIME_EXPORTS | _SUBTYPE_EXPORTS)

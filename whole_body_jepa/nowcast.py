"""Validated same-anchor completion from related physiological measurements."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd


SIBLING_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"hemoglobin", "hematocrit"}),
    frozenset({"ph", "bicarbonate"}),
    frozenset({"anion_gap", "sodium", "chloride", "bicarbonate"}),
    frozenset({"serum_osmolality", "sodium", "glucose", "bun"}),
    frozenset(
        {
            "creatinine",
            "bun",
            "urine_output",
            "urine_output_per_kg_hour",
            "urine_output_3h_per_kg_hour",
        }
    ),
    frozenset({"bilirubin", "bilirubin_direct"}),
)


def related_siblings(target: str) -> frozenset[str]:
    related: set[str] = set()
    for group in SIBLING_GROUPS:
        if target in group:
            related.update(group)
    related.discard(target)
    return frozenset(related)


def measurement_pure_feature_columns(
    available_columns: Iterable[str],
    variables: Iterable[str],
    target: str,
) -> tuple[str, ...]:
    """Return current-value inputs without target, siblings, ages, or workflow fields."""

    available = set(available_columns)
    excluded = related_siblings(target) | {target}
    return tuple(
        f"{variable}_t"
        for variable in variables
        if variable not in excluded and f"{variable}_t" in available
    )


def derived_current_completions(values: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Compute current values only where a fixed physiological formula applies."""

    numeric = {
        key: float(value)
        for key, value in values.items()
        if value is not None and np.isfinite(float(value))
    }
    output: dict[str, dict[str, Any]] = {}
    if "map" not in numeric and {"sbp", "dbp"}.issubset(numeric):
        output["map"] = {
            "value": numeric["dbp"] + (numeric["sbp"] - numeric["dbp"]) / 3.0,
            "source": "derived_formula",
            "inputs": ["sbp", "dbp"],
        }
    if "anion_gap" not in numeric and {"sodium", "chloride", "bicarbonate"}.issubset(numeric):
        output["anion_gap"] = {
            "value": numeric["sodium"] - numeric["chloride"] - numeric["bicarbonate"],
            "source": "derived_formula",
            "inputs": ["sodium", "chloride", "bicarbonate"],
        }
    if "serum_osmolality" not in numeric and {"sodium", "glucose", "bun"}.issubset(numeric):
        output["serum_osmolality"] = {
            "value": 2.0 * numeric["sodium"] + numeric["glucose"] / 18.0 + numeric["bun"] / 2.8,
            "source": "derived_formula",
            "inputs": ["sodium", "glucose", "bun"],
        }
    return output


def latest_nowcast_row(model_frame: pd.DataFrame, feature_columns: Iterable[str]) -> pd.DataFrame:
    """Build one runtime row in the exact feature order stored in an audited head."""

    row = model_frame.iloc[[-1]].copy()
    return row.reindex(columns=list(feature_columns))

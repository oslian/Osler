"""Choose the deployable recipe for a site from the data the site actually provides.

The treatment-context ridge is defined by pre-anchor treatment records; without
them it is not the validated recipe.  When a site records none of the target's
permitted treatment features, the cell falls back to the measurement-pure ridge,
which needs measurements only.  Other recipes tolerate missing treatment
features (imputed with training medians) and are kept.  The rule is
deterministic and uses inputs only, never outcomes.
"""

from __future__ import annotations

import pandas as pd

from osler_jepa.treatment_units import treatment_feature_actions


TREATMENT_DEFINED_RECIPES = frozenset({"observed_treatment_context_ridge"})
FALLBACK_RECIPE = "measurement_pure_ridge"
MIN_AVAILABLE_SHARE = 0.01


def permitted_treatment_columns(columns, target: str) -> list[str]:
    actions = treatment_feature_actions(target)
    return [
        column for column in columns
        if any(column == f"hist_{action}" or column.startswith(f"hist_{action}_") for action in actions)
    ]


def treatment_context_available(frame: pd.DataFrame, target: str, min_share: float = MIN_AVAILABLE_SHARE) -> bool:
    for column in permitted_treatment_columns(frame.columns, target):
        if pd.to_numeric(frame[column], errors="coerce").notna().mean() >= min_share:
            return True
    return False


def deployable_recipe(catalog_recipe: str, site_frame: pd.DataFrame, target: str) -> tuple[str, str]:
    """Return (recipe, reason) for this site."""

    if catalog_recipe in TREATMENT_DEFINED_RECIPES and not treatment_context_available(site_frame, target):
        return FALLBACK_RECIPE, "treatment_context_not_recorded_at_site"
    return catalog_recipe, "catalog"

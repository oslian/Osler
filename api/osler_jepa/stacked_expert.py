"""Bounded residual over a frozen target-horizon expert cell (AGENTS.md, second layer).

Each cell's output is its frozen expert forecast plus a bounded residual learned
by gradient boosting from what is known at anchor time about a target: its
current value, the hours since it was last measured, the frozen expert's
forecast change for every horizon and, optionally, its recent trajectory
(change from its value at t - L for each lookback L).  The formal residual
reads its own target only (``exchange=False``); reading every target
(``exchange=True``, the stacked single-target form of multi-target regression,
Spyromitros-Xioufis et al., 2016) is an ablation.  Missing inputs stay missing
(native handling of ``HistGradientBoostingRegressor``).  Research runtime only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

RESIDUAL_SETTINGS = dict(loss="absolute_error", learning_rate=0.05, max_iter=300, max_leaf_nodes=15,
                         min_samples_leaf=200, l2_regularization=1.0, early_stopping=False, random_state=0)
LIMIT_QUANTILE = 0.99


def target_columns(target: str, horizons: tuple[int, ...], lags: tuple[int, ...] = ()) -> list[str]:
    """A target's inputs: current value, measurement age, forecast change per horizon, trajectory per lookback."""

    return ([f"current__{target}", f"age__{target}"] + [f"change__{target}@{h}h" for h in horizons]
            + [f"trend__{target}@-{lag}h" for lag in lags])


def feature_columns(target: str, targets: tuple[str, ...], horizons: tuple[int, ...], lags: tuple[int, ...] = (), *,
                    exchange: bool) -> list[str]:
    return [column for other in (targets if exchange else (target,)) for column in target_columns(other, horizons, lags)]


@dataclass
class StackedCell:
    columns: list[str]
    limit: float
    model: HistGradientBoostingRegressor

    def predict(self, features: pd.DataFrame, anchor: np.ndarray) -> np.ndarray:
        residual = self.model.predict(features[self.columns].to_numpy(dtype=np.float64))
        return anchor + np.clip(residual, -self.limit, self.limit)


def fit_cell(features: pd.DataFrame, anchor: np.ndarray, future: np.ndarray, columns: list[str]) -> StackedCell:
    """Bounded residual of the frozen expert, fitted on the given (fit) rows."""

    residual = future - anchor
    limit = max(float(np.quantile(np.abs(residual), LIMIT_QUANTILE)), 1e-6)
    model = HistGradientBoostingRegressor(**RESIDUAL_SETTINGS).fit(features[columns].to_numpy(dtype=np.float64), residual)
    return StackedCell(columns=list(columns), limit=limit, model=model)


def group_importance(cell: StackedCell, features: pd.DataFrame, anchor: np.ndarray, future: np.ndarray,
                     groups: dict[str, list[str]], *, seed: int) -> dict[str, float]:
    """Relative MAE increase when one group's inputs are permuted jointly across rows (association only)."""

    base = float(np.mean(np.abs(cell.predict(features, anchor) - future)))
    order = np.random.default_rng(seed).permutation(len(features))
    output = {}
    for name, columns in groups.items():
        shuffled = features.copy()
        shuffled[columns] = features[columns].to_numpy()[order]
        output[name] = float(np.mean(np.abs(cell.predict(shuffled, anchor) - future)) / base - 1.0)
    return output

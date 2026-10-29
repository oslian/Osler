"""A shared deployment clock for replaying retrospective ICU stays as a live, concurrent stream.

Retrospective databases carry no usable cross-patient calendar (eICU offsets
and MIMIC-IV dates are per-stay shifted), so a deployment is simulated: each
hospital's stays arrive as a Poisson process whose rate keeps on average
``census`` stays in the unit at once (arrival rate = census / mean stay span);
an anchor's clock time is its stay's arrival time plus its hours since the
stay's first anchor.  Stays therefore overlap, as in a real ICU, and an outcome
can be used for adaptation only once the clock passes its availability time
(anchor + horizon + |label offset|), never merely because another stay began.

The first ``min(warmup_stays, warmup_share x stays)`` arrivals form a warm-up
cohort: their inputs may fit day-one covariate weights, and they are excluded
from scoring.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


DEFAULT_CENSUS = 12.0
WARMUP_STAYS = 200
WARMUP_SHARE = 0.25


@dataclass(frozen=True)
class ClockedStream:
    order: np.ndarray        # positions into the supplied rows, in clock order
    clock_hours: np.ndarray  # clock time of each supplied row (aligned with the input rows)
    warmup: np.ndarray       # True for rows of warm-up stays (aligned with the input rows)


def deployment_clock(stays: np.ndarray, hours_since_onset: np.ndarray, *, seed: int,
                     census: float = DEFAULT_CENSUS, warmup_stays: int = WARMUP_STAYS,
                     warmup_share: float = WARMUP_SHARE) -> ClockedStream:
    stays = np.asarray(stays).astype(str)
    onset = np.asarray(hours_since_onset, dtype=np.float64)
    if not np.all(np.isfinite(onset)):
        raise ValueError("hours_since_onset must be finite for every row")
    frame = pd.DataFrame({"stay": stays, "onset": onset})
    spans = frame.groupby("stay", sort=False)["onset"].max() + 1.0
    rng = np.random.default_rng(seed)
    arrival_order = rng.permutation(spans.index.to_numpy())
    rate = float(census) / float(spans.mean())
    arrivals = pd.Series(np.cumsum(rng.exponential(1.0 / rate, size=len(arrival_order))), index=arrival_order)
    clock = arrivals.reindex(stays).to_numpy() + onset
    warm_count = max(1, min(int(warmup_stays), int(warmup_share * len(arrival_order))))
    warm = set(arrival_order[:warm_count])
    order = np.lexsort((onset, clock))
    return ClockedStream(order=order, clock_hours=clock, warmup=np.array([s in warm for s in stays]))

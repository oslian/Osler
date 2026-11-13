"""Source-count-only bounded site sampling; no physiology or outcomes used."""
from collections import Counter
import numpy as np


def bounded_weights(sites, lower=.5, upper=2.):
    if not len(sites) or not 0 < lower <= 1 <= upper or lower == upper:
        raise ValueError('Nonempty sites and bounds bracketing one required')
    counts = Counter(sites)
    raw = np.array([len(sites)/(len(counts)*counts[s]) for s in sites], dtype=float)
    lo, hi = 0., upper / raw.min()
    for _ in range(100):
        mid = (lo+hi)/2
        if np.clip(mid*raw,lower,upper).mean() < 1:
            lo=mid
        else:
            hi=mid
    result=np.clip((lo+hi)/2*raw,lower,upper)
    if not np.isclose(result.mean(),1.,rtol=0,atol=1e-12):
        raise ValueError('Weight normalization failed')
    return result


def sample_indices(weights, rng, draws):
    weights=np.asarray(weights,dtype=float)
    if weights.ndim!=1 or not len(weights) or not np.isfinite(weights).all() or (weights<=0).any() or draws<1:
        raise ValueError('Invalid sampling distribution')
    return rng.choice(len(weights),size=draws,replace=True,p=weights/weights.sum())

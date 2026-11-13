"""
synthetic_rct_loop.py

The simplest complete RCT: one control loop (insulin -> glucose), but built with
the two features that make RCTs necessary:

  1. UNOBSERVED confounder U (illness severity): raises insulin dose AND raises
     glucose -> biases any observational estimate.
  2. NONLINEAR true response: insulin lowers glucose along a SATURATING curve
     (effect flattens at high dose), not a straight line.

Two worlds are generated from the SAME truth:
  - OBSERVATIONAL: dose driven by glucose + U (confounded), like MIMIC.
  - RCT: dose RANDOMIZED by a coin flip, independent of glucose and U.

We then ask, in each world:
  - does a naive fit recover the true response? (tests confounding)
  - does a linear vs a nonlinear fit recover the SHAPE? (tests nonlinearity)

Ground truth is set by us, so we can check both.
"""
import argparse
import json

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures


def true_response(insulin):
    # saturating: big drop at first, flattens out. E_max model.
    Emax, K = -60.0, 4.0
    return Emax * insulin / (K + insulin)


def generate(n, world, seed):
    rng = np.random.default_rng(seed)
    glucose = rng.normal(180, 40, n).clip(70, 400)
    U = rng.normal(0, 1, n)                       # unobserved severity
    if world == "observational":
        dose = (0.05 * (glucose - 120) + 3.0 * U + rng.normal(0, 1, n)).clip(0, 12)
    else:                                          # RCT: randomized, independent of glucose & U
        dose = rng.choice([0.0, 2.0, 4.0, 6.0, 8.0], n)   # random arms (graded)
    dg = true_response(dose) + 0.15 * (110 - glucose) + 10.0 * U + rng.normal(0, 12, n)
    return glucose, dose, dg


def fit_and_probe(glucose, dose, dg, tag):
    # features available to the analyst: glucose + dose (U is UNOBSERVED)
    grid = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
    Xg = np.column_stack([np.full_like(grid, glucose.mean()), grid])

    lin = Ridge(1e-6).fit(np.column_stack([glucose, dose]), dg)
    lin_curve = lin.predict(Xg) - lin.predict(Xg[[0]])   # response relative to dose 0

    poly = PolynomialFeatures(2, include_bias=False)
    Xp = poly.fit_transform(np.column_stack([glucose, dose]))
    nl = Ridge(1e-6).fit(Xp, dg)
    nl_curve = nl.predict(poly.transform(Xg)) - nl.predict(poly.transform(Xg[[0]]))

    truth = true_response(grid) - true_response(grid[0])
    print(f"\n=== {tag} ===")
    print(f"  dose grid      : {grid}")
    print(f"  TRUE response  : {np.round(truth,1)}")
    print(f"  linear fit     : {np.round(lin_curve,1)}   (err {np.abs(lin_curve-truth).mean():.1f})")
    print(f"  nonlinear fit  : {np.round(nl_curve,1)}   (err {np.abs(nl_curve-truth).mean():.1f})")
    return {"true": truth.tolist(),
            "linear": [round(float(x),2) for x in lin_curve],
            "nonlinear": [round(float(x),2) for x in nl_curve],
            "linear_err": round(float(np.abs(lin_curve-truth).mean()),3),
            "nonlinear_err": round(float(np.abs(nl_curve-truth).mean()),3)}


def run(n, seed):
    print(f"ground truth: SATURATING insulin->glucose response (Emax model)")
    print(f"truth at doses [0,2,4,6,8]: {np.round(true_response(np.array([0,2,4,6,8.])),1)}")
    out = {}
    for world in ["observational", "RCT"]:
        g, d, dg = generate(n, world, seed)
        out[world] = fit_and_probe(g, d, dg, world.upper())
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="/tmp/synthetic_rct_loop.json")
    a = ap.parse_args()
    o = run(a.n, a.seed)
    with open(a.output, "w") as f:
        json.dump(o, f, indent=1)
    print(f"\nwrote {a.output}")

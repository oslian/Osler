"""
perturbation_response_belief.py

A PERTURBATION-RESPONSE belief for the glucose control loop.

Instead of passively observing glucose, this belief probes the loop with the
insulin/dextrose the patient actually received, and estimates a PER-PATIENT
hidden state: insulin responsiveness (how much glucose drops per unit insulin)
and dextrose responsiveness -- from that patient's OWN prior perturbation
responses within the stay.

Online / no leakage: at each anchor the sensitivity is estimated from a running
regression over STRICTLY PRIOR anchors' (dose -> observed glucose change); the
current anchor's future is added to the accumulator only AFTER emitting.

Gate (same discipline): does adding the per-patient response belief improve 6h
glucose forecasting beyond a baseline that ALREADY has the raw dose, AND beyond a
capacity-matched placebo, across 7 patient-held-out splits?

This is factual: it estimates the observed dose-response, not a causal effect of
a different dose.
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error


def _resolve_action_column(frame, requested, aliases, label):
    candidates = [requested] if requested else []
    candidates.extend(aliases)
    for column in candidates:
        if column and column in frame.columns:
            return column
    raise ValueError(
        f"no {label} action column found; tried {', '.join(c for c in candidates if c)}"
    )


def build_response_belief(
    df,
    time_col,
    future_time_col="t_plus",
    insulin_col=None,
    dextrose_col=None,
    min_prior=3,
    lam=1.0,
):
    n = len(df)
    ins_sens = np.full(n, np.nan)
    dex_sens = np.full(n, np.nan)
    pred_delta = np.full(n, np.nan)
    n_prior = np.zeros(n)

    g_t = pd.to_numeric(df["glucose_t"], errors="coerce").to_numpy()
    g_tp = pd.to_numeric(df["glucose_tp6"], errors="coerce").to_numpy()
    insulin_col = _resolve_action_column(
        df,
        insulin_col,
        ["act_insulin_amount_like_sum", "act_insulin_total"],
        "insulin",
    )
    dextrose_col = _resolve_action_column(
        df,
        dextrose_col,
        ["act_dextrose_amount_like_sum", "act_dextrose_total"],
        "dextrose",
    )
    ins = pd.to_numeric(df[insulin_col], errors="coerce").fillna(0).to_numpy()
    dex = pd.to_numeric(df[dextrose_col], errors="coerce").fillna(0).to_numpy()
    tvals = df[time_col].to_numpy()
    if future_time_col not in df.columns:
        raise ValueError(
            f"{future_time_col!r} is required: response labels must be completed "
            "no later than the current anchor to avoid overlap leakage"
        )
    future_tvals = df[future_time_col].to_numpy()

    for _, g in df.groupby("stay_id", sort=False):
        pos = g.index.to_numpy()
        pos = pos[np.argsort(tvals[pos])]
        XtX = np.zeros((3, 3)); Xty = np.zeros(3); cnt = 0
        pending = []
        for p in pos:
            # A prior anchor is not enough: its 6-hour outcome must already be
            # available by the current anchor. This blocks overlapping-window
            # leakage when anchors are closer than the forecast horizon.
            ready = []
            for prior, item in enumerate(pending):
                future_time = item[2]
                try:
                    is_ready = bool(pd.notna(future_time) and future_time <= tvals[p])
                except TypeError:
                    is_ready = False
                if is_ready:
                    ready.append(prior)
            for prior in ready:
                x_prior, dg_prior, _ = pending[prior]
                XtX += np.outer(x_prior, x_prior)
                Xty += x_prior * dg_prior
                cnt += 1
            if ready:
                ready_set = set(ready)
                pending = [item for i, item in enumerate(pending) if i not in ready_set]
            x = np.array([1.0, ins[p], dex[p]])
            if cnt >= min_prior:
                beta = np.linalg.solve(XtX + lam * np.eye(3), Xty)
                ins_sens[p] = beta[1]
                dex_sens[p] = beta[2]
                pred_delta[p] = float(x @ beta)
                n_prior[p] = cnt
            if np.isfinite(g_tp[p]) and np.isfinite(g_t[p]) and pd.notna(future_tvals[p]):
                dg = g_tp[p] - g_t[p]
                pending.append((x, dg, future_tvals[p]))
    return pd.DataFrame({
        "resp_insulin_sensitivity": ins_sens,
        "resp_dextrose_sensitivity": dex_sens,
        "resp_predicted_delta": pred_delta,
        "resp_n_prior": n_prior,
    }, index=df.index)


def run(
    cohort,
    time_col,
    id_col,
    future_time_col,
    seeds,
    min_rows,
    insulin_col=None,
    dextrose_col=None,
):
    df = pd.read_parquet(cohort)
    df = df.sort_values(["stay_id", time_col]).reset_index(drop=True)
    seed_values = list(seeds)
    print(f"rows={len(df)}  subjects={df[id_col].nunique()}  stays={df['stay_id'].nunique()}")

    print("building per-patient insulin/dextrose response belief (prior-only) ...")
    Bel = build_response_belief(
        df,
        time_col,
        future_time_col=future_time_col,
        insulin_col=insulin_col,
        dextrose_col=dextrose_col,
    )
    print(f"anchors with an estimated response state: {int(Bel['resp_n_prior'].gt(0).sum())}")

    snap = [c for c in df.columns if c.endswith("_t") and df[c].notna().any()]
    dose = [
        "act_insulin_amount_like_sum", "act_insulin_total",
        "act_dextrose_amount_like_sum", "act_dextrose_total",
        "hist_insulin_amount_like_sum", "hist_dextrose_amount_like_sum",
    ]
    dose = [c for c in dose if c in df.columns]
    base_cols = snap + dose

    y = pd.to_numeric(df["glucose_tp6"], errors="coerce").to_numpy()
    cur = pd.to_numeric(df["glucose_t"], errors="coerce").to_numpy()
    Xbase = df[base_cols].apply(pd.to_numeric, errors="coerce")
    Bcols = list(Bel.columns)

    base_mae, cand_mae, plac_mae, per_mae = [], [], [], []
    for s in seed_values:
        rng = np.random.default_rng(s)
        subs = df[id_col].unique(); rng.shuffle(subs)
        te = set(subs[: int(0.3 * len(subs))])
        is_te = df[id_col].isin(te).to_numpy()
        ok = np.isfinite(y) & np.isfinite(cur)
        tr = (~is_te) & ok; ev = is_te & ok
        if ev.sum() < min_rows:
            continue

        med = Xbase[tr].median()
        Xb = Xbase.fillna(med).to_numpy()
        Bm = Bel.fillna(Bel[tr].median()).to_numpy()
        scB = StandardScaler().fit(Xb[tr]); Xb = np.nan_to_num(scB.transform(Xb))
        scb = StandardScaler().fit(Bm[tr]); Bm = np.nan_to_num(scb.transform(Bm))
        P = rng.normal(size=(len(df), Bm.shape[1]))

        yt = y[ev]
        per_mae.append(mean_absolute_error(yt, cur[ev]))
        base_mae.append(mean_absolute_error(yt, Ridge(1.0).fit(Xb[tr], y[tr]).predict(Xb[ev])))
        Xc = np.hstack([Xb, Bm]); Xp = np.hstack([Xb, P])
        cand_mae.append(mean_absolute_error(yt, Ridge(1.0).fit(Xc[tr], y[tr]).predict(Xc[ev])))
        plac_mae.append(mean_absolute_error(yt, Ridge(1.0).fit(Xp[tr], y[tr]).predict(Xp[ev])))

    per_mae, base_mae, cand_mae, plac_mae = map(np.array, (per_mae, base_mae, cand_mae, plac_mae))
    bb = int((cand_mae < base_mae).sum()); bp = int((cand_mae < plac_mae).sum()); N = len(cand_mae)
    print(f"\n{'model':22s} {'mean MAE':>10s}")
    print(f"{'persistence':22s} {per_mae.mean():10.3f}")
    print(f"{'ridge (snapshot+dose)':22s} {base_mae.mean():10.3f}")
    print(f"{'+ placebo':22s} {plac_mae.mean():10.3f}")
    print(f"{'+ response belief':22s} {cand_mae.mean():10.3f}")
    complete = N == len(seed_values)
    validated = bool(complete and bb == N and bp == N)
    print(f"\nresponse belief beats baseline: {bb}/{N}   beats placebo: {bp}/{N}")
    print(f"splits evaluated: {N}/{len(seed_values)}")
    print(f"VALIDATED: {validated}")
    return {"targets": "glucose_tp6", "beats_baseline": f"{bb}/{N}", "beats_placebo": f"{bp}/{N}",
            "splits_evaluated": N,
            "splits_requested": len(seed_values),
            "complete_seed_gate": complete,
            "validated": validated,
            "persistence_mae": round(float(per_mae.mean()), 4),
            "baseline_mae": round(float(base_mae.mean()), 4),
            "candidate_mae": round(float(cand_mae.mean()), 4),
            "placebo_mae": round(float(plac_mae.mean()), 4)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--time-col", default="t")
    ap.add_argument("--future-time-col", default="t_plus")
    ap.add_argument("--insulin-col", default=None)
    ap.add_argument("--dextrose-col", default=None)
    ap.add_argument("--id-col", default="subject_id")
    ap.add_argument("--seeds", type=int, default=7)
    ap.add_argument("--min-rows", type=int, default=500)
    ap.add_argument("--output", default="/tmp/perturbation_response_belief.json")
    a = ap.parse_args()
    out = run(
        a.cohort,
        a.time_col,
        a.id_col,
        a.future_time_col,
        range(a.seeds),
        a.min_rows,
        insulin_col=a.insulin_col,
        dextrose_col=a.dextrose_col,
    )
    with open(a.output, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {a.output}")

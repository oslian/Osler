"""
learned_worldmodel_vs_router.py

Exploratory head-to-head for a static learned transition predictor:
does a learned MLP beat persistence, raw-state Ridge, and the existing
hand-crafted belief-feature Ridge on real transitions, patient-held-out?

This is not the production JEPA dynamics model. It is a controlled comparator
for deciding whether a learned tabular representation adds value at all.

Three contenders on the same held-out patients, same targets, same MAE metric:
  1. persistence          -- predict "no change" (current value)         [floor]
  2. per-target ridge     -- one ridge per target on raw current features
  3. belief ridge          -- raw current features + validated online beliefs
  4. multi-target MLP     -- ONE network predicts all targets jointly    [learned shared representation]
  5. per-target MLP       -- one learned model per target                 [capacity reference]

The scientific question: can a learned representation capture cross-variable
structure better than hand-crafted belief features + per-target ridge?

Discipline: patient-held-out (grouped by subject), 7 seeds, per-target reporting,
treatment features are FACTUAL INPUT ONLY (no causal claim), no future *_tp* leakage
except the target being predicted. Sparse/slow variables are expected to stay at
persistence -- learning does not create signal that is not in the data.
"""
import argparse
import json
import re

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error

try:
    from aki_renal_belief import renal_belief_state_v2_features
    from cardiovascular_belief import cardiovascular_belief_state_features
    from electrolyte_belief import electrolyte_belief_state_features
    from respiratory_belief import respiratory_belief_state_features
    from endocrine_belief import endocrine_belief_state_features
    REAL_BELIEF_BUILDERS = [
        renal_belief_state_v2_features,
        cardiovascular_belief_state_features,
        electrolyte_belief_state_features,
        respiratory_belief_state_features,
        endocrine_belief_state_features,
    ]
except Exception:
    REAL_BELIEF_BUILDERS = []


def detect_target_pairs(df, future_suffix, current_suffix="_t"):
    cols = set(df.columns)
    pairs = []
    for col in df.columns:
        if col.endswith(future_suffix):
            stem = col[: -len(future_suffix)]
            current = stem + current_suffix
            if current in cols:
                pairs.append((current, col))
    return pairs


def is_future_feature(column, configured_suffix):
    """Reject all recognizable future-state columns from model inputs.

    A cohort can contain several horizons at once. Dropping only the selected
    suffix (for example ``_tp6``) would leak ``_tp24`` or ``_tp48`` into a
    nominal 6-hour comparison.
    """
    name = str(column).lower()
    suffix = str(configured_suffix).lower()
    if suffix and name.endswith(suffix):
        return True
    return bool(
        re.search(r"(?:^|_)(?:tp\d+|future|next)(?:_|$)", name)
    )


def build_xy(df, target_pairs, future_suffix, id_col, extra_drop):
    target_future = [f for _, f in target_pairs]
    drop = set(target_future) | {id_col} | set(extra_drop)
    # features = every current-state numeric column (incl. current target values,
    # which is exactly what persistence uses) + hist_*/act_* treatment context.
    feat_cols = [
        c for c in df.columns
        if c not in drop and not is_future_feature(c, future_suffix)
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    return feat_cols


def compute_real_beliefs(df):
    """Build the existing online-compatible belief features when available."""
    if "stay_id" not in df.columns or not REAL_BELIEF_BUILDERS:
        return None
    blocks = []
    for builder in REAL_BELIEF_BUILDERS:
        try:
            block = builder(df).reindex(df.index)
            blocks.append(block)
        except Exception as exc:
            print(f"  belief builder skipped: {exc}")
    if not blocks:
        return None
    return pd.concat(blocks, axis=1)


def run(cohort, future_suffix, id_col, seeds, hidden, min_rows, extra_drop, current_suffix="_t", mlp_cap=40000):
    df = pd.read_parquet(cohort)
    target_pairs = detect_target_pairs(df, future_suffix, current_suffix)
    extra_drop = list(extra_drop) + ["stay_id", "t", "t_plus", "first_careunit", "last_careunit"]
    feat_cols = build_xy(df, target_pairs, future_suffix, id_col, extra_drop)
    feat_cols = [c for c in feat_cols if df[c].notna().any()]
    future_inputs = [c for c in df.columns if is_future_feature(c, future_suffix)]
    belief_frame = compute_real_beliefs(df)
    print(f"rows={len(df)}  subjects={df[id_col].nunique()}  "
          f"features={len(feat_cols)}  target_pairs={len(target_pairs)}  "
          f"future_columns_excluded={len(future_inputs)}  "
          f"belief_features={0 if belief_frame is None else belief_frame.shape[1]}")

    results = {}
    for seed in range(seeds):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
        tr_idx, te_idx = next(gss.split(df, groups=df[id_col]))
        tr, te = df.iloc[tr_idx], df.iloc[te_idx]

        Xtr_raw = tr[feat_cols].apply(pd.to_numeric, errors="coerce")
        Xte_raw = te[feat_cols].apply(pd.to_numeric, errors="coerce")
        med = Xtr_raw.median()
        keep = [c for c in feat_cols if np.isfinite(med[c])]
        med = med[keep]
        Xtr = Xtr_raw[keep].fillna(med).to_numpy()
        Xte = Xte_raw[keep].fillna(med).to_numpy()
        sc = StandardScaler().fit(Xtr)
        Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)

        if belief_frame is not None:
            Btr_raw = belief_frame.iloc[tr_idx].apply(pd.to_numeric, errors="coerce")
            Bte_raw = belief_frame.iloc[te_idx].apply(pd.to_numeric, errors="coerce")
            Bmed = Btr_raw.median().fillna(0.0)
            Btr = Btr_raw.fillna(Bmed).to_numpy()
            Bte = Bte_raw.fillna(Bmed).to_numpy()
            Bsc = StandardScaler().fit(Btr)
            Xtr_belief = np.hstack([Xtr_s, np.nan_to_num(Bsc.transform(Btr))])
            Xte_belief = np.hstack([Xte_s, np.nan_to_num(Bsc.transform(Bte))])
        else:
            Xtr_belief = Xte_belief = None

        # multi-target MLP: ONE shared network for ALL targets jointly.
        # ICU labs are sparsely co-measured, so almost no row has every future
        # target. Train on all rows; impute missing TRAINING targets with the
        # train median (sparse targets then behave near-median = honest weak
        # fallback), and EVALUATE only on truly-present values in the loop below.
        bases = [b for b, _ in target_pairs]
        futs = [f for _, f in target_pairs]
        Ytr = tr[futs].apply(pd.to_numeric, errors="coerce")
        ymed = Ytr.median()
        keept = [i for i in range(len(bases)) if np.isfinite(ymed[futs[i]])]
        bases = [bases[i] for i in keept]
        futs = [futs[i] for i in keept]
        Ytr = Ytr[futs].fillna(ymed[futs])
        yscaler = StandardScaler().fit(Ytr)
        mlp = MLPRegressor(hidden_layer_sizes=hidden, alpha=1e-3, max_iter=200,
                           early_stopping=True, random_state=seed)
        mlp.fit(Xtr_s, yscaler.transform(Ytr))
        mlp_pred_all = pd.DataFrame(yscaler.inverse_transform(mlp.predict(Xte_s)),
                                    columns=bases, index=te.index)

        for base, fut in target_pairs:
            if base not in mlp_pred_all.columns:
                continue
            yte = pd.to_numeric(te[fut], errors="coerce")
            ok = yte.notna() & te[base].notna()
            if ok.sum() < min_rows:
                continue
            yt = yte[ok].to_numpy()
            # 1. persistence
            per = mean_absolute_error(yt, pd.to_numeric(te[base], errors="coerce")[ok].to_numpy())
            # 2. per-target ridge
            ytr = pd.to_numeric(tr[fut], errors="coerce")
            trok = ytr.notna().to_numpy()
            if trok.sum() < min_rows:
                continue
            rid = Ridge(alpha=1.0).fit(Xtr_s[trok], ytr[trok].to_numpy())
            rmae = mean_absolute_error(yt, rid.predict(Xte_s[ok.to_numpy()]))
            if Xtr_belief is not None:
                rid_belief = Ridge(alpha=1.0).fit(
                    Xtr_belief[trok], ytr[trok].to_numpy()
                )
                rbmae = mean_absolute_error(
                    yt, rid_belief.predict(Xte_belief[ok.to_numpy()])
                )
            else:
                rbmae = np.nan
            # 3. shared MLP (jointly trained across all targets, median-imputed)
            mmae = mean_absolute_error(yt, mlp_pred_all[base][ok].to_numpy())
            # 4. per-target MLP: FULL capacity, ONLY present rows, no imputation.
            #    This removes both handicaps of the shared MLP = the fair test.
            tr_rows = np.where(trok)[0]
            if len(tr_rows) > mlp_cap:
                tr_rows = np.random.default_rng(seed).choice(tr_rows, mlp_cap, replace=False)
            ptm = MLPRegressor(hidden_layer_sizes=hidden, alpha=1e-3, max_iter=300,
                               early_stopping=True, random_state=seed)
            ptm.fit(Xtr_s[tr_rows], ytr.to_numpy()[tr_rows])
            ptmae = mean_absolute_error(yt, ptm.predict(Xte_s[ok.to_numpy()]))
            results.setdefault(base, {
                "persistence": [], "ridge": [], "ridge_belief": [],
                "mlp_share": [], "mlp_pt": [],
            })
            results[base]["persistence"].append(per)
            results[base]["ridge"].append(rmae)
            results[base]["ridge_belief"].append(rbmae)
            results[base]["mlp_share"].append(mmae)
            results[base]["mlp_pt"].append(ptmae)

    summary = {}
    print(f"\n{'target':22s} {'persist':>8s} {'ridge':>8s} {'belief':>8s} "
          f"{'mlp_sh':>8s} {'mlp_pt':>8s}  {'pt<persist':>10s} {'pt<belief':>9s}")
    for base, r in sorted(results.items()):
        p, rd, rb, ms, mp = map(np.array, (
            r["persistence"], r["ridge"], r["ridge_belief"],
            r["mlp_share"], r["mlp_pt"],
        ))
        valid_rb = np.isfinite(rb)
        rb_mean = float(rb[valid_rb].mean()) if valid_rb.any() else float("nan")
        bp = int((mp < p).sum()); br = int((mp < rd).sum()); n = len(mp)
        bb = int((mp[valid_rb] < rb[valid_rb]).sum()) if valid_rb.any() else 0
        nb = int(valid_rb.sum())
        summary[base] = {
            "persistence": round(float(p.mean()), 4),
            "ridge": round(float(rd.mean()), 4),
            "ridge_belief": round(rb_mean, 4) if np.isfinite(rb_mean) else None,
            "mlp_share": round(float(ms.mean()), 4),
            "mlp_pt": round(float(mp.mean()), 4),
            "mlp_pt_beats_persistence": f"{bp}/{n}",
            "mlp_pt_beats_ridge": f"{br}/{n}",
            "mlp_pt_beats_ridge_belief": f"{bb}/{nb}" if nb else "0/0",
            "mlp_pt_wins": bool(n == seeds and bp == n and br == n),
        }
        flag = "  PT-MLP WINS" if summary[base]["mlp_pt_wins"] else ""
        print(f"{base:22s} {p.mean():8.3f} {rd.mean():8.3f} {rb_mean:8.3f} "
              f"{ms.mean():8.3f} {mp.mean():8.3f}  {bp:>6d}/{n} {bb:>5d}/{nb}{flag}")

    wins = [k for k, v in summary.items() if v["mlp_pt_wins"]]
    print(f"\nper-target MLP beats BOTH persistence and ridge on: {len(wins)}/{len(summary)} targets")
    print(f"  {wins}")
    return {"n_targets": len(summary), "mlp_pt_wins": wins, "per_target": summary}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, help="real transition parquet (state_t + *_tp6 + treatment)")
    ap.add_argument("--future-suffix", default="_tp6")
    ap.add_argument("--id-col", default="subject_id")
    ap.add_argument("--seeds", type=int, default=7)
    ap.add_argument("--hidden", default="128,64")
    ap.add_argument("--min-rows", type=int, default=200)
    ap.add_argument("--drop", default="", help="comma-separated extra columns to exclude from features")
    ap.add_argument("--mlp-cap", type=int, default=40000, help="max training rows per per-target MLP")
    ap.add_argument("--output", default="/tmp/learned_worldmodel_vs_router.json")
    a = ap.parse_args()
    hidden = tuple(int(x) for x in a.hidden.split(","))
    extra_drop = [c for c in a.drop.split(",") if c]
    out = run(a.cohort, a.future_suffix, a.id_col, a.seeds, hidden, a.min_rows, extra_drop,
              mlp_cap=a.mlp_cap)
    with open(a.output, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {a.output}")

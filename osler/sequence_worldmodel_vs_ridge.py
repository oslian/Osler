"""
sequence_worldmodel_vs_ridge.py

The decisive test: does a learned sequence model (GRU over within-stay history)
beat the ACTUAL current system = ridge + hand-crafted BELIEF state features
(which already encode level/slope/innovation of the same history)?

Contenders per dense target, patient-held-out, MAE:
  1. persistence     -- current value (no change)                         [floor]
  2. ridge_snapshot  -- ridge on current state only                       [snapshot]
  3. ridge_belief    -- ridge on current state + real belief features     [DEPLOYED system]
  4. gru             -- GRU over the last K states (learned history rep)   [learned]

If gru < ridge_belief -> the learned representation beats the hand-crafted belief
  slopes on the SAME history = first real "learning > feature engineering".
If gru >= ridge_belief -> the belief states already captured the history; learning
  adds no net gain -> clean close.

Discipline: patient-held-out by subject; belief features from the real belief
builders (grouped by stay, online-compatible); targets median-imputed for training
only, evaluated on truly-present values; masked GRU loss; no future leakage.
"""
import argparse
import json
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

try:
    from aki_renal_belief import renal_belief_state_v2_features
    from cardiovascular_belief import cardiovascular_belief_state_features
    from electrolyte_belief import electrolyte_belief_state_features
    from respiratory_belief import respiratory_belief_state_features
    from endocrine_belief import endocrine_belief_state_features
    BELIEF_BUILDERS = [
        ("renal", renal_belief_state_v2_features),
        ("cardiovascular", cardiovascular_belief_state_features),
        ("electrolyte", electrolyte_belief_state_features),
        ("respiratory", respiratory_belief_state_features),
        ("endocrine", endocrine_belief_state_features),
    ]
except Exception as exc:
    print("belief import failed:", exc)
    BELIEF_BUILDERS = []

DENSE = ["glucose", "potassium", "sodium", "chloride", "bicarbonate", "anion_gap",
         "bun", "creatinine", "calcium", "magnesium", "phosphate", "heart_rate",
         "map", "o2sat", "respiratory_rate", "temperature", "lactate",
         "hemoglobin", "hematocrit", "wbc", "platelets"]


class GRUNet(nn.Module):
    def __init__(self, n_feat, n_out, hidden=64):
        super().__init__()
        self.gru = nn.GRU(n_feat, hidden, batch_first=True)
        self.head = nn.Linear(hidden, n_out)

    def forward(self, x):
        _, h = self.gru(x)
        return self.head(h[-1])


class DiffBelief(nn.Module):
    """Differentiable predict-update belief: a Holt (level + trend) filter with
    LEARNABLE smoothing parameters per channel. Same interpretable predict-update
    structure as the hand-set belief, but its parameters are learned end-to-end
    instead of hand-tuned. Few params (2 per channel + a linear head)."""

    def __init__(self, n_ch, n_out):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(n_ch))
        self.beta = nn.Parameter(torch.full((n_ch,), -1.0))
        self.head = nn.Linear(2 * n_ch, n_out)

    def forward(self, x):
        a = torch.sigmoid(self.alpha)
        b = torch.sigmoid(self.beta)
        level = x[:, 0, :]
        trend = torch.zeros_like(level)
        for t in range(1, x.shape[1]):
            prev = level
            level = a * x[:, t, :] + (1 - a) * (level + trend)
            trend = b * (level - prev) + (1 - b) * trend
        return self.head(torch.cat([level, trend], dim=1))


def vicreg_terms(z):
    """VICReg anti-collapse: variance term keeps each embedding dim's std >= 1
    (stops dimensions collapsing to constants); covariance term decorrelates
    dimensions (stops rank collapse)."""
    std = torch.sqrt(z.var(dim=0) + 1e-4)
    var_loss = torch.relu(1.0 - std).mean()
    zc = z - z.mean(dim=0)
    cov = (zc.T @ zc) / (z.shape[0] - 1)
    off = cov - torch.diag(torch.diagonal(cov))
    cov_loss = (off ** 2).sum() / z.shape[1]
    return var_loss, cov_loss


class VICRegNet(nn.Module):
    """GRU encoder + projection to an embedding, trained with a supervised head
    PLUS VICReg regularization on the embedding to prevent representation
    collapse (the failure mode seen in the earlier JEPA)."""

    def __init__(self, n_feat, n_out, hidden=64, emb=64):
        super().__init__()
        self.gru = nn.GRU(n_feat, hidden, batch_first=True)
        self.proj = nn.Linear(hidden, emb)
        self.head = nn.Linear(emb, n_out)

    def forward(self, x):
        _, h = self.gru(x)
        z = self.proj(h[-1])
        return self.head(z), z


def build_sequences(df, k, seq_group, time_col):
    seq_idx = np.zeros((len(df), k), dtype=np.int64)
    tvals = df[time_col].to_numpy()
    for _, g in df.groupby(seq_group, sort=False):
        pos = g.index.to_numpy()
        pos = pos[np.argsort(tvals[pos])]
        for j, p in enumerate(pos):
            lo = max(0, j - k + 1)
            hist = pos[lo:j + 1]
            if len(hist) < k:
                hist = np.concatenate([np.full(k - len(hist), hist[0]), hist])
            seq_idx[p] = hist
    return seq_idx


def compute_beliefs(df):
    blocks = []
    for name, builder in BELIEF_BUILDERS:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                blocks.append(builder(df).reset_index(drop=True))
        except Exception as exc:
            print(f"  belief {name} skipped: {exc}")
    if not blocks:
        return None
    return pd.concat(blocks, axis=1)


def _std_matrix(raw, tr_rows):
    med = raw.iloc[tr_rows].median().fillna(0.0)
    M = raw.fillna(med).to_numpy(dtype=np.float32)
    sc = StandardScaler().fit(M[tr_rows])
    return np.nan_to_num(sc.transform(M)).astype(np.float32)


def run(cohort, k, hidden, epochs, cap, seed, id_col, time_col, future_suffix, max_stays):
    df = pd.read_parquet(cohort)
    if "stay_id" not in df.columns:
        df["stay_id"] = df[id_col]
    if max_stays and df["stay_id"].nunique() > max_stays:
        keep = pd.Series(df["stay_id"].unique()).sample(max_stays, random_state=seed)
        df = df[df["stay_id"].isin(set(keep))]
    df = df.sort_values(["stay_id", time_col]).reset_index(drop=True)

    feat_cols = [c for c in df.columns if c.endswith("_t") and df[c].notna().any()]
    targets = [(f"{d}_t", f"{d}{future_suffix}") for d in DENSE
               if f"{d}_t" in df.columns and f"{d}{future_suffix}" in df.columns]
    tgt_fut = [f for _, f in targets]
    print(f"rows={len(df)}  subjects={df[id_col].nunique()}  stays={df['stay_id'].nunique()}  "
          f"seq_feats={len(feat_cols)}  dense_targets={len(targets)}  K={k}")

    print("computing real belief features (grouped by stay) ...")
    B = compute_beliefs(df)
    print(f"belief features: {0 if B is None else B.shape[1]}")

    seq_idx = build_sequences(df, k, "stay_id", time_col)

    rng = np.random.default_rng(seed)
    subs = df[id_col].unique(); rng.shuffle(subs)
    te_subs = set(subs[: int(0.3 * len(subs))])
    is_te = df[id_col].isin(te_subs).to_numpy()
    tr_full = np.where(~is_te)[0]; te_rows = np.where(is_te)[0]

    F = _std_matrix(df[feat_cols].apply(pd.to_numeric, errors="coerce"), tr_full)
    Bmat = (_std_matrix(B.apply(pd.to_numeric, errors="coerce"), tr_full)
            if B is not None else np.zeros((len(df), 0), dtype=np.float32))

    Y = df[tgt_fut].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    ymask = ~np.isnan(Y)
    ymed = np.nanmedian(np.where(ymask, Y, np.nan)[tr_full], axis=0)
    Yf = np.where(ymask, Y, ymed).astype(np.float32)
    ysc = StandardScaler().fit(Yf[tr_full]); Yn = ysc.transform(Yf).astype(np.float32)

    tr_rows = tr_full if len(tr_full) <= cap else rng.choice(tr_full, cap, replace=False)
    Xtr = torch.tensor(F[seq_idx[tr_rows]]); Ytr = torch.tensor(Yn[tr_rows])
    Mtr = torch.tensor(ymask[tr_rows].astype(np.float32))
    Xte = torch.tensor(F[seq_idx[te_rows]])

    def train_predict(net, tag):
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        n = len(tr_rows); bs = 512
        for ep in range(epochs):
            net.train(); perm = torch.randperm(n)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]; opt.zero_grad()
                loss = (((net(Xtr[idx]) - Ytr[idx]) ** 2) * Mtr[idx]).sum() / Mtr[idx].sum().clamp(min=1)
                loss.backward(); opt.step()
            print(f"  [{tag}] epoch {ep+1}/{epochs}  masked_mse={loss.item():.4f}")
        net.eval()
        with torch.no_grad():
            return ysc.inverse_transform(net(Xte).numpy())

    gpred = train_predict(GRUNet(len(feat_cols), len(targets), hidden), "gru")
    dpred = train_predict(DiffBelief(len(feat_cols), len(targets)), "diff_belief")

    def train_vicreg(net, lam_var=1.0, lam_cov=0.1):
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        n = len(tr_rows); bs = 512
        for ep in range(epochs):
            net.train(); perm = torch.randperm(n)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]; opt.zero_grad()
                pred, z = net(Xtr[idx])
                sup = (((pred - Ytr[idx]) ** 2) * Mtr[idx]).sum() / Mtr[idx].sum().clamp(min=1)
                vloss, closs = vicreg_terms(z)
                (sup + lam_var * vloss + lam_cov * closs).backward(); opt.step()
            print(f"  [vicreg] epoch {ep+1}/{epochs}  sup={sup.item():.4f}  var={vloss.item():.4f}  cov={closs.item():.4f}")
        net.eval()
        with torch.no_grad():
            pred, z = net(Xte)
            zc = z - z.mean(0)
            s = torch.linalg.svdvals(zc); pnorm = s / s.sum()
            erank = float(torch.exp(-(pnorm * torch.log(pnorm + 1e-9)).sum()))
            print(f"  [vicreg] embedding effective_rank={erank:.2f} / {z.shape[1]}  (high = NOT collapsed)")
            return ysc.inverse_transform(pred.numpy())

    vpred = train_vicreg(VICRegNet(len(feat_cols), len(targets), hidden))

    Xtr_snap, Xte_snap = F[tr_full], F[te_rows]
    Xtr_bel = np.hstack([F[tr_full], Bmat[tr_full]])
    Xte_bel = np.hstack([F[te_rows], Bmat[te_rows]])

    print(f"\n{'target':16s} {'persist':>8s} {'ridge_belief':>12s} {'gru':>8s} {'diff_bel':>9s} {'vicreg':>8s}  winner")
    summary = {}
    for ti, (cur, fut) in enumerate(targets):
        name = cur[:-2]
        yte = Y[te_rows, ti]
        ok = ~np.isnan(yte) & df[cur].iloc[te_rows].notna().to_numpy()
        if ok.sum() < 200:
            continue
        yt = yte[ok]
        per = mean_absolute_error(yt, df[cur].iloc[te_rows].to_numpy()[ok])
        ytr = Y[tr_full, ti]; trok = ~np.isnan(ytr)
        bel = mean_absolute_error(yt, Ridge(1.0).fit(Xtr_bel[trok], ytr[trok]).predict(Xte_bel[ok]))
        gru = mean_absolute_error(yt, gpred[ok, ti])
        dbl = mean_absolute_error(yt, dpred[ok, ti])
        vic = mean_absolute_error(yt, vpred[ok, ti])
        scores = {"persistence": per, "ridge_belief": bel, "gru": gru,
                  "diff_belief": dbl, "vicreg": vic}
        best = min(scores, key=scores.get)
        summary[name] = {kk: round(float(v), 4) for kk, v in scores.items()} | {"winner": best}
        print(f"{name:16s} {per:8.3f} {bel:12.3f} {gru:8.3f} {dbl:9.3f} {vic:8.3f}  {best}")

    vic_best = [k2 for k2, v in summary.items() if v["winner"] == "vicreg"]
    vic_beats_belief = [k2 for k2, v in summary.items() if v["vicreg"] < v["ridge_belief"]]
    print(f"\nvicreg outright best on: {len(vic_best)}/{len(summary)}  {vic_best}")
    print(f"vicreg beats ridge_BELIEF on: {len(vic_beats_belief)}/{len(summary)}  {vic_beats_belief}")
    return {"targets": len(summary), "vicreg_best": vic_best,
            "vicreg_beats_belief": vic_beats_belief, "per_target": summary}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--cap", type=int, default=150000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-stays", type=int, default=30000,
                    help="subsample stays to keep belief computation tractable")
    ap.add_argument("--id-col", default="subject_id")
    ap.add_argument("--time-col", default="t")
    ap.add_argument("--future-suffix", default="_tp6")
    ap.add_argument("--output", default="/tmp/sequence_worldmodel_vs_ridge_belief.json")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    out = run(a.cohort, a.k, a.hidden, a.epochs, a.cap, a.seed, a.id_col, a.time_col,
              a.future_suffix, a.max_stays)
    with open(a.output, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {a.output}")

# Observation Representation Repair Findings

## Decision

The three proposed repairs were tested together without modifying the release.
No candidate passes a complete point, prospective-risk, or interval gate across
eICU patient-heldout, eICU hospital-heldout, and NWICU. The experiment remains
candidate-only.

The result narrows the failure: explicit observation state carries some real
transfer information for glucose, and raw cross-variable input is often better
than the frozen latent for glucose. Neither is stable enough to beat the same-
capacity target-history baseline across source and target domains. The old
combined gate was not hiding a broad success.

## Fixed Comparison

The release JEPA remained frozen and measurement-pure. All learned arms used the
same HGB estimator settings and the same seven target-history summaries:

| Arm | Additional information |
|---|---|
| `history` | Target's own pre-anchor history only |
| `raw` | Current normalized non-target physiology |
| `aware_raw` | Raw physiology plus explicit observed masks and ages |
| `latent` | Frozen 311-dimensional shared JEPA feature contract |
| `aware_latent` | Frozen JEPA plus explicit non-target masks and ages |

Every added block was compared with a same-width wrong-patient placebo while
retaining the target patient's own history. No future value, future measurement
time, hospital identifier, or care-unit identifier was supplied as input.

## Point Forecast

MAE, lower is better:

| Target / cohort | History | Release | Raw | Aware raw | Frozen latent | Aware latent |
|---|---:|---:|---:|---:|---:|---:|
| Glucose / eICU patient | 27.115 | 30.054 | 27.517 | 27.933 | 28.302 | 28.369 |
| Glucose / eICU hospital | 29.875 | 32.878 | 29.454 | 30.711 | 30.588 | 31.086 |
| Glucose / NWICU | 28.794 | 28.761 | 28.775 | **28.470** | 29.127 | 28.838 |
| MAP / eICU patient | **9.676** | 10.082 | 9.823 | 9.839 | 9.825 | 9.865 |
| MAP / eICU hospital | 10.261 | 10.654 | **10.220** | 10.291 | 10.479 | 10.514 |
| MAP / NWICU | 9.023 | 9.352 | **8.910** | 8.923 | 8.911 | 8.922 |

No arm passes the point gate. Glucose `aware_raw` beats its wrong-patient
placebo on NWICU (paired MAE CI [-1.311,-0.234]) but does not significantly beat
history or release. On eICU patient-heldout, it is worse than history by 0.818
MAE and the CI crosses zero. Its source-hospital result is also not stable.

For MAP, the raw, latent, and observation-aware arms all beat release on NWICU,
but none significantly beats history or its matched placebo. On eICU patient-
heldout, raw is significantly worse than history (CI [0.014,0.281]). This is the
earliest observed failure in this audit.

The raw-versus-latent pattern is target-specific. Raw is descriptively better
for glucose in all three cohorts, locating some loss in frozen compression, but
the raw information itself still fails to add stable value beyond target history.
MAP shows no meaningful raw-versus-latent separation.

## Prospective Change Risk

No arm passes the separately scored risk gate. At a matched 5% FPR, no candidate
has a significant recall advantage over history, release, and its placebo in all
cohorts. Source-frozen thresholds also transfer poorly for glucose: NWICU FPR is
9.30% for `aware_raw` and 8.53% for `raw`. This failure is independent of point
MAE and interval calibration.

## Interval Calibration

No arm passes the separate interval gate. Marginal coverage often looks close to
90%, but care-unit coverage and interval-score superiority do not transfer.

- Glucose `aware_raw`: NWICU coverage 90.0%, mean width 126.19 mg/dL. Care-unit
  coverage ranges from 81.3% in MICU to 97.4% in NSICU, and interval score does
  not significantly beat history, release, and placebo together.
- MAP `raw`: NWICU coverage 92.6%, mean width 42.15 mmHg. CTICU coverage is 100%,
  outside the prespecified band, and interval-score comparisons cross zero.

Separating the gates therefore corrects the semantics of evaluation but does
not manufacture a promotion. Calibration can improve versus the old release in
some source roles while point and risk capabilities remain unsupported.

## What The Three Repairs Established

1. **Observation semantics:** partially supported. Missing/fresh/stale metadata
   helps external glucose relative to a wrong-patient placebo, but not reliably
   across source roles or against personal history.
2. **Representation attribution:** partially supported for glucose. Raw cross-
   variable input is consistently better than the frozen latent descriptively,
   yet neither gives stable incremental value beyond personal history. A full
   encoder retrain is therefore not justified by this two-target audit alone.
3. **Independent capability gates:** necessary and implemented. The failures are
   real in all three components; the previous combined screen was not the sole
   reason candidates remained unpromoted.

## Updated Failure Location

The first software-level information loss is still the runtime's neutralized
measurement representation: measured median and missing map to the same numeric
input, and fresh and stale equal-valued observations map identically. This audit
shows that restoring those semantics only at the readout is insufficient.

The first performance failure is before external transfer: on source patient-
heldout data, additional whole-body inputs do not reliably improve over the
patient's own target history. The evidence therefore supports two contributors,
not a single root cause:

- frozen shared compression loses some glucose-relevant cross-variable detail;
- the remaining pre-anchor cross-variable context has weak and heterogeneous
  incremental information about the 3h future after own-target history is known.

This does not prove that a new end-to-end observation-aware JEPA cannot help.
It does prove that simply appending masks/ages to a frozen representation, adding
more quantiles, or changing the acceptance bookkeeping is insufficient.

## Reproduction

```sh
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  LOKY_MAX_CPU_COUNT=4 \
  /Users/chunyouchang/Desktop/llm_project/medical_jepa/.venv/bin/python -B -u \
  -m scripts.audit_representation_repair \
  --output artifacts/representation_repair_20260907
```

Machine-readable results are in
`artifacts/representation_repair_20260907/report.json`; fitted weights are
research-only and were frozen before NWICU evaluation. NWICU has been examined
previously, original encoder-training overlap is not fully audited, and no
clinical or causal claim is allowed.

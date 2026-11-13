# Observation Representation Repair Protocol

This is a candidate-only diagnosis. It does not modify the release runtime,
checkpoint, registry, or serving intervals.

## Questions

1. Does explicit pre-anchor measurement state add information beyond target
   history when missing, stale, and freshly measured values are distinguishable?
2. Is useful cross-variable information present in raw observations but lost in
   the frozen shared JEPA representation?
3. Do point prediction, prospective change ranking, and interval calibration
   reach different conclusions when assessed independently?

## Fixed Comparison

Glucose@3h and MAP@3h use the existing patient/hospital-disjoint eICU role
definition and a newly hash-selected 500-patient sample from the previously
examined NWICU transfer dataset. Every learned arm uses the same
hist-gradient-boosting settings and the same seven target-history summaries.

- `history`: target history only.
- `raw`: history plus current normalized non-target values.
- `aware_raw`: raw plus explicit observed masks and measurement ages.
- `latent`: history plus the frozen 311-dimensional JEPA feature contract.
- `aware_latent`: latent plus the explicit non-target masks and ages.

Each added block has a capacity-matched wrong-patient placebo. Target anchor
values remain the patient's own. No future measurement, hospital identifier, or
care-unit identifier is an input. The frozen encoder remains measurement-pure;
measurement state is tested only in the readout so this audit cannot create an
out-of-distribution encoder input.

## Separate Gates

- Point: candidate must significantly beat history, release, and its matched
  wrong-patient placebo, while respecting the stable-case harm limit.
- Risk: at matched 5% FPR, recall must significantly beat the same references;
  the source-frozen cutoff must also remain at or below 5% FPR.
- Interval: 90% marginal and care-unit coverage must be within 0.87-0.93,
  interval score must significantly beat the references, and point MAE must be
  noninferior to history within 2% of the research change threshold.

No method is selected on NWICU. All methods and all three gates are reported.
NWICU is previously examined and is not an untouched lockbox; therefore this
experiment cannot promote a release cell even if a diagnostic gate passes.

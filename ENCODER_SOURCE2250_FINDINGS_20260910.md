# Source Learning Curve: 2,250 Patients

The prespecified second expansion stage completed all28 fits (four arms,
seven seeds,32epochs). Each seed has2,250 fitting and the original136 tuning
patients. Same availability-safe glucose@6h labels, original hospital/MIMIC
query hashes, initialization, step budget and source-tune checkpoint selection.
See ENCODER_LEARNING_CURVE_PROTOCOL_20260910.md.

## Results

Median patient-equal MAE (mg/dL), identity decoder:

| Input/domain | 250fit | 1000fit | 2250fit |
|---|---:|---:|---:|
| Own / hospital |36.679|36.617|36.998|
| Whole / hospital |38.070|37.116|37.108|
| Own / MIMIC |31.690|31.211|31.214|
| Whole / MIMIC |33.497|32.286|32.223|

Versus the matched250-fit reference, whole significantly improves6/7 seeds
in each domain. Own improves0/7hospital and4/7MIMIC. Whole-vs-own remains0/7
in both domains, and whole source-tune MAE is numerically worse in7/7.
The1000->2250 median difference is small; this descriptive comparison is NOT
a paired significance test or a convergence proof.

MIMIC Trauma SICU still passes0/7 vs persistence for both identity arms.
The2014-2016 subgroup passes7/7own,1/7whole. These are descriptive time
subgroups, not prospective time-heldout validation. No interval calibration.

## Interpretation

More source support improved whole-context generalization relative to250,
but under this fixedcompute schedule further expansion does not close the
incremental context gap. The first observed failure remains source-tune
generalization, not exclusively MIMIC transfer. Source population composition,
support changes and per-patient training exposure prevent attributing all
improvement or remaining failure to N alone. More data is not yet a complete
repair; no claim that additional data cannot help.

Both own and whole selected epochs vary18-32. The next optimization question
is distinct from the earlier250-fit8->32epoch trial: whether the expanded
cohort is underexposed at the fixed512optimizer steps. This is untested here.
Do not tune that budget on MIMIC; use source fit/tune evidence.

Candidate_only. No release weights, registry, intervals or runtime modified.
MIMIC remains repeatedly inspected validation; pretraining overlap remains
unexcluded. This glucose-only diagnostic does not establish all-target repair.

## Artifacts

- artifacts/encoder_source2250_20260910.json
- artifacts/encoder_source2250_comparison_20260910.json
- artifacts/encoder_source2250_strata_20260910.json

Training exited0 and both aggregate audits completed. Private candidate
weights and patient predictions remain outside the repository.

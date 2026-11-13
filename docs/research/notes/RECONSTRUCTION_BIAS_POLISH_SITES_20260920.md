# Bias Polish: Evidence Failure Versus Opposing Site Effects

Frozen all33bias-polished candidate, same cached evaluation pairs. Reuse
site_summary and patient bootstrap helpers. Reproduce prior mean deltas exactly;
do not remove sites, adjust gate thresholds or choose outputs from this audit.

## Different Failure Types

BUN selected:9/11site means improve, equal-site delta-.053806. Every
leave-one-site-out average remains negative(range-.085175,-.033199).
Only site-evidence gate fails; this is not proof of no benefit nor proof that
more samples would guarantee passage. Broad BUN7/7site means improve and passes
the existing incremental patient/site gate. No sufficiently supported site has
a nominal95%harm interval for BUN in either group.

Bicarbonate broad:2/7sites improve,5worsen. Patient-pooled delta-.008345 but
equal-site delta+.003388; leave-one-site-out range[-.003932,.015593] crosseszero.
Three n>=30sites have nominal unadjusted harm intervals. This is opposing site
behavior, not merely an otherwise uniform benefit missing significance.
Selected bicarbonate11/11site means improve and incrementally passes.

Creatinine broad:5/7sites worsen; allleave-one-site-out means remain positive.
Magnitude is small(.001708pooled normalized MAE), but calibration did not repair
it. Training-risk reduction does not guarantee held-out error reduction.

## All-Target Scope

Selected failures (overlapping categories):11patient-mean,12site-mean,
29patient-evidence,32site-evidence. Broad:15,13,22,32respectively.
Patient/equal-site weighting flips signs for13selected and6broadtargets.
14selected/15broad targets have favorable bothmeans but fail evidence gates.
Three selected/26broad targets contain at least one n>=30nominal harmful site.
These last counts are exploratory and NOT multiplicity-adjusted harm claims.
Small cohorts may simply be unable to expose subgroup harms; do not compare
these counts as population risk rates.

## Decision

There is no single gate problem: BUN supports further independent confirmation,
whereas bicarbonate and other targets require correction transfer to be improved.
Do not broadly install source-fit offsets or relabel all failures as low power.
Future repair must be selected on source-held-out hospitals and validated on
independent patients, with no evaluation-derived hospital-specific corrections.
Existing scalar bias polish remains candidate-only. No runtime changes.

Audit completed; originalmean checks exact; existing site-summary and bias tests
rerun. Evidence: artifacts/native_reconstruction_bias_polish_sites_20260920.json.

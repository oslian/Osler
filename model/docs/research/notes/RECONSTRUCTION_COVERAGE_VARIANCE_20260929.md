# Conditional Group Variance Diagnosis

47targets, all original outer patient-target pairs; no model changes or cohort
exclusions. For each reference, compute normalized paired error differences,
component means, estimated variance of each mean(sample variance/n), and1000
paired patient bootstrap replicates per component. This is conditional on fitted
models, not retraining uncertainty. Shared fitted weights can induce dependence.

## Results

Of30 failed targets, only9 permit full variance decomposition:21 contain at
least one singleton component. Singleton mean variance is UNKNOWN, not zero.
No singleton is dropped. Among the9estimable targets, median estimated sampling
fraction is .6766 versus process and .5574 versus population. Positive residual
between-component variance remains for5/9 and6/9 respectively. These9 are not
representative of all30; no conclusion that noise explains two thirds overall.

18/30 have at least one >=30person component whose nominal95% bootstrap interval
indicates harm versus process;19/30 versus population. These are exploratory
unadjusted subgroup intervals, NOT multiplicity-corrected discoveries of harm.
They motivate caution and targeted source-only checks, not clinical conclusions.

## MAP

Two singleton components prevent estimating the full noise contribution.
Among >=30person components:0nominal harm and17nominal benefit intervals versus
process; the same counts versus population. This does not prove no harmful
domains, nor supersede its failed family-adjusted component gate.
MAP appears evidence-limited rather than demonstrated broad harm in this audit;
the result cannot be extrapolated to other targets with adverse supported groups.

## Consequence

There are at least two distinct situations, not one universal encoder failure:
limited precision for source-group estimates, and plausible adverse domains for
some targets. Do not pool singletons as zero-variance sources, discard them,
or change the gate after seeing results. A source-only hierarchical uncertainty
analysis can retain small groups with explicit assumptions; it must be labeled
model-based and sensitivity-tested, not used to retroactively promote MAP.
Representation repair should be evaluated separately from statistical reporting.

Artifact:`artifacts/native_reconstruction_coverage_variance_20260929.json`.
Reproduce:`python -m scripts.diagnose_coverage_variance`.
Three tests verify variance subtraction, negative excess retention and unknown
singleton variance. Runtime, registry and calibrated intervals remain unchanged.

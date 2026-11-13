# Coverage: Source-Only Support Diagnosis

Fixed47targets x5selection folds. Join the hash-verified target/sibling-masked
raw snapshot to frozen coverage real/process predictions. Positive advantage
means smaller absolute error for real versus numeric-zero process, normalized
by the fit-only target scale. No fitted selector, exclusions or outer-data tuning.

Features fixed before this run: observed variable count, count aged<=1h,
count aged<=6h, mean age of observed variables. Ages use anchor minus actual
event time plus token age, in hours. Missing variables do not enter mean age.
Both feature and advantage are centered within provenance component before
Spearman correlation. This is descriptive, patient-weighted association, not
component-equal inference, effect size of an intervention, or power analysis.

| Feature | Comparisons | Median rho | Positive | Absolute rho >=0.1 |
| --- | ---: | ---: | ---: | ---: |
| Observed count | 235 | 0.02044 | 175 | 17 |
| Fresh <=1h count | 235 | 0.01559 | 164 | 12 |
| Fresh <=6h count | 235 | 0.02175 | 174 | 13 |
| Mean observed age | 235 | -0.00317 | 110 | 8 |

Simple aggregate support has weak association with numeric advantage. These
results do not justify a general support/freshness gate as the fix. They do not
rule out variable-specific support, nonlinear interactions, source differences,
or age errors. Repeated selection patients and overlapping training folds mean
235 comparisons are not235 independent experiments.

Next diagnostic should isolate numeric value use while holding masks/ages and
capacity fixed, using source-only matched value perturbations. Existing process
controls are separately trained; their comparison does not isolate reliance of
the real encoder on values. Check prior perturbation experiments first, retain
all targets, and do not tune to the already inspected outer failures.

Artifact: `artifacts/native_reconstruction_coverage_support_20260929.json`.
Reproduce: `python -m scripts.diagnose_coverage_support`.
No serving, weights, calibration, registry or promotion changed.

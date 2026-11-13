# Fixed-Hospital Reconstruction Stability

## Question And Scope

Does variation across the seven completed fits account for most reconstruction
error? This is a posthoc diagnostic, not a new model selection or promotion.
The task remains fresh current-value reconstruction with the target and its
algebraic siblings masked throughout history, not future forecasting.

`scripts/audit_reconstruction_fit_stability.py` reads the durable private copies
of saved predictions. Their SHA256 values match the original evaluation reports.
The seven-split summary's dependencies are verified, and saved predictions
reproduce the original patient counts and real/process/population MAEs in both
evaluation roles. For each target, hospital patient keys and truths match exactly
across all available fits; no silent intersection or patient exclusion is used.
This verifies prediction evidence, not a fresh replay of all original training.

All 39 catalog targets are retained. Thirty-five have seven fits and at least
30 hospital patients. Anion gap has only four eligible fits; fibrinogen, LDH and
TSH have 26, 27 and 17 hospital patients respectively. These remain unsupported
for the corresponding complete comparison, not evidence of no information.

## Result

For each target, equal-weight fit/patient MSE decomposes exactly into:

`mean fit MSE = between-fit prediction variance + mean-prediction residual MSE`.

Across the 35 supported seven-fit targets, the median between-fit fraction is
**2.59%**. The median MAE reduction of the arithmetic mean prediction relative
to mean individual-fit MAE is **2.40%**. These are descriptive target medians,
not a pooled clinical error reduction or a replacement for the original MAE gate.

| Target | Hospital patients | Between-fit share of MSE | Fits with lower mean MAE than process-only |
|---|---:|---:|---:|
| Albumin | 172 | 9.98% | 7/7 |
| AST | 172 | 0.95% | 7/7 |
| Total protein | 172 | 9.09% | 7/7 |
| Creatinine | 193 | 2.59% | 7/7 |
| Bicarbonate | 187 | 5.40% | 6/7 |
| Heart rate | 194 | 2.47% | 6/7 |
| MAP | 177 | 1.82% | 6/7 |
| Glucose | 190 | 0.88% | 5/7 |
| BUN | 194 | 5.40% | 7/7 |
| Bilirubin | 156 | 4.97% | 0/7 |
| O2 saturation | 186 | 1.01% | 0/7 |

The last column is numerical direction only, NOT significance or seven-split
promotion. It describes the fixed hospital group, not the changing source
evaluation patients. Creatinine's mean increment is consistently favorable here
but still failed the full original comparison in most splits. Bilirubin and O2
saturation have consistently unfavorable numerical increments against the
trained process-only comparator, so calling every failure a near-miss caused
only by random fitting is unsupported.

## What Is And Is Not Located

Most error in these seven fits remains after averaging their predictions.
Thus variability alone is not a sufficient general explanation, and a broad
ensemble sweep should not be presented as the main repair. The hypothesis that
fixed-patient training repeats will remove most error now has weaker motivation.

The residual is NOT a measurement-noise estimate, irreducible physiology, or
proof the data contain no more information. Shared bias, limited representation,
incomplete convergence, and insufficient conditional information can all remain.
MSE is tail-sensitive; the original MAE remains the performance criterion.
Fit composition, source medians, initialization, sampling and checkpoint
selection vary together. No pure optimizer-variance attribution is made.

The next discriminating diagnosis should examine target-specific source fit and
tune errors of the selected models, using the same masking and comparators:
does numerical reconstruction fail already on fitting patients, or only on
unseen patients? The existing global tune scalar cannot answer this. Only after
this distinction should a training/representation repair versus a generalization
repair be chosen. Do not select targets, epochs or new heads on hospital results.

## Verification And Boundaries

- Five synthetic tests cover the exact decomposition, identical biased outputs,
  perfect predictions, nonfinite rejection, and strict patient/truth alignment.
- Original artifacts and bound training scripts were not modified.
- Machine report: `artifacts/native_reconstruction_fit_stability_20260917.json`.
- Only aggregates are written to public artifacts; patient rows remain private.
- No weights, runtime routing, intervals, causal claims or promotion changed.

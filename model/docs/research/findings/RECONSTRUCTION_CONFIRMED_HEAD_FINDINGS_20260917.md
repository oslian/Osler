# Source-confirmed Heads: Negative Efficacy Result

Completed the fixed protocol for all 28 four-role supported targets, retaining
all 39 catalog statuses. Fitting-site five-fold head nomination was followed by
separate readout-group confirmation against original and fit-median predictions.
Patient and equal-component error comparisons used fixed .05/39 correction.
Three decision-helper tests passed. Evaluation original MAEs reproduce to1e-10.

No nominee passed. All 33 trained targets therefore retain original outputs;
macro MAE is unchanged at2.9342321433. This is prevention of unconfirmed changes,
NOT an accuracy repair, new physiology signal, or promotion.

## What this distinguishes

- BUN nominates ridge1000. Confirmation patient delta is-0.0440 but equal-site
  delta is+0.0185 against original; versus population, corresponding values are
  -0.1785 and+0.0074. Patient-pooled advantage does not imply equal-site advantage.
- Phosphate nominates ridge100 from fitting-site CV, whereas the prior
  minimum-error readout-group selection chose1000. The new nominee's confirmation
  delta is-0.00848 versus original and+0.04158 versus population. Hyperparameter
  preference does not reproduce across these source roles.
- Bicarbonate nominates1000, with confirmation improvements versus original
  (patient-0.06824/site-0.12904) and population(-0.16722/-0.06494). Its site p value
  versus population is0.2703. This is inadequate evidence, not established harm.

Only seven confirmation hospital components are present, with unequal per-target
support. Site t tests have weak power/normal-mean assumptions, and Bonferroni is
conservative. Do not infer lack of physiology from non-significance or optimize
thresholds against previously inspected evaluation. The fixed encoder saw the
fitting patients used in head CV; this CV nominates only and does not independently
validate representation. All data remain reused development, not a new lockbox.

The next unresolved issue is robust cross-site readout learning/support, not merely
allowing more minimum-MAE substitutions. No runtime, weights or intervals changed.

Evidence: `artifacts/native_reconstruction_confirmed_heads_20260917.json` and
`artifacts/native_reconstruction_confirmed_choices_20260917.json`.

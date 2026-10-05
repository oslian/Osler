# Source-Only Readout Selection Result

All33targets compared original, fit median and five body ridge strengths.
Only source tune MAE selected each target; choices saved before evaluation.
The original and all-probe evaluation scores reproduce within1e-9. Allsix
unsupported targets retained, no patient exclusions or runtime changes.

Choices:9original,6population,18ridge. Macro target-equal normalized MAE:
original2.934232, selected2.966656, all-probe2.966618, population3.093444.
The selected mean is worse than original, not an overall repair.

Versus original, selected nominal paired95% gains: bicarbonate, phosphate,
lipase. Lipase uses population, so its benefit is abstention, not shared-state
information. Selected harms: BUN, platelets, troponin I. Bicarbonate uses
ridge100 and phosphate ridge1000, consistent with source-driven choices rather
than handpicked evaluation winners, but neither is formally promoted.

Original is selected for triglycerides; it remains worse than population in
evaluation. This demonstrates that including a safe constant candidate alone
does not ensure source selection recognizes when to abstain. The source-tune
ordering need not transfer to different patients/sites.

Source selection restores ALT/bilirubin compared with all-probe, but worsens
triglycerides. Creatinine selects ridge1000 without significant paired harm;
absence of significance is not equivalence or a retention guarantee.

## Limits And Decision

The same tune previously selected encoder checkpoints, so it is not an
independent readout-selection gate. Reused evaluation is development evidence;
nominal intervals are not multiplicity-adjusted. This does not justify
loosening gates or selecting individual evaluation winners into runtime.

The tested selector is not a demonstrated broad repair. Next validation must
separate encoder selection from readout selection and measure hospital-level
selection stability; merely offering more heads is not enough. A truly new
selection split requires retraining without those selection patients/sites,
not relabeling data already used by the encoder.

Three source-choice helper tests pass; full score reproduction provides the
integration check. No weights, intervals, serving registry or promotion changed.
Artifacts: `artifacts/native_reconstruction_readout_choices_20260917.json` and
`artifacts/native_reconstruction_readout_selection_20260917.json`.

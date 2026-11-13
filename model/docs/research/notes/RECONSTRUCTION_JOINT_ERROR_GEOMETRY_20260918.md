# Joint-source Error Geometry

All33targets/two sourcegroups/fit and encoder-selection roles audited from saved
predictions; all mean error increments reproduce. No external data, fitting,
checkpoint selection or acceptance-rule changes.

Selected-cohort Spearman fit->selection:
CPK .714->-.110;lipase .862->-.080;glucose .186->.107;
troponin .554->.271. CPK/lipase selection sizes30/28 are small.
Broad CPK .618->.049(N12),lipase .820->.357(N7),troponin .714->.660(N18).
Thus broad troponin retains a positive sample ranking despite MAE harm, unlike
selected CPK/lipase. Different targets need different diagnoses, not one assumed
universal loss/bias cause. Correlation uncertainty is not quantified here.

Harm also exists inside the training truth5th-95th percentile range:
selected glucose+.05144,CPK+.41042,lipase+.16255,troponin+.51495 normalized MAE.
Individual harm fractions57%,63%,61%,65% respectively. Therefore these observed
failures are not solely a few out-of-range tail labels. Truth-range strata are
retrospective diagnostics, NEVER inference-time selectors or evidence of a causal
mechanism. Offset correction cannot restore ranking but may still lower MAE.

The next repair should distinguish unstable learned ordering from calibration
and limited support. This audit is not permission to tune on validation labels,
discard sparse targets or claim nonpredictability. Runtime unchanged.
Evidence:artifacts/native_reconstruction_joint_error_geometry_20260918.json.

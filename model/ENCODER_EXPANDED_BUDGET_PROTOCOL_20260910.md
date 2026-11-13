# Expanded Source Optimization Budget

Decision based on source-only encoder_exposure_20260910.json:2250fit whole
late source-tune MAE improves versus epochs9-16 in7/7seeds. Actual supervised
row coverage is about22%, future-event coverage24%, median7 draws/patient.
This is a reason to test additional optimization, not proof of undertraining.

Keep identical2250fit/136tune source roles, input hashes, initialization,
sampling, four own/whole bounded/identity arms and seven seeds. Extend32 to64
epochs, same16steps/batch32/AdamW settings. Verify first32 epoch logs exactly
reproduce original2250run. Select best source-tune checkpoint including0;
no external checkpoint selection. No learning-rate or architecture change.

Compare same-row patient-bootstrap predictions against32epoch each arm,
whole versus own and persistence. Original descriptive care-unit/time audits
unchanged. No conformal, all-target retention or pristine lockbox is created.
MIMIC remains validation. Candidate_only, no runtime or causal promotion.

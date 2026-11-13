# Source Fit / Tune Failure Localization

Read-only evaluation of the original32-epoch seven-seed identity checkpoints.
All14 source-tune MAEs reproduce stored selection scores within1e-5. Input
and checkpoint hashes are checked. Source-fit errors are in-sample; tune
errors were used to select checkpoints, not an independent new test.

Each split has250 fitting patients and136 tuning patients. Rows are repeated
observations, not independent new people. Whole versus own is significantly
better in6/7 fitting comparisons, but worse numerically in7/7 tuning
comparisons (significantly worse6/7). Seed71 is worse in both; its whole
checkpoint was selected at epoch5, unlike own epoch32.

This is direct evidence of an in-sample versus new-patient reversal for most
splits. Failure begins inside eICU before MIMIC domain transfer. Cross-database
numerical sensitivity remains real but is not a necessary explanation of all
the extra-context deficit. A diagnosis centered only on MIMIC harmonization
misses this earlier failure. These are repeated development diagnoses, not
independent statistical replications or proof that all physiology is absent.

Selected source-tune whole-minus-own MAE is positive in all28 inspected
splits across budget, numeric-rescale, context-dropout and history-mask
reports. That tally is descriptive; these runs share data. It is not a
28-independent-test significance claim.

The current pipeline inherits its subject roles from a bounded prior
creatinine-anchor replay before rebuilding glucose availability anchors.
It is not training this diagnostic on all available eICU patients. More
source-independent patients and patient-level learning curves are therefore
an untested repair at this exact corrected contract. The observed reversal
does not by itself prove small sample size is the sole cause; inherited
pretraining, feature dimensionality and noisy associations may contribute.

No model was trained or promoted here. No runtime or calibrator changed.
Evidence: artifacts/encoder_fit_gap_20260910.json and
scripts/audit_encoder_fit_gap.py. This evidence changes the next experiment
from another external-specific head to a source-side generalization test.

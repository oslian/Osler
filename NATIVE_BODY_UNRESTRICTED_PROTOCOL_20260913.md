# Baseline-Independent Raw History Control

Keep seed7 partition, random seed20260913, one fresh current MAP query per
patient, target/sibling masking, ridge alpha grid and source-tune selection
from the expanded raw probe. Do not use future pairing to select patients.
Remove nested baseline checkpoint and its paired-fit support/affine transform.
Use the original cached measured channels and the same summary calculation.
StandardScaler and Ridge are fitted on reconstruction-fit patients only.
Preserve support-only, deranged-patient placebo and population controls.
All original event times must be finite, ordered and pre-anchor.

This tests representation filtering, not a new neural encoder or promotion.
eICU heldout roles remain reused research validation. No threshold changes.

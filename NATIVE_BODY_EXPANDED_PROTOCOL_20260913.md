# Expanded Current-Only Masked History Probe

Use all seed7 role patients with finite current MAP and nonnegative age<=1h,
no future paired requirement. Select one query per patient with RNG20260913,
then reject absent independent histories with same masking contract. Same
ridge summaries/support/placebo, source-only scaling and alpha grid as previous
pilot. Existing input transforms use nested baseline-fit subset of sourcefit,
not external labels. No neural predictions/latents are used as features.
Changing cohort means old and new MAEs are not paired evidence of a repair.
Report postmask counts and all heldout comparisons. Reused holdouts not lockbox.
No model promotion or forecast inference. Multi-target support still untested.

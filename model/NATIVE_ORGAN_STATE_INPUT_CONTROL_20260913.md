# Native Replay State Input Control

Predeclared training-matched diagnostic, not promotion. Reuse the exact seed7
pilot patient selection (64fit,32tune,64evaluation,64hospital), initialization,
8epochs, losses, optimizer and frozen baseline. Full-state identity graph versus
identical heads with state coordinates zeroed during both training and inference.
Time, anchor and mask remain. Apply to value AND latent heads. Parameter shapes
are matched; the control deliberately cannot train state through these heads.
Select epoch0..8 using source tune only. Evaluation/hospital are diagnostic,
previously inspected roles, not fresh lockboxes. Compare with fullstate, baseline
and persistence using patient paired bootstrap. No serving or calibration changes.
Existing full-state tune trajectory must reproduce before interpreting control.

# Fixed Monitored Continuation

Diagnostic extension of the completed47target coverage experiment, not a claim
that prior failed continuation was never tried. Existing20260919continuation
used a different cohort/representation; no outcome from it is erased.

All existing folds and real/process arms continue from exact epoch8state to
epoch10. Two more epochs, same1072draws/target, sampler queues, optimizer/RNG,
loss, LR and masking. No early stopping or best-epoch selection. Independent
folds0/1/3/4; fold2 reuse requires separate explicit equivalence certification.
Stage execution may stop at saved epoch9, but final experiment requires10.

Keep epoch8untouched. Save immutable new checkpoints per epoch, then evaluate
all fixed monitor identities at epochs8/9/10. Reject changed dependencies and
role leakage. Training may never consume selection/calibration/test labels.
Do not interpret online loss as a fixed-cohort learning curve. Report alltarget
monitor MAE and loss, both patient and component weighting. Monitor is sampled
development, not full outer evaluation or promotion. Final epoch10, regardless
of monitor result, is the sole continuation endpoint for later paired comparison.
Existing outer data are reused development, never an untouched lockbox.

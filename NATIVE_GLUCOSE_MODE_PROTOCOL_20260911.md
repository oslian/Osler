# Student training-mode contrast

Prior availability/native trials deliberately used eval-mode training, which
keeps gradients active but disables the existing Transformer dropout=0.1.
This is a documented policy, not an accidental gradient disconnection. The
prior variable-block dropout experiment is not this activation/attention dropout.

Fixed contrast: native glucose own and whole, all seven original partitions,
same initialization, events, labels, 32 epochs, sampler, optimizer, losses and
tune-only checkpoint selection. Before each epoch call model.train(); existing
model override retains EMA teacher eval mode. Evaluation calls predict(), which
uses eval mode. Assertions ensure this boundary; no dropout probability search.
Original complete saved predictions must reproduce before fitting each arm.

Primary paired patient-bootstrap comparisons: train-mode vs old eval-mode for
each view, and whole-train vs own-train. Persistence retained. New source and
held-out hospital evaluation, 2,000 bootstrap draws. Overlapping seeds are not
independent tests; no new lockbox, conformal or multi-target retention claims.

This tests whether omitted regularization contributes to the observed fit/new-
patient reversal, not whether activation dropout guarantees generalization.
No runtime, causal, interval or promotion change without full validation.

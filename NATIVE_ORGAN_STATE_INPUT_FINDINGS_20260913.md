# Training-Matched State Input Diagnostic

Completed the predeclared seed7 64fit/32tune pilot comparison. Identical heads,
initialization, baseline, eight epochs and patient order; control zeros state
coordinates at BOTH value and JEPA latent heads during training/inference.
Time, current anchor and mask remain. Same parameter shapes do not imply equal
effective capacity: control deliberately disconnects state-to-head learning.

All eight full-state tune scores exactly reproduce the original pilot. All
direct artifact bindings verify. Tune baseline MAE8.34714 becomes8.41926 with
state and8.42216 without state at epoch8. Both select epoch0. Selected predictions
equal baseline on both previously inspected held-out roles. No new promotion.

Failure in this pilot does not require state inputs: anchor/time correction
alone also fails source tuning. State adds no demonstrated generalizing benefit;
this is not proof of equality, irrelevance of physiology, or a universal cause.
Together with `artifacts/native_organ_head_attribution_20260913.json` (shuffling
state changes output only~0.001mmHg), this weakens a graph-specific explanation
of the small pilot's failure. Attribution intervention alone is off-distribution
and not causal evidence. Pilot support/optimization remain unresolved.

Next useful question: does the organ state encode patient-varying changes before
the head, and can it predict residual beyond an explicitly fitted anchor-only
control on source-disjoint patients? Do not enlarge the graph as a presumed fix.
No release weights, registry, intervals or runtime behavior changed.

Evidence: `artifacts/native_organ_state_input_control_20260913.json`.

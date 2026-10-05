# Native teacher semantic check

CodeGraph and source trace confirm `teacher_target` is the single-target
specialization of `_encode_target_specific_values`, not a separate incompatible
body teacher. An existing equivalence test already covers zero anchors.
Extended regression covers nonzero anchors, positive/negative/zero delta,
non-target values, joint anchor/future translation, and stop-gradient.
All5availability-encoder tests pass.

No target absolute-value versus delta mismatch found in this helper. Equality
does not demonstrate informative teacher geometry, sufficient JEPA gradients,
generalization or successful full persistent-organ training. Prior teacher
geometry and gradient findings remain applicable. No new training, runtime,
weights, calibration or promotion; do not repeat this as an untested cause.

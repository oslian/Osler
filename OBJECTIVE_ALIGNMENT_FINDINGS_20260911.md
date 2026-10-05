# Local Objective Alignment

Real source-fit patient-uniform batches, three fixed-seed batches per selected
checkpoint. Seven whole glucose and14 own/whole potassium checkpoints. Current
value SmoothL1 and normalized latent SmoothL1 differentiated separately with
respect to trainable shared-context parameters. No training or external reads.

| Group | Negative cosine / probes | Median cosine | JEPA/value gradient norm | Combined/value cosine |
|---|---:|---:|---:|---:|
| Glucose whole | 4/21 | .11683 | .01137 | .999949 |
| Potassium own | 13/21 | -.29032 | .02450 | .999812 |
| Potassium whole | 11/21 | -.03871 | .02056 | .999850 |

Teacher helper encodes normalized truth minus normalized current and matches
the native target-specific single-variable contract. No absolute/delta reversal
was identified. Prior teacher-grid discriminability experiment already rules
out literal constant encoding on its tested glucose grid; not repeated here.

These local raw gradients do not support strong cancellation as the dominant
immediate effect. They indicate weak JEPA influence relative to numerical loss
at the selected checkpoints. Adam momentum, preconditioning, earlier training
and future dynamics are not diagnosed by this cosine. No claim that JEPA is
irrelevant or that increasing its weight solves generalization. Related source
gradient balancing was already tested and failed to fully repair glucose.

This separates numerical-smallness and directional conflict from a proven root
cause. No runtime, checkpoint, interval or promotion changes. Next interventions
must demonstrate benefit, not rely on this gradient statistic as proof.

Artifact: `artifacts/objective_alignment_20260911.json`.
Command: `python -m scripts.audit_objective_alignment`.
The report contains only aggregate probes and file hashes, no patient rows.

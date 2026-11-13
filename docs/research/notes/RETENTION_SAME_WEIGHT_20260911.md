# Same-weight retention control

Fixed saved weights, same source-fit batches, same primary loss and teacher;
only the retention target mask changes. Seven seeds, three batches each at
both uniform-trained and selected-trained checkpoints. No fitting or external
evaluation. All42 original diagonal probes reproduce exactly; bindings verified.

| Checkpoint | Retention mask | Opposed /21 | Median cosine | Norm ratio |
|---|---|---:|---:|---:|
| Uniform-trained | Uniform | 12 | -0.1878 | 0.3574 |
| Uniform-trained | Selected | 9 | 0.0363 | 0.1772 |
| Selected-trained | Uniform | 8 | 0.0823 | 0.9442 |
| Selected-trained | Selected | 6 | 0.2070 | 0.1764 |

Selection reduces local opposition at either fixed checkpoint family, so the
prior diagonal difference is not solely different trained weights. However,
the effect is not universal and the prior accuracy experiment still failed
stable whole-versus-own improvement. Uniform retention at selected checkpoints
has a much larger norm: unprotected outputs diverged from their initial
reference, not necessarily from truth. Its force would pull them back without
distinguishing useful adaptation from harmful drift.

This supports separating reference quality from update interference. It does
not justify treating every negative dot product as an error to project away,
or claiming an optimization repair without a matched outcome experiment.
Shared-state generalization remains unresolved. No runtime, interval, weight,
promotion or causal claim. These checkpoints use the base world-model class,
not the full persistent-organ subclass. Source cohort and historical pretraining
limitations remain unchanged.

Runner: `python -m scripts.audit_retention_same_weight`.
Artifact: `artifacts/retention_same_weight_20260911.json`.

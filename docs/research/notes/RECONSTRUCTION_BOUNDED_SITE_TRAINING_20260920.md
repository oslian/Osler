# Matched Bounded-Site Training Results

Both real-input arms completed eight epochs, 283008 updates each. Same initial
source scores reproduced within1e-7, same patient/target pools, group/target
budget, seed19, model architecture, masking and AdamW. Both use replacement
sampling; the intervention only changes per-patient sampling probabilities.
Seven sampling/schedule tests pass, including matched group/target order and
RNG replay. Per-epoch unique exposure remains recorded in checkpoints/report.

Both arms select epoch7 on the unchanged original90+90 source encoder patients.

| Source score | Uniform | Bounded site |
|---|---:|---:|
| Equal group/target macro | 1.813590196 | 1.792912755 |
| Selected group macro | 1.822615956 | 1.797984335 |
| Broad group macro | 1.804564436 | 1.787841174 |

Relative macro change -1.140%, normalized MAE, not accuracy. Each group has20/33
numerically improved targets. These are checkpoint-selected source scores,
not heldout effectiveness or promotion.

| Target | Selected uniform -> bounded | Broad uniform -> bounded |
|---|---:|---:|
| Creatinine | 1.627367 -> 1.585220 | 1.565659 -> 1.637265 |
| BUN | 1.257114 -> 1.105465 | .984129 -> 1.116262 |
| Glucose | 1.039886 -> 1.042394 | .742094 -> .732543 |
| Lipase | 3.113574 -> 3.121779 | 5.975782 -> 5.970054 |
| Troponin I | 3.998733 -> 4.007102 | 1.589464 -> 1.644926 |

Site sampling improves the overall source score but does not remove source
group tradeoffs. No significance claimed from these small selection groups.
Next freeze both selected checkpoints and evaluate identical existing heldout
pairs. New matched process/placebo and full gates remain required; this is not
full39-target training, a new external lockbox, or future forecasting.

Evidence: artifacts/native_reconstruction_bounded_site_training_20260920.json.
No runtime weights, intervals, registry or promotion changed.

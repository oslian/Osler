# Same-Map Relaxation: Partial Fit Repair, No Strong-Comparator Promotion

## Fixed Intervention and Verification

All47 targets/all5 source folds/four arms completed:940 fits,1,797,884 selection
occurrences,720,470 unique patient-target pairs,29,580 people. Only C changed,
from100/n to1000/n. Same saved scaler, RBF map, bandwidth,128landmarks, weights,
median offset, labels, component shuffle and frozen body encoder. Coefficients
and intercept were relearned, not globally amplified after fitting.

All940 solvers converged in77-4517iterations, below the unchanged50,000cap.
Original fit and query predictions reproduce before comparing new outputs.
Same-patient references and allfold aggregate baselines reproduce to1e-12.
Seven paired comparisons per candidate,2*47*7 family, both patient/component
gates unchanged. No parameter selection, outside queries or favorable-subset
promotion. These source selection rows are reused development, NOT a lockbox.

## Improvement Is Real but Limited

Normalized target-macro MAE, not physical units:

| Readout | Patient equal | Component equal |
|---|---:|---:|
| Existing raw HGB |0.825737|0.787947|
| Existing body HGB |0.857139|0.811241|
| Native neural |0.882389|0.831593|
| Scaled body, original C |0.886598|0.835116|
| Scaled body, C times10 |0.873760|0.823452|
| Scaled raw, original C |0.913889|0.863185|
| Scaled raw, C times10 |0.902195|0.851674|
| Relaxed process control |0.947142|0.893080|
| Relaxed shuffled-body control |0.953670|0.901658|

Body gains1.448%/1.397% and raw gains1.280%/1.333% versus their original scaled
kernel. Both have real improvements, not only a feature-distance change.
Body/raw/process fitting mean MAE improves235/235; shuffle230/235. Median fit
MAE reductions body2.555%, raw3.497%, process1.112%. Thus the earlier penalty
was limiting fit, and relearning directions helps more than the scalar-ray-only
diagnosis suggested. These fit and selection gains are not new JEPA capabilities.

Corrected both-level wins over SAME-map old C:
- Body3targets: ALT, phosphate, WBC.
- Raw6targets: AST, glucose, MAP, phosphate, total_protein, WBC.

Full conjunction0/47 for body and0/47 for raw. Neither candidate has a corrected
both-level win over native, body HGB or raw HGB. Body macro is numerically better
than native but that does not satisfy target-level evidence. Body still loses
body HGB on31patient/30component target means; raw loses raw HGB on46/42means.
Both body and raw still lose HGB FITTING mean MAE in232/235 combinations.
This remains partly a fitting/readability issue, not solely an external-transfer
failure or acceptance threshold problem.

Selected patient-equal errors, all targets remain in the full artifact:

| Target | Old scaled body | Relaxed body | Body HGB | Relaxed raw | Raw HGB |
|---|---:|---:|---:|---:|---:|
| BUN |0.645720|0.626076|0.614053|0.628853|0.578286|
| Creatinine |0.924149|0.902730|0.873193|0.906750|0.820478|
| Glucose |0.707676|0.702757|0.701064|0.709385|0.689013|
| Hemoglobin |0.459517|0.450195|0.444886|0.461904|0.425513|
| MAP |0.561358|0.556646|0.556953|0.558024|0.550190|

MAP's tiny mean advantage over body HGB is NOT a corrected patient/component
pass. Do not promote it based on this table. Some source target means regress:
body11patient/9component, raw4patient/6component, versus old C. Relaxed process
and shuffled controls worsen in aggregate despite fitting gains, illustrating
that weaker regularization can increase nontransferable fitting.

## Interpretation and Branch Decision

The chain is now measured, not guessed:
1. Raw lag units dominated the original kernel metric; fit-only scaling fixed
   that imbalance and modestly reduced error.
2. Regularization compressed predictions; scalar rescaling explained only a
   small fraction of the HGB fitting gap.
3. Same-map weaker regularization and coefficient relearning reduce more error,
   but still do not beat stronger original readouts or produce full passes.

One finite C does not prove no kernel could work. Nevertheless, this branch
has not earned more query-driven tuning. Stop this kernel-repair branch:
do not extend C/landmark/bandwidth sweeps on these same selection answers.
Keep original stronger references and return to evidence about the actual
shared state, rather than treating failure of this diagnostic kernel as proof
that the original JEPA has no signal or needs a wholly new architecture.
Full shared-state/generalization objective remains unresolved; no unique global
root cause has been established by these local readout experiments.

23focused tests pass. A synthetic high-C fixture initially invalidly assumed
every optimization must converge; corrected it to verify unchanged parameters
and honest status reporting, and added an explicit one-iteration nonconvergence
test. No production iteration budget or acceptance rule was relaxed. Full-data
940fits independently all converged. Parent models remained unmodified.

No runtime weights, registry, calibrators or predictions replaced. No external,
forecast, clinical or causal claim. Private row-level outputs/heads remain local.
Evidence: RECONSTRUCTION_KERNEL_RELAXATION_PROTOCOL_20261003.md;
scripts/train_relaxed_kernel.py, scripts/score_relaxed_kernel.py;
artifacts/native_reconstruction_relaxed_kernel_{fold0..4,assessment}_20261003.json.

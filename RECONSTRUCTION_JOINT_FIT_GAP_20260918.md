# Joint Fit Versus Selection Gap

Replayed the selected real checkpoint on every joint-source fit/encoder example.
All66encoder-selection scores reproduce to1e-7. No fitting, new selection or
external labels. Both groups improve population MAE on32/33fit targets; O2sat
is the only fit exception and improves on encoder-selection in both groups.

All9selected-group and8broad-group targets that lose the initial population
baseline on encoder-selection DO beat that baseline on fitting patients.
Even the never-improving encoder-selection targets are not universally unable
to fit their training labels. The observable failure is a source generalization
gap, before MIMIC/NWICU or promotion gates, not just failure to optimize training.

| Group/target | Fit model-minus-prior MAE | Selection model-minus-prior |
|---|---:|---:|
| Selected glucose | -.06336 | +.04441 |
| Selected CPK | -.44723 | +.30650 |
| Selected troponin | -.51764 | +.67943 |
| Broad CPK | -.31959 | +.04232 |
| Broad troponin | -.82904 | +.53385 |

Normalized units. This does not prove memorization is the only cause: site/case
mix shift, noisy sparse labels and high-dimensional estimation remain competing
explanations. Sparse selection counts(CPK12,troponin18broad) limit certainty.
No claim of statistical significance from these means. No runtime promotion.
Earlier wording 'never learned' must refer to selection improvement, not training
fit. Next repair should target generalization, not an assumed universal inability
to fit or a larger readout by default.

Evidence:artifacts/native_reconstruction_joint_fit_gap_20260918.json.

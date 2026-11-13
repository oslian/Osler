# Where the native treatment-number increment first weakens

Re-inferred14 frozen clock-corrected checkpoints on unchanged query histories.
Source evaluation and hospital MAEs reproduce within1e-7. No fitting or new
selection. `artifacts/native_fit_gap_20260911.json`.

| Median patient MAE | Real numbers | Support control | Significant gain/harm |
|---|---:|---:|---:|
| Fit | .373898 | .373873 | 2/7,0/7 |
| Tune | .391823 | .392081 | 0/7,0/7 |
| Evaluation | .388720 | .388724 | 1/7,1/7 |
| Hospital | .387887 | .387864 | 0/7,0/7 |

In six of seven fits, real-minus-support paired fit MAE has absolute magnitude
under .0001. Seed7 differs more (-.003247), with different selected epochs;
do not attribute all of that difference uniquely to treatment values. Fit/tune
scores are descriptive, not independent efficacy tests. Difference of medians
does not equal a paired effect. Tiny statistically significant effects must
not be presented as clinically useful gains.

The numeric treatment increment is already small in fitting patients. This is
not the same pattern as the older glucose whole-context in-sample improvement
followed by source reversal. Hospital shift alone cannot explain this specific
native treatment contrast. It still does not distinguish weak conditional
information from insufficient optimization of the new event pathway.

Next discriminating control: known source-defined artificial event increment
through this same full native JEPA path, with support-only equal-capacity
control and unchanged patient roles. Prior simple/tree positive controls are
not this model/path test. Synthetic success would establish learnability of
the injected signal only, not biological validity or useful real forecasts;
synthetic failure would motivate training/path diagnosis before new data claims.
No runtime, uncertainty, or promotion changes.

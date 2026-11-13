# Selected-epoch versus training-path diagnosis

All28arm logs have32ordered epochs. Every nonzero selected checkpoint matches
the minimum recorded source-tune MAE and reported score. No selection-index
bug is found. This does not validate the sampling precision of source-tune.

Old/clean full models select the same epoch in4/7partitions:19,31,59,71.
Both significant hospital regressions occur inside that set:
seed31 epoch32 pairedMAE+1.160514 CI[0.541011,1.802345];
seed71 epoch26 +1.151337 CI[0.276680,2.031680].
Consequently changing selected epoch is not a sufficient explanation of the
observed harm. The new training trajectory can differ even at identical epoch.

Source-tune curves differ from the first logged epoch in every partition.
This is expected sensitivity to changed input, not proof the first update is
incorrect or evidence of divergent optimization. Earlier fixed-input/scale
crossing supports a checkpoint-mediated effect; local gradient/sampler
influence remains unmeasured. Do not declare one optimizer or representation
the root cause from this descriptive audit alone.

No fitting, new gate, runtime, calibration or promotion.
Artifact: `artifacts/native_conflict_selection_20260911.json`.

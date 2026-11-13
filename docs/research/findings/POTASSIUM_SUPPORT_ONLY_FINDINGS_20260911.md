# Matched support-only training result

Completed14fits64epochs. All seven own-arm training logs, selected epochs and
tune/evaluation predictions exactly reproduce prior three-way controls. Same
785/262/262 patient roles, seed, capacity, loss and optimization. The whole arm
retains cross-variable masks/ages/presence, but no cross-variable numerical
variation in either training or evaluation. Potassium history remains real.

Support-only versus full-numeric evaluation significant gain0/7 and harm0/7;
full-numeric is numerically better5/7, not significantly. This is not an
equivalence test and cannot establish equal models or information absence.
Support-only vs own gain0/7,harm1/7; vs persistence gain7/7. Median evaluation
MAE support-only .380441,full-numeric .374244,own .377129,persistence .437907.
No candidate correction or deletion is justified for runtime.

Compared with the fixed-weight off-manifold probe, the ablation is now learned
under its own input contract. The failure to establish incremental whole
benefit is not explained simply by an inference-only missing-block mismatch.
Neither observation support alone nor current full numerical encoding reliably
beats own history. This does not prove workflow-only learning, a universal
information ceiling, or a unique biological cause. Source data is reused and
pretraining overlap unexcluded; no new external/calibration/promotion evidence.

Artifact: `artifacts/potassium_support_only_20260911.json` (complete=true).
Protocol: `POTASSIUM_SUPPORT_ONLY_PROTOCOL_20260911.md`.
Command: `python -m scripts.train_potassium_matched_encoder --support-only`.
Runtime weights, registry, intervals and serving unchanged.

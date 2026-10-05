# Masked Current MAP Multi-Scale Probe

Same current-only patients, seed, mask, preprocessing and ridge grid as the
baseline-independent raw control. Add fixed 1/6/24h windows plus all history
for each remaining measured variable. Concatenate values then support features.
Keep retrained support-only and fixed patient-deranged equal-capacity controls.
Retrain the all-history incumbent and require its heldout MAEs reproduce to
1e-9. Selection uses source tune only; evaluate paired bootstrap against
incumbent, population, support and placebo. No threshold changes/promotion.

Unlike MAP_EVENT_HISTORY_20260912 this is target/sibling-blocked current
reconstruction, not own/full future forecasting. It tests whether aggregate
history obscures recency, not whether a new neural architecture is required.
This reused single-split research result is not pristine external validation.

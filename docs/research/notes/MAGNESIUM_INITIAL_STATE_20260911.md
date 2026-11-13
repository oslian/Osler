# Initialization identity correction

Reconstructed seven matched native candidates before training. All2285state
tensors are bit-identical across seeds. New projection columns are zeroed,
old columns/bias copied, and new NativeEventFilter loads every inherited tensor.
Thus no random initialization difference survives this warm-start path.

The previous ensemble report's phrase "partitions AND seeds" must not imply
initial-weight variability. Seeds change sampling order, fit/tune membership,
source-dependent event statistics/support and selected epoch. Their separate
contributions remain unisolated. Deterministic dropout-disabled policy is
intentional and matched in training/inference, not a new missing-train bug.
No model fitting/runtime/promotion. Artifact magnesium_initial_state_20260911.json.

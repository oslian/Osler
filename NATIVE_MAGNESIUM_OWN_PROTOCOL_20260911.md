# Magnesium own-history benchmark

Same query/patient partitions, restricted to observed magnesium AND glucose
future pairs to match multi-task eligibility. Only magnesium input/history,
no event values. Same model capacity,32epochs,patient-uniform sampling,AdamW1e-4.
Primary magnesium value+JEPA loss and magnesium source-tune checkpoint selection.
Compare frozen real-auxiliary magnesium predictions on identical evaluation rows.

This is a task-specific benchmark, not an isolated input ablation: multitask
uses glucose selection and downweighted auxiliary objectives, and sees additional
glucose-only rows. Do not infer causal cross-organ contribution from ranking.
No runtime/calibration/new lockbox/promotion. Both current and future measured,
same inherited initialization and native clock/source limitations.

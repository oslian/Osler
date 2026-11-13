# Complete Exposure Continuation

Continue the expanded sequential real/process/placebo arms from their FINAL
epoch8 model, AdamW, patient queues and RNG states. No reset, new learning rate,
weighting, group or target changes. Original33targets retained to isolate added
training; six untrained targets remain pending rather than silently omitted.

Stop at ceil(maximum broad target support/536)=19 total epochs, fixed from
source support before continuation. This guarantees every available broad-fit
patient-target example has been consumed at least once under the existing queues.
Sparse targets and the smaller selected group repeat more often: report that
tradeoff, do not claim this is a pure diversity intervention.

Preserve original90+90selection and the exact original macro rule including
epoch0-8 candidates. All checkpoints remain private. Expanded3323source selection
can be replayed separately under its recorded protocol, not silently substituted
mid-training. Frozen old and broad evaluation must remain separate and cannot
choose the new budget. No evaluation-informed early stopping or gate changes.

Completion is training through19 plus source/evaluation evidence, not an automatic
promotion. Full controls, old-cohort retention, patient/site inference and later
external/interval requirements still apply. Runtime unchanged during experiment.

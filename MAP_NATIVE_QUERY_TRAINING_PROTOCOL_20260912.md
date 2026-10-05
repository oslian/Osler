# MAP Availability-Native Neural Contrast

Question: does full history improve on own MAP history on the newly rebuilt
MAP availability-anchor task? This is not an old/new-anchor paired effect,
because eligible query distributions differ.

Frozen before fitting: seven existing patient partitions; fit/tune/evaluation/
hospital roles; source-only transforms already bound in the preparation report;
same initialization per arm; own MAP versus full supported history; historical
deterministic dropout policy; AdamW 1e-4/weight decay 1e-4; 32 epochs with 16
batches of 32 uniformly sampled patients and one query per sampled patient;
SmoothL1 value plus target-specific EMA JEPA loss; gradient clipping 1. Select
epoch zero or later solely by patient-equal source-tune MAE. Full tune queries
are evaluated, without filtering on potassium labels.

Both arms see the identical new eligible MAP labels and patient sampling stream.
Own excludes other variables and treatment channels; full includes eligible
history, so a gain would still require a support-only attribution control.
The full 48-hour query-replayed native history is retained without truncation.
The horizon contract remains the prepared nominal 6h contract. No adjustment
based on held-out errors, no future treatment and no test-selected threshold.

Evaluate patient-equal MAE and paired 20,000-draw 95% bootstrap differences
against own and persistence in new source patients and held-out hospitals.
Seven splits are required for stability; seed7 alone is not a pass. These
datasets have been repeatedly inspected and are not a fresh lockbox. No
automatic promotion: calibration, broader gates and untouched replication are
outside this comparison. This is the established native JEPA diagnostic, not
the full persistent anatomical body-state architecture validation.

Runner: `scripts/train_map_native_queries.py`. Checkpoints/predictions remain
private. Public reports contain aggregate results and provenance hashes.

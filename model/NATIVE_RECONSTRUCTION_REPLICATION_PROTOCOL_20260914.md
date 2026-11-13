# Repeated Patient-Split Reconstruction

Use existing partitions19/31/43/59/71/83, frozen raw-as-of samples, and original
39target catalog. Remap each shard by patient identity using the new partition.
Old shard-role labels must not determine fitting. Exclude from fitting only
targets with fewer than30source fit or tune patients; keep them explicitly
unsupported in reports. Do not replace seeds or use heldout effect selection.

Each split starts from the same release with that split seed for fresh shared
state initialization and complete-exposure schedules. Recompute fit medians.
Real, trained process-only and target-wise fit-patient-deranged placebo receive
the same eight epochs, AdamW1e-4, SmoothL1, clip1, zero-start median readout and
source-tune selection including epoch0. Process neutralizes numbers in fit,
tune and evaluation; placebo labels only within fit target. No future labels.

Save optimizer and RNG at each epoch. An epoch1 execution tranche is permitted
for end-to-end verification, but does not complete the eight-epoch experiment.
Heldout accuracy stays uninspected until all three arms complete eight epochs.
All datasets/hospital holdouts are reused; these are not independent studies.
No model, interval or runtime promotion under this protocol alone.

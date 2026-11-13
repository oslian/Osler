# Pretraining Provenance Boundary

Release checkpoint records a canonical eICU examples path,563736training rows,
3286subjects; current file contains580980rows3291subjects before training filters.
Current MAP queries contain1763subjects. Exact non-null string subject/stay
intersection with the current recorded pretraining source is zero for both.
No identifiers were exported. Current files and checkpoint are hash-bound.

This does NOT verify historical source bytes or identity remapping. Checkpoint
does not bind its original training source hash. Therefore do not claim all
pretraining was untouched; also do not assert overlap without evidence.

The separate verified in-sample residual issue remains: native MAP fine-tuning
used all64 residual-fit patients. A nested patient-disjoint retraining comparison
must exclude residual development patients from that baseline optimization and
selection, use source-only transforms, and retain this warm-start caveat.
No new training, promotion or runtime/calibration changes.

Evidence: `artifacts/native_pretraining_overlap_20260913.json`.

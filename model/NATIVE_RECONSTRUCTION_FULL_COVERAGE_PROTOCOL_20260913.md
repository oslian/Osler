# Full-Coverage Matched Training

Reuse exact saved patient/target examples and eligible35targets from sampling
pilot. No cohort, labels, normalization, initialization or validation changes.
Each balanced epoch randomly permutes patients per target and cycles each
target to max_target_patient_count draws, in shuffled round-robin target order.
Assert every fit patient/target key consumed each epoch. Smaller targets may
repeat; this trades lower support for equal target exposure, not more evidence.
Event-weighted control gets exactly the same number of updates each epoch.
8epochs, otherwise same AdamW/SmoothL1/clip/source-tune selection including0.
No threshold/promotion changes. Both schedule and budget differ from prior
512step pilot; their separate causal contributions are not isolated here.
All heldout scores remain small reused-source pilot evidence, not full release.

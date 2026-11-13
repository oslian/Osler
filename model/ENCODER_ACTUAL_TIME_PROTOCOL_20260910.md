# Actual Elapsed-Time Training

Matched2250fit/136tune,7seeds,4own/whole bounded/identity arms,64epochs.
Same input physiology, sampling, labels, optimizer and initialization as
encoder_source2250_64. Source supervised and tune queries now request actual
elapsed time (5-7h), not nominal6h. Teacher still encodes observed value delta.
Selection is source-tune actual-time MAE including epoch0. This changes
training and source-selection query semantics; no new architecture.

Primary operational comparison remains fixed6h query on identical validation
rows with the existing5-7h truth tolerance; this is approximate6h scoring,
not exact6h ground truth. Save additional actual-time queries separately.
Do not present actual-time-query scores as prospective prediction of when a
measurement will occur. For matched actual-time evaluation the old model
must also be queried at actual time, not just reuse old6h predictions.

eICU selects everything. MIMIC repeatedly inspected validation, not lockbox.
No new calibration, runtime, retention or causal promotion. Candidate_only.

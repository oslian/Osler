# MAP Clock Matched Training

Fixed seven-seed own/full/support-control comparison, initialized identically
to original native MAP training. Only the two native history clock coordinates
change to recovered unbounded event hours/24 and /168. Query context, future
labels, source-only normalization, support, event values, EMA/value objective,
32 epochs, patient sampler, learning rate, dropout policy and tune selection
remain unchanged. Start epoch0 is eligible; no evaluation-based settings.

Record all arms versus their old counterpart and new own/control using
patient-equal MAE and paired bootstrap. This is a reused source/hospital
diagnostic, not a fresh lockbox. No automatic promotion, calibration or
release changes. Do not infer complete repair from a single split. The old
potassium clock experiment does not substitute for this matched MAP trial.

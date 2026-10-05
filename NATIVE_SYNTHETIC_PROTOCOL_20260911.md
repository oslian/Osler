# Same-path artificial signal positive control

This is NOT physiological training or an RCT. Use original native history
clock and same7splits/32epochs/16steps/batch32/EMA loss/source tune selection.
Assign independent fixed-seed Rademacher(-1,+1) values to each known-rate token,
preserving all rate masks/update positions, all laboratory values and query
contexts. Reused event identity has the same assigned value. Never use outcome
information to generate signals. Add0.5 times the last known event sign to the
real future potassium label; queries without a known rate receivezero shift.

Fit synthetic_real and capacity-matched synthetic_support, latter retains
known masks but no event numbers. Evaluate on the same artificial labels in
each patient role. Primary contrast real versus support with existing patient
bootstrap. Frozen whole and persistence are secondary task controls only;
their artificial-label scores are not the original biological scores.

This tests a strong known historical-event signal under the actual JEPA
training path. Success does not prove small real effects learnable, sufficient
memory for remote events, causal effect, or clinical validity. Failure does not
establish information absence. No deployment, calibrator, registry or biological
promotion. Keep synthetic outputs separately named and marked.

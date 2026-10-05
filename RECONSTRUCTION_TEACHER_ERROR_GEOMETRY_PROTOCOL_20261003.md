# Teacher Error Geometry: Frozen Source Diagnostic

No fitting, new prediction, selection, clipping or external query. Reuse all47
targets/all5source folds of the completed state-teacher experiment. Purpose:
distinguish faithfully following a worse teacher from failing to approximate
a better teacher. This is not the September18 tail/ranking geometry audit.

Let B=original body100, T=raw teacher, S=state student, Y=real current truth.
Teacher quality is defined RETROSPECTIVELY by |T-Y|-|B-Y|: better,worse,equal.
No clinical stable/abnormal meaning, feature, calibration or legal router is
attached to these outcome-defined groups.

Partition movement from B to S into unchanged, teacher stationary, away from
teacher, inside the B-to-T segment, or beyond T. Separately partition movement
relative to Y into unchanged, away from truth, approaching truth without crossing,
or crossing truth. With a=|S-B| and z=(Y-B)*sign(S-B), the exact MAE increment is
a - 2*clip(z,0,a). Verify row-level equality and all sums before reporting.

Report every target, per-fold counts and loss contributions. For each category
and teacher-quality x teacher-position intersection, average indicator, gross
harm=max(delta,0), gross benefit=max(-delta,0) and net delta over a patient's
original fold queries FIRST, then equally over patients or provenance components.
Contributions use the whole target population denominator, not just members of
the stratum. Counts may overlap across strata for repeated patients; occurrence
counts are not independent sample sizes. Report gross contributions so small
net differences do not hide large cancelling benefits and harms.

Original overall truth/imitation MAE must reproduce. Categories are descriptive,
post-outcome and post-result diagnostics, not prespecified efficacy hypotheses
or a new statistical gate. Do not drop outliers or adjust thresholds. No model
promotion, causal claim, encoder change or runtime modification is authorized.

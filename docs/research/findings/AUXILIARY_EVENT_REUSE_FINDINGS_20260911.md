# Future Event Reuse Audit

Added selected future source ID to the label builder. Rebuilt all53 source
columns and exactly reproduced current, future, age, delta_t and pair masks
against the original49,823-query artifact. Original artifact unchanged. IDs
remain local; public report is aggregate. Equal-valued distinct measurements
retain distinct IDs in a regression test. No model input receives future IDs.

Potassium6174pairs/1309patients correspond to4660 distinct stay/event keys;
24.52% of rows are excess over unique outcomes. Max8queries per event,p95=3.
Glucose49823pairs correspond to42864outcomes (13.97% excess,max10,p95=2).
Other auxiliary targets show22.41%-27.89% excess. This is not effective sample
size: distinct outcomes are still correlated within a patient.

Different legal prediction anchors may legitimately share future truth, so this
is not leakage or a mandate to discard queries. Patient-level bootstrap already
accounts for within-patient clustering in uncertainty. However patient-uniform,
then row-uniform training gives repeated outcomes more probability within a
patient. Whether event-balanced sampling repairs error remains untested here.
It must preserve legitimate anchors and compare on the same evaluation rows;
future event IDs must never enter inference or deployment selection.

This repairs missing audit provenance, not forecast accuracy. No runtime,
weights, calibration or promotion change. No sole root cause established.
Artifact: `artifacts/auxiliary_event_reuse_20260911.json`.
Command: `python -m scripts.audit_auxiliary_event_reuse`.

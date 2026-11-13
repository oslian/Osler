# Pre-anchor bedside history support

Fixed descriptive split: at least5 observed native groups for EACH of HR,
MAP,O2,RR,temperature in the existing48h history, versus fewer. No outcome or
future event used to define the split, no threshold search or trained router.
Same existing real-number and support-only predictions,20k common patient draws.

Real-support CI gains/harms: source higher-support0/0, lower-support0/1;
hospital higher-support0/1, lower-support0/1, all out of7. Higher-support
hospital subgroup has70patients; lower has147, overlapping because a patient's
support can change across queries. Source higher subgroup180-211patients.

No reliable numeric gain emerges in the defined richer-history group. This
rejects this simple support split as an observed rescue, not all data-quality
or sampling-power explanations. Five events is a diagnostic threshold, not a
claim of clinically sufficient history or a validated deployment rule. No
model selection, fit, runtime, calibration or promotion changed.

Artifact: `artifacts/bedside_history_support_20260912.json`.

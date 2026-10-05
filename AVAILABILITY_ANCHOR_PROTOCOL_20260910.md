# Availability-Anchored Glucose Diagnostic

Freeze before outcomes: retain prior patient roles, rebuild all queries at raw
glucose result availability; current is the latest measured known version.
Current age must be at most 6h for scoring. Rebuild truth as the nearest actual
measurement 5-7h AFTER the new anchor, ties by measurement time then native ID.
All queries persist regardless of future label support. Final versions are used
only as retrospective future truth; unknown results never enter history.

Bounded scope: laboratory glucose only. Other cached features, old labels and
old frozen-JEPA predictions are NOT carried to shifted anchors. This is a timing
contract and source-trained signal diagnostic, not a full JEPA repair/promotion.
eICU revised-entry time is conservative, not proof of first provisional availability.
The cohort remains selected by the previous experiment, not a general ICU sample.

Compare persistence, current+age ridge residual, and current+age+6/24h history
ridge residual. Histories have level, variation, slope and explicit validity.
No treatment, hospital ID, future actual delta_t or measurement-time prediction
enters features. Seven existing patient split seeds; source-only alpha 1/10/100
selection; patient-equal training weights and paired patient bootstrap (1000).
External eICU hospital and already-inspected MIMIC never select settings.
No clipping or result-driven parameter revision. No interval/promotion claim.
Differences from old scores cannot isolate a model improvement: the task changed.

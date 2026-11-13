# Event Summary Quality Repair

## Reproduced Defect

`event_windows` respected result availability, source revisions and same-time
conflicting values, but did not inspect `quality_valid`. A finite invalid
900 reading was included as the maximum despite its false quality flag.
The new regression failed with maximum900 instead of70 before the repair.

## Repair

After selecting the latest available source versions, exclude a measurement
timestamp if any of its sources has false or missing explicit quality status.
Retain historical valid observations as historical summary members, not as a
replacement current measurement. Do not delete prior evidence based on a
revision that was unavailable at the query. Explicit valid zero is retained.
Events without the optional quality column keep their existing contract.

Tests cover the invalid spike, a valid zero revised invalid later, prefix
invariance, and agreeing sources with contradictory quality. Prior same-time
value-conflict and distinct event identity tests remain.

## Scope

This fixes the event-summary feature path before using it with the newly
qualified bedside events. Neither the completed native MAP JEPA nor its
fixed-lag tree control calls this helper. Consequently this is NOT their
observed failure cause and does not claim accuracy improvement. Earlier
event-summary artifacts retain historical code hashes and are not validation
of the updated helper. New summary caches must be rebuilt before training.

No weights, runtime registry, serving behavior or interval artifact changed.

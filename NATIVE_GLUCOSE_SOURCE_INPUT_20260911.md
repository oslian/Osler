# Native glucose source representation

Question: does retaining the known source name throughout history help the
same JEPA learn a transferable glucose correction? A current-source offset
already failed; this is a different input representation, not another offset.

The builder joins raw lab IDs to source names and uses the exact deduplicated
native schedule ordering. Only the latest measured glucose accepted by the
existing token can carry a source indicator. Conflicting/unknown source names
remain unknown. Bedside glucose and glucose are source labels, not verified
specimen or assay identities. No future source names enter features.

Existing token values, masks, clocks, labels, partitions and weights are not
changed. The new two-column array is private and hash-bound; the aggregate
artifact is `artifacts/native_glucose_source_20260911.json`.

Completed cache: 128,236 native groups, 53,752 accepted glucose groups;
41,639 bedside glucose, 12,112 glucose, one accepted glucose group with unknown
or ambiguous source. Full history-index/query-pointer equality checks passed.
The one unknown is retained, not assigned by guessing. These are event-group
counts, not independent patients or scored future pairs.

Reproduce with `python -m scripts.build_native_glucose_source` (fresh output).
Six unittest checks cover source selection, ambiguity, unknowns, rejected
observations, latest draw selection and existing token semantics. Pytest is not
installed in the current runtime; these tests use the existing unittest suite.

## Next accuracy contrast

Keep the same source-defined seven patient partitions, glucose labels,
optimizer and selection procedure. Compare source-aware history with an
equal-width zero-source control for both own and whole views. Insert the two
channels BEFORE the existing six event channels to preserve the native filter's
observed-rate mask position. Initialize new projection columns to zero; require
zero-channel baseline reproduction before fitting. Select only on source tune,
then evaluate new source patients and held-out hospitals against both controls
and persistence. Do not use future source switching to select patients/routes.

No accuracy training has been completed for these source-aware inputs. Input
correctness is not forecast improvement or promotion. Historical pretraining
membership remains uncertified; evaluation cohorts are reused. No runtime,
interval, causal or organ-coupling claim changes.

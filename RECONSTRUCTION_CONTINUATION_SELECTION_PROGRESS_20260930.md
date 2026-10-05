# Full Epoch-10 Source Selection: Progress

The complete source-selection query grid is being evaluated using the same
frozen epoch-10 real/process weights as the completed outer assessment. No
refitting, epoch selection, exclusions, or new promotion rule is introduced.
The purpose is to replace the small source monitor as a diagnostic of remaining
component harms, not to reinterpret reused outer data as an untouched lockbox.

## Completed

| Fold | Pairs | Patients | Targets | Inference |
| --- | ---: | ---: | ---: | --- |
| 0 | 363,526 | 14,845 | 47 | Certified reuse of identical outer-fold-2 computation |
| 1 | 363,526 | 14,845 | 47 | Full newly executed inference |
| 2 | 356,944 | 14,735 | 47 | Full new queries using certified fit-0 weights |
| 3 | 356,944 | 14,735 | 47 | Full newly executed inference |
| 4 | 356,944 | 14,735 | 47 | Full newly executed inference |

Folds 0 and 1 use the same query patients with different fitted weights. Do not
sum them as 29,690 independent people or two independent replications.
Fold 0 reuse checks exact fitted-population identity, continued-weight digests,
all fit-only scaler fields, query identities, original truth/provenance/control
values, complete source eligibility and native-history binding. Fold metadata
alone is changed. No selection outcome enters model fitting.

The first two completed files passed SHA256, source patient role, unique paired keys,
finite output, dependency and complete-grid checks. Their predictions reproduce
every target's original epoch-10 fixed monitor for both arms, using the full
source file filtered by the original monitor identities. Maximum absolute
metric difference is 2.22e-16 (fold 0) and 1.11e-16 (fold 1).

Private prediction hashes:

- Fold 0: `700d4032f9bedc9411417dd640e3e68f0a60d36f8fdfff25ed64a713ff433cf6`
- Fold 1: `91096236c2a79b0828c66112e46a404a91c60a4b666c21dcd62a21846eb04bcc`
- Fold 2: `8cc0bf0d1a0723ff1040a36929404884ca45e3285e3b4ac10049ba0d62ef5625`
- Fold 3: `4d3aeaf9ee6bd11e9ba55f92bbfc23c8838c653ad15531eb0530f083f4a9048f`
- Fold 4: `1a8c6fd1192d08709d5e19160be4677a893b9c941bade2ea30664fcd23239f3d`

Fold 1 ran to completion without interruption. Four new reuse tests and four
existing fixed-monitor tests pass. The first reuse draft assumed a `role` key
in an older metadata document; it stopped before output. Role validation uses
the authoritative patient partition and complete eligibility grid instead.
This was a diagnostic implementation assumption, not a historical model defect.

## Completed Follow-Up (2026-10-01)

All five folds now have complete inference and repeatable integrity verification.
Folds 2/3/4 completed without interruption. Their monitor parity differences are
2.22e-16 / 4.44e-16 / 2.22e-16. Five aggregate verification artifacts bind the
query payloads, roles, dependencies, full eligibility grid and original monitors.
Eighteen tests pass across summary, verification, reuse and monitor behavior.

All-fold paired descriptive assessment is complete in
`RECONSTRUCTION_FULL_SELECTION_FINDINGS_20261001.md`. The 1,797,884 query
occurrences represent 720,470 distinct patient-target pairs and 29,580 distinct
people, not 1.8 million independent observations. Serving weights, intervals
and registry remain unchanged. No job from this stage remains live.

## Commands

`python -m scripts.certify_continuation_selection0`

`python -m scripts.infer_continuation_selection --fold 1`

The latter was also used for folds 2/3/4; existing output files are rejected,
not overwritten. The fixed historical outer inference script is unchanged
because completed inference reports bind its exact source digest. All patient
rows remain under the ignored local `continuation_selection_20260930` directory.

`python -m scripts.verify_continuation_selection --fold N` verifies each completed
fold before `python -m scripts.summarize_continuation_selection` can assess all
five. Verification and assessment outputs are immutable; existing files are
rejected. The original first-two-fold description above records that earlier
stage, not the current completion status.

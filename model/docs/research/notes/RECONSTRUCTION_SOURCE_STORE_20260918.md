# Expanded Source Loading Verification

The fixed-budget queue preflight is complete. Each target receives 4288 draws;
unique exposure is min(available patients, 4288). This is sampler replay, not
model training. Dense targets do not consume the entire 10201-patient pool.

The new SourceStore retains compact label/key metadata and loads patient histories
through a bounded standard-library LRU cache (16 shards). It verifies SHA256 and
patient identity. Source roles are checked against the current nested split.
Historical tensor role tags predate that split and must not be interpreted as
current roles. An initial new-loader check incorrectly required these tags to
match; verification stopped before fitting, and the check was corrected to use
the authoritative split. This is not a newly demonstrated cause of old model
failures, nor proof that historically inspected data are an untouched lockbox.

| Mode/group | Fit patients | Fit pairs | Encoder patients | Encoder pairs |
|---|---:|---:|---:|---:|
| Small/selected | 536 | 12993 | 90 | 2470 |
| Small/broad | 536 | 12147 | 90 | 2025 |
| Expanded/selected | 536 | 12993 | 90 | 2470 |
| Expanded/broad | 10201 | 234618 | 90 | 2025 |

Counts concern the original 33 trained targets, not all 39 available targets.
The extra 3233 broad encoder patients are not included. All real source shards
and parent report bindings verified, with one lazy sample per target checked.
Seven tests cover bounded loading, changed files, patient mismatch, forbidden
nested roles, fixed selection membership, queue cycling and exact queue resume.

Evidence: artifacts/native_reconstruction_source_store_20260918.json and
artifacts/native_reconstruction_full_exposure_preflight_20260918.json.
No expanded-source optimizer run, new accuracy result, promotion, calibration,
or runtime change. Next implement the paired small/expanded trainer with the
same persistent queues and budget; preserve original evidence files.

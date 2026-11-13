# Time-qualified bedside events

Same 5,526 existing queries; no label, patient split or prediction changed.
Built 1,347,546 private canonical nursing events from the existing numeric
cache. Exact category/label/name mapping; Celsius only, explicit source IDs,
separate measured and entry time. Entry-before-measurement rejected rather
than clamped. Equal-valued new measurements remain distinct; exact event
duplicates collapse. Invasive and noninvasive MAP retain method provenance,
and same-time disagreement remains missing under the existing snapshot policy.

| Variable | Queries with observation within 6h | Patients |
|---|---:|---:|
| Heart rate | 5366 | 1724 |
| O2 saturation | 5247 | 1698 |
| Respiratory rate | 5233 | 1708 |
| MAP | 4507 | 1522 |
| Temperature | 4803 | 1621 |

These are time-qualified, finite numeric observations, not yet model-ready
clinical-quality inputs. Temperature C is explicit in the label; other units
are label conventions, not an explicit raw unit field. Zeros are retained as
observations (HR 68, O2 15, RR 1652, MAP 28 events), not converted to missing.
Their plausibility requires a separate quality policy before training.
Six-hour freshness is the existing comparison contract, not newly optimized.

Eleven bedside/raw-event tests pass, covering timestamp legality, zero,
remeasurement, duplicate identity, C/F exclusion, ambiguous oxygen label and
conflicting MAP sources. Extraction and same-anchor support finished; range
qualification, native stream integration and matched retraining remain pending.
This demonstrates missing channels can be populated for many existing anchors;
it does not demonstrate an accuracy gain or explain all previous failures.

No runtime weights, registry, calibration or promotion changed. Aggregate
artifact: `artifacts/native_bedside_events_20260911.json`. Patient rows stay
private outside Git.

## Native encoder wiring

The native schedule and token encoder previously recognized only lab_available;
bedside_available would have been silently ignored as a measurement. Both now
share an explicit observation-kind whitelist. Event identity keeps source-kind
namespaces, while same-variable/same-measurement-time conflict detection spans
sources. Synthetic raw nursing -> available-time replay -> normalized token
test verifies arrival masks, value and measurement age. Separate tests verify
bedside zero and cross-source conflict handling. This enables new inputs; it
does not explain historical failures on lab-only data, where no bedside rows
were supplied. New private caches and trained source support remain required.
Historical artifact code hashes must not be rewritten to claim new-code runs.

## Quality and query input build

Applied existing eICU sepsis extractor PLAUSIBLE bounds without tuning them.
2,352 finite events fail these bounds; they remain in the private stream with
quality_valid=false. A latest rejected observation blocks a current estimate
instead of resurrecting an older normal value. These are operational quality
bounds, not proof of physiological impossibility. Snapshot and native token
readers honor the flag. Absent quality columns preserve the original lab policy;
the merge explicitly marks original lab rows true, not NaN.

Rebuilt identical 5,526 query inputs. All non-bedside current/mask/age columns
are exactly equal to the prior conflict-checked inputs. Qualified query support:
HR5366, MAP4505, O25245, RR5195, temperature4787. Laboratory labels and
partitions unchanged. Artifacts: native_bedside_quality_20260911.json and
native_bedside_inputs_20260911.json under artifacts. Private expanded input
cache exists; expanded native history tokens and matched training still pending.
No accuracy or promotion claim. Non-temperature unit provenance remains
label-conventional rather than explicitly measured raw unit metadata.

## Expanded native history cache completed

Combined 2,127,267 existing lab/insulin and bedside events using availability
time; same 5,526 queries, 48h replay window. Encoded 207,499 unique simultaneous
groups into the existing input schema. Per-query history median68/max385;
no history truncation or resampling. Observed token counts: HR77993, MAP64372,
O275225, RR72631, temperature36276. Insulin remains the original record contract,
not newly verified administration timing. All token numbers finite; rejected
quality values are not encoded as observations.

Artifact: `artifacts/native_bedside_tokens_20260911.json`; private
native_tokens_bedside.npz. New query and native-history caches are now both
ready. Prior training-pending statements still apply: no matched retraining,
calibration or promotion has occurred. Need source-fit support recomputation
for the newly observed channels, not reuse the old absent-channel whitelist.

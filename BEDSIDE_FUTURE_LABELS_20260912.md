# Bedside future-label contract

Recent bedside trials predict glucose only; their negative increment must not
be generalized to the added vitals' own forecastability. Constructed additional
measured targets on exactly the same5,526queries, without selecting new patients
or shifting anchors. Existing6h +/-1h, current age<=6h diagnostic contract.

| Target | Paired queries | Patients |
|---|---:|---:|
| Heart rate |5095|1595|
| MAP |3925|1303|
| O2 saturation |4873|1537|
| Respiratory rate |4814|1559|
| Temperature |3521|1324|

Fixed a newly exposed helper gap before using bedside truth: future_pairs did
not honor quality_valid. It now rejects a nearest future group if any source
has invalid quality, without skipping to a farther valid measurement. Physical
zero remains valid when explicitly quality-valid; no zero-as-missing rule.
Snapshots already honor quality. Twelve related tests pass. Historical lab
datasets have no new quality flag and these bedside labels were not used by
the previous glucose training: this is not claimed as its failure cause.

No neural training or promotion yet. Proposed next matched task is MAP with
own history versus full history; new eligibility must be applied equally to
all arms. New label support does not establish accuracy, independence from
measurement process, or untouched external validity. Native time-based inputs
and fixed-horizon labels remain separate; recorded timestamps do not imply
uniform future support. No release/calibrator/registry change.

Artifact: `artifacts/bedside_future_labels_20260912.json`; private labels outside Git.

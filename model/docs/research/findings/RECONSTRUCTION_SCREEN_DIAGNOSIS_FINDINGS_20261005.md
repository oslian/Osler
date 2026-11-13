# E41: zero screens has more than one meaning

This is a retrospective audit of E40 current masked reconstruction, not a new
forecast test, candidate fit, acceptance policy or release promotion.
The 53 targets are not the release's target-by-horizon forecast cells.

## First unmet condition

The reporting order is support, paired mean direction, CI, nominal t-test,
family-adjusted test. Every overlapping failure is also retained in the JSON.
These labels describe evidence; they are not causal explanations of physiology.

| Inventory category | Targets | What this does and does not show |
|---|---:|---|
| No scored selection observations | 7 | No evaluation evidence, not model error |
| Scored but below 30 patients or 3 components | 19 | Not enough support under the existing contract |
| Sufficient support, fails to beat every control in both mean weightings | 27 | Cannot attribute the full failure only to statistical significance |
| Meets all mean comparisons but blocked only by later inference | 0 | No such target in this fixed candidate |
| Full screen | 0 | Unchanged original decision |

There are 46 scored targets. All 46, including low-support targets, fail at
least one of the ten controls at the descriptive mean stage. This is not a
claim of significant harm, and does not mean each target loses to every control.

## Separate the strong raw comparator

Within the 27 sufficiently supported targets:

| Candidate versus raw observations | Count |
|---|---:|
| Not lower under BOTH patient-equal and component-equal MAE | 23 |
| Lower under both means, but paired CI requirement fails | 4 |
| Passed | 0 |

The four mean gains are heart_rate, lactate, potassium and respiratory_rate.
They are descriptive exceptions, not newly selected heads or promotions.
Against PCGrad alone, 15/27 fail the mean condition and 12/27 first fail CI.

Across all 270 sufficiently supported target/control comparisons, only four
have multiplicity as their first unmet condition: phosphate versus paired mean,
sodium versus bound_start, wbc versus immediate, and wbc versus unbridged.
Those targets still fail other comparisons earlier. Adjusting multiplicity
alone cannot make this fixed candidate pass the complete screen.

## What was repaired

Failure reporting previously exposed only zero full screens without this
complete disjoint inventory. The new diagnostic makes support, point-error
direction, interval uncertainty and multiplicity explicit. It does NOT change
the old gate or falsely turn an inconclusive target into a failed prediction.

The suspected Monte Carlo p-resolution problem is absent: the implementation
uses a one-sided paired t-test for p_less and bootstrap for the CI. Synthetic
clear improvement passes the original threshold. This is not an endorsement
of every statistical assumption; no alternative test was selected here.

No interpretation that "the gate alone prevents success" is supported for
E40. Equally, these results do not prove every unsupported target is unlearnable.
Optimization, conditional support and representation/readout limitations
remain incompletely separated. The true model repair remains unfinished.

## Verification and boundaries

- All five saved selection files, 5120 prediction occurrences, all53 inventory
  and all ten controls retained; all460 scored comparisons audited.
- Independent scalar patient collapse, component collapse, MAEs and paired
  t-statistics reconstructed from private rows. Maximum difference3.916e-13.
- Original CIs reused after E40's independent replay verification; no new seeds
  chosen. Every original comparator screen and full intersection reproduced.
- 81 source/code/protocol/test bindings match. All classification totals match
  the 53-target inventory; no IDs or row-level data are in the public report.
- 11 new/existing tests pass. Existing saved model weights were not modified.
- MIMIC/NWICU remain reused development. Fold0/2 identity and historical encoder
  exposure limitations remain. No new independent external confirmation.
- No training, runtime, interval or promotion change. No parameter sweep.

[Protocol](../../../RECONSTRUCTION_SCREEN_DIAGNOSIS_PROTOCOL_20261005.md),
[aggregate report](../../../artifacts/native_screen_failures_20261005.json),
[E40 source findings](RECONSTRUCTION_ACTUAL_STEP_FINDINGS_20261005.md).

# Full Source BUN History Coverage Replication

## Fixed Scope

Same bounded-site epoch7 and masking policy as the latest-coverage diagnostic.
All90 original selected source-selection patients and3086 accepted BUN patients
from the existing3323-person broad encoder-selection pool. No fit overlap,
duplicate patients, new fitting, evaluation-based selection or new external
lockbox. All original90selected/83broad per-patient errors and event counts
reproduce within1e-7, not merely the aggregate score.

Compare full history, latest observation event per independent variable, and
that union plus last8 events. Target and siblings removed over full history;
retain original timestamps, values, masks, ages and simultaneous groups.

## Result

Normalized current-reconstruction MAE, not future forecast errors:

| Source scope | Patients | Full | Latest each | Latest-minus-full95% interval |
|---|---:|---:|---:|---|
| Selected |90|1.105465|1.157538|[+.000476,+.105321]|
| Broad all |3086|.893612|.857813|[-.042463,-.029139]|
| Broad original |83|1.116262|1.055578|[-.101102,-.020621]|
| Broad additional |3003|.887458|.852347|[-.041676,-.028643]|

Broad reduction4.006% vs selected increase4.711%. Additional3003 patients
replicate the direction seen in the original83: broad benefit is not confined
to the tiny original selection subset. The selectedgroup remains90 and its
interval is borderline/unadjusted; this is not equally powered replication in
both groups.

Last8-plus-latest also has opposite signs: selected change+.044686
CI[-.000475,+.089939], broad-.030236CI[-.035696,-.024785]. Mean retained event
counts selected61.27->6.07latest, broad33.15->4.75latest. Event frequency, age,
case mix and history usefulness have not been independently isolated.

## What This Localizes

More repeated history is not monotonically beneficial for this fixed BUN
shared-state reconstruction. The broad-group harm from retaining it survives
a larger source replay. A universal truncation fix still fails the selected
population and must not enter runtime.

This is not evidence that real older physiology is inherently uninformative:
deletion changes initialization, transition gaps, observation sequence and
training distribution. The next distinction is whether the harm repeats across
source sites and pre-anchor history-support strata, before learning any
retention gate. Do not use retrospective target truth to choose history length.
Any eventual repair must be trained on fitting patients and checked against
full history, latest-coverage and matched placebo controls on separate roles.

Five summary/coverage tests pass; original-pair replay and dependency hashes
checked. No runtime weights, intervals, registry or promotion changed. Full
project scope remains33trained plus6pending, not reduced to BUN.

Evidence: artifacts/native_reconstruction_bun_full_history_20260920.json.
Private patient rows remain under ignored data/.

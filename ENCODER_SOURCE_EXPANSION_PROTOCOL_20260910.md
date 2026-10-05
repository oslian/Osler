# Independent Source Patient Expansion

The prior glucose diagnostic uses386 development patients,250 per fit, inherited
from a bounded creatinine replay. This protocol builds additional source support
without changing old validation queries, weights or feature artifacts.

Identify source hospitals from all old development query stays and held-out
hospitals from all old hospital query stays. Require disjoint hospital sets.
Exclude every patient in either old role and every patient having any stay in
any held-out hospital, even if another stay is in a source hospital.

Hash-order eligible unique patients with a fixed source-expansion-v1 salt;
take the first4000 BEFORE inspecting labs, outcomes or pairing support. Keep
only their stays in source hospitals. No external patient characteristics or
errors may influence selection. Preserve the old retrospective cohort's lack
of a new age filter; this is not a new disease-specific cohort claim.

Reuse canonical-unit extraction, actual result availability and the exact
existing glucose anchor/5-7h label builder. All outcomes are used only for
training eligibility/labels, not selection of the4000. Keep unscored queries.
Store raw labs, queries and subject manifest privately under /private/tmp;
write only aggregate counts/hashes to the repository. Old caches must not
be overwritten. Report overlaps and independent patient count explicitly.

Next learning curve must retain identical old tune patients and validation
roles and compare own/whole at matched source support. Additional source
patients are a broader cohort, so improvement would not isolate sample size
from composition perfectly. This preprocessing trial makes no accuracy or
promotion claim. Existing pretraining overlap is still not excluded. No
MIMIC/NWICU tuning and no new pristine lockbox assertion.

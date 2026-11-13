# E22: Separate time evolution from later observation assimilation

## Question and difference

E20 found last-known measurement readability gaps at immediate assimilation and
later along the state path. E21's fixed retention objective improved some known
content but not masked inference. Neither separates subsequent advance/update.
E16 removed dynamics globally in an older model with its existing final readout;
this experiment branches AFTER a variable's last known observation in epoch10,
and fits the SAME reader specification separately for each representation.

## Fixed design before results

- Reuse E19's exact five fold hash samples, fitting/selection patients, masks,
  scalers and 177 supported variable-folds. All 53 variables retain support status.
- Event JEPA and state weights frozen. No additional outer database query.
- At the last observation of variable v, preserve the exact post-update snapshot.
  Continue four paths to the original anchor:
  1. post: snapshot unchanged (existing E20 reader).
  2. advance: original segmented time integration, no later observations.
  3. update: actual clock and gate delta progress, original observation updates,
     but no latent time dynamics or uncertainty growth between observations.
  4. body: original full path (existing E19/E20 reader).
- Removing later observations does NOT remove v's last observation. All inputs
  still respect original target/sibling masking and anchor availability.
- Labels remain the last-known value, not unobserved current physiology. Holding
  history may aid memory while harming current inference; no deployment claim.
- Every representation is 64 body dimensions plus identical 159 process inputs.
  Use existing component-bounded weights, Ridge(alpha=100) with fit-only scaler,
  and the existing fixed 100-iteration HGB. Two new representations x two readers
  x 177 variable-folds = 708 new models. No hyperparameter/target selection.
- Preserve post/body predictions exactly, and reproduce original states against
  saved caches (atol=1e-6, rtol=0). Abort on identity/hash/role mismatch.

## Fixed analysis

For EACH reader, evaluate MAE differences: post-minus-advance,
post-minus-update, advance-minus-body, update-minus-body. Negative means the
first has lower error. Repeated-fold loss is first averaged per patient, then
per source component. Report patient and component paired 95% bootstrap CI and
one-sided tests using existing paired_summary. Fixed family 53 x 4 x 2;
screen requires both CI upper<0 and p<0.05/424, >=30 patients and >=3 components.
Report all signs; reverse-direction means are descriptive, not screened claims.
Report difference-in-differences descriptively to expose nonadditivity.
No clinical, causal-organ, irreversible information-loss or promotion claims.
All cohorts are reused development; five folds are not independent replications.

## Counterexamples and decision

- If suppressing updates improves known-content decoding, subsequent assimilation
  contributes to this reader-specific gap; it does not prove updates are useless.
- If suppressing dynamics improves it, dynamics contributes; do not globally
  delete it (E16 demonstrated harms on other targets).
- If only one reader supports a contrast, retain coordinate/readout confounding.
- If both single paths preserve content but full does not, investigate interaction,
  not either component alone. Null results do not prove no information loss.
- Further repair must improve the original masked inference, not just remember
  a previously seen value. No more retention-weight or readout sweeps here.

Save private states, models and patient predictions under ignored data/research_recovery;
only code, aggregate reports and findings may be versioned. Independently reload
saved models, reproduce predictions and recompute assessment before conclusions.

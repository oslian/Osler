# E28: final organ observation reachability

## Question

Does the E27 body-only readout have a computational path from the final organ
proposal when the query immediately follows that observation? This is a fixed
weight structural diagnostic, not a new accuracy trial or causal physiology.

E26 tested identity binding, E22 probed later content retention, E25 fitted direct
organ readers, and E27 trained a binding intervention. None establishes that
the last organ update reaches a same-time body readout. Their negative results
remain valid; this diagnostic does not reopen their training branches.

## Fixed checks

1. For every E27 trained arm in all five fold mappings, reproduce the state
   update with synthetic, finite body/organ inputs and full presence. Keep body,
   prior state, masks and weights fixed; change only observed organs.
2. At zero elapsed time after update, compare body and organ states, readout,
   and autograd dependence on organ and body proposals. At one hour, check that
   the organ branch can reach body. Positive-time sensitivity is a control,
   not a claim that the changed value is correct or clinically meaningful.
3. Inventory all original fit and selection contexts without fitting or reading
   outcomes to choose examples. Count exact-zero final anchor gaps, single-event
   histories, and nonzero native-to-binding proposal differences at zero gap.
   Keep all contexts and target support, including unsupported targets.
4. Verify saved reports, checkpoints and caches against their bound hashes.
   Output aggregates only. Fold 2 reuses fold 0 initial weights; counts are not
   independent replications. Test synthetic failure cases and zero-time parity.

## Interpretation and stop

A zero final-organ Jacobian with nonzero body Jacobian supports branch-specific
same-time inaccessibility, not total information loss: body proposals already
encode the same observed values. Zero-gap exposure is not an error-attribution
fraction. A positive-time path does not establish useful organ coupling.

If the branch is reachable at zero time, reject this local hypothesis. If it is
not, record the affected boundary and assess the body bypass before proposing
a repair. Do not advance time artificially, change labels, add an unvalidated
organ residual, or deploy old weights on changed inputs to manufacture a gain.
No loss, gate, checkpoint, runtime, forecast or interval change in this audit.

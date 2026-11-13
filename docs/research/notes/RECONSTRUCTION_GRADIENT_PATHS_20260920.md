# Full Reconstruction Gradient Reachability

Fixed bounded-site selected epoch7, one source-fit patient per target/group:
66 probes across33targets and two groups. Allrequires-grad parameters queried
with autograd; no optimizer steps, evaluation labels or weight changes. Source
shard integrity and training/checkpoint bindings verified.

Of2245 requires-grad parameter tensors,99 connect to at least one probe loss.
This is not a99/2245 model quality metric: the wrapper owns many unused
forecast/auxiliary branches. The current reconstruction loss does not supervise
future value heads, future latent heads or uncertainty growth, as expected.

Every tensor in these persistent components receives nonzero gradients in at
least one probe: readout, body dynamics, body update gate, organ dynamics,
organ message and organ update gate. JEPA variable/module mixers, token/state
encoders, embeddings and body/presence projections also receive gradients.
The belief filter has11connected tensors out of15,10nonzero. Each event is
encoded as a one-step proposal, with temporal recurrence performed by the outer
persistent state; absent one-step recurrence gradients alone are not a bug.

## Interpretation

This rules out a blanket disconnected-body/organ training path on these probes.
It does not establish enough gradient signal, stable optimization, complete
parameter utilization across all data, causal organ dynamics or generalization.
Neither the count of unused tensors nor raw gradient magnitude justifies
changing architecture or amplifying gradients. No claim of a unique root cause.

Evidence: artifacts/native_reconstruction_gradient_paths_20260920.json.
The next diagnostic should test what information survives state updates or
whether useful cross-variable information exists under matched observation
support, rather than assuming the persistent path never learned at all.
No runtime, registry, interval, weights or promotion changes.

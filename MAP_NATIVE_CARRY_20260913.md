# MAP Latest-Observation Direct Carry

Seven source-fit probes, 96 distinct sampled patients per seed, compared own,
full and support-control checkpoints. All predictions reproduce within the
declared numerical tolerance. Two synthetic tests verify the audit's last-
target boundary and inactive padding handling.

For each history, the audit multiplies (1 - update_gate) after its last MAP
observation. This measures only a direct mixing path; it excludes decay and
innovation-mediated retention, not total memory or the recurrent Jacobian.

Every arm/seed has median zero subsequent active updates and median direct
carry 1.0. Means are 1.0 except full/control seed59 (0.99637/0.99653). All sampled
histories contain a MAP observation. The new query task is anchored to MAP
availability, so a long chain of post-latest-MAP updates is generally absent.

Thus the simple claim that unrelated events routinely erase the latest MAP
after it arrives is not supported by this sample. The overall expanded-history
gap remains, but this probe cannot diagnose earlier-history retention, the
latest update's use of preceding state, optimization or current-body readout.
The result must not be misreported as all historical memory preserved.

No fitting, accuracy repair, release change or promotion. Repeated source-fit
probes are not a new held-out test. Evidence:
`artifacts/map_native_carry_20260913.json`.

# Rate-sequence subgroup has too few independent evaluation patients

Frozen rate-history checkpoints reproduced all pooled MAEs; no fitting or
weight modification. Input-defined groups: at least two nonconflicting same-label
insulin rate observations before query, versus no such supported sequence.
No future truth used to choose groups. Exploratory after pooled results, not
prespecified promotion or external validation.

Although341source windows support sequences, held-out source patients by seed
are only10/8/3/2/6/1/6 (rows35/33/38/2/18/1/16). All subgroup paired CIs are
unsupported under the existing bootstrap minimum. Do not count unsupported
as seven failed efficacy tests or seven zero-effect observations.

Real input does vary: insulin range nonzero274windows, signed first-last
change nonzero259. Saved tree-node inspection shows own real heads use rate
history in all7fits; full real heads in3/7. This rules out an entirely dead
input path, not weak or heterogeneous predictive signal.

The without-sequence group has257-261patients per evaluation and retains
weak1/7own-vs-original/placebo gain. Known-sequence efficacy is unresolved:
nominal row count overstated the independent subgroup support. Whole-cohort
negative increment remains valid. Do not loosen patient CI support or deploy
subgroup routing from this result.

The two input groups can contain different windows of the same patient;
their patient counts are not additive. Neither group is clinical healthy/sick.
No unique universal root cause or forecast improvement established.

Artifact: `artifacts/potassium_rate_subgroup_20260911.json`.
Runner: `scripts.audit_potassium_treatment_context --inspect-rate-history`.
No external, runtime, interval or promotion changes.

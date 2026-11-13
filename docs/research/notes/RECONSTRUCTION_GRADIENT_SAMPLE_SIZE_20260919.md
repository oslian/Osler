# Gradient Reliability Versus Patient Count

Frozen expanded real checkpoint7; original33targets, sourcefit patients only.
Two disjoint nested pools per group-target; no replacement. Mean body-update-gate
gradients evaluated after8/16/32/64patients per pool. This new nested partition
differs from the previous8-patient probe; only within this experiment is sample
size paired. No parameter or optimizer update occurred.

| Patients/pool | Supported tasks | Same-task cosine median | Negative pairs | Common63 median |
|---|---:|---:|---:|---:|
| 8 | 66 | .0273 | 32 | .0243 |
| 16 | 66 | .0898 | 26 | .0752 |
| 32 | 66 | .2137 | 21 | .2039 |
| 64 | 63 | .3725 | 14 | .3725 |

Common63 fixes the task population across sample sizes. Selected CPK/lipase/
troponin have fewer than128patients and cannot supply two64person pools; they
remain included at smaller sizes and are not silently classified as no-conflict.
Among30targets with both sourcegroups supported at64, none has negative cross-
group cosine in BOTH probes. Smaller sizes produce different conflicting lists.
This is not proof that all cross-organ task pairs are compatible, nor that no
conflict exists in fullencoder or Adam updates. Remaining14negative same-task
pairs demonstrate residual estimation variability. One checkpoint and one split
are insufficient for a universal convergence claim.

Inference supported: small-patient gradient signs are unstable, and larger
patient averaging improves reproducibility on this shared-gate parameter block.
Gradient surgery based on small probes is not presently well justified. A
distinct repair hypothesis is balanced multi-patient gradient accumulation,
compared with the current one-example Adam updates at explicitly matched data
exposure; changed optimizer step count/effective learning rate must be reported
and controlled. Better cosine alone does not establish lower held-out MAE.

Four geometry/partition tests pass. Source-only, no runtime, interval, gate or
promotion change. All39scope retained33trained/6pending.
Evidence: artifacts/native_reconstruction_gradient_sample_size_20260919.json.

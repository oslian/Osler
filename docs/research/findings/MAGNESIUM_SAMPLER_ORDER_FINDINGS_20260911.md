# Fixed partition sampling-order results

Fourteen fits complete: own/whole x7samplingseeds, all on partition7.
Seed7 predictions reproduce original own/whole bit-for-bit. All bindings verify.
Source evaluation medianMAE own .199772,whole .201988; hospital .180267,.183292.
Whole-own significant gain0/7harm0/7both. These are seven optimization repeats
on one partition, NOT seven patient-split generalization tests.

Common-hospital patient-equal prediction variance own .00118172,whole .00135400
(~1.15x), versus mixed-partition previous .00099354,.00278031 (~2.80x).
Fixed-partition whole variability is smaller, own is not. This suggests
partition-dependent factors contribute to prior excess variability, but one
partition cannot identify independent additive variance components. Do not
subtract these as a causal percentage. Sampling and resulting epoch selection
still vary; initialization and fit statistics are fixed.
No stable repair, calibration, runtime or promotion.
Artifact: `artifacts/magnesium_sampler_order_20260911.json`.

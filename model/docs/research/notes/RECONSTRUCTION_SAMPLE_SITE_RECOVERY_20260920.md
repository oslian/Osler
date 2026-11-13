# Sample Site Recovery

Recovered all 45 ambiguous broad-source patients, 1,286 accepted patient-target
examples, including four patients inherited from the original matched source.
Every recovered example matches its historical tensor, event-time vector,
anchor hour, query position, target index and normalized truth exactly.
All 45 patients have accepted target examples spanning multiple hospitals.

The old matched cohort used a shared RNG. Recovery therefore replayed its
complete original patient ordering and candidate draws, not a new per-patient
seed. Newly expanded patients used their original patient-hashed seed. No
alternative anchor was selected after acceptance or comparison. Event shards
and original sample bindings were checked. No existing samples were written.

The private sidecar contains exact target-specific stay/site/anchor provenance
for these 45 people and unique-site mappings for the other 13,479 source people.
It does not reconstruct a new stay label for singleton patients; their site is
unambiguous from original provenance. Patient identity and source role remain
unchanged even when different target samples originate in different hospitals.

## Meaning

The provenance omission is now repaired for broad-source site-aware sampling.
Assigning one representative hospital per patient would have misattributed some
examples for all 45 affected people. This does NOT show their physiological
histories mixed hospitals: reconstruction uses each query's own stay, and exact
historical inputs were preserved. It also does not establish a cause of forecast
or reconstruction error: site had not been used by the physiology encoder.

Next intervention can use verified per-example sites in the training sampler,
not as physiology inputs, while keeping patient isolation. Original selected
source group provenance must also be checked before claiming a complete
two-group sampling repair. No patients are excluded from recovered broad data.

Four focused tests pass. Evidence:
artifacts/native_reconstruction_sample_sites_20260920.json.
Private sidecar remains under ignored data/. No runtime, weights, intervals,
registry, promotion or forecast performance changed.

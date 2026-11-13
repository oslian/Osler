# Private update allocation result

Fourteen32epoch fits complete. Same native glucose roles and own/whole views,
only three target-private modules trained. All nonprivate tensors and all52
other forecast outputs exactly preserved for both arms in all seven seeds.
All105bindings verified. This demonstrates isolation, not accuracy of retained
outputs. No new architecture or runtime change.

| Patient-equal glucose MAE | Source evaluation | Hospital |
|---|---:|---:|
| Original own | 33.9762 | 41.4025 |
| Original whole | 34.8408 | 42.4007 |
| Private-update own | 34.4857 | 42.8374 |
| Private-update whole | 34.9507 | 42.5642 |
| Persistence | 56.8777 | 77.3038 |

Private whole beats matched private own0/7source (harm4/7),2/7hospital.
Versus original whole:0/7source with harm1/7,1/7hospital. Versus original own:
0/7both, harm3/7source and1/7hospital. Private own harms original own4/7hospital.
Both whole-persistence roles7/7 are not proof of improvement over existing
JEPA. Seven split CIs are exploratory and overlap, not independent replications.

The preservation defect can be eliminated mechanically, but doing so does not
repair incremental generalization. The apparently improved whole-own hospital
ranking partly reflects a worse restricted own comparator; original own remains
better. Do not promote from that relative ranking or call shared updates wholly
harmful. Some adaptation beyond private heads appears useful in the tested
contract, while unrestricted primary-only training fails other-target retention.

Frozen new-event projection remains zero; this arm cannot learn numerical rate
inputs. It is not a test that treatment information is useless. The teacher is
fixed with its frozen encoder. Earlier context-frozen pilot allowed shared
future updates and is not identical. Still base world-model, not full persistent
organ architecture. No calibration, untouched external test or causal claim.

Artifact: `artifacts/native_private_update_20260911.json`.
Protocol: `NATIVE_PRIVATE_UPDATE_PROTOCOL_20260911.md`.
Do not repeat complete freezing as an untested universal repair. Remaining
research must preserve useful adaptation, not merely lock old outputs.

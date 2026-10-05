# Native MAP Source Training Transforms

Prepared all seven source-only transform sets for the new MAP-native query
task. Each split fits numerical centers/scales only from its unique eligible
source-fit native history groups. As in the historical runner, only the five
bedside coordinates are rescaled and MAP's target coordinate is unchanged.
Input support uses distinct fit patients with any supported fixed-lag history,
matching the existing policy, not a new held-out support selection.

Seeds 7,19,31,43,59,71,83 have respectively 114445,118113,113695,123359,
117807,116459,120946 eligible fit queries. Fit patient counts are
893,906,922,888,922,890,898. All seven have 46 supported variables. Fit patient
identities are disjoint from tune, evaluation and hospital roles. Treatment
normalization is fitted only on native history groups reached by fit queries.

`transformed_storage` reuses the established rescaling, support masking and
event-feature helpers. Tests compare the compact and dense training batches
exactly for own and full modes, including padding, reordered global query IDs,
event centers/scales and preserving the original input. Three compact batch
tests pass. This is preprocessing preparation/equivalence, not neural efficacy.

The output transform archive is private. Source/artifact bindings are in
`artifacts/map_native_training_transforms_20260912.json`. No new model weights
were trained or saved here; no serving, calibration or promotion change.
The matched seven-seed own/full neural fitting and evaluation remain pending.

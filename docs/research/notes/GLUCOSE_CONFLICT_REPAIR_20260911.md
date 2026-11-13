# Joint conflict contract and controlled accuracy test

Rebuilt current/future labels after both conflict repairs on the original5526
queries. Verified current finite mask exactly matches repaired snapshot mask;
denormalized current agrees, retained future/time/IDs unchanged. Total removed
pairs:glucose92,bicarbonate118,creatinine22,Mg1. New combined private label cache
is distinct from the earlier future-only repair cache and original artifacts.

Controlled raw-history tree test: old inputs+old fit labels versus repaired
inputs+clean fit labels, own/full arms, identical original patient roles and
hyperparameter grid. BOTH selected on clean source tune and evaluated on the
same clean rows. No apparent gain from comparing different test truths.
This changes both training input and training eligibility, not an isolated
input-only intervention. Actual recurrent JEPA was not retrained here.

| Median glucose MAE | Old own | Clean own | Old full | Clean full | Persistence |
|---|---:|---:|---:|---:|---:|
| Source evaluation |35.5755|35.2362|35.3713|35.5690|56.7791|
| Hospital |45.6532|44.9898|45.3844|45.5482|77.1678|

Clean-full versus old-full significant gain/harm0/7both. Clean-own versus
old-own gain0/7source1/7hospital. Clean-full versus clean-own gain1/7source,
0/7hospital. Clean-full versus persistence7/7both. Thus a proven contract
repair does not establish a stable forecast repair in this raw-history test.
Do not revert correct conflict handling to recover old scores. No native-stream
or all-target accuracy claim, new lockbox, intervals, runtime change or promotion.

Artifacts: `artifacts/joint_conflict_contract_20260911.json` and
`artifacts/glucose_conflict_repair_20260911.json`.

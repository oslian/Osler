# Matched raw fixed-lag diagnostic

Same Mg eligible queries/patient roles as native JEPA. Existing fixed-lag
raw current/mask/age features, patient-weighted ridge with source-tune alpha
grid1..10000. Own/full/support (other values removed, observation semantics
retained). No JEPA latent used. This is a fixed-lag summary, not the entire
native event stream; no claim of model-independent information impossibility.

| Median MAE | Own | Full | Support | Persistence |
|---|---:|---:|---:|---:|
| Source evaluation | .233711 | .274704 | .255600 | .277679 |
| Hospital | .204721 | .232304 | .210853 | .238434 |

Full-own gain0/7both, harm7/7source5/7hospital. Full-support gain0/7both,
harm5/7both. Own-persistence gain7/7source3/7hospital. Full-persistence
gain3/7source0/7hospital. Merely bypassing latent with this raw linear readout
does not repair increment. Thus JEPA compression is not required for the
observed failure pattern; nonlinear/event representations may still differ.
This is diagnostic, not a proposed replacement architecture or promotion.
Artifact: `artifacts/magnesium_raw_history_20260911.json`.

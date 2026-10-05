# Independent Model Snapshot / 獨立模型交付

Destination: private repository **oslian/Model**, release `snapshot-20261005`.

This is a source-and-weights snapshot, not a new model promotion.

## Included

- Current WholeBody-JEPA source, including the reviewed uncommitted working-tree snapshot; no source Git history.
- Runtime checkpoint, heads, registries, selectors and calibrators as actual binary files in `models/`.
- Four sanitized epoch-10 coverage checkpoints at the repository root, with their complete native-history input builder and synthetic examples.
- Extractors, training/evaluation/inference scripts, tests, aggregate findings and the root-cause ledger.
- 22 release archives containing 617 neural/parameter containers and 2,891 learned reader files. Some containers hold multiple models; these are file counts, not independent promoted models.
- SHA-256 manifests, restore/verify tooling, bilingual input documentation, units and field definitions.

Latest completed research evidence captured: E44. E45 was ongoing in the source checkout; it is not claimed completed or validated in this snapshot. Historical candidates and negative results remain candidate-only.

## Verified Locally

- Runtime end-to-end: 41 cells, 27 movable and 14 persistence in the synthetic release fixture.
- 188 fold-target coverage bundle parity checks.
- Synthetic CSV-to-native-history-to-estimate inference.
- 19 input/runtime/archive regression tests.
- Pure model tensors preserved by export; private training metadata removed.

These are delivery and implementation checks, not new clinical accuracy or external-validation evidence. Refer to `PUBLICATION_VALIDATION.json` for the tested environment and `RESEARCH_ASSETS.json` for archive hashes.

## Intentionally Not Uploaded

- Licensed patient datasets, raw waveforms, patient CSVs, private histories, prediction/replay rows, credentials and local caches.
- 705 learned reader containers that retain training support vectors or exemplar data; 50 query/replay containers. The exclusions are enumerated in `RESEARCH_WEIGHTS_MANIFEST.json`.
- Optimizer/RNG state, patient queues and training identity sets from exported neural containers. This is not an exact training-resume bundle.
- Unrelated computer folders and separate historical repositories outside the current WholeBody-JEPA checkout.

Removing direct records is not a differential-privacy guarantee. Keep this repository private and respect the original data and code licenses.

## Use

Start with `README.md`, then `RUNTIME_GUIDE.md` for forecasting or `COVERAGE_CANDIDATE_USAGE.md` for masked current-state reconstruction. They are different interfaces. Research archives can be verified and restored using `scripts.restore_research_assets`; original licensed datasets are still required to reproduce patient-level research.

The main research checkout's branch, staging area and remotes were not changed by this publication task. No training or research rerun was authorized by this upload.

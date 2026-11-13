"""Verify surviving reconstruction evidence after temporary source-file loss."""
import json
from pathlib import Path
import torch
from whole_body_jepa.multisource_intervals import sha256


def main():
    source = Path('artifacts/native_reconstruction_expanded_20260913.json')
    original = Path('/private/tmp/rate_anchored_potassium_20260911')
    backup = Path('data/research_recovery/rate_anchored_potassium_20260911')
    report = json.loads(source.read_text())
    missing, changed = [], []
    for name, digest in report['bindings'].items():
        p = Path(name)
        if not p.exists():
            missing.append(name)
        elif sha256(p) != digest:
            changed.append(name)
    if changed:
        raise ValueError('existing source evidence changed')
    cache = 'map_native_anchor_support/expanded_reconstruction_20260913'
    index = torch.load(backup / cache / 'index.pt', weights_only=False)
    verified = 0
    for shard in index['shards']:
        old = Path(shard['path'])
        new = backup / old.relative_to(original)
        if sha256(new) != shard['sha256']:
            raise ValueError('backup shard mismatch')
        verified += 1
    copied = 0
    for path in original.rglob('*'):
        if path.is_file():
            new = backup / path.relative_to(original)
            if not new.is_file() or sha256(path) != sha256(new):
                raise ValueError('backup does not exactly match surviving file')
            copied += 1
    out = Path('artifacts/native_reconstruction_recovery_20260916.json')
    if out.exists():
        raise ValueError('existing recovery evidence')
    out.write_text(json.dumps(dict(complete=True, missing_source_dependencies=missing,
        changed_source_dependencies=changed, verified_original_shards=verified,
        exact_backup_files=copied, seed71_training_started=False,
        runtime_modified=False, source_recovery_complete=False, promotion_allowed=False,
        limitation='Missing upstream sources are not reconstructed by surviving masked samples.',
        bindings={str(p):sha256(p) for p in (source, Path(__file__), backup/cache/'index.pt')}), indent=2)+'\n')
    print('verified backup files', copied, 'shards', verified, 'missing upstream', len(missing))


if __name__ == '__main__':
    main()

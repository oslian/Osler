"""Export completed research weights without patient indices or optimizer state."""
import ast
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

import numpy as np
import torch


def local_sources(initial):
    pending = list(initial)
    found = set()
    while pending:
        path = pending.pop()
        if path in found or not path.is_file():
            continue
        found.add(path)
        for parent in path.parents:
            if parent == Path('.'):
                break
            init = parent/'__init__.py'
            if init.is_file():
                pending.append(init)
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [n.name for n in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    package = list(path.parent.parts)
                    prefix = package[:len(package)-node.level+1]
                    base = '.'.join(prefix + ([node.module] if node.module else []))
                else:
                    base = node.module or ''
                names = [base] + [base+'.'+n.name for n in node.names if base]
            for name in names:
                module = Path(*name.split('.'))
                for candidate in (module.with_suffix('.py'), module/'__init__.py'):
                    if candidate.is_file():
                        pending.append(candidate)
    return sorted(found)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    root = Path('data/research_recovery/additional_development_domains_20260921')
    original = torch.load('models/whole_body_joint_jepa.pt', weights_only=False)
    schema = {k: original[k] for k in ('variables', 'architecture', 'input_dim', 'modules', 'hidden', 'latent')}
    schema['state_dict'] = {k: original['state_dict'][k].cpu() for k in (
        'variable_embedding', 'variable_module_ids', 'variable_module_membership', 'horizon_bucket_values')}
    # Embedding is used only for its shape by fresh_encoder, not as trained weights.
    schema['state_dict']['variable_embedding'] = torch.zeros_like(schema['state_dict']['variable_embedding'])
    scalers = torch.load(root/'scaling_20260922.pt', weights_only=False)['scalers']
    out = args.output
    out.mkdir(exist_ok=True)
    weights = []
    checkpoint_hashes = {}
    for fold in (0, 1, 3, 4):
        source = args.checkpoint_dir/f'fold{fold}_real_epoch10.pt'
        checkpoint = torch.load(source, map_location='cpu', weights_only=False)
        if checkpoint['epoch'] != 10 or checkpoint['fold'] != fold or checkpoint['arm'] != 'real':
            raise ValueError(f'Unexpected checkpoint identity: {source}')
        scaler = {k: torch.as_tensor(np.asarray(scalers[fold][k])).cpu()
                  for k in ('location', 'scale', 'counts')}
        payload = dict(format='masked-current-reconstruction-coverage-v1',
            state_dict={k: v.detach().cpu() for k, v in checkpoint['state_dict'].items()},
            schema=schema, scaler=scaler, variables=list(checkpoint['variables']),
            targets=list(checkpoint['targets']), fold=fold, epoch=checkpoint['epoch'],
            candidate_only=True, causal_claim_allowed=False)
        path = out/f'coverage_fold{fold}_real_epoch10.pt'
        torch.save(payload, path)
        if torch.load(path, weights_only=True)['fold'] != fold:
            raise ValueError(f'Export verification failed: {path}')
        weights.append(path)
        checkpoint_hashes[str(fold)] = hashlib.sha256(source.read_bytes()).hexdigest()
    sources = local_sources([Path('scripts/predict_coverage_csv.py'),
                             Path('scripts/verify_coverage_bundle.py'),
                             Path('tests/test_coverage_input.py'),
                             Path('scripts/train_encoder_crossfit.py'),
                             Path('scripts/encoder_fold_scaling.py'),
                             Path('whole_body_jepa/native_body_reconstruction.py'),
                             Path('whole_body_jepa/native_organ_state_residual.py')])
    archive = out/'coverage_candidate_source.tar.gz'
    extra = [Path('requirements.txt'), Path('COVERAGE_CANDIDATE_USAGE.md'),
             Path('examples/coverage_synthetic_events.csv'),
             Path('scripts/load_coverage_candidate.py'), Path(__file__).resolve().relative_to(Path.cwd()),
             Path('scripts/train_coverage_reconstruction.py'), Path('scripts/coverage_sampling.py'),
             Path('scripts/continue_monitored_reconstruction.py'),
             Path('scripts/fixed_reconstruction_monitor.py'),
             Path('RECONSTRUCTION_MONITORED_CONTINUATION_PROTOCOL_20260929.md'),
             Path('RECONSTRUCTION_CONTINUATION_OUTER_FINDINGS_20260930.md'),
             Path('RECONSTRUCTION_COVERAGE_OUTER_FINDINGS_20260929.md'),
             Path('RECONSTRUCTION_COVERAGE_TRAINING_PROTOCOL_20260925.md')]
    sources = local_sources(sources + [p for p in extra if p.suffix == '.py'])
    with tarfile.open(archive, 'w:gz') as tar:
        for path in sorted(set(sources+extra)):
            tar.add(path, arcname=str(path), recursive=False)
    manifest = dict(candidate_only=True, epoch=10, independently_trained_folds=[0, 1, 3, 4],
        equivalent_reused_fold={'2': 0}, checkpoint_sha256=checkpoint_hashes,
        purpose='Masked current-value reconstruction; not future forecast or runtime promotion.',
        contains_patient_rows=False, contains_optimizer=False,
        input_entrypoint='python -m scripts.predict_coverage_csv',
        input_format='Canonical measurement-event CSV; not unconverted hospital exports',
        synthetic_example='examples/coverage_synthetic_events.csv',
        files={p.name: dict(bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
               for p in (*weights, archive)})
    (out/'MANIFEST.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()

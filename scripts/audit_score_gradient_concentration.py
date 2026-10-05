"""Compare score concentration to saved local gradients without refitting."""
import json
from pathlib import Path
import numpy as np
from whole_body_jepa.multisource_intervals import sha256


def subset_geometry(norms, cosine, mask):
    norms, cosine, mask = np.asarray(norms), np.asarray(cosine), np.asarray(mask, dtype=bool)
    n = len(norms)
    if cosine.shape != (n, n) or mask.shape != (n,) or not np.isfinite(norms).all() or not np.isfinite(cosine).all():
        raise ValueError('Invalid geometry')
    if (norms < 0).any() or not np.allclose(cosine, cosine.T, atol=1e-10):
        raise ValueError('Invalid norms or symmetry')
    gram = norms[:, None] * cosine * norms[None, :]
    dot = gram.mean(axis=1)
    norm_sum = norms.sum()
    dot_sum = dot.sum()
    return dict(norm_sum_share=float(norms[mask].sum() / norm_sum) if norm_sum > 0 else None,
                signed_mean_direction_share=float(dot[mask].sum() / dot_sum) if dot_sum > 1e-20 else None,
                dot_with_average=dot.tolist())


def main():
    src = Path('artifacts/native_reconstruction_trained_state_gradients_20260919.json')
    scores = Path('artifacts/native_reconstruction_selection_concentration_20260919.json')
    out = Path('artifacts/native_reconstruction_score_gradient_concentration_20260919.json')
    if out.exists():
        raise ValueError('Existing audit')
    r, s = [json.loads(p.read_text()) for p in (src, scores)]
    # Validate exactly the source artifacts being interpreted.
    for report in (r, s):
        for path, digest in report['bindings'].items():
            if sha256(Path(path)) != digest:
                raise ValueError(f'Binding changed: {path}')
    per = s['arms']['expanded_sequential']['per_target']
    top = sorted(per, key=lambda t: per[t]['selected_score_share'], reverse=True)[:5]
    mask = [t.split('/')[1] in top for t in r['tasks']]
    results = []
    for probe in r['probes']:
        result = subset_geometry(probe['norm'], probe['cosine'], mask)
        np.testing.assert_allclose(result['dot_with_average'], probe['dot_with_average'], atol=1e-12, rtol=1e-10)
        result['largest_norm_tasks'] = [r['tasks'][i] for i in np.argsort(probe['norm'])[-10:][::-1]]
        results.append(result)
    out.write_text(json.dumps(dict(top_score_targets=top,
        score_share=sum(per[t]['selected_score_share'] for t in top), probes=results,
        parameter_count=r['parameter_count'], patients_per_task_per_probe=r['patients_per_task_per_probe'],
        bindings={str(p): sha256(p) for p in (src, scores, Path(__file__))},
        runtime_modified=False,
        limitation='Two small source probes, body-update gate only. Norm shares and signed projections are not Adam update attribution, full-gradient evidence, or causal failure proof.'), indent=2) + '\n')
    print('Top score targets', top)
    for result in results:
        print({k: v for k, v in result.items() if k != 'dot_with_average'})


if __name__ == '__main__':
    main()

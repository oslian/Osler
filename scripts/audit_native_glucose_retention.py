"""Other-target retention under glucose-only native fine-tuning."""
import json
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scripts.train_availability_encoder import batch
from scripts.audit_event_aware_patient_state import paired
from scripts.audit_recurrent_event_state_loop import patient_equal_mae
from whole_body_jepa.native_event_encoder import native_arrays, NativeEventFilter
from whole_body_jepa.history_event_input import prepare_events, append_events, expand_history_projection
from whole_body_jepa.availability_inputs import restrict_input_support
from whole_body_jepa.runtime import load_release
from whole_body_jepa.multisource_intervals import sha256


def all_points(model, arrays, rows, medians, scales):
    model.eval(); points = []
    with torch.inference_mode():
        for start in range(0, len(rows), 256):
            points.append(model(*batch(arrays, rows[start:start+256]))['value'].numpy())
    return np.concatenate(points) * scales + medians


def main():
    root = Path('/private/tmp/rate_anchored_potassium_20260911')
    output = Path('artifacts/native_glucose_retention_20260911_v2.json')
    if output.exists(): raise ValueError('fresh output required')
    ref = json.loads(Path('artifacts/rate_glucose_jepa_20260911.json').read_text())
    for name, digest in ref['binding'].items():
        if sha256(Path(name)) != digest: raise ValueError('input changed')
    def load(name):
        with np.load(root / name) as f: return {k: f[k] for k in f.files}
    base, native, labels = load('body_arrays.npz'), load('native_tokens.npz'), load('multitarget_labels.npz')
    valid = load('labels.npz')['paired'][:, 0]
    q = pd.read_parquet(root / 'queries.parquet').loc[valid].reset_index(drop=True).rename(columns={'uniquepid': 'subject_id'})
    assets = load_release(Path('models')); variables = assets.checkpoint['variables']; glucose = variables.index('glucose')
    eligible = labels['paired'][:, glucose]
    arrays, rows, values = native_arrays(base, native, len(variables), base['presence'].shape[1])
    report = dict(complete=False, training_performed=False, runtime_modified=False, seeds=[], binding=dict(ref['binding']))
    for path in (Path(__file__), Path('models/whole_body_joint_jepa.pt'), Path('whole_body_joint_jepa.py')):
        report['binding'][str(path)] = sha256(path)
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    for entry in ref['seeds']:
        seed = entry['seed']; roles = {a: b[valid] & eligible for a, b in load(f'partition_{seed}.npz').items()}
        checkpoint = torch.load(root / f'glucose_native_real_{seed}.pt', map_location='cpu', weights_only=False)
        events, _, _ = prepare_events(values, np.unique(arrays['history'][rows[roles['fit']]]))
        view = append_events(restrict_input_support(arrays, np.asarray(checkpoint['allowed_variables']), assets.model.variable_module_ids.numpy()), events)
        torch.manual_seed(seed); model = deepcopy(assets.model); expand_history_projection(model, 6)
        nf = NativeEventFilter(model.input_dim, model.belief_filter.hidden); nf.load_state_dict(model.belief_filter.state_dict()); model.belief_filter = nf
        model.value_delta_activation = torch.nn.Identity(); model.belief_input_mode = 'observed_steps'
        initial = all_points(model, view, rows, assets.scaler.medians, assets.scaler.scales)
        before = deepcopy(model.state_dict())
        model.load_state_dict(checkpoint['state_dict'])
        final = all_points(model, view, rows, assets.scaler.medians, assets.scaler.scales)
        np.testing.assert_allclose(final[:, glucose], load(f'glucose_native_predictions_{seed}.npz')['real'], atol=1e-4, rtol=1e-6)
        groups = {'other_value_heads': lambda n: n.startswith('value_heads.') and int(n.split('.')[1]) != glucose,
                  'state_encoder': lambda n: n.startswith('state_encoder.'),
                  'variable_mixer': lambda n: n.startswith('variable_mixer.'),
                  'module_mixer': lambda n: n.startswith('module_mixer.')}
        drift = {}
        for name, include in groups.items():
            keys = [k for k in before if include(k)]
            delta = sum(float((checkpoint['state_dict'][k] - before[k]).double().square().sum()) for k in keys)
            norm = sum(float(before[k].double().square().sum()) for k in keys)
            drift[name] = dict(tensors=len(keys), relative_l2=(delta / norm) ** .5)
        item = dict(seed=seed, roles={}, parameter_drift=drift)
        for role in ('evaluation', 'hospital'):
            result = {}
            for col, variable in enumerate(variables):
                m = roles[role] & labels['paired'][:, col]
                f = q.loc[m].copy(); f['truth'] = labels['future'][m, col]
                patients = int(f.subject_id.nunique())
                if patients < 30:
                    result[variable] = dict(patients=patients, supported=False); continue
                result[variable] = dict(patients=patients, supported=True,
                    initial_mae=patient_equal_mae(f, f.truth, initial[m, col]),
                    final_mae=patient_equal_mae(f, f.truth, final[m, col]),
                    persistence_mae=patient_equal_mae(f, f.truth, labels['current'][m, col]),
                    final_vs_initial=paired(f, final[m, col], initial[m, col], f'retention:{seed}:{role}:{variable}', 2000))
            item['roles'][role] = result
        report['seeds'].append(item)
    report['complete'] = True; output.write_text(json.dumps(report, indent=2) + '\n')
    for role in ('evaluation', 'hospital'):
        print(role)
        for variable in variables:
            items = [s['roles'][role][variable] for s in report['seeds']]
            if not all(i['supported'] for i in items): continue
            gain = sum(i['final_vs_initial']['ci95'][1] < 0 for i in items)
            harm = sum(i['final_vs_initial']['ci95'][0] > 0 for i in items)
            print(variable, gain, harm, *[round(float(np.median([i[k] for i in items])), 4) for k in ('initial_mae','final_mae','persistence_mae')])


if __name__ == '__main__': main()

"""Separate inherited own/whole gap from the effect of native fine-tuning."""
import json
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scripts.train_availability_encoder import predict
from scripts.audit_event_aware_patient_state import paired
from scripts.audit_recurrent_event_state_loop import patient_equal_mae
from whole_body_jepa.native_event_encoder import native_arrays, NativeEventFilter
from whole_body_jepa.history_event_input import prepare_events, append_events, expand_history_projection
from whole_body_jepa.availability_inputs import restrict_input_support
from whole_body_jepa.runtime import load_release
from whole_body_jepa.multisource_intervals import sha256


def main():
    root = Path('/private/tmp/rate_anchored_potassium_20260911')
    output = Path('artifacts/native_glucose_initialization_20260911.json')
    if output.exists(): raise ValueError('fresh output required')
    ref = json.loads(Path('artifacts/rate_glucose_jepa_20260911.json').read_text())
    for name, digest in ref['binding'].items():
        if sha256(Path(name)) != digest: raise ValueError('bound input changed')
    def load(name):
        with np.load(root / name) as f: return {k: f[k] for k in f.files}
    base, native, labels = load('body_arrays.npz'), load('native_tokens.npz'), load('multitarget_labels.npz')
    valid = load('labels.npz')['paired'][:, 0]
    q = pd.read_parquet(root / 'queries.parquet').loc[valid].reset_index(drop=True).rename(columns={'uniquepid': 'subject_id'})
    assets = load_release(Path('models')); col = assets.checkpoint['variables'].index('glucose')
    median, scale = assets.scaler.medians[col], assets.scaler.scales[col]
    eligible = labels['paired'][:, col]; q['truth'] = labels['future'][:, col]
    arrays, rows, values = native_arrays(base, native, len(assets.checkpoint['variables']), base['presence'].shape[1])
    report = dict(complete=False, training_performed=False, runtime_modified=False, seeds=[], binding=dict(ref['binding']))
    report['binding'][str(Path(__file__))] = sha256(Path(__file__))
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    for entry in ref['seeds']:
        seed = entry['seed']; roles = {a: b[valid] & eligible for a, b in load(f'partition_{seed}.npz').items()}
        events, _, _ = prepare_events(values, np.unique(arrays['history'][rows[roles['fit']]]))
        old = load(f'glucose_native_predictions_{seed}.npz'); points = dict(own_final=old['own'], whole_final=old['real'])
        for arm, original in [('own', 'own'), ('whole', 'real')]:
            checkpoint = torch.load(root / f'glucose_native_{original}_{seed}.pt', map_location='cpu', weights_only=False)
            view = restrict_input_support(arrays, np.asarray(checkpoint['allowed_variables']), assets.model.variable_module_ids.numpy())
            e = events.copy()
            if arm == 'own': e[:] = 0
            view = append_events(view, e); torch.manual_seed(seed)
            model = deepcopy(assets.model); expand_history_projection(model, 6)
            nf = NativeEventFilter(model.input_dim, model.belief_filter.hidden); nf.load_state_dict(model.belief_filter.state_dict()); model.belief_filter = nf
            model.value_delta_activation = torch.nn.Identity(); model.belief_input_mode = 'observed_steps'
            points[arm + '_initial'] = predict(model, view, rows, col, median, scale)
        item = dict(seed=seed, roles={})
        for role, m in roles.items():
            f = q.loc[m]
            comparisons = [('whole_initial', 'own_initial'), ('whole_final', 'whole_initial'), ('own_final', 'own_initial'), ('whole_final', 'own_final')]
            item['roles'][role] = dict(mae={a: patient_equal_mae(f, f.truth, p[m]) for a, p in points.items()}, comparisons={f'{a}_vs_{b}': paired(f, points[a][m], points[b][m], f'initial:{seed}:{role}:{a}:{b}', 2000) for a, b in comparisons})
        report['seeds'].append(item)
    report['complete'] = True; output.write_text(json.dumps(report, indent=2) + '\n')
    for role in ('fit', 'tune', 'evaluation', 'hospital'):
        print(role, {a: float(np.median([s['roles'][role]['mae'][a] for s in report['seeds']])) for a in points}, flush=True)


if __name__ == '__main__': main()

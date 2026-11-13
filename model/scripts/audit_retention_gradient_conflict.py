"""Source-fit local primary/retention gradient diagnostic; no weight updates."""
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from scripts.train_native_glucose_retention import retention_loss
from scripts.train_availability_encoder import batch, configure, teacher_target
from whole_body_jepa.native_event_encoder import native_arrays, NativeEventFilter
from whole_body_jepa.history_event_input import prepare_events, append_events, expand_history_projection
from whole_body_jepa.availability_inputs import restrict_input_support, source_patient_support
from whole_body_jepa.runtime import load_release
from whole_body_jepa.multisource_intervals import sha256
from whole_body_joint_jepa import _masked_latent_prediction_loss


def geometry(a, b):
    aa = sum(float(x.double().square().sum()) for x in a if x is not None)
    bb = sum(float(x.double().square().sum()) for x in b if x is not None)
    dot = sum(float(x.double().mul(y.double()).sum()) for x, y in zip(a, b)
              if x is not None and y is not None)
    return dict(primary_norm=aa**.5, retention_norm=bb**.5,
                cosine=dot/(aa*bb)**.5 if aa*bb > 0 else None,
                retention_to_primary=(bb/aa)**.5 if aa > 0 else None)


def main():
    root = Path('/private/tmp/rate_anchored_potassium_20260911')
    output = Path('artifacts/retention_gradient_conflict_20260911.json')
    if output.exists(): raise ValueError('fresh output required')
    ref = json.loads(Path('artifacts/native_selected_retention_20260911.json').read_text())
    for name, digest in ref['binding'].items():
        if sha256(Path(name)) != digest: raise ValueError('changed input: '+name)
    def load(name):
        with np.load(root/name) as f: return {k:f[k] for k in f.files}
    base, native, labels = load('body_arrays.npz'), load('native_tokens.npz'), load('multitarget_labels.npz')
    valid = load('labels.npz')['paired'][:, 0]
    q = pd.read_parquet(root/'queries.parquet').loc[valid].reset_index(drop=True)
    assets = load_release(Path('models')); variables = assets.checkpoint['variables']; col = variables.index('glucose')
    arrays, rows, values = native_arrays(base, native, len(variables), base['presence'].shape[1])
    truth = (labels['future'][:, col]-assets.scaler.medians[col])/assets.scaler.scales[col]
    report = dict(runtime_modified=False, training_performed=False, causal_claim_allowed=False,
                  role='source_fit_only', binding=ref['binding'].copy(), probes=[])
    report['binding'][str(Path(__file__))] = sha256(Path(__file__))
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    for entry in ref['seeds']:
        seed = entry['seed']; fit = load(f'partition_{seed}.npz')['fit'][valid] & labels['paired'][:, col]
        pools = [np.flatnonzero(fit & q.uniquepid.eq(s).to_numpy()) for s in sorted(set(q.loc[fit, 'uniquepid']))]
        rng = np.random.default_rng(seed+20260911)
        batches = [np.array([rng.choice(pools[i]) for i in rng.integers(len(pools), size=32)]) for _ in range(3)]
        initial = load(f'glucose_retention_predictions_{seed}.npz')['initial']
        target = torch.tensor((initial-assets.scaler.medians)/assets.scaler.scales, dtype=torch.float32)
        support = source_patient_support(base, base['query_rows'], q.uniquepid, fit)>=30
        support[col] = False
        for arm, prefix in [('uniform', 'glucose_retention'), ('selected', 'glucose_selected_retention')]:
            path = root/f'{prefix}_{seed}.pt'; report['binding'][str(path)] = sha256(path)
            cp = torch.load(path, map_location='cpu', weights_only=False)
            events, center, scale = prepare_events(values, np.unique(arrays['history'][rows[fit]]))
            np.testing.assert_allclose(center, cp['event_center']); np.testing.assert_allclose(scale, cp['event_scale'])
            view = append_events(restrict_input_support(arrays, np.asarray(cp['allowed_variables']), assets.model.variable_module_ids.numpy()), events)
            model = deepcopy(assets.model); expand_history_projection(model, 6)
            model.belief_filter = NativeEventFilter(model.input_dim, model.belief_filter.hidden)
            model.value_delta_activation = torch.nn.Identity(); model.belief_input_mode = 'observed_steps'
            model.load_state_dict(cp['state_dict']); configure(model, True)
            mask = support.copy()
            if arm == 'selected': mask &= np.isin(variables, entry['retained_variables'])
            masks = torch.tensor((view['mask'][rows]>0) & mask[None, :])
            named = [(n,p) for n,p in model.named_parameters() if p.requires_grad and
                     n.split('.')[0] in ('belief_filter','variable_mixer','module_mixer','state_encoder')]
            for bi, chosen in enumerate(batches):
                inputs = batch(view, rows[chosen]); result = model(*inputs)
                y = torch.tensor(truth[chosen], dtype=torch.float32)
                teacher = teacher_target(model, inputs[1], y, col)
                primary = F.smooth_l1_loss(result['value'][:,col], y)+_masked_latent_prediction_loss(result['predicted_target_latent'][:,col], teacher, torch.ones(len(y),dtype=torch.bool))
                retention = retention_loss(result['value'], target[chosen], masks[chosen])
                a = torch.autograd.grad(primary, [p for _,p in named], retain_graph=True, allow_unused=True)
                b = torch.autograd.grad(retention, [p for _,p in named], allow_unused=True)
                groups = {'all':geometry(a,b)}
                for group in sorted(set(n.split('.')[0] for n,_ in named)):
                    ids = [i for i,(n,_) in enumerate(named) if n.startswith(group+'.')]
                    groups[group] = geometry([a[i] for i in ids], [b[i] for i in ids])
                report['probes'].append(dict(seed=seed,arm=arm,batch=bi,primary_loss=float(primary.detach()),retention_loss=float(retention.detach()),groups=groups))
        print('complete', seed, flush=True)
    output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__': main()

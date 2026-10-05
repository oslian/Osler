"""All saved epochs on expanded source-only selection patients."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import torch
from scripts.reconstruction_source_store import SourceStore, source_shards
from scripts.train_reconstruction_replication_v2 import input_tokens
from whole_body_jepa.history_event_input import expand_history_projection
from whole_body_jepa.native_body_reconstruction import NativeBodyReconstruction
from whole_body_jepa.native_organ_state_residual import NativeOrganStateResidual
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.runtime import load_release


def summarize(rows, targets):
    result = {}
    for scope in ('original', 'additional', 'all'):
        chosen = [r for r in rows if scope=='all' or r['scope']==scope]
        per = {}
        for t in targets:
            rr = [r for r in chosen if r['target']==t]
            if not rr:
                raise ValueError('Missing target support')
            per[t] = dict(patients=len(rr), mae=float(np.mean([abs(r['prediction']-r['truth']) for r in rr])))
        result[scope] = dict(targets=per, macro=float(np.mean([x['mae'] for x in per.values()])))
    return result


def main():
    root = Path('data/research_recovery/broad_source_20260917')
    out = Path('artifacts/native_reconstruction_expanded_selection_20260919.json')
    if out.exists():
        raise ValueError('Completed replay exists')
    tp = Path('artifacts/native_reconstruction_source_exposure_pair_20260918.json')
    scopep = Path('artifacts/native_reconstruction_complete_source_scope_20260919.json')
    reports = [json.loads(p.read_text()) for p in (tp, scopep)]
    for r in reports:
        for p, h in r['bindings'].items():
            if sha256(Path(p)) != h:
                raise ValueError(f'Binding changed: {p}')
    ip, ep = root/'joint_source_manifest.pt', root/'full_samples/index.pt'
    sp = Path('data/research_recovery/reconstruction_nested_selection_20260917.pt')
    joint, full, split = [torch.load(p, weights_only=False) for p in (ip, ep, sp)]
    shards = source_shards(joint, full, 'expanded', split['roles'])
    original = {s['patient'] for s in shards if s['group']=='broad' and s['role']=='encoder_tune'}
    forbidden = {s['patient'] for s in shards if s['role']=='fit' or s['group']=='selected'}
    tune = [s for s in full['shards'] if s['role']=='encoder_tune']
    ids = {s['patient'] for s in tune}
    if len(ids)!=len(tune) or ids & forbidden or not original <= ids:
        raise ValueError('Source roles not isolated')
    variables, targets = joint['variables'], joint['targets']
    columns = [variables.index(t) for t in targets]
    store = SourceStore(tune, columns)
    cp = root/'source_exposure_pair/expanded/real.pt'
    ck = torch.load(cp, weights_only=False)
    if ck['bindings'] != reports[0]['bindings'] or ck['epoch'] != 8:
        raise ValueError('Checkpoint mismatch')
    epochpaths = [root/f'source_exposure_pair/expanded/real_epoch{i}.pt' for i in range(1,9)]
    paths = [Path(__file__), tp, scopep, ip, ep, sp, cp, *epochpaths,
             Path('RECONSTRUCTION_EXPANDED_SELECTION_PROTOCOL_20260919.md'),
             Path('scripts/reconstruction_source_store.py')]
    bindings = {str(p): sha256(p) for p in paths}
    private = root/'expanded_selection_replay'; private.mkdir(exist_ok=True)
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    assets = deepcopy(load_release(Path('models')))
    expand_history_projection(assets.model, 6); assets.model.belief_input_mode='observed_steps'
    model = NativeBodyReconstruction(NativeOrganStateResidual(assets.model, variables, ('map',))).eval()
    allscores = {}
    for epoch in range(9):
        saved = private/f'epoch{epoch}.pt'
        expected = ck['initial_scores'] if epoch==0 else ck['log'][epoch-1]['group_target_scores']
        if saved.exists():
            record = torch.load(saved, weights_only=False)
            if record['bindings'] != bindings:
                raise ValueError('Resume bindings changed')
            rows = record['rows']
        else:
            if epoch:
                state = torch.load(epochpaths[epoch-1], weights_only=False)
                if state['epoch']!=epoch or state['bindings']!=ck['bindings']:
                    raise ValueError('Epoch mismatch')
                model.load_state_dict(state['state_dict'])
            rows = []
            with torch.inference_mode():
                for key, truth in store.labels.items():
                    if epoch:
                        e=store[key]
                        prediction=float(model(input_tokens(e,'real',len(variables)), e['event_hours'],e['anchor_hour'],variables[key[1]])[0])
                    else:
                        prediction=ck['medians'][key[1]]
                    rows.append(dict(patient=key[0],target=variables[key[1]],truth=truth,
                                     prediction=prediction,scope='original' if key[0] in original else 'additional'))
            temp=saved.with_suffix('.tmp'); torch.save(dict(rows=rows,bindings=bindings),temp);temp.replace(saved)
        if len(rows)!=len(store.labels) or {(r['patient'], variables.index(r['target'])) for r in rows} != set(store.labels):
            raise ValueError('Incomplete or duplicate replay')
        scores=summarize(rows,targets)
        for t in targets:
            if not np.isclose(scores['original']['targets'][t]['mae'],expected['broad'][t],atol=1e-7,rtol=0):
                raise ValueError(f'Original source score mismatch: {epoch}/{t}')
        scores['equal_group_macro']=(scores['all']['macro']+np.mean(list(expected['selected'].values())))/2
        allscores[epoch]=scores
        print('Epoch',epoch,{k:scores[k]['macro'] for k in ('original','additional','all')},'equalgroup',scores['equal_group_macro'],flush=True)
    out.write_text(json.dumps(dict(epochs=allscores,bindings=bindings,
        private_predictions={str(p):sha256(p) for p in private.glob('epoch*.pt')},
        original_patients=len(original),additional_patients=len(ids-original),
        trained_targets=targets, runtime_modified=False,promotion_allowed=False,
        limitation='Source-only epoch replay; selection size and composition change together. Not external validation.'),indent=2)+'\n')


if __name__=='__main__':
    main()

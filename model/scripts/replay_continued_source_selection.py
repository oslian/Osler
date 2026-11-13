"""Extend the frozen source-selection replay to continuation epochs9-19."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import torch
from scripts.replay_expanded_source_selection import summarize
from scripts.reconstruction_source_store import SourceStore, source_shards
from scripts.train_reconstruction_replication_v2 import input_tokens
from whole_body_jepa.history_event_input import expand_history_projection
from whole_body_jepa.native_body_reconstruction import NativeBodyReconstruction
from whole_body_jepa.native_organ_state_residual import NativeOrganStateResidual
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.runtime import load_release


def main():
    root=Path('data/research_recovery/broad_source_20260917')
    tp=Path('artifacts/native_reconstruction_coverage_continuation_20260919.json')
    previous=Path('artifacts/native_reconstruction_expanded_selection_20260919.json')
    out=Path('artifacts/native_reconstruction_continued_selection_20260920.json')
    if out.exists(): raise ValueError('Completed replay exists')
    training,old=[json.loads(p.read_text()) for p in (tp,previous)]
    for r in (training,old):
        for p,h in r['bindings'].items():
            if sha256(Path(p))!=h: raise ValueError(f'Changed binding: {p}')
    ip,ep=root/'joint_source_manifest.pt',root/'full_samples/index.pt'
    sp=Path('data/research_recovery/reconstruction_nested_selection_20260917.pt')
    joint,full,split=[torch.load(p,weights_only=False) for p in (ip,ep,sp)]
    shards=source_shards(joint,full,'expanded',split['roles'])
    original={s['patient'] for s in shards if s['group']=='broad' and s['role']=='encoder_tune'}
    forbidden={s['patient'] for s in shards if s['role']=='fit' or s['group']=='selected'}
    tune=[s for s in full['shards'] if s['role']=='encoder_tune'];ids={s['patient'] for s in tune}
    if len(ids)!=len(tune) or ids & forbidden or not original<=ids: raise ValueError('Role overlap')
    variables,targets=joint['variables'],joint['targets']
    if targets!=old['trained_targets']: raise ValueError('Target scope changed')
    store=SourceStore(tune,[variables.index(t) for t in targets])
    cp=root/'coverage_continuation/real.pt';ck=torch.load(cp,weights_only=False)
    if ck['epoch']!=19 or ck['bindings']!=training['bindings']: raise ValueError('Checkpoint mismatch')
    epochpaths={i:root/f'coverage_continuation/real_epoch{i}.pt' for i in range(9,20)}
    paths=[Path(__file__),tp,previous,ip,ep,sp,cp,*epochpaths.values(),
           Path('scripts/replay_expanded_source_selection.py'),Path('scripts/reconstruction_source_store.py'),
           Path('RECONSTRUCTION_CONTINUED_SELECTION_PROTOCOL_20260920.md')]
    bindings={str(p):sha256(p) for p in paths}
    private=root/'continued_selection_replay';private.mkdir(exist_ok=True)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    assets=deepcopy(load_release(Path('models')));expand_history_projection(assets.model,6)
    assets.model.belief_input_mode='observed_steps'
    model=NativeBodyReconstruction(NativeOrganStateResidual(assets.model,variables,('map',))).eval()
    scores=dict(old['epochs'])
    for epoch,p in epochpaths.items():
        saved=private/f'epoch{epoch}.pt'
        if saved.exists():
            rec=torch.load(saved,weights_only=False)
            if rec['bindings']!=bindings: raise ValueError('Resume mismatch')
            rows=rec['rows']
        else:
            state=torch.load(p,weights_only=False)
            if state['epoch']!=epoch or state['bindings']!=ck['bindings']: raise ValueError('Epoch mismatch')
            model.load_state_dict(state['state_dict']);rows=[]
            with torch.inference_mode():
                for key,truth in store.labels.items():
                    e=store[key]
                    pred=float(model(input_tokens(e,'real',len(variables)),e['event_hours'],e['anchor_hour'],variables[key[1]])[0])
                    rows.append(dict(patient=key[0],target=variables[key[1]],truth=truth,prediction=pred,
                                     scope='original' if key[0] in original else 'additional'))
            temp=saved.with_suffix('.tmp');torch.save(dict(rows=rows,bindings=bindings),temp);temp.replace(saved)
        if len(rows)!=len(store.labels) or {(r['patient'],variables.index(r['target'])) for r in rows}!=set(store.labels):
            raise ValueError('Incomplete pairs')
        result=summarize(rows,targets);expected=ck['log'][epoch-1]['group_target_scores']
        for t in targets:
            if not np.isclose(result['original']['targets'][t]['mae'],expected['broad'][t],atol=1e-7,rtol=0):
                raise ValueError('Original score mismatch')
        result['equal_group_macro']=(result['all']['macro']+np.mean(list(expected['selected'].values())))/2
        scores[str(epoch)]=result
        print('Epoch',epoch,'full broad',result['all']['macro'],'equalgroup',result['equal_group_macro'],flush=True)
    out.write_text(json.dumps(dict(epochs=scores,bindings=bindings,
        private_predictions={str(p):sha256(p) for p in private.glob('epoch*.pt')},
        original_patients=len(original),additional_patients=len(ids-original),trained_targets=targets,
        runtime_modified=False,promotion_allowed=False,
        limitation='Source-only replay; reused source selection, not independent validation or promotion.'),indent=2)+'\n')


if __name__=='__main__':main()

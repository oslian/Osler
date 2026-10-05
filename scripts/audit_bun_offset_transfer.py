"""Source-fit intercept diagnostic; not a deployed BUN calibrator."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import torch
from scripts.audit_bun_dynamics_branches import disabled_dynamics
from scripts.audit_continued_selection_pairs import interval
from scripts.reconstruction_source_store import SourceStore
from whole_body_jepa.history_event_input import expand_history_projection
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.native_body_reconstruction import NativeBodyReconstruction
from whole_body_jepa.native_organ_state_residual import NativeOrganStateResidual
from whole_body_jepa.runtime import load_release


def offsets(rows):
    if not rows: raise ValueError('Empty fitting group')
    result = {}
    for arm in ('full', 'no_body'):
        residual = np.array([r['truth']-r['predictions'][arm] for r in rows])
        if not np.isfinite(residual).all(): raise ValueError('Nonfinite residual')
        result[arm] = float(np.median(residual))
    return result


def main():
    root=Path('data/research_recovery/broad_source_20260917')
    fp=root/'bun_dynamics_fit_20260920.pt'
    fr=Path('artifacts/native_reconstruction_bun_dynamics_fit_20260920.json')
    bp=root/'bun_dynamics_branches_20260920.pt'
    br=Path('artifacts/native_reconstruction_bun_dynamics_branches_20260920.json')
    cp=root/'bounded_site_training/bounded.pt'
    mp=root/'site_balance_manifest_20260920.pt'
    jp=root/'joint_source_manifest.pt'
    out=Path('artifacts/native_reconstruction_bun_offset_transfer_20260920.json')
    private=root/'bun_offset_transfer_20260920.pt'
    if out.exists() or private.exists(): raise ValueError('Existing diagnostic')
    for report,payload in ((fr,fp),(br,bp)):
        r=json.loads(report.read_text())
        if sha256(payload)!=r['private_sha256']: raise ValueError('Private hash mismatch')
        for p,h in r['bindings'].items():
            if sha256(Path(p))!=h: raise ValueError('Dependency changed')
    fit,previous,ck,manifest,joint=[torch.load(p,weights_only=False) for p in (fp,bp,cp,mp,jp)]
    correction={g:offsets(rows) for g,rows in fit.items()}
    train_ids={r['patient'] for rows in fit.values() for r in rows}
    tune_ids={r['patient'] for rows in previous.values() for r in rows}
    if train_ids & tune_ids: raise ValueError('Fitting/selection overlap')
    variables=joint['variables']; col=variables.index('bun')
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    assets=deepcopy(load_release(Path('models')))
    expand_history_projection(assets.model,6);assets.model.belief_input_mode='observed_steps'
    model=NativeBodyReconstruction(NativeOrganStateResidual(assets.model,variables,('map',))).eval()
    model.load_state_dict(ck['best_state_dict'])
    raw,result={},{}
    with torch.inference_mode():
        for group,old in previous.items():
            store=SourceStore([s for s in manifest['shards'] if s['group']==group and s['role']=='encoder_tune'],[col])
            if set(store.labels)!={(r['patient'],col) for r in old}: raise ValueError('Pair mismatch')
            rows=[]
            for r in old:
                e=store[r['patient'],col];predictions={}
                for arm in ('full','no_body'):
                    with disabled_dynamics(model.trajectory.layer,body=arm=='no_body'):
                        predictions[arm]=float(model(e['tokens'],e['event_hours'],e['anchor_hour'],'bun')[0])
                    if not np.isclose(abs(predictions[arm]-e['truth_normalized']),r['errors'][arm],atol=1e-7,rtol=0):
                        raise ValueError('Baseline mismatch')
                rows.append(dict(patient=r['patient'],truth=float(e['truth_normalized']),predictions=predictions))
            raw[group]=rows; truth=np.array([r['truth'] for r in rows])
            pred={a:np.array([r['predictions'][a] for r in rows]) for a in ('full','no_body')}
            base=np.abs(pred['full']-truth)
            tests={}
            for source,shift in correction.items():
                errors={a:np.abs(pred[a]+shift[a]-truth) for a in pred}
                tests[source]=dict(mae={a:float(v.mean()) for a,v in errors.items()},
                    full_corrected_vs_original=interval(errors['full']-base),
                    corrected_no_body_vs_corrected_full=interval(errors['no_body']-errors['full']))
            result[group]=dict(patients=len(rows),signed_mean_error={a:float((v-truth).mean()) for a,v in pred.items()},
                body_contribution_mean=float((pred['full']-pred['no_body']).mean()),
                body_contribution_std=float((pred['full']-pred['no_body']).std()),
                by_offset_training_group=tests)
            print(group,result[group],flush=True)
    torch.save(raw,private)
    out.write_text(json.dumps(dict(offsets=correction,groups=result,
        bindings={str(p):sha256(p) for p in (Path(__file__),fp,fr,bp,br,cp,mp,jp)},private_sha256=sha256(private),
        runtime_modified=False,promotion_allowed=False,
        limitation='Offsets fitted only on source-fit residuals. Selection patients disjoint but previously reused. Group-specific and cross-group comparisons descriptive, no winner selected or runtime correction.'),indent=2)+'\n')


if __name__=='__main__':main()

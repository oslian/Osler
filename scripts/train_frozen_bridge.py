"""E34: train bridge only from the identical E29 epoch10 initialization."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from scripts.audit_trained_bridge import check_bindings, same_source_examples
from scripts.diagnose_organ_assimilation import require_verification
from scripts.organ_assimilation import CHANNELS
from scripts.probe_assimilation_transfer import DEST as TRANSFER, ROOT
from scripts.score_additional_reconstruction import paired_summary
from scripts.train_content_retention import replay
from scripts.train_organ_assimilation import AssimilatedReconstruction, DEST as JOINT, SEED, validate_identities
from scripts.train_organ_binding import prepare, select_arm, space_required, write_json
from scripts.verify_organ_assimilation import direct_prediction
from whole_body_jepa.multisource_intervals import sha256

DEST = ROOT/'frozen_bridge_20261005'
PROTOCOL = Path('RECONSTRUCTION_FROZEN_BRIDGE_PROTOCOL_20261005.md')
OUTPUT = Path('artifacts/native_frozen_bridge_assessment_20261005.json')
ROLES = ('fit', 'selection', 'same_source')
CONTROLS = ('immediate', 'fixed_delayed', 'fixed_body', 'bound_start', 'unbridged', 'frozen', 'raw')
NEW = tuple('fixed_'+c for c in CHANNELS)


def assert_frozen(candidate, original):
    for key, value in original.state_dict().items():
        torch.testing.assert_close(candidate.model.state_dict()[key], value, atol=0, rtol=0)


def fit_bridge(original, examples, bound, channel):
    if not examples or len(examples) != len(bound):
        raise ValueError('Aligned fitting contexts required')
    torch.manual_seed(SEED)
    candidate = AssimilatedReconstruction(original, channel).eval()
    candidate.requires_grad_(False)
    candidate.bridge.projection.weight.requires_grad_(True)
    parameters = [candidate.bridge.projection.weight]
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=.01)
    rng = np.random.default_rng(SEED)
    history = []
    for epoch in range(2):
        losses, gradients = [], []
        for i in rng.permutation(len(examples)):
            e = select_arm(examples[i], bound[i], 'binding')
            prediction = candidate(e)
            loss = F.smooth_l1_loss(prediction, prediction.new_tensor(e['truth']))
            if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach())); gradients.append(float(norm))
        assert_frozen(candidate, original)
        history.append(dict(epoch=epoch+1, steps=len(losses), loss=float(np.mean(losses)),
            maximum_gradient=max(gradients), nonzero_gradients=sum(x > 0 for x in gradients)))
        print('E34 fit', channel, history[-1], flush=True)
    return candidate, history


def describe(frame, inventory):
    validate_identities(frame)
    names = [c for c in (*NEW, 'bound_start', 'immediate', 'delayed', 'body', 'unbridged', 'frozen', 'raw') if c in frame]
    if (set(NEW) - set(frame) or 'bound_start' not in frame or 'unbridged' not in frame
            or not np.isfinite(frame[['truth', *names]].to_numpy()).all()
            or not set(frame.variable) <= set(inventory)):
        raise ValueError('Invalid predictions/inventory')
    results = {}
    for v in inventory:
        f = frame[frame.variable == v]
        if not len(f):
            results[v] = dict(status='unsupported'); continue
        losses = f[names].subtract(f.truth, axis=0).abs().groupby(f.patient).mean()
        link = f.drop_duplicates('patient').set_index('patient').component
        groups = losses.groupby(link.reindex(losses.index)).mean()
        results[v] = dict(status='scored', patients=len(losses), components=len(groups),
            occurrences=len(f), normalized_mae=dict(patient=losses.mean().to_dict(), component=groups.mean().to_dict()))
    return results


def assess(frame, seed, repetitions=5000):
    if frame.variable.nunique() != 1: raise ValueError('Single target required')
    result = describe(frame, list(frame.variable.unique()))[frame.variable.iloc[0]]
    if not set(CONTROLS) <= set(frame): raise ValueError('Missing comparison')
    names = ['fixed_immediate', *CONTROLS]
    losses = frame[names].subtract(frame.truth, axis=0).abs().groupby(frame.patient).mean()
    links = frame.drop_duplicates('patient').set_index('patient').component
    groups = losses.groupby(links.reindex(losses.index)).mean()
    comparisons = {}
    for i, c in enumerate(CONTROLS):
        stats = dict(patient=paired_summary(losses.fixed_immediate-losses[c], seed+i, repetitions),
            component=paired_summary(groups.fixed_immediate-groups[c], seed+100+i, repetitions))
        passed = len(losses) >= 30 and len(groups) >= 3 and all(
            s['ci95'] is not None and s['ci95'][1] < 0 and s['p_less'] is not None
            and s['p_less'] < .05/(53*len(CONTROLS)) for s in stats.values())
        comparisons[c] = dict(paired=stats, screen_pass=bool(passed))
    result['comparisons'] = comparisons
    return result


def run_fold(fold, parents):
    variables, examples, bound, original, parent, extraction, _ = prepare(fold)
    rp = Path(f'artifacts/native_organ_assimilation_fold{fold}_20261004.json')
    tp = Path(f'artifacts/native_assimilation_transfer_fold{fold}_20261005.json')
    for label, path in (('joint', rp), ('transfer', tp)):
        if sha256(path) != parents[label]['bindings'][str(path)]: raise ValueError('Changed parent report')
    joint, transfer = json.loads(rp.read_text()), json.loads(tp.read_text())
    if joint['contract']['parent'] != parent or joint['contract']['extraction'] != extraction:
        raise ValueError('Different E29 initialization/input contract')
    check_bindings(joint['contract']['bindings']); check_bindings(transfer['bindings'])
    frames, paths = {}, [rp, tp, PROTOCOL, Path(__file__), Path('tests/test_frozen_bridge.py'),
        Path('scripts/audit_trained_bridge.py'), Path('scripts/verify_organ_assimilation.py'),
        Path('scripts/train_organ_assimilation.py'), Path('scripts/organ_assimilation.py')]
    for role in ROLES:
        path = TRANSFER/f'fold{fold}'/'predictions.parquet' if role == 'same_source' else JOINT/f'fold{fold}'/f'{role}_predictions.parquet'
        expected_hash = transfer['prediction_sha256'] if role == 'same_source' else joint['prediction_hashes'][role]
        if sha256(path) != expected_hash: raise ValueError('Changed parent predictions')
        frames[role] = pd.read_parquet(path); validate_identities(frames[role]); paths.append(path)
        if len(frames[role]) != (2048 if role == 'fit' else 1024): raise ValueError('Role size changed')
    if any(set(frames[a].patient) & set(frames[b].patient)
           for a,b in (('fit','selection'),('fit','same_source'),('selection','same_source'))):
        raise ValueError('Patient role overlap')
    dest = DEST/f'fold{fold}'; dest.mkdir(parents=True, exist_ok=True)
    contract = dict(fold=fold, parent=parent, extraction=extraction, seed=SEED,
        passes=2, lr=1e-4, weight_decay=.01, trainable=['bridge.projection.weight'],
        bindings={str(p): sha256(p) for p in paths})
    cp = dest/'contract.json'
    if cp.exists() and json.loads(cp.read_text()) != contract: raise ValueError('Changed frozen experiment')
    if not cp.exists(): write_json(cp, contract)
    models, fits = {}, {}
    for c in CHANNELS:
        path, mp = dest/f'{c}.pt', dest/f'{c}.json'
        if not mp.exists():
            if path.exists(): raise ValueError('Unverified partial checkpoint exists')
            space_required()
            candidate, history = fit_bridge(original, examples['fit'], bound['fit'], c)
            space_required()
            tmp = path.with_suffix('.tmp'); torch.save(candidate.state_dict(), tmp); tmp.replace(path)
            write_json(mp, dict(sha256=sha256(path), contract_sha256=sha256(cp), channel=c,
                training=history, trainable_parameters=candidate.bridge.projection.weight.numel()))
        info = json.loads(mp.read_text())
        if info['sha256'] != sha256(path) or info['contract_sha256'] != sha256(cp) or info['channel'] != c:
            raise ValueError('Changed fixed checkpoint')
        candidate = AssimilatedReconstruction(original, c).eval()
        candidate.load_state_dict(torch.load(path, weights_only=False), strict=True)
        candidate.requires_grad_(False); assert_frozen(candidate, original)
        models[c], fits[c] = candidate, info
    if len({i['trainable_parameters'] for i in fits.values()}) != 1: raise ValueError('Capacity differs')
    meta_path = Path(f'artifacts/native_frozen_bridge_fold{fold}_20261005.json')
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta['contract'] != contract or meta['fits'] != fits: raise ValueError('Changed completed fold')
        check_bindings(meta['outputs'])
        return {r: pd.read_parquet(dest/f'{r}.parquet') for r in ROLES}, variables, meta_path
    parity = {r: {c: 0. for c in CHANNELS} for r in ROLES}
    with torch.inference_mode():
        for role, frame in frames.items():
            if role == 'same_source': rows = same_source_examples(frame, original, variables, fold)
            else:
                expected = [(e['patient'],e['component'],e['target'],e['truth']) for e in examples[role]]
                if list(frame[['patient','component','context','truth']].itertuples(index=False,name=None)) != expected:
                    raise ValueError('Cached identities/truth differ')
                rows = (select_arm(e,b,'binding') for e,b in zip(examples[role],bound[role]))
            records = []
            for i,e in enumerate(rows):
                item = dict(bound_start=float(original.readout(replay(original.trajectory.layer,e))[0,e['target']]))
                for c, model in models.items():
                    point = float(model(e))
                    check = direct_prediction(model,e,e['organs'])
                    np.testing.assert_allclose(point,check,atol=1e-7,rtol=0)
                    if not np.isfinite([point,check,item['bound_start']]).all(): raise ValueError('Nonfinite prediction')
                    parity[role][c] = max(parity[role][c],abs(point-check))
                    item['fixed_'+c] = point
                records.append(item)
                if (i+1)%512 == 0: print('E34 replay',fold,role,i+1,len(frame),flush=True)
            if len(records) != len(frame): raise ValueError('Incomplete inference')
            for col in records[0]: frame[col] = [r[col] for r in records]
            describe(frame,variables)
    for c, model in models.items():
        assert_frozen(model,original)
        saved = torch.load(dest/f'{c}.pt',weights_only=False)
        for k,v in model.state_dict().items(): torch.testing.assert_close(v,saved[k],atol=0,rtol=0)
    check_bindings(contract['bindings'])
    outputs = {}
    space_required()
    for role,frame in frames.items():
        path = dest/f'{role}.parquet'; tmp = path.with_suffix('.tmp')
        frame.to_parquet(tmp,index=False); tmp.replace(path); outputs[str(path)] = sha256(path)
    write_json(meta_path,dict(contract=contract,fits=fits,outputs=outputs,parity=parity,
        runtime_modified=False,promotion_allowed=False))
    print('E34 fold complete',fold,flush=True)
    return frames,variables,meta_path


def main():
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    if OUTPUT.exists(): raise ValueError('Completed experiment exists')
    parents,bindings = {},{str(p):sha256(p) for p in (PROTOCOL,Path(__file__))}
    for label,stem,date in (('joint','native_organ_assimilation','20261004'),('transfer','native_assimilation_transfer','20261005')):
        suffix = '_assessment' if label == 'joint' else ''
        ap,vp = Path(f'artifacts/{stem}{suffix}_{date}.json'),Path(f'artifacts/{stem}_verification_{date}.json')
        require_verification(ap,json.loads(vp.read_text())); parents[label] = json.loads(ap.read_text())
        bindings.update({str(p):sha256(p) for p in (ap,vp)})
    data = {r:[] for r in ROLES}
    for fold in range(5):
        frames,variables,path = run_fold(fold,parents)
        for r in ROLES: data[r].append(frames[r])
        bindings[str(path)] = sha256(path)
    frames = {r:pd.concat(fs,ignore_index=True) for r,fs in data.items()}
    results = {r:describe(f,variables) for r,f in frames.items()}
    for i,v in enumerate(variables):
        f = frames['selection'][frames['selection'].variable==v]
        if len(f): results['selection'][v] = assess(f,SEED+i*1000)
    scored = {v:r for v,r in results['selection'].items() if r['status']=='scored'}
    screens = {c:[v for v,r in scored.items() if r['comparisons'][c]['screen_pass']] for c in CONTROLS}
    summary = dict(occurrences={r:len(f) for r,f in frames.items()},screens=screens,
        full_screen=sorted(set.intersection(*(set(v) for v in screens.values()))))
    write_json(OUTPUT,dict(results=results,summary=summary,bindings=bindings,
        runtime_modified=False,promotion_allowed=False,independent_summary_verified=False,
        limitation='Reused development; optimization-scope ablation, not forecast or untouched external validation.'))
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__': main()

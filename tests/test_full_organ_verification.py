from copy import deepcopy
import json
import math
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import joblib
import torch

from scripts.score_full_organ_readout import summarize
from scripts import verify_full_organ_readout as verifier
from scripts.train_full_organ_readout import features


class InputWitness:
    """Detect any input permutation, not just equality of constant outputs."""
    def __init__(self, x):
        self.x = x
        self.n_features_in_ = x.shape[1]
        self.n_iter_ = 100

    def get_params(self):
        return dict(verifier.PARAMETERS)

    def predict(self, x):
        np.testing.assert_array_equal(x, self.x)
        return x[:, 0]


class FullOrganVerificationTest(unittest.TestCase):
    @staticmethod
    def predictions():
        rng = np.random.default_rng(22)
        frames = []
        for fold in range(5):
            n = 80-fold*9
            frame = pd.DataFrame(dict(patient=[f'p{i:03}' for i in range(n)],
                component=[f's{i%4}' for i in range(n)], variable='v0', fold=fold,
                truth=rng.normal(size=n)))
            for arm in verifier.ARMS:
                frame[arm] = frame.truth+rng.normal(size=n)
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    def test_independent_statistics_match_scorer_with_unequal_support(self):
        frame = self.predictions()
        variables = [f'v{i}' for i in range(53)]
        actual = verifier.independent_assessment(frame, variables, repetitions=200)
        expected = summarize(frame, variables, repetitions=200)
        self.assertLess(verifier.compare_results(list(actual), list(expected)), 1e-12)
        self.assertEqual(actual[0]['v0']['patients'], 80)
        self.assertEqual(actual[0]['v52'], dict(status='unsupported'))

    def test_patient_then_component_equalization_not_row_weighting(self):
        rows = pd.DataFrame(dict(patient=['a', 'a', 'a', 'b', 'c'],
            component=['s0', 's0', 's0', 's0', 's1'], variable='v0',
            fold=[0, 1, 2, 0, 0], truth=0.))
        for arm in verifier.ARMS:
            rows[arm] = [0., 3., 6., 1., 10.]
        targets, _ = verifier.independent_assessment(rows, [f'v{i}' for i in range(53)], 100)
        self.assertAlmostEqual(targets['v0']['normalized_mae']['patient']['organs'], 14/3)
        self.assertEqual(targets['v0']['normalized_mae']['component']['organs'], 6.)
        self.assertFalse(targets['v0']['comparisons']['organs_minus_body']['screen_pass'])

    def test_large_group_sum_preserves_small_nonnegative_losses(self):
        values = np.r_[1e16, np.ones(20000), 2e16, np.full(10000, 3.)][:, None]
        keys = np.r_[np.repeat('a', 20001), np.repeat('b', 10001)]
        expected = np.array([[math.fsum(values[:20001, 0])/20001],
                             [math.fsum(values[20001:, 0])/10001]])
        unique, means = verifier.group_means(keys, values)
        np.testing.assert_array_equal(unique, ['a', 'b'])
        np.testing.assert_array_equal(means, expected)
        order = np.random.default_rng(7).permutation(len(keys))
        np.testing.assert_array_equal(verifier.group_means(keys[order], values[order])[1], expected)
        naive = np.zeros((2, 1))
        np.add.at(naive, (keys == 'b').astype(int), values)
        self.assertGreater(np.max(abs(naive/np.array([20001, 10001])[:, None]-expected)), .1)

    def test_duplicate_identity_component_change_and_nonfinite_rejected(self):
        frame = self.predictions()
        bad_component = frame.copy(); bad_component.loc[0, 'component'] = 'wrong'
        bad_value = frame.copy(); bad_value.loc[0, 'organs'] = np.inf
        bad_fold = frame.copy(); bad_fold.loc[0, 'fold'] = 5
        for bad in (pd.concat([frame, frame.iloc[[0]]]), bad_component, bad_value, bad_fold):
            with self.assertRaises(ValueError):
                verifier.independent_assessment(bad, [f'v{i}' for i in range(53)], 10)

    def test_roundoff_never_changes_screens_counts_or_inventory(self):
        base = dict(mae=1., screen=False, count=53, ci=[-.2, -.1], unsupported=None)
        rounded = deepcopy(base); rounded['mae'] += 1e-15
        self.assertLess(verifier.compare_results(rounded, base), 1e-12)
        for key, value in [('screen', True), ('count', 52), ('mae', 1.0001),
                           ('mae', np.nan), ('ci', [-.2, .1]), ('unsupported', 'passed')]:
            changed = deepcopy(base); changed[key] = value
            with self.assertRaises(ValueError):
                verifier.compare_results(changed, base)
        with self.assertRaises(ValueError):
            verifier.compare_results(dict(base, unexpected=1), base)

    @staticmethod
    def replay_fixture():
        ids = ['a', 'b', 'c']
        rng = np.random.default_rng(73)
        data = dict(body=rng.normal(size=(3, 64)), raw=rng.normal(size=(3, 265)),
                    organs=rng.normal(size=(3, 448)), truth=np.zeros(3), prediction=np.ones(3))
        projection = np.random.default_rng(20261003).normal(size=(64, 448))/8
        models = {arm: InputWitness(features(data, ids, arm, projection)) for arm in verifier.ARMS[:4]}
        bundle = dict(target='v0', parameters=verifier.PARAMETERS, models=models, projection=projection)
        frame = pd.DataFrame(dict(patient=ids, component=['s0', 's1', 's2'], variable='v0', fold=0,
                                  truth=data['truth'], native=data['prediction']))
        for arm, model in models.items():
            frame[arm] = model.predict(model.x)
        frame['raw'] = data['raw'][:, 0]
        return bundle, InputWitness(data['raw']), frame, ids, data, dict(zip(ids, frame.component))

    def test_replay_independent_input_assembly_all_five_readers(self):
        bundle, raw, frame, ids, data, components = self.replay_fixture()
        self.assertEqual(verifier.replay_selection(bundle, raw, frame, ids, data,
            'v0', 0, components, (14, 32)), 5)
        changed = frame.copy(); changed.loc[0, 'organs'] += .1
        with self.assertRaises(AssertionError):
            verifier.replay_selection(bundle, raw, changed, ids, data, 'v0', 0, components, (14, 32))

    def test_replay_rejects_changed_truth_order_or_projection(self):
        bundle, raw, frame, ids, data, components = self.replay_fixture()
        changed = frame.copy(); changed.loc[0, 'truth'] = .1
        for wrong in (changed, frame.iloc[::-1]):
            with self.assertRaises(AssertionError):
                verifier.replay_selection(bundle, raw, wrong, ids, data, 'v0', 0, components, (14, 32))
        bundle['projection'][0, 0] += .1
        with self.assertRaises(AssertionError):
            verifier.replay_selection(bundle, raw, frame, ids, data, 'v0', 0, components, (14, 32))

    def test_no_assessment_means_no_verification_or_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'verification.json'
            with patch.object(verifier, 'ASSESSMENT', Path(tmp)/'missing.json'), \
                    patch.object(verifier, 'OUTPUT', output), patch.object(verifier, 'verify_full_grid') as grid:
                with self.assertRaisesRegex(ValueError, 'Full E25 assessment required'):
                    verifier.main()
                grid.assert_not_called()
            self.assertFalse(output.exists())

    def test_main_replays_complete_synthetic_inventory_and_saves_aggregate_only(self):
        variables = [f'v{i}' for i in range(53)]
        bundle, raw, frame, ids, data, components = self.replay_fixture()
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            root, dest = Path(tmp)/'raw', Path(tmp)/'organs'
            readers = dest/'readers'
            artifacts = Path(tmp)/'artifacts'; artifacts.mkdir()
            root.mkdir(); dest.mkdir()
            pd.DataFrame(dict(uniquepid=ids, component=[components[p] for p in ids])).to_parquet(root/'provenance.parquet')
            roles = [dict.fromkeys(ids, 'selection') for _ in range(5)]
            torch.save(dict(folds=roles), root/'partitions_20260922.pt')
            bindings, grid, frames = {}, {}, []
            for fold in range(5):
                rd = readers/f'fold{fold}'; rd.mkdir(parents=True)
                raw_dir = root/f'nonlinear_probe_20261001/fold{fold}_probes'; raw_dir.mkdir(parents=True)
                for role in ('fit', 'selection'):
                    grid[fold, role] = dict(expected=47*len(ids), support={v: dict(expected=len(ids)) for v in variables[:47]})
                sqlite3.connect(root/f'nonlinear_probe_20261001/fold{fold}_selection.sqlite').close()
                sqlite3.connect(dest/f'fold{fold}_selection.sqlite').close()
                projection = np.random.default_rng(20261003+fold).normal(size=(64, 448))/8
                models = {arm: InputWitness(features(data, ids, arm, projection)) for arm in verifier.ARMS[:4]}
                targets = {}
                for v in variables:
                    if v not in variables[:47]:
                        targets[v] = dict(status='unsupported', fit_patients=0, selection_patients=0)
                        continue
                    current = frame.copy(); current['variable'] = v; current['fold'] = fold
                    saved = dict(bundle, target=v, models=models, projection=projection)
                    mp, pp, raw_path = rd/f'{v}.joblib', rd/f'{v}.parquet', raw_dir/f'{v}.joblib'
                    joblib.dump(saved, mp); current.to_parquet(pp, index=False)
                    joblib.dump(dict(models=dict(raw_probe=raw)), raw_path)
                    targets[v] = dict(model_sha256=verifier.sha256(mp), prediction_sha256=verifier.sha256(pp),
                                      raw_model_sha256=verifier.sha256(raw_path))
                    bindings.update({str(p): verifier.sha256(p) for p in (mp, pp)})
                    frames.append(current)
                # The bound fold metadata commits to the raw model hash even
                # when the top-level scorer does not list that file directly.
                rp = artifacts/f'native_reconstruction_full_organ_readers_fold{fold}_20261003.json'
                rp.write_text(json.dumps(dict(complete=True, contract=dict(variables=variables), targets=targets)))
                bindings[str(Path('artifacts')/rp.name)] = verifier.sha256(rp)
            targets, summary = summarize(pd.concat(frames, ignore_index=True), variables)
            report = dict(bindings=bindings, targets=targets, summary=summary, runtime_modified=False, promotion_allowed=False)
            assessment = artifacts/'assessment.json'; assessment.write_text(json.dumps(report))
            output = artifacts/'verification.json'
            scripts = Path(tmp)/'scripts'; scripts.mkdir()
            for name in ('score_additional_reconstruction.py', 'train_full_organ_readout.py'):
                (scripts/name).write_text('# Synthetic binding fixture\n')
            try:
                os.chdir(tmp)
                with patch.object(verifier, 'ROOT', root), patch.object(verifier, 'DEST', dest), \
                        patch.object(verifier, 'READERS', readers), patch.object(verifier, 'ASSESSMENT', assessment), \
                        patch.object(verifier, 'OUTPUT', output), \
                        patch.object(verifier, 'verify_full_grid', return_value=(grid, variables, (14, 32), {})), \
                        patch.object(verifier, 'paired_data', return_value=(ids, data)):
                    verifier.main()
            finally:
                os.chdir(original_cwd)
            result = json.loads(output.read_text())
            self.assertEqual(result['models_replayed'], 47*5*5)
            self.assertEqual(result['target_folds'], 47*5)
            self.assertTrue(result['independent_statistical_verification_complete'])
            self.assertFalse(result['promotion_allowed'])
            self.assertNotIn('patient', result)


if __name__ == '__main__':
    unittest.main()

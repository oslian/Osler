import sqlite3
import tempfile
import unittest
from pathlib import Path

import torch

from whole_body_jepa.reconstruction_cache import ReconstructionCache


class ReconstructionCacheTests(unittest.TestCase):
    def sample(self):
        return dict(tokens=torch.tensor([[0.,1.,7.],[1.125,0.,0.]],dtype=torch.float32),
            event_hours=torch.tensor([1.0000000001,2.],dtype=torch.float64),
            truth_normalized=0.,anchor_hour=2.,target_index=3)

    def test_lossless_readonly_and_atomic_completion(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'cache.sqlite';c=ReconstructionCache(path,{'a':'b'})
            e=self.sample();c.add('p',e);c.finish_shard('one');c.add('q',e);c.close()
            c=ReconstructionCache(path)
            actual=c.get('p',3)
            for k in ('tokens','event_hours'):
                self.assertEqual(actual[k].dtype,e[k].dtype)
                self.assertTrue(torch.equal(actual[k],e[k]))
            self.assertEqual(actual['truth_normalized'],0.)
            with self.assertRaises(KeyError):c.get('q',3)
            with self.assertRaises(sqlite3.OperationalError):c.add('r',e)
            c.close()
            with self.assertRaisesRegex(ValueError,'Changed'):ReconstructionCache(path,{'a':'c'})

    def test_duplicate_and_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            c=ReconstructionCache(Path(folder)/'c.sqlite',{})
            c.add('p',self.sample());c.finish_shard('one')
            with self.assertRaises(sqlite3.IntegrityError):c.add('p',self.sample())
            c.db.execute("UPDATE examples SET payload=x'00'");c.db.commit()
            with self.assertRaisesRegex(ValueError,'Corrupt'):c.get('p',3)
            c.close()

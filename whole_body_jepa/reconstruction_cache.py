"""Lossless, bounded random access to raw masked patient histories."""
import hashlib
import io
import json
import sqlite3
from pathlib import Path

import numpy as np
import torch


def pack_example(example):
    buffer=io.BytesIO()
    np.savez_compressed(buffer,
        tokens=example['tokens'].detach().cpu().numpy(),
        event_hours=example['event_hours'].detach().cpu().numpy(),
        truth=np.float64(example['truth_normalized']),
        anchor=np.float64(example['anchor_hour']),
        target=np.int64(example['target_index']))
    return buffer.getvalue()


def unpack_example(payload):
    with np.load(io.BytesIO(payload),allow_pickle=False) as data:
        return dict(tokens=torch.from_numpy(data['tokens'].copy()),
            event_hours=torch.from_numpy(data['event_hours'].copy()),
            truth_normalized=float(data['truth']),anchor_hour=float(data['anchor']),
            target_index=int(data['target']))


class ReconstructionCache:
    """Writes commit explicitly; interrupted transactions never mark a shard done."""
    def __init__(self,path,bindings=None):
        path=Path(path)
        if bindings is None:
            self.db=sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True)
        else:
            self.db=sqlite3.connect(path)
            self.db.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY,value TEXT NOT NULL)')
            self.db.execute('CREATE TABLE IF NOT EXISTS examples (patient TEXT NOT NULL,target INTEGER NOT NULL,truth REAL NOT NULL,payload BLOB NOT NULL,digest TEXT NOT NULL,PRIMARY KEY(patient,target))')
            self.db.execute('CREATE TABLE IF NOT EXISTS completed (shard TEXT PRIMARY KEY)')
            encoded=json.dumps(bindings,sort_keys=True)
            row=self.db.execute("SELECT value FROM metadata WHERE key='bindings'").fetchone()
            if row is not None and row[0]!=encoded:
                self.db.close();raise ValueError('Changed cache dependencies')
            self.db.execute("INSERT OR IGNORE INTO metadata VALUES ('bindings',?)",(encoded,))
            self.db.commit()

    def add(self,patient,example):
        payload=pack_example(example)
        self.db.execute('INSERT INTO examples VALUES (?,?,?,?,?)',
            (str(patient),example['target_index'],example['truth_normalized'],
             payload,hashlib.sha256(payload).hexdigest()))

    def get(self,patient,target):
        row=self.db.execute('SELECT payload,digest FROM examples WHERE patient=? AND target=?',
                            (str(patient),int(target))).fetchone()
        if row is None:raise KeyError((patient,target))
        if hashlib.sha256(row[0]).hexdigest()!=row[1]:raise ValueError('Corrupt cached example')
        result=unpack_example(row[0])
        if result['target_index']!=target:raise ValueError('Wrong cached target')
        return result

    def finish_shard(self,name):
        self.db.execute('INSERT INTO completed VALUES (?)',(name,))
        self.db.commit()

    def close(self):
        self.db.close()

"""Event-identity-safe lab replay; explicit observation and query separation."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
from torch import nn


def canonical_events(frame):
    """Deduplicate exact source versions, never deduplicate equal measured values."""
    out = frame.copy()
    for col in ('measured_at', 'available_at'):
        out[col] = pd.to_datetime(out[col], errors='coerce')
    valid = out[['stay_id', 'source_id', 'measured_at', 'available_at', 'value']].notna().all(axis=1)
    valid &= out.available_at >= out.measured_at
    valid &= np.isfinite(pd.to_numeric(out.value, errors='coerce'))
    out = out.loc[valid].copy()
    if 'variable' in out.columns:
        identity = out.groupby(['stay_id', 'source_id'], dropna=False)
        if out.variable.isna().any() or (identity.variable.nunique() > 1).any():
            raise ValueError('event identity changes variable across source versions')
    key = ['stay_id', 'source_id', 'available_at']
    versions = out.groupby(key, dropna=False)
    if (versions.value.nunique() > 1).any() or (versions.measured_at.nunique() > 1).any():
        raise ValueError('conflicting contents for the same event version')
    return out.drop_duplicates(key).sort_values(['stay_id', 'available_at', 'source_id'], kind='stable').reset_index(drop=True)


@dataclass
class Replay:
    tokens: np.ndarray
    updates: np.ndarray
    queries: np.ndarray
    rows: np.ndarray
    event_ids: tuple


def build_replay(events, queries, *, center, scale, repeat_at_query=False):
    """Process all available raw events, and score only original query anchors.

    Both arms receive identical raw events. The control additionally re-ingests
    the last event AFTER each query. Elapsed time belongs to query/readout, not
    to artificial repeated physiological transitions.
    """
    events = canonical_events(events)
    tokens, updates, q, rows, ids = [], [], [], [], []
    cursor = 0
    raw = list(events.itertuples())
    last = None
    previous_available = None
    for index, row in queries.sort_values('t', kind='stable').iterrows():
        while cursor < len(raw) and raw[cursor].available_at <= row.t:
            e = raw[cursor]
            age = (e.available_at-e.measured_at).total_seconds()/3600
            elapsed = 0. if previous_available is None else (e.available_at-previous_available).total_seconds()/3600
            token = np.array([(e.value-center)/scale, np.log1p(age), np.log1p(elapsed)], dtype=np.float32)
            tokens.append(token); updates.append(True); q.append([0., 0., 0.]); rows.append(-1)
            ids.append((str(e.source_id), e.available_at.isoformat()))
            previous_available = e.available_at
            last = (e, token)
            cursor += 1
        if last is None:
            continue
        e, token = last
        age = (row.t-e.measured_at).total_seconds()/3600
        since = (row.t-e.available_at).total_seconds()/3600
        tokens.append(np.zeros(3, dtype=np.float32)); updates.append(False)
        q.append([(e.value-center)/scale, np.log1p(age), np.log1p(since)])
        rows.append(index); ids.append(None)
        if repeat_at_query:
            tokens.append(token); updates.append(True); q.append([0., 0., 0.]); rows.append(-1)
            ids.append((str(e.source_id), e.available_at.isoformat()))
    return Replay(np.asarray(tokens, dtype=np.float32).reshape(-1, 3), np.asarray(updates),
                  np.asarray(q, dtype=np.float32).reshape(-1, 3), np.asarray(rows, dtype=int), tuple(ids))


class RawLabResidual(nn.Module):
    """Shared patient recurrence, with query readout that never mutates state."""
    def __init__(self, width=16):
        super().__init__()
        self.width = width
        self.encoder = nn.Sequential(nn.Linear(3, width), nn.Tanh())
        self.update = nn.GRUCell(width, width)
        self.head = nn.Sequential(nn.Linear(width+3, width), nn.SiLU(), nn.Linear(width, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, tokens, updates, queries):
        h = torch.zeros(tokens.shape[0], self.width, device=tokens.device)
        encoded = self.encoder(tokens)
        outputs = []
        for t in range(tokens.shape[1]):
            updated = self.update(encoded[:, t], h)
            h = torch.where(updates[:, t, None], updated, h)
            outputs.append(4*torch.tanh(self.head(torch.cat([h, queries[:, t]], dim=1)).squeeze(-1)))
        return torch.stack(outputs, dim=1)


def batch_replays(sequences):
    n = max(len(s.rows) for s in sequences)
    x, q = np.zeros((len(sequences), n, 3), np.float32), np.zeros((len(sequences), n, 3), np.float32)
    updates = np.zeros((len(sequences), n), bool)
    rows = np.full((len(sequences), n), -1, int)
    for i, s in enumerate(sequences):
        stop = len(s.rows)
        x[i, :stop], q[i, :stop], updates[i, :stop], rows[i, :stop] = s.tokens, s.queries, s.updates, s.rows
    return torch.from_numpy(x), torch.from_numpy(updates), torch.from_numpy(q), rows


def predictions(model, sequences, frame, residual_scale):
    pred = frame.frozen.to_numpy(float).copy()
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(sequences), 32):
            x, u, q, rows = batch_replays(sequences[start:start+32])
            delta = model(x, u, q).numpy()
            valid = rows >= 0
            pred[rows[valid]] += residual_scale * delta[valid]
    return pred

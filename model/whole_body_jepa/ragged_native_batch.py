"""Lossless native-query storage without a dense history for every event row."""
import numpy as np
import torch


class RaggedNativeBatch:
    def __init__(self, shards, query_count):
        tokens, values = [], []
        sequences = [None] * query_count
        contexts = None
        offset = 1
        for base, native in shards:
            ids = native['global_query_rows']
            np.testing.assert_array_equal(ids, base['global_query_rows'])
            ptr, history = native['query_ptr'], native['history_indices']
            if len(ptr) != len(ids) + 1 or ptr[0] != 0 or ptr[-1] != len(history) or (np.diff(ptr) < 0).any():
                raise ValueError('invalid query pointers')
            x = native['input']
            if len(history) and (history.min() < 0 or history.max() >= len(x)):
                raise ValueError('invalid native index')
            if contexts is None:
                contexts = {k: np.empty((query_count, *base[k].shape[1:]), base[k].dtype)
                    for k in ('input', 'current', 'mask', 'ages', 'presence', 'horizon')}
            for i, index in enumerate(ids):
                if index < 0 or index >= query_count or sequences[index] is not None:
                    raise ValueError('invalid or repeated global query')
                sequences[index] = history[ptr[i]:ptr[i+1]] + offset
                for key in contexts:
                    contexts[key][index] = base[key][base['query_rows'][i]]
            tokens.append(x)
            values.append(native['event_values'])
            offset += len(x)
        if any(s is None for s in sequences):
            raise ValueError('missing global queries')
        self.query = contexts
        self.input = np.concatenate([np.zeros((1, contexts['input'].shape[1]), np.float32),
                                     *tokens])
        self.event_values = np.concatenate([np.full((1, 3), np.nan), *values])
        self.sequences = sequences
        self.width = max(map(len, sequences))

    def batch(self, rows):
        rows = np.asarray(rows, dtype=np.int64)
        indices = np.zeros((len(rows), self.width), np.int64)
        for i, row in enumerate(rows):
            sequence = self.sequences[row]
            if len(sequence):
                indices[i, -len(sequence):] = sequence
        return (torch.from_numpy(self.input[indices]),
            *[torch.from_numpy(self.query[k][rows])
              for k in ('current', 'mask', 'ages', 'presence', 'horizon')])

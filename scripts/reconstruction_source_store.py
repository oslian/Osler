"""Bounded, integrity-checked patient history access for source-only training."""
from functools import lru_cache
from pathlib import Path

import torch

from whole_body_jepa.multisource_intervals import sha256


class SourceStore:
    def __init__(self, shards, columns, cache_size=16):
        self.shards = {}
        self.labels = {}
        self.groups = {c: [] for c in columns}
        if cache_size < 1:
            raise ValueError("Positive bounded cache required")
        for shard in shards:
            patient = shard["patient"]
            if patient in self.shards:
                raise ValueError("Duplicate patient in source store")
            self.shards[patient] = dict(shard)
            data = self._read(patient)
            for c, example in data["samples"].items():
                if c in self.groups:
                    key = (patient, c)
                    self.groups[c].append(key)
                    self.labels[key] = float(example["truth_normalized"])
        self._cached_read = lru_cache(maxsize=cache_size)(self._read)

    def _read(self, patient):
        shard = self.shards[patient]
        path = Path(shard["path"])
        if sha256(path) != shard["sha256"]:
            raise ValueError("Patient shard changed")
        data = torch.load(path, weights_only=False)
        # Historical tensors predate the nested split; roles belong to the
        # independently verified source manifest, not the embedded old tag.
        if data["patient"] != patient:
            raise ValueError("Patient mismatch")
        return data

    def __getitem__(self, key):
        if key not in self.labels:
            raise KeyError(key)
        return self._cached_read(key[0])["samples"][key[1]]


def source_shards(joint, expanded, mode, selected_roles):
    """Change broad fit only; preserve both original encoder-selection groups."""
    if mode not in ("small", "expanded"):
        raise ValueError("Unknown source mode")
    if joint["variables"] != expanded["variables"]:
        raise ValueError("Variable schema mismatch")
    result = [dict(s) for s in joint["shards"]
              if not (mode == "expanded" and s["group"] == "broad"
                      and s["role"] == "fit")]
    if mode == "expanded":
        result.extend(dict(s, group="broad") for s in expanded["shards"]
                      if s["role"] == "fit")
    patients = set()
    for s in result:
        if s["role"] not in ("fit", "encoder_tune"):
            raise ValueError("Non-source role")
        if s["group"] not in ("selected", "broad"):
            raise ValueError("Unknown source group")
        if s["group"] == "selected" and selected_roles.get(s["patient"]) != s["role"]:
            raise ValueError("Nested source role mismatch")
        if s["patient"] in patients:
            raise ValueError("Patient crosses source groups or roles")
        patients.add(s["patient"])
    return result

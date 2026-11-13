"""Which files hold lockbox data, and the one reader for every parquet input that must not.

Every lockbox extractor (SICdb, HiRID, AmsterdamUMCdb, INSPIRE) writes its database name into the ``hospitalid`` column of
every row.  ``hospital_identities`` returns every distinct value of that support column (never a label), missing
values included; ``lockbox_sources`` keeps the lockbox names among them.  A file or dataset without the column (MIMIC
training files, caches, examples) or without rows has no identity: this detects the project's extraction formats, it
is not a general provenance check.  A lockbox read is admitted only for a protocol-frozen, addendum-recorded file whose
identities are exactly its database (``lockbox_scoring``).  ``read_non_lockbox`` is the reader of every other parquet
input of the project: it refuses a lockbox file before any other column is read.  The few authorized direct readers
are listed, with their role, in ``test_lockbox_scoring.AUTHORIZED_DIRECT_READERS``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds

# Zigong (2026-10-18) and Emory (2026-10-20) were scored once and are now development data; SICdb and HiRID were
# scored under v10/v11 and AmsterdamUMCdb under v12, and stay guarded; INSPIRE is the protocol-v13 lockbox.
LOCKBOX_DATABASES = frozenset({"SICdb", "HiRID", "AmsterdamUMCdb", "INSPIRE"})


def hospital_identities(path: str | Path) -> set:
    """Every distinct ``hospitalid`` value of a parquet file or dataset (missing values as None); empty without the
    column or without rows."""

    data = ds.dataset(path, format="parquet")
    if "hospitalid" not in data.schema.names:
        return set()
    return set(data.to_table(columns=["hospitalid"]).column(0).unique().to_pylist())


def lockbox_sources(path: str | Path) -> set[str]:
    """The lockbox databases whose rows a parquet file or dataset holds."""

    return hospital_identities(path) & LOCKBOX_DATABASES


def read_non_lockbox(path: str | Path, **kwargs) -> pd.DataFrame:
    """A transition file in a training or development role; a lockbox file is refused before its data are read."""

    sources = lockbox_sources(path)
    if sources:
        raise RuntimeError(f"{path} holds lockbox rows ({sorted(sources)}); it is read only by a claimed lockbox scoring step")
    return pd.read_parquet(path, **kwargs)

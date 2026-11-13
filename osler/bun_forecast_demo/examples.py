"""Synthetic example trajectories exposed by the BUN forecast website."""

from __future__ import annotations

from pathlib import Path
from typing import Any


EXAMPLE_DIR = Path(__file__).resolve().parent / "examples"

EXAMPLES: tuple[dict[str, Any], ...] = (
    {
        "id": "stable",
        "filename": "patient_1.csv",
        "title": {"zh-Hant": "病人 1", "en": "Patient 1"},
    },
    {
        "id": "progressive",
        "filename": "patient_2.csv",
        "title": {"zh-Hant": "病人 2", "en": "Patient 2"},
    },
    {
        "id": "recovering",
        "filename": "patient_3.csv",
        "title": {"zh-Hant": "病人 3", "en": "Patient 3"},
    },
    {
        "id": "sparse",
        "filename": "patient_4.csv",
        "title": {"zh-Hant": "病人 4", "en": "Patient 4"},
    },
    {
        "id": "rrt",
        "filename": "patient_5.csv",
        "title": {"zh-Hant": "病人 5", "en": "Patient 5"},
    },
)

EXAMPLE_BY_ID = {example["id"]: example for example in EXAMPLES}


def example_catalog() -> list[dict[str, Any]]:
    """Return public metadata without filesystem paths."""

    return [
        {
            "id": example["id"],
            "filename": example["filename"],
            "title": example["title"],
            "synthetic": True,
            "download_url": f"/examples/{example['id']}.csv",
        }
        for example in EXAMPLES
    ]


def example_path(example_id: str) -> Path:
    """Resolve one allow-listed synthetic example or raise ``KeyError``."""

    example = EXAMPLE_BY_ID[example_id]
    path = EXAMPLE_DIR / str(example["filename"])
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


__all__ = ["EXAMPLES", "EXAMPLE_BY_ID", "example_catalog", "example_path"]

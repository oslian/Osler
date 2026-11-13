"""Canonical functional-system and anatomical-organ contracts.

System nodes describe physiological functions such as perfusion, acid-base
control, and coagulation. Organ nodes describe anatomical structures such as
the heart, lungs, kidneys, and liver. The two taxonomies are intentionally
separate and communicate through an explicit many-to-many interface.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


CANONICAL_SYSTEM_VARIABLES: dict[str, tuple[str, ...]] = {
    "cardiovascular_perfusion": ("heart_rate", "map", "lactate", "bnp"),
    "respiratory_exchange": ("o2sat", "respiratory_rate"),
    "renal_clearance": ("bun", "creatinine", "uric_acid", "urine_output"),
    "acid_base_electrolyte": (
        "anion_gap",
        "bicarbonate",
        "calcium",
        "chloride",
        "ionized_calcium",
        "magnesium",
        "ph",
        "phosphate",
        "potassium",
        "serum_osmolality",
        "sodium",
    ),
    "endocrine_metabolic": (
        "cortisol",
        "free_t4",
        "glucose",
        "triglycerides",
        "tsh",
    ),
    "hepatic_synthesis_detoxification": (
        "albumin",
        "alkaline_phos",
        "alt",
        "ast",
        "bilirubin",
        "bilirubin_direct",
        "prealbumin",
        "total_protein",
    ),
    "digestion_absorption": ("amylase", "lipase"),
    "hematopoiesis_coagulation": (
        "fibrinogen",
        "hematocrit",
        "hemoglobin",
        "inr",
        "platelets",
        "ptt",
    ),
    "immune_inflammation": (
        "crp",
        "crp_hs",
        "esr",
        "ferritin",
        "temperature",
        "wbc",
    ),
    "cardiac_injury": ("ck_mb", "troponin_i"),
    "musculoskeletal_injury": ("cpk", "ldh", "myoglobin"),
}


# Variable ownership is disjoint so an observation is never silently copied
# into multiple organ tokens. Empty organs remain explicit anatomical nodes;
# they can receive system messages but are never presented as observed.
CANONICAL_ORGAN_VARIABLES: dict[str, tuple[str, ...]] = {
    "heart": ("heart_rate", "bnp", "ck_mb", "troponin_i"),
    "blood_vessels": ("map", "lactate"),
    "lungs": ("o2sat", "respiratory_rate"),
    "kidneys": (
        "anion_gap",
        "bicarbonate",
        "bun",
        "calcium",
        "chloride",
        "creatinine",
        "ionized_calcium",
        "magnesium",
        "ph",
        "phosphate",
        "potassium",
        "serum_osmolality",
        "sodium",
        "uric_acid",
        "urine_output",
    ),
    "liver": (
        "albumin",
        "alkaline_phos",
        "alt",
        "ast",
        "bilirubin",
        "bilirubin_direct",
        "crp",
        "crp_hs",
        "ferritin",
        "fibrinogen",
        "inr",
        "prealbumin",
        "ptt",
        "total_protein",
    ),
    "pancreas": ("amylase", "glucose", "lipase", "triglycerides"),
    "stomach": (),
    "intestines": (),
    "thyroid": ("free_t4", "tsh"),
    "adrenal_glands": ("cortisol",),
    "bone_marrow": ("esr", "hematocrit", "hemoglobin", "platelets", "wbc"),
    "spleen": (),
    "brain": ("temperature",),
    "skeletal_muscle": ("cpk", "ldh", "myoglobin"),
}


# Functional systems may depend on more than one anatomical organ. This map is
# the only legal system/organ interface used by the adjacent hierarchy.
CANONICAL_SYSTEM_TO_ORGANS: dict[str, tuple[str, ...]] = {
    "cardiovascular_perfusion": ("heart", "blood_vessels", "kidneys"),
    "respiratory_exchange": ("lungs", "heart"),
    "renal_clearance": ("kidneys",),
    "acid_base_electrolyte": ("kidneys", "lungs"),
    "endocrine_metabolic": ("pancreas", "liver", "thyroid", "adrenal_glands"),
    "hepatic_synthesis_detoxification": ("liver",),
    "digestion_absorption": ("stomach", "intestines", "pancreas", "liver"),
    "hematopoiesis_coagulation": ("bone_marrow", "liver", "spleen"),
    "immune_inflammation": ("bone_marrow", "spleen", "liver", "brain"),
    "cardiac_injury": ("heart",),
    "musculoskeletal_injury": ("skeletal_muscle", "kidneys"),
}


CANONICAL_ORGAN_COUPLINGS: tuple[tuple[str, str], ...] = (
    ("heart", "blood_vessels"),
    ("heart", "lungs"),
    ("heart", "kidneys"),
    ("heart", "thyroid"),
    ("heart", "adrenal_glands"),
    ("blood_vessels", "kidneys"),
    ("blood_vessels", "liver"),
    ("lungs", "kidneys"),
    ("kidneys", "liver"),
    ("kidneys", "skeletal_muscle"),
    ("liver", "pancreas"),
    ("liver", "stomach"),
    ("liver", "intestines"),
    ("liver", "bone_marrow"),
    ("pancreas", "stomach"),
    ("pancreas", "intestines"),
    ("bone_marrow", "spleen"),
)


def canonical_organ_adjacency(
    organ_names: Iterable[str],
) -> tuple[tuple[float, ...], ...]:
    """Return the sparse anatomical organ graph in the requested order."""

    names = tuple(str(value) for value in organ_names)
    indices = {name: index for index, name in enumerate(names)}
    adjacency = [[0.0 for _ in names] for _ in names]
    for left, right in CANONICAL_ORGAN_COUPLINGS:
        if left not in indices or right not in indices:
            continue
        left_index = indices[left]
        right_index = indices[right]
        adjacency[left_index][right_index] = 1.0
        adjacency[right_index][left_index] = 1.0
    return tuple(tuple(row) for row in adjacency)


def canonical_organ_contracts(
    available_variables: Iterable[str],
) -> dict[str, tuple[str, ...]]:
    """Return disjoint anatomical organ contracts for a cohort."""

    available = set(available_variables)
    contracts = {
        organ: tuple(variable for variable in variables if variable in available)
        for organ, variables in CANONICAL_ORGAN_VARIABLES.items()
    }
    return contracts


def canonical_target_organ_membership(
    target_names: Iterable[str],
    organ_names: Iterable[str],
) -> np.ndarray:
    """Map functional target cells to anatomical organs without label copying."""

    targets = tuple(str(value) for value in target_names)
    organs = tuple(str(value) for value in organ_names)
    organ_index = {name: index for index, name in enumerate(organs)}
    variable_systems: dict[str, list[str]] = {}
    for system, variables in CANONICAL_SYSTEM_VARIABLES.items():
        for variable in variables:
            variable_systems.setdefault(variable, []).append(system)

    membership = np.zeros((len(targets), len(organs)), dtype=np.float32)
    direct_owner = {
        variable: organ
        for organ, variables in CANONICAL_ORGAN_VARIABLES.items()
        for variable in variables
    }
    for target_index, target in enumerate(targets):
        for system in variable_systems.get(target, ()):  # usually one system
            for organ in CANONICAL_SYSTEM_TO_ORGANS[system]:
                if organ in organ_index:
                    membership[target_index, organ_index[organ]] = 1.0
        if not membership[target_index].any():
            owner = direct_owner.get(target)
            if owner is not None and owner in organ_index:
                membership[target_index, organ_index[owner]] = 1.0
    return membership


def assert_disjoint_contracts(
    contracts: dict[str, tuple[str, ...]],
) -> None:
    """Raise when an observation is assigned to multiple organ tokens."""

    owners: dict[str, str] = {}
    for organ, variables in contracts.items():
        for variable in variables:
            previous = owners.setdefault(variable, organ)
            if previous != organ:
                raise ValueError(
                    f"{variable!r} belongs to both {previous!r} and {organ!r}"
                )

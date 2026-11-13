"""Frozen deployment form for exported JEPA predictions (used by ``jepa_gate.py``).

Input: one system column (the frozen expert anchor or a matched hierarchy
model) of the per-row export of ``teacher_anchored_joint_jepa_audit.py
--export-predictions``; calibration rows come from the training database, test
rows from one external database.  Every system goes through the same frozen
deployment form:

- point: ``online_point_correction`` with the frozen config, scale = median
  calibration residual of that system;
- interval: v2 quintile value-bin scale fitted on the calibration set's
  selection patients and DtACI (meta rate 20, theta0 0.15) on its conformal
  patients (``osler_jepa.patient_split``; the same patients that chose the
  coupling path never give conformal scores), network-shared across the
  external hospitals, uniform calibration weights for both systems (event rows
  carry no day-one covariates);
- each hospital is replayed on the shared deployment clock
  (``osler_jepa.deployment_clock``, as in ``site_adaptive_external_audit``): stays
  overlap, and an outcome updates the trackers only once the clock passes its
  availability time ``label_available_hr`` (horizon + |measurement time - target
  time|, exported by the audit); warm-up stays are not scored.

Research-only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mimic_to_eicu_49_cell_external_audit import stable_seed
from osler_jepa.interval_scale import fit_interval_scale
from osler_jepa.patient_split import selection_patients
from osler_jepa.predictive_distribution import MultiQTConfig, mondrian_multiqt_stream, value_bins
from threshold_event_form import resolution
from osler_jepa.site_adaptive_conformal import dtaci_stream_widths
from osler_jepa.site_adaptive_point import online_point_correction
from osler_jepa.deployment_clock import deployment_clock


def deploy(calibration: pd.DataFrame, test: pd.DataFrame, system: str, cell: str, point_config, dtaci_config,
           network_config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Frozen deployment form on the shared clock; returns corrected predictions, widths and the scored mask.

    Network-shared DtACI only when the external database has several hospitals.
    """

    residual = np.abs(calibration[system].to_numpy() - calibration["future"].to_numpy())
    choosing = selection_patients(calibration["subject_id"].to_numpy())   # scales: selection patients; scores: conformal
    current_cal = calibration["current"].to_numpy()
    point_scale = float(np.median(residual[choosing]))
    scale = fit_interval_scale("quintile", current_cal[choosing], np.full(int(choosing.sum()), np.nan), residual[choosing],
                               calibration["subject_id"].to_numpy()[choosing])
    cal_scores = residual[~choosing] / scale(current_cal[~choosing], np.full(int((~choosing).sum()), np.nan))
    availability = test["label_available_hr"].to_numpy(dtype=float)
    if not np.all(np.isfinite(availability)):
        raise ValueError(f"{cell}: label availability times missing; re-export predictions")
    stays, units = test["stay_id"].astype(str).to_numpy(), test["careunit"].astype(str).to_numpy()
    onset = test["hours_since_onset"].to_numpy(dtype=float)
    current, future = test["current"].to_numpy(), test["future"].to_numpy()
    clock, warm = np.full(len(test), np.nan), np.zeros(len(test), dtype=bool)
    corrected, widths = np.full(len(test), np.nan), np.full(len(test), np.nan)
    hospitals = test["hospitalid"].astype(str).to_numpy()
    network = len(np.unique(hospitals)) > 1

    def stream(order, levels, config):
        return dtaci_stream_widths(
            calibration_scores=cal_scores, calibration_weights=np.ones(len(cal_scores)), stays=stays[order],
            levels={"care_unit": levels[order]}, anchor_hours=clock[order], resolve_hours=clock[order] + availability[order],
            scales=scale(current[order], np.full(len(order), np.nan)), predictions=corrected[order], observed=future[order],
            config=config).widths

    for hospital in np.unique(hospitals):
        rows = np.flatnonzero(hospitals == hospital)
        clocked = deployment_clock(stays[rows], onset[rows], seed=stable_seed(cell, 0, hospital, "clock"))
        order = rows[clocked.order]
        clock[rows], warm[rows] = clocked.clock_hours, clocked.warmup
        corrected[order] = online_point_correction(
            stays=stays[order], levels={"care_unit": units[order]}, anchor_hours=clock[order],
            resolve_hours=clock[order] + availability[order], current=current[order],
            prediction=test[system].to_numpy()[order], observed=future[order], scale=point_scale, config=point_config)
        if not network:
            widths[order] = stream(order, units, dtaci_config)
    if network:                                             # one network on one clock
        order = np.lexsort((onset, clock))
        widths[order] = stream(order, hospitals, network_config)
    return corrected, widths, ~warm


def deploy_distribution(calibration: pd.DataFrame, test: pd.DataFrame, system: str, cell: str, point_config,
                        multiqt: MultiQTConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Online predictive distribution of one system on the shared clock; returns corrected points, the issued
    quantiles (rows x ``multiqt.levels``) and the scored mask.

    Same clock, warm-up and online point correction as ``deploy``.  Distribution: Mondrian MultiQT per hospital
    (``osler_jepa.predictive_distribution``) from day-one signed errors of the calibration set's conformal patients,
    normalized by the v2 quintile scale fitted on its selection patients and floored at the target's recording
    resolution (``threshold_event_form.resolution`` of the calibration current values; amendment
    online_joint_dev_rule_amendment_20261031.json: a bin whose median |error| is 0 would otherwise give a 1e-9 scale
    and exploding scores), with uniform weights; categories are the ``tail8`` bins of the current value on the
    calibration set.
    """

    signed = calibration["future"].to_numpy() - calibration[system].to_numpy()
    subjects = calibration["subject_id"].to_numpy()
    choosing = selection_patients(subjects)
    current_cal = calibration["current"].to_numpy()
    point_scale = float(np.median(np.abs(signed[choosing])))
    no_change = lambda n: np.full(n, np.nan)
    fitted = fit_interval_scale("quintile", current_cal[choosing], no_change(int(choosing.sum())), np.abs(signed[choosing]),
                                subjects[choosing])
    floor = resolution(current_cal)
    scale = lambda values, change: np.maximum(fitted(values, change), floor)
    scores = signed[~choosing] / scale(current_cal[~choosing], no_change(int((~choosing).sum())))
    edges = value_bins(current_cal)
    availability = test["label_available_hr"].to_numpy(dtype=float)
    if not np.all(np.isfinite(availability)):
        raise ValueError(f"{cell}: label availability times missing")
    stays, units = test["stay_id"].astype(str).to_numpy(), test["careunit"].astype(str).to_numpy()
    onset = test["hours_since_onset"].to_numpy(dtype=float)
    current, future = test["current"].to_numpy(), test["future"].to_numpy()
    hospitals = test["hospitalid"].astype(str).to_numpy()
    corrected, warm = np.full(len(test), np.nan), np.zeros(len(test), dtype=bool)
    quantiles = np.full((len(test), len(multiqt.levels)), np.nan)
    for hospital in np.unique(hospitals):
        rows = np.flatnonzero(hospitals == hospital)
        clocked = deployment_clock(stays[rows], onset[rows], seed=stable_seed(cell, 0, hospital, "clock"))
        order = rows[clocked.order]
        clock = np.empty(len(test))
        clock[rows], warm[rows] = clocked.clock_hours, clocked.warmup
        corrected[order] = online_point_correction(
            stays=stays[order], levels={"care_unit": units[order]}, anchor_hours=clock[order],
            resolve_hours=clock[order] + availability[order], current=current[order],
            prediction=test[system].to_numpy()[order], observed=future[order], scale=point_scale, config=point_config)
        stream = mondrian_multiqt_stream(
            calibration_scores=scores, calibration_weights=np.ones(len(scores)),
            calibration_groups=np.digitize(current_cal[~choosing], edges), calibration_subjects=subjects[~choosing],
            groups=np.digitize(current[order], edges), predictions=corrected[order],
            scales=scale(current[order], no_change(len(order))), observed=future[order], anchor_hours=clock[order],
            resolve_hours=clock[order] + availability[order], config=multiqt)
        quantiles[order] = stream.quantiles()
    return corrected, quantiles, ~warm

"""Truth-bin-supervised routing over one frozen shared JEPA body state."""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, Laplace, MixtureSameFamily
from torch.nn import functional as F

from scripts.audit_release_masked_cells import donor_indices
from scripts.masked_generalization_methods import STEPS, selection_score
from scripts.masked_mixture_readouts import SCALE_FLOOR, mixture_median
from whole_body_jepa.multisource_intervals import SEEDS, stable_hash


FAMILIES = {"guided_3": 3, "guided_5": 5}
GATE_WEIGHT = .25


class GuidedMixtureReadout(nn.Module):
    def __init__(self, input_dim, fit_truth, components):
        super().__init__()
        if components not in (3, 5):
            raise ValueError("only prespecified 3- or 5-expert readers are allowed")
        self.components = components
        width = 48 if components == 3 else 44
        self.net = nn.Sequential(nn.Linear(input_dim, width), nn.SiLU(), nn.Linear(width, 3 * components))
        truth = torch.as_tensor(fit_truth).detach().float()
        if truth.ndim != 1 or not len(truth) or not torch.isfinite(truth).all():
            raise ValueError("finite source fitting truth required")
        probabilities = torch.linspace(0, 1, components + 1)[1:-1]
        self.register_buffer("boundaries", torch.quantile(truth, probabilities))
        locations = torch.quantile(truth, (torch.arange(components) + .5) / components)
        scale = (truth - truth.median()).abs().mean().clamp_min(.1)
        raw_scale = scale + torch.log(-torch.expm1(-scale))
        with torch.no_grad():
            self.net[-1].weight.zero_()
            self.net[-1].bias.zero_()
            self.net[-1].bias[components:2 * components] = locations
            self.net[-1].bias[2 * components:] = raw_scale

    def values(self, state):
        return self.net(state).chunk(3, dim=-1)

    def distribution(self, state):
        logits, locations, raw_scales = self.values(state)
        scales = F.softplus(raw_scales) + SCALE_FLOOR
        return MixtureSameFamily(Categorical(logits=logits), Laplace(locations, scales))

    def labels(self, truth):
        return torch.bucketize(truth.contiguous(), self.boundaries)

    def loss(self, state, truth):
        logits, _, _ = self.values(state)
        return -self.distribution(state).log_prob(truth).mean() + GATE_WEIGHT * F.cross_entropy(logits, self.labels(truth))


@torch.no_grad()
def components(heads, states, routing="legal", route_indices=None):
    if routing not in {"legal", "uniform", "shuffled"}:
        raise ValueError("illegal routing")
    if len(heads) != len(states) or not heads:
        raise ValueError("one context per reader required")
    if routing == "shuffled" and (route_indices is None or len(route_indices) != len(heads)):
        raise ValueError("route donors required")
    weights, locations, scales = [], [], []
    for index, (head, state) in enumerate(zip(heads, states)):
        distribution = head.distribution(state)
        probabilities = distribution.mixture_distribution.probs
        if routing == "uniform":
            probabilities = torch.full_like(probabilities, 1 / head.components)
        elif routing == "shuffled":
            probabilities = probabilities[route_indices[index]]
        weights.append(probabilities / len(heads))
        locations.append(distribution.component_distribution.loc)
        scales.append(distribution.component_distribution.scale)
    return tuple(torch.cat(value, dim=1) for value in (weights, locations, scales))


@torch.no_grad()
def predict(heads, states, routing="legal", route_indices=None):
    return mixture_median(components(heads, states, routing, route_indices))


def fit_curve(component_count, state, truth, seed, steps):
    torch.manual_seed(seed)
    head = GuidedMixtureReadout(state.shape[1], truth, component_count)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.003, weight_decay=.001)
    rng = np.random.default_rng(seed)
    snapshots = {}
    for step in range(1, max(steps) + 1):
        rows = rng.integers(0, len(state), 128)
        optimizer.zero_grad(set_to_none=True)
        loss = head.loss(state[rows], truth[rows])
        if not torch.isfinite(loss):
            raise ValueError("nonfinite guided-mixture loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
        optimizer.step()
        if step in steps:
            snapshots[step] = copy.deepcopy(head).eval()
    return snapshots


def fit_family(family, fit, tune, target, steps=STEPS, seeds=SEEDS):
    if family not in FAMILIES:
        raise ValueError("unknown guided family")
    if set(fit["subjects"]) & set(tune["subjects"]):
        raise ValueError("fit/tune patient overlap")
    real, placebo = {step: [] for step in steps}, {step: [] for step in steps}
    tune_placebo = []
    for seed in seeds:
        draw = stable_hash("guided_mixture_fit", target, family, seed) % (2 ** 32)
        fit_donors = donor_indices(fit["subjects"], stable_hash("guided_fit_placebo", target, family, seed))
        tune_donors = donor_indices(tune["subjects"], stable_hash("guided_tune_placebo", target, family, seed))
        fitted = fit_curve(FAMILIES[family], fit["latent"], fit["y"], draw, steps)
        control = fit_curve(FAMILIES[family], fit["latent"][fit_donors], fit["y"], draw, steps)
        tune_placebo.append(tune["latent"][tune_donors])
        for step in steps:
            real[step].append(fitted[step])
            placebo[step].append(control[step])
    curve, placebo_curve = {}, {}
    for step in steps:
        curve[step] = selection_score(tune["y"], predict(real[step], [tune["latent"]] * len(seeds)), tune["units"])
        placebo_curve[step] = selection_score(tune["y"], predict(placebo[step], tune_placebo), tune["units"])
    chosen = min(steps, key=lambda step: (curve[step]["selection_score"], step))
    chosen_placebo = min(steps, key=lambda step: (placebo_curve[step]["selection_score"], step))
    return {"heads": real[chosen], "placebo_heads": placebo[chosen_placebo],
            "matched_placebo_heads": placebo[chosen]}, {
                "chosen_steps": chosen, "placebo_chosen_steps": chosen_placebo,
                "parameters_per_head": sum(p.numel() for p in real[chosen][0].parameters()),
                "curve": curve, "placebo_curve": placebo_curve}


@torch.no_grad()
def gate_diagnostics(heads, states, truth):
    accuracies, usage = [], []
    for head, state in zip(heads, states):
        logits, _, _ = head.values(state)
        probabilities = logits.softmax(1)
        accuracies.append((probabilities.argmax(1) == head.labels(truth)).float().mean().item())
        usage.append(probabilities.mean(0).numpy())
    return {"mean_gate_bin_accuracy": float(np.mean(accuracies)),
            "mean_component_usage": np.mean(usage, axis=0).tolist()}

"""Distributional readers of ONE frozen, target-masked JEPA context."""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, Laplace, MixtureSameFamily
from torch.nn import functional as F

from scripts.audit_release_masked_cells import donor_indices
from scripts.masked_generalization_methods import STEPS, selection_score
from whole_body_jepa.multisource_intervals import SEEDS, stable_hash


FAMILIES = ("wide_point", "single_laplace", "mixture_laplace")
SCALE_FLOOR = .05


class MaskedReadout(nn.Module):
    def __init__(self, family, input_dim, fit_truth):
        super().__init__()
        if family not in FAMILIES:
            raise ValueError("unknown reader")
        self.family = family
        self.components = 3 if family == "mixture_laplace" else 1
        width, outputs = {"wide_point": (54, 1), "single_laplace": (53, 2),
                          "mixture_laplace": (48, 9)}[family]
        self.net = nn.Sequential(nn.Linear(input_dim, width), nn.SiLU(), nn.Linear(width, outputs))
        truth = torch.as_tensor(fit_truth).detach().float()
        if truth.ndim != 1 or not torch.isfinite(truth).all() or not len(truth):
            raise ValueError("finite source fitting truth required")
        with torch.no_grad():
            self.net[-1].weight.zero_()
            self.net[-1].bias.zero_()
            if family == "wide_point":
                self.net[-1].bias[0] = truth.median()
            else:
                loc = torch.quantile(truth, torch.tensor([.1, .5, .9])) if self.components == 3 else truth.median()[None]
                scale = (truth - truth.median()).abs().mean().clamp_min(.1)
                raw_scale = scale + torch.log(-torch.expm1(-scale))
                start = self.components if self.components == 3 else 0
                self.net[-1].bias[start:start + self.components] = loc
                self.net[-1].bias[start + self.components:] = raw_scale

    def forward(self, state):
        return self.net(state)

    def distribution(self, state):
        value = self(state)
        if self.family == "wide_point":
            raise ValueError("point reader has no density")
        if self.components == 3:
            logits, loc, raw_scale = value.chunk(3, dim=-1)
        else:
            loc, raw_scale = value.chunk(2, dim=-1)
            logits = torch.zeros_like(loc)
        scale = F.softplus(raw_scale) + SCALE_FLOOR
        return MixtureSameFamily(Categorical(logits=logits), Laplace(loc, scale))

    def loss(self, state, truth):
        if self.family == "wide_point":
            return F.l1_loss(self(state).flatten(), truth)
        return -self.distribution(state).log_prob(truth).mean()


@torch.no_grad()
def ensemble_components(heads, states, routing="legal", route_indices=None):
    """Shuffled routing changes ONLY probabilities, never patient experts."""
    if routing not in {"legal", "uniform", "shuffled"}:
        raise ValueError("illegal routing")
    if len(heads) != len(states) or not heads:
        raise ValueError("one context tensor per ensemble member required")
    if routing == "shuffled" and (route_indices is None or len(route_indices) != len(heads)):
        raise ValueError("explicit route donors required")
    weights, locations, scales = [], [], []
    for i, (head, state) in enumerate(zip(heads, states)):
        distribution = head.distribution(state)
        probs = distribution.mixture_distribution.probs
        if routing == "uniform":
            probs = torch.full_like(probs, 1 / head.components)
        elif routing == "shuffled":
            probs = probs[route_indices[i]]
        weights.append(probs / len(heads))
        locations.append(distribution.component_distribution.loc)
        scales.append(distribution.component_distribution.scale)
    return tuple(torch.cat(parts, dim=1) for parts in (weights, locations, scales))


@torch.no_grad()
def mixture_median(components):
    weights, loc, scale = components
    if not all(torch.isfinite(v).all() for v in components) or not bool((scale > 0).all()):
        raise ValueError("invalid distribution parameters")
    if not torch.allclose(weights.sum(1), torch.ones(len(weights)), atol=1e-6):
        raise ValueError("unnormalized mixture")
    lower = (loc - 20 * scale).min(1).values
    upper = (loc + 20 * scale).max(1).values
    law = Laplace(loc, scale)
    for _ in range(40):
        middle = (lower + upper) / 2
        cdf = (weights * law.cdf(middle[:, None])).sum(1)
        lower = torch.where(cdf < .5, middle, lower)
        upper = torch.where(cdf >= .5, middle, upper)
    return ((lower + upper) / 2).numpy()


@torch.no_grad()
def predict(heads, states, routing="legal", route_indices=None):
    if heads[0].family == "wide_point":
        if routing != "legal":
            raise ValueError("point reader has no router")
        return torch.stack([h(x).flatten() for h, x in zip(heads, states)]).mean(0).numpy()
    return mixture_median(ensemble_components(heads, states, routing, route_indices))


def oracle_diagnostic(components, legal, truth):
    """Outcome-aware bound over component locations OR legal output; never served."""
    choices = np.column_stack([legal, components[1].numpy()])
    nearest = np.abs(choices - np.asarray(truth)[:, None]).argmin(1)
    return choices[np.arange(len(choices)), nearest]


def routing_diagnostics(components):
    weights, loc, scale = [v.numpy() for v in components]
    # Collapse is measured within each seed, not the seven-seed mixture itself.
    per_seed = weights.reshape(len(weights), -1, 3) * (weights.shape[1] / 3)
    entropy = -(per_seed * np.log(np.maximum(per_seed, 1e-12))).sum(2)
    gap = np.ptp(loc.reshape(len(loc), -1, 3), axis=2)
    width = scale.reshape(len(scale), -1, 3).mean(2)
    return {"effective_components_per_seed": float(np.exp(entropy).mean()),
            "mean_max_route_probability": float(per_seed.max(2).mean()),
            "mean_component_usage": per_seed.mean((0, 1)).tolist(),
            "location_span_over_scale": float(np.mean(gap / width))}


def fit_curve(family, state, truth, seed, steps):
    torch.manual_seed(seed)
    head = MaskedReadout(family, state.shape[1], truth)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.003, weight_decay=.001)
    rng = np.random.default_rng(seed)
    snapshots = {}
    state, truth = state.detach(), truth.detach()
    for step in range(1, max(steps) + 1):
        rows = rng.integers(0, len(state), size=128)
        optimizer.zero_grad(set_to_none=True)
        loss = head.loss(state[rows], truth[rows])
        if not torch.isfinite(loss):
            raise ValueError("nonfinite reader loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
        optimizer.step()
        if step in steps:
            snapshots[step] = copy.deepcopy(head).eval()
    return snapshots


def fit_family(family, fit, tune, target, steps=STEPS, seeds=SEEDS):
    if set(fit["subjects"]) & set(tune["subjects"]):
        raise ValueError("fit/tune patient overlap")
    real, placebo = {s: [] for s in steps}, {s: [] for s in steps}
    tune_states = []
    for seed in seeds:
        draw = stable_hash("mixture_readout_fit", target, seed) % (2 ** 32)
        fi = donor_indices(fit["subjects"], stable_hash("fit_placebo", target, seed))
        ti = donor_indices(tune["subjects"], stable_hash("tune_placebo", target, seed))
        fitted = fit_curve(family, fit["latent"], fit["y"], draw, steps)
        control = fit_curve(family, fit["latent"][fi], fit["y"], draw, steps)
        tune_states.append(tune["latent"][ti])
        for step in steps:
            real[step].append(fitted[step])
            placebo[step].append(control[step])
    curve, pc = {}, {}
    for step in steps:
        curve[step] = selection_score(tune["y"], predict(real[step], [tune["latent"]] * len(seeds)), tune["units"])
        pc[step] = selection_score(tune["y"], predict(placebo[step], tune_states), tune["units"])
    chosen = min(steps, key=lambda s: (curve[s]["selection_score"], s))
    chosen_placebo = min(steps, key=lambda s: (pc[s]["selection_score"], s))
    return {"heads": real[chosen], "placebo_heads": placebo[chosen_placebo],
            "matched_placebo_heads": placebo[chosen]}, {
                "chosen_steps": chosen, "placebo_chosen_steps": chosen_placebo,
                "parameters_per_head": sum(p.numel() for p in real[chosen][0].parameters()),
                "curve": curve, "placebo_curve": pc}

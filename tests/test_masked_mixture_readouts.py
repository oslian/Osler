import inspect
import unittest

import numpy as np
import torch

from scripts.masked_mixture_readouts import (
    FAMILIES, MaskedReadout, ensemble_components, fit_family, mixture_median,
    oracle_diagnostic, predict, routing_diagnostics,
)


def test_real_parameter_matching_and_gradients():
    torch.manual_seed(73)
    counts = []
    x, y = torch.randn(24, 64), torch.randn(24)
    for family in FAMILIES:
        head = MaskedReadout(family, 64, y)
        counts.append(sum(p.numel() for p in head.parameters()))
        opt = torch.optim.Adam(head.parameters(), lr=.01)
        for _ in range(2):
            opt.zero_grad()
            loss = head.loss(x, y)
            assert torch.isfinite(loss)
            loss.backward()
            opt.step()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
        assert all(p.grad.abs().sum() > 0 for p in head.parameters())
    assert counts == [3565, 3553, 3561]
    assert max(counts) / min(counts) < 1.004


def test_median_not_component_mean():
    weights = torch.tensor([[1.], [1.]])
    loc, scale = torch.tensor([[0.], [3.]]), torch.ones(2, 1)
    np.testing.assert_allclose(mixture_median((weights, loc, scale)), [0, 3], atol=1e-6)
    weights, loc, scale = torch.tensor([[.9, .1]]), torch.tensor([[0., 100.]]), torch.ones(1, 2)
    result = mixture_median((weights, loc, scale))[0]
    assert abs(result - .117783) < 1e-4
    assert result < 1  # Component mean would be 10.


def test_shuffle_only_moves_routes_and_oracle_is_separate():
    torch.manual_seed(83)
    head = MaskedReadout("mixture_laplace", 64, torch.linspace(-2, 2, 20))
    torch.nn.init.normal_(head.net[-1].weight, std=.1)
    x = torch.randn(12, 64)
    ix = np.roll(np.arange(12), 1)
    legal = ensemble_components([head], [x])
    shuffled = ensemble_components([head], [x], "shuffled", [ix])
    torch.testing.assert_close(shuffled[0], legal[0][ix])
    torch.testing.assert_close(shuffled[1], legal[1])
    torch.testing.assert_close(shuffled[2], legal[2])
    assert not torch.equal(shuffled[0], legal[0])
    assert "truth" not in inspect.signature(predict).parameters
    p = predict([head], [x])
    y = np.linspace(-3, 3, 12)
    oracle = oracle_diagnostic(legal, p, y)
    assert np.all(np.abs(oracle - y) <= np.abs(p - y) + 1e-6)
    stats = routing_diagnostics(legal)
    assert 1 <= stats["effective_components_per_seed"] <= 3.000001
    with unittest.TestCase().assertRaises(ValueError):
        predict([head], [x], "oracle")


def test_scale_floor_zero_truth_and_extreme_nll():
    head = MaskedReadout("mixture_laplace", 64, torch.zeros(20))
    x = torch.zeros(10, 64)
    d = head.distribution(x)
    assert (d.component_distribution.scale >= .05).all()
    assert torch.isfinite(d.log_prob(torch.full((10,), 1e6))).all()
    np.testing.assert_allclose(predict([head], [x]), 0, atol=1e-6)


def test_fit_is_deterministic_no_context_update_and_independent_placebo():
    torch.set_num_threads(1)
    torch.manual_seed(19)
    x = torch.randn(80, 64)
    y = x[:, 0] + .1 * torch.randn(80)
    fit = {"latent": x[:40].clone(), "y": y[:40], "subjects": np.arange(40), "units": np.repeat("unit", 40)}
    tune = {"latent": x[40:].clone(), "y": y[40:], "subjects": np.arange(40, 80), "units": np.repeat("unit", 40)}
    before = fit["latent"].clone()
    a, ai = fit_family("mixture_laplace", fit, tune, "test", steps=(2, 4), seeds=(7, 19))
    b, bi = fit_family("mixture_laplace", fit, tune, "test", steps=(2, 4), seeds=(7, 19))
    assert ai == bi
    torch.testing.assert_close(before, fit["latent"])
    np.testing.assert_array_equal(predict(a["heads"], [tune["latent"]] * 2), predict(b["heads"], [tune["latent"]] * 2))
    assert a["heads"][0] is not a["placebo_heads"][0]
    with unittest.TestCase().assertRaisesRegex(ValueError, "overlap"):
        fit_family("mixture_laplace", fit, fit, "test", steps=(2,), seeds=(7,))


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(fn) for name, fn in sorted(globals().items())
                              if name.startswith("test_") and callable(fn))

"""Joint dependence layer over per-cell marginal forecasts (Gaussian copula, ensemble copula coupling).

Each cell keeps its own marginal predictive distribution (point + empirical
quantiles of its calibration errors).  Members of a joint ensemble are drawn
by mapping correlated latent normals through those marginals (ensemble
copula coupling with a parametric copula, Moller et al. 2013).  The latent
correlation comes from pairwise Spearman correlations of the cells' errors
(``2 sin(pi rho / 6)``), so cells that are rarely labelled together are
allowed; the matrix is repaired to a valid correlation matrix.  A Student-t
copula with the same correlation adds symmetric tail dependence: its degrees
of freedom are chosen by pairwise composite pseudo-likelihood with the
correlation fixed (Demarta and McNeil 2005).  Scores:
variogram score (Scheuerer and Hamill 2015) and energy score, both restricted
to the labelled components of each vector.  Research runtime only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.stats import norm, t as student_t


def latent_correlation(errors: pd.DataFrame, min_pairs: int) -> np.ndarray:
    """Gaussian-copula correlation of the columns of ``errors`` (NaN = not labelled); unsupported pairs get 0."""

    spearman = errors.corr(method="spearman", min_periods=min_pairs).to_numpy()
    latent = np.where(np.isfinite(spearman), 2.0 * np.sin(np.pi * spearman / 6.0), 0.0)
    np.fill_diagonal(latent, 1.0)
    return nearest_correlation(latent)


def nearest_correlation(matrix: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    """Clip eigenvalues at ``floor`` and rescale to a unit diagonal."""

    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2.0)
    repaired = (vectors * np.maximum(values, floor)) @ vectors.T
    scale = np.sqrt(np.diag(repaired))
    return repaired / np.outer(scale, scale)


def latent_draws(rows: int, members: int, correlation: np.ndarray, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """(independent, correlated) standard-normal draws of shape (rows, members, d) from the same base draws."""

    base = np.random.default_rng(seed).standard_normal((rows, members, len(correlation)))
    return base, base @ np.linalg.cholesky(correlation).T


def t_copula_log_density(x: np.ndarray, y: np.ndarray, rho: float, df: float) -> np.ndarray:
    """Bivariate t-copula log density at t(df) quantiles x, y."""

    one_minus = 1.0 - rho ** 2
    quadratic = (x ** 2 - 2.0 * rho * x * y + y ** 2) / one_minus
    joint = (gammaln((df + 2.0) / 2.0) - gammaln(df / 2.0) - np.log(df * np.pi) - 0.5 * np.log(one_minus)
             - (df + 2.0) / 2.0 * np.log1p(quadratic / df))
    return joint - student_t.logpdf(x, df) - student_t.logpdf(y, df)


def t_degrees_of_freedom(errors: pd.DataFrame, correlation: np.ndarray, min_pairs: int,
                         grid: tuple[float, ...]) -> tuple[float, dict[float, float]]:
    """Degrees of freedom maximizing the pairwise composite pseudo-likelihood over pairs with >= ``min_pairs``
    co-observed rows; pseudo-observations are each column's mid-ranks / (n + 1) over its observed rows."""

    pseudo = errors.rank(method="average").to_numpy() / (errors.notna().sum().to_numpy() + 1.0)
    observed = np.isfinite(pseudo)
    pairs = [(i, j, observed[:, i] & observed[:, j]) for i in range(pseudo.shape[1]) for j in range(i + 1, pseudo.shape[1])]
    pairs = [(i, j, both) for i, j, both in pairs if both.sum() >= min_pairs]
    profile = {}
    for df in grid:
        quantiles = student_t.ppf(pseudo, df)
        profile[df] = float(sum(np.sum(t_copula_log_density(quantiles[both, i], quantiles[both, j], correlation[i, j], df))
                                for i, j, both in pairs))
    return max(profile, key=profile.get), profile


def t_latent(correlated: np.ndarray, df: float, *, seed: int) -> np.ndarray:
    """Student-t copula draws on the normal scale: correlated normals / sqrt(W / df), W ~ chi2(df) per row and
    member (shared by the components), mapped through Phi^-1(T_df(.)) so ``marginal_members`` reads t levels."""

    mixing = np.random.default_rng(seed).chisquare(df, correlated.shape[:2]) / df
    values = correlated / np.sqrt(mixing)[..., None]
    return np.where(values < 0, norm.ppf(student_t.cdf(values, df)), -norm.ppf(student_t.sf(values, df)))


def marginal_members(point: np.ndarray, sorted_errors: list[np.ndarray], latent: np.ndarray) -> np.ndarray:
    """Members = point + empirical error quantile at Phi(latent), per component."""

    levels = norm.cdf(latent)
    output = np.empty_like(latent)
    for component, errors in enumerate(sorted_errors):
        positions = (np.arange(len(errors)) + 0.5) / len(errors)
        output[..., component] = point[:, None, component] + np.interp(levels[..., component], positions, errors)
    return output


def conditional_members(point: np.ndarray, scale: np.ndarray, groups: np.ndarray, sorted_scores: list[dict[int, np.ndarray]],
                        latent: np.ndarray) -> np.ndarray:
    """Members = point + scale * empirical quantile at Phi(latent) of the scores of the row's category, per component
    (Mondrian, scale-normalized marginal); ``sorted_scores[c]`` maps every category of component c to its sorted scores."""

    levels = norm.cdf(latent)
    output = np.empty_like(latent)
    for component, by_group in enumerate(sorted_scores):
        for group, scores in by_group.items():
            rows = groups[:, component] == group
            positions = (np.arange(len(scores)) + 0.5) / len(scores)
            output[rows, :, component] = (point[rows, None, component]
                                          + scale[rows, None, component] * np.interp(levels[rows, :, component], positions, scores))
    return output


def variogram_score(members: np.ndarray, observed: np.ndarray, labelled: np.ndarray, order: float = 0.5) -> np.ndarray:
    """Per row: sum over labelled pairs i < j of (|y_i - y_j|^p - E|X_i - X_j|^p)^2."""

    rows, _, dimension = members.shape
    score = np.zeros(rows)
    for i in range(dimension):
        for j in range(i + 1, dimension):
            both = labelled[:, i] & labelled[:, j]
            expected = np.mean(np.abs(members[:, :, i] - members[:, :, j]) ** order, axis=1)
            score += np.where(both, (np.abs(observed[:, i] - observed[:, j]) ** order - expected) ** 2, 0.0)
    return score


def energy_score(members: np.ndarray, observed: np.ndarray, labelled: np.ndarray, chunk: int = 2000) -> np.ndarray:
    """Per row: E||X - y|| - 0.5 E||X - X'|| over the labelled components (rows in chunks: the pair term is M x M)."""

    output = np.empty(len(members))
    for start in range(0, len(members), chunk):
        part, mask = members[start:start + chunk], labelled[start:start + chunk, None, :]
        spread = np.where(mask, part - np.nan_to_num(observed[start:start + chunk])[:, None, :], 0.0)
        centred = np.where(mask, part, 0.0)
        pairwise = centred[:, :, None, :] - centred[:, None, :, :]
        output[start:start + chunk] = (np.mean(np.sqrt(np.sum(spread ** 2, axis=2)), axis=1)
                                       - 0.5 * np.mean(np.sqrt(np.sum(pairwise ** 2, axis=3)), axis=(1, 2)))
    return output


def couple(sorted_values: np.ndarray, latent: np.ndarray) -> np.ndarray:
    """Ensemble copula coupling (Schefzik et al. 2013): give member k of each component the sorted marginal value
    whose rank equals the rank of member k's latent draw; the marginals are unchanged, only their pairing."""

    ranks = np.argsort(np.argsort(latent, axis=1, kind="stable"), axis=1, kind="stable")
    return np.take_along_axis(sorted_values, ranks, axis=1)

"""Policy estimators in outcome units per eligible customer."""

import numpy as np


def transformed_outcome(y, t, propensity=0.5):
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    if (
        y.ndim != 1
        or t.shape != y.shape
        or not np.isfinite(y).all()
        or not np.isin(t, [0, 1]).all()
        or not 0 < propensity < 1
    ):
        raise ValueError("Invalid outcomes, assignment or propensity")
    return y * (t / propensity - (1 - t) / (1 - propensity))


def aipw(y, t, mu0, mu1, p=0.5):
    y = np.asarray(y, float)
    t = np.asarray(t, float)
    transformed_outcome(y, t, p)
    mu0 = np.asarray(mu0, float)
    mu1 = np.asarray(mu1, float)
    if mu0.shape != y.shape or mu1.shape != y.shape or not np.isfinite(mu0 + mu1).all():
        raise ValueError("Invalid nuisance predictions")
    return mu1 - mu0 + t * (y - mu1) / p - (1 - t) * (y - mu0) / (1 - p)


def top_policy(scores, capacity, seed=71):
    scores = np.asarray(scores, float)
    if scores.ndim != 1 or not np.isfinite(scores).all() or not 0 <= capacity <= 1:
        raise ValueError("Invalid scores/capacity")
    rank = np.lexsort((np.random.default_rng(seed).random(len(scores)), -scores))
    mask = np.zeros(len(scores))
    mask[rank[: int(np.floor(len(scores) * capacity + 0.5))]] = 1
    return mask


def gain(scores, y, t, capacity):
    return float(np.mean(top_policy(scores, capacity) * transformed_outcome(y, t)))


def curve(scores, y, t):
    z = transformed_outcome(y, t)
    n = len(z)
    if n == 0:
        raise ValueError("Empty curve")
    rank = np.lexsort((np.random.default_rng(71).random(n), -np.asarray(scores)))
    q = np.arange(n + 1) / n
    g = np.r_[0, np.cumsum(z[rank])] / n
    # Right-continuous step gain G(q)=G(floor(n*q)/n), minus continuous random line.
    area = float(g[:-1].sum() / n - z.mean() / 2)
    return q, g, area


def bootstrap_policies(scores, t, signals, q=0.4, draws=2000, seed=82):
    """Arm-stratified paired bootstrap, fixed model scores, re-rank each replicate.
    Returns each policy's gain and differences versus expected random/response/heuristic.
    Conditional uncertainty excludes training variability and dependence between customers.
    """
    names = list(scores)
    n = len(t)
    signals = np.asarray(signals, float)
    if signals.ndim == 1:
        signals = signals[:, None]
    arrays = np.column_stack([scores[k] for k in names])
    t = np.asarray(t)
    groups = [np.flatnonzero(t == a) for a in (0, 1)]
    if any(len(g) == 0 for g in groups):
        raise ValueError("Both arms required")

    def estimate(idx):
        masks = np.column_stack(
            [top_policy(arrays[idx, j], q) for j in range(len(names))]
        )
        random = np.full((len(idx), 1), masks[:, 0].mean())
        masks = np.c_[masks, random, np.ones(len(idx)), np.zeros(len(idx))]
        return np.einsum("ij,ik->jk", masks, signals[idx]) / len(idx)

    observed = estimate(np.arange(n))
    out = np.empty((draws, *observed.shape))
    rng = np.random.default_rng(seed)
    for b in range(draws):
        out[b] = estimate(
            np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        )
    return names + ["Random", "Contact all", "Contact none"], observed, out

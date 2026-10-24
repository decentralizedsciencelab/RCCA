"""Endpoints, accepted-set bound, and multiplicity."""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple
import numpy as np
from scipy.stats import beta


def clopper_pearson_upper(k: int, n: int, alpha: float = 0.05) -> float:
    """One-sided (1-alpha) Clopper-Pearson upper bound U(k, n; alpha)."""
    if n <= 0:
        return 1.0
    if k >= n:
        return 1.0
    return float(beta.ppf(1 - alpha, k + 1, n - k))


def min_accepts_for_bound(target: float, k: int, alpha: float = 0.05, n_max: int = 5000) -> int:
    """Smallest N_A such that U(k, N_A; alpha) <= target."""
    for n in range(k + 1, n_max + 1):
        if clopper_pearson_upper(k, n, alpha) <= target:
            return n
    raise ValueError("not reachable within n_max")


def _check_strata(strata: np.ndarray) -> np.ndarray:
    """Every stratum needs >= 2 tasks to carry a variance. A singleton would
    otherwise contribute zero, understating the SE and making the gate and the
    primary test anti-conservative. Merge or drop it upstream instead."""
    labels = np.unique(strata)
    for s in labels:
        k = int(np.count_nonzero(strata == s))
        if k < 2:
            raise ValueError(
                f"stratum {s!r} has {k} task(s); merge it with another stratum or "
                "drop it before bootstrapping (a singleton stratum has no variance)")
    return labels


def _strat_resample(rng: np.random.Generator, strata: np.ndarray, B: int) -> np.ndarray:
    """B x n matrix of resampled indices, resampling tasks within each stratum.

    Each resampled column is written back to its OWN stratum's positions, so
    column j of the result always belongs to stratum `strata[j]`. `_t_stats`
    slices this matrix by the original index positions, so a contiguous-block
    layout would put foreign strata in every slice; with unequal stratum means
    that inflates the bootstrap SE, collapses the t quantiles toward zero and
    makes the studentized limit equal the point estimate.
    """
    n = strata.shape[0]
    out = np.empty((B, n), dtype=np.int64)
    for s in _check_strata(strata):
        idx = np.flatnonzero(strata == s)
        out[:, idx] = idx[rng.integers(0, idx.size, size=(B, idx.size))]
    return out


def _strat_se(x: np.ndarray, strata: np.ndarray) -> float:
    """Analytic SE of a stratified mean with proportional allocation."""
    n = x.size; v = 0.0
    for s in _check_strata(strata):
        xs = x[strata == s]; k = xs.size
        v += (k / n) ** 2 * xs.var(ddof=1) / k
    return math.sqrt(v)


@dataclass
class BootResult:
    estimate: float
    lower: float
    upper: float
    p_one_sided_less: float     # for H1: mean < 0
    p_one_sided_greater: float
    n_clusters: int
    B: int
    method: str


def _t_stats(x: np.ndarray, strata: np.ndarray, B: int, rng: np.random.Generator):
    n = x.size
    idx = _strat_resample(rng, strata, B)
    xb = x[idx]
    means = xb.mean(axis=1)
    se_b = np.zeros(B)
    for s in _check_strata(strata):
        cols = np.flatnonzero(strata == s); k = cols.size
        se_b += (k / n) ** 2 * xb[:, cols].var(axis=1, ddof=1) / k
    se_b = np.sqrt(se_b)
    t = (means - x.mean()) / np.where(se_b > 0, se_b, np.inf)
    return means, t


def cluster_bootstrap(task_values: np.ndarray, strata: Optional[np.ndarray] = None,
                      B: int = 2000, alpha: float = 0.05, seed: int = 0,
                      method: str = "t") -> BootResult:
    """Task-level cluster bootstrap within repository strata.

    `task_values` holds one number per task (already averaged over repetitions
    and authoring modes). `method` 't' gives the studentized (bootstrap-t)
    interval, the default; 'percentile' gives the percentile interval. The
    one-sided p-value is the smallest alpha at which the one-sided limit excludes 0.
    """
    x = np.asarray(task_values, float)
    n = x.size
    strata = np.zeros(n, dtype=int) if strata is None else np.asarray(strata)
    rng = np.random.default_rng(seed)
    est = float(x.mean())
    means, t = _t_stats(x, strata, B, rng)
    if method == "t":
        se0 = _strat_se(x, strata)
        lo = est - np.quantile(t, 1 - alpha / 2) * se0
        hi = est - np.quantile(t, alpha / 2) * se0
        # one-sided p for H1: mean < 0 is P*(est - t* se0 >= 0) = P*(t* <= est/se0)
        p_less = float(np.mean(t <= est / se0)) if se0 > 0 else float(est >= 0)
        p_greater = float(np.mean(t >= est / se0)) if se0 > 0 else float(est <= 0)
    else:
        lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
        p_less = float(np.mean(means >= 0.0))
        p_greater = float(np.mean(means <= 0.0))
    return BootResult(est, float(lo), float(hi), p_less, p_greater, n, B, method)


def one_sided_upper(task_values: np.ndarray, strata: Optional[np.ndarray], alpha: float,
                    B: int = 2000, seed: int = 0) -> float:
    """Percentile one-sided upper limit."""
    x = np.asarray(task_values, float)
    n = x.size
    strata = np.zeros(n, dtype=int) if strata is None else np.asarray(strata)
    rng = np.random.default_rng(seed)
    means = x[_strat_resample(rng, strata, B)].mean(axis=1)
    return float(np.quantile(means, 1 - alpha))


def one_sided_upper_t(task_values: np.ndarray, strata: Optional[np.ndarray], alpha: float,
                      B: int = 2000, seed: int = 0) -> float:
    """Studentized (bootstrap-t) one-sided upper limit for the stratified mean."""
    x = np.asarray(task_values, float)
    n = x.size
    strata = np.zeros(n, dtype=int) if strata is None else np.asarray(strata)
    rng = np.random.default_rng(seed)
    se0 = _strat_se(x, strata)
    if se0 == 0:
        return float(x.mean())
    _, t = _t_stats(x, strata, B, rng)
    return float(x.mean() - np.quantile(t, alpha) * se0)


@dataclass
class GateResult:
    delta_S_hat: float
    upper: float
    margin: float
    alpha_G: float
    passed: bool
    method: str


def noninferiority_gate(delta_S_tasks: np.ndarray, m_S: float, alpha_G: float = 0.05,
                        strata: Optional[np.ndarray] = None, B: int = 2000, seed: int = 0,
                        method: str = "t") -> GateResult:
    """H0: Delta_S >= m_S vs H1: Delta_S < m_S; passes if the one-sided upper
    (1-alpha_G) limit for Delta_S = S_CI - S_RCCA lies below m_S.
    `method` 't' (default) uses the studentized bootstrap; the percentile
    interval was anti-conservative in the operating-characteristic simulation."""
    f = one_sided_upper_t if method == "t" else one_sided_upper
    up = f(delta_S_tasks, strata, alpha_G, B, seed)
    return GateResult(float(np.mean(delta_S_tasks)), up, m_S, alpha_G, up < m_S, method)


def holm(pvals: Sequence[float], alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Holm step-down: returns (adjusted p-values, reject flags)."""
    p = np.asarray(pvals, float)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj, adj <= alpha


def csu(accepted: np.ndarray, unsafe: np.ndarray) -> Dict[str, float]:
    """C, S and U from per-candidate accept flags and hidden unsafe labels."""
    accepted = np.asarray(accepted, bool); unsafe = np.asarray(unsafe, bool)
    N = accepted.size
    NS, NU = int((~unsafe).sum()), int(unsafe.sum())
    AS, AU = int((accepted & ~unsafe).sum()), int((accepted & unsafe).sum())
    return {"C": (AS + AU) / N, "S": AS / NS if NS else float("nan"),
            "U": AU / NU if NU else float("nan"), "N": N, "N_S": NS, "N_U": NU}

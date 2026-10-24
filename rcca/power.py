"""Power and joint operating-characteristic simulation for the noninferiority gate.

Primary endpoint: task-level paired RCCA - CI difference in severity-weighted
unsafe automatic acceptance L_i, averaged over repetitions, then over the two
authoring modes within a task, then over test tasks, tested one-sided with a
task-level cluster bootstrap within repository strata. The safe-change
noninferiority gate on Delta_S = S_CI - S_RCCA precedes it in a fixed sequence.

Clustering: a task random effect u_t ~ N(0, sigma^2) on the logit scale shared
by both policies and all candidates of the task (the paired design cancels it);
ICC = sigma^2/(sigma^2 + pi^2/3). Marginal acceptance rates are matched
exactly by solving for the intercept.

Usage: python -m rcca.power [n_sim] [B] [t|percentile] [out.json] [n_repos] [u_ci]
"""
from __future__ import annotations
import json, math, sys, time
from dataclasses import dataclass, asdict
from typing import Dict, List
import numpy as np
from scipy.optimize import brentq
from scipy.special import expit

from .stats import one_sided_upper, one_sided_upper_t


@dataclass
class Design:
    n_tasks: int = 90
    n_repos: int = 4
    modes: int = 2
    reps: int = 3
    attrition: float = 0.10
    alpha: float = 0.05
    alpha_G: float = 0.05
    B: int = 1000
    n_sim: int = 1000
    severity_beta: tuple = (2.0, 2.0)   # w_i ~ Beta(2,2), mean 0.5
    method: str = "t"                    # 't' (studentized) or 'percentile'
    u_ci: float = 0.29                   # measured standard-CI miss rate


def _sigma_from_icc(icc: float) -> float:
    if icc <= 0:
        return 0.0
    return math.sqrt(icc * (math.pi ** 2 / 3) / (1 - icc))


_GH_X, _GH_W = np.polynomial.hermite.hermgauss(60)


def _marginal(beta0: float, sigma: float) -> float:
    z = math.sqrt(2) * _GH_X
    return float(np.sum(_GH_W * expit(beta0 + sigma * z)) / math.sqrt(math.pi))


def _intercept(target: float, sigma: float) -> float:
    if target <= 0: return -40.0
    if target >= 1: return 40.0
    return brentq(lambda b: _marginal(b, sigma) - target, -40, 40)


def _upper(x: np.ndarray, strata: np.ndarray, alpha: float, d: Design, rng: np.random.Generator) -> float:
    seed = int(rng.integers(0, 2**31 - 1))
    if d.method == "t":
        return one_sided_upper_t(x, strata, alpha, d.B, seed)
    return one_sided_upper(x, strata, alpha, d.B, seed)


def _task_strata(n_tasks: int, n_repos: int) -> np.ndarray:
    return np.arange(n_tasks) % n_repos


def simulate_primary(U_CI: float, U_RCCA: float, icc: float, d: Design, weighted: bool,
                     seed: int = 0) -> Dict[str, float]:
    """Rejection rate of H0 (no reduction) at one-sided alpha: power when
    U_RCCA < U_CI, type-I error when U_RCCA == U_CI."""
    rng = np.random.default_rng(seed)
    sigma = _sigma_from_icc(icc)
    b_ci, b_r = _intercept(U_CI, sigma), _intercept(U_RCCA, sigma)
    strata_all = _task_strata(d.n_tasks, d.n_repos)
    n_keep = int(round(d.n_tasks * (1 - d.attrition)))
    rejects = 0
    ests = []
    for _ in range(d.n_sim):
        keep = np.sort(rng.choice(d.n_tasks, n_keep, replace=False))
        strata = strata_all[keep]
        u = rng.normal(0, sigma, size=n_keep)[:, None, None]
        shape = (n_keep, d.modes, d.reps)
        a_ci = rng.random(shape) < expit(b_ci + u)
        a_r = rng.random(shape) < expit(b_r + u)
        if weighted:
            w = rng.beta(*d.severity_beta, size=(n_keep, d.modes, 1))   # adjudicated per defect
        else:
            w = np.ones((n_keep, d.modes, 1))
        L_ci = (w * a_ci).mean(axis=2).mean(axis=1)                    # reps, then modes
        L_r = (w * a_r).mean(axis=2).mean(axis=1)
        diff = L_r - L_ci
        rejects += int(_upper(diff, strata, d.alpha, d, rng) < 0)
        ests.append(diff.mean())
    return {"U_CI": U_CI, "U_RCCA": U_RCCA, "icc": icc, "weighted": weighted,
            "tasks_analyzed": n_keep, "reject_rate": rejects / d.n_sim,
            "mean_estimate": float(np.mean(ests)), "n_sim": d.n_sim, "method": d.method}


def simulate_joint(U_CI: float, U_RCCA: float, S_CI: float, delta_S_true: float, m_S: float,
                   icc: float, d: Design, weighted: bool = True, seed: int = 0) -> Dict[str, float]:
    """Joint operating characteristics of gate + primary in fixed sequence."""
    rng = np.random.default_rng(seed)
    sigma = _sigma_from_icc(icc)
    b_ci, b_r = _intercept(U_CI, sigma), _intercept(U_RCCA, sigma)
    s_ci, s_r = _intercept(S_CI, sigma), _intercept(S_CI - delta_S_true, sigma)
    strata_all = _task_strata(d.n_tasks, d.n_repos)
    n_keep = int(round(d.n_tasks * (1 - d.attrition)))
    gate_pass = prim_rej = both = 0
    for _ in range(d.n_sim):
        keep = np.sort(rng.choice(d.n_tasks, n_keep, replace=False))
        strata = strata_all[keep]
        u = rng.normal(0, sigma, size=n_keep)[:, None, None]
        v = rng.normal(0, sigma, size=n_keep)[:, None, None]
        shape = (n_keep, d.modes, d.reps)
        acc_ci_s = rng.random(shape) < expit(s_ci + v)
        acc_r_s = rng.random(shape) < expit(s_r + v)
        dS = (acc_ci_s.mean(axis=(1, 2)) - acc_r_s.mean(axis=(1, 2)))
        g = _upper(dS, strata, d.alpha_G, d, rng) < m_S
        a_ci = rng.random(shape) < expit(b_ci + u)
        a_r = rng.random(shape) < expit(b_r + u)
        w = rng.beta(*d.severity_beta, size=(n_keep, d.modes, 1)) if weighted else np.ones((n_keep, d.modes, 1))
        diff = (w * a_r).mean(axis=2).mean(axis=1) - (w * a_ci).mean(axis=2).mean(axis=1)
        p = _upper(diff, strata, d.alpha, d, rng) < 0
        gate_pass += int(g); prim_rej += int(p); both += int(g and p)
    return {"U_CI": U_CI, "U_RCCA": U_RCCA, "S_CI": S_CI, "delta_S_true": delta_S_true,
            "m_S": m_S, "icc": icc, "weighted": weighted, "tasks_analyzed": n_keep,
            "gate_pass_rate": gate_pass / d.n_sim, "primary_reject_rate": prim_rej / d.n_sim,
            "joint_pass_rate": both / d.n_sim,
            "conditional_power": (both / gate_pass) if gate_pass else float("nan"),
            "n_sim": d.n_sim, "method": d.method}


def main(out_path: str, n_sim: int = 1000, B: int = 1000, method: str = "t",
         n_repos: int = 4, u_ci: float = 0.29) -> dict:
    d = Design(n_sim=n_sim, B=B, method=method, n_repos=n_repos, u_ci=u_ci)
    t0 = time.time()
    res: Dict[str, List[dict]] = {"design": asdict(d), "primary": [], "type1": [], "gate": [], "joint": []}
    # Primary power: design assumption (0.40 -> 0.25); the measured baseline `u_ci` at the same
    # 37.5% relative reduction and at the same 0.15 absolute reduction; and the mutation-only
    # seeding floor (0.174 -> 0.109). The baseline is a parameter, not a constant: the probe
    # measured 0.29 under its own incidental coverage mix of 199 covered to 101 uncovered sites,
    # while the frozen covered_share = 0.50 implies 0.31 (docs/findings/2026-09-22-u-ci-probe.md).
    # 0.40 and 0.174 are the probe's two pure arms and do not move with the baseline.
    rel, ab = round(u_ci * 0.625, 2), round(u_ci - 0.15, 2)
    scen = [(0.40, 0.25), (u_ci, rel), (u_ci, ab), (0.174, 0.109)]
    k = 0
    for U_CI, U_R in scen:
        for icc in (0.05, 0.50):
            for weighted in (False, True):
                res["primary"].append(simulate_primary(U_CI, U_R, icc, d, weighted, seed=1000 + k)); k += 1
    for U in (0.40, u_ci):
        for icc in (0.05, 0.50):
            res["type1"].append(simulate_primary(U, U, icc, d, True, seed=2000 + k)); k += 1
    S_CI = 0.98
    for m_S in (0.03, 0.05, 0.10):
        for icc in (0.05, 0.50):
            res["gate"].append(simulate_joint(u_ci, u_ci, S_CI, m_S, m_S, icc, d, True, seed=3000 + k)); k += 1
            for U_CI, U_R in ((0.40, 0.25), (u_ci, rel)):
                res["joint"].append(simulate_joint(U_CI, U_R, S_CI, 0.02, m_S, icc, d, True, seed=4000 + k)); k += 1
    res["elapsed_s"] = time.time() - t0
    with open(out_path, "w") as f:
        json.dump(res, f, indent=1)
    return res


def report(path: str) -> str:
    r = json.load(open(path))
    out = [f"method={r['design']['method']} n_sim={r['design']['n_sim']} B={r['design']['B']} elapsed={r['elapsed_s']:.0f}s",
           "PRIMARY POWER (U_CI->U_RCCA, icc, weighted): reject rate, mean estimate"]
    for x in r["primary"]:
        out.append(f"  {x['U_CI']:.3f}->{x['U_RCCA']:.3f} icc={x['icc']:.2f} w={int(x['weighted'])} power={x['reject_rate']:.3f} est={x['mean_estimate']:+.3f}")
    out.append("TYPE-I (no effect)")
    for x in r["type1"]:
        out.append(f"  U={x['U_CI']:.2f} icc={x['icc']:.2f} alpha_hat={x['reject_rate']:.4f}")
    out.append("GATE FALSE-PASS at boundary (true Delta_S = m_S)")
    for x in r["gate"]:
        out.append(f"  m_S={x['m_S']:.2f} icc={x['icc']:.2f} false_pass={x['gate_pass_rate']:.4f}")
    out.append("JOINT OC under H1 (S_CI=0.98, true Delta_S=0.02)")
    for x in r["joint"]:
        out.append(f"  {x['U_CI']:.2f}->{x['U_RCCA']:.2f} m_S={x['m_S']:.2f} icc={x['icc']:.2f} gate={x['gate_pass_rate']:.3f} primary={x['primary_reject_rate']:.3f} joint={x['joint_pass_rate']:.3f} cond={x['conditional_power']:.3f}")
    return "\n".join(out)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    B = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    method = sys.argv[3] if len(sys.argv) > 3 else "t"
    n_repos = int(sys.argv[5]) if len(sys.argv) > 5 else 4
    u_ci = float(sys.argv[6]) if len(sys.argv) > 6 else 0.29
    suffix = ("" if n_repos == 4 else f"_{n_repos}strata") + \
             ("" if u_ci == 0.29 else f"_u{u_ci:g}".replace(".", ""))
    out = sys.argv[4] if len(sys.argv) > 4 else f"analysis/power_results_{method}{suffix}.json"
    main(out, n_sim=n, B=B, method=method, n_repos=n_repos, u_ci=u_ci)
    print(report(out))

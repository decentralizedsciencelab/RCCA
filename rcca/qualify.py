"""Accrual and accepted-set qualification operating characteristics.

The protocol fixes M = 820 sampled calibration changes,
a target of N_A = 300 automatically accepted observations, and stopping that
depends only on counts: RCCA processes the stream in temporal order until the
300th acceptance or the 820th change, whichever comes first. The bound is the
one-sided 95 % Clopper-Pearson limit U(K_A, N_A; 0.05) against 1 %, 2 % and 5 %.

Two consequences follow before any run, and this module quantifies both.

1. **Accrual can stop short.** Because stopping is at the 300th acceptance,
   `N_A = min(300, Binomial(820, C))` where C is the policy's automatic-acceptance
   coverage on the stream. So the design carries an acceptance-coverage floor of
   300/820 = 0.366. Below it, accrual misses the target and the protocol's own
   rule makes the result a feasibility outcome -- however safe the policy is.

2. **A full accrual is not a qualification.** At N_A = 300 the 5 % target needs
   K_A <= 8, the 2 % target K_A <= 1 and the 1 % target K_A = 0, so the
   probability of qualifying is a function of the true theta that the design
   cannot control.

Usage: python -m rcca.qualify [n_sim] [out.json]
"""
from __future__ import annotations
import json, sys, time
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Sequence
import numpy as np

from .stats import clopper_pearson_upper


@dataclass
class Design:
    m_sampled: int = 820            # M, sampled calibration changes
    n_target: int = 300             # N_A target
    alpha: float = 0.05
    n_sim: int = 2000
    targets: Sequence[float] = (0.01, 0.02, 0.05)


def simulate(coverage: float, theta: float, d: Design, seed: int = 0) -> Dict[str, object]:
    """One (C, theta) cell: accrual, then the bound on what accrued."""
    rng = np.random.default_rng(seed)
    accepted = rng.binomial(d.m_sampled, coverage, size=d.n_sim)
    n_a = np.minimum(accepted, d.n_target)
    k_a = rng.binomial(n_a, theta)
    bounds = np.array([clopper_pearson_upper(int(k), int(n), d.alpha) if n else 1.0
                       for k, n in zip(k_a, n_a)])
    reached = n_a >= d.n_target
    return {
        "coverage": coverage, "theta": theta,
        "reach_rate": float(reached.mean()),
        "n_a_median": int(np.median(n_a)),
        "n_a_p05": int(np.quantile(n_a, 0.05)),
        "k_a_mean": float(k_a.mean()),
        "bound_median": float(np.median(bounds)),
        # qualification is the conjunction the protocol states: a full accrual AND
        # a bound at or under the target. A short accrual is a feasibility outcome.
        "qualify": {f"{t}": float((reached & (bounds <= t)).mean()) for t in d.targets},
        "bound_only": {f"{t}": float((bounds <= t).mean()) for t in d.targets},
        "n_sim": d.n_sim,
    }


def critical_coverage(d: Design) -> float:
    """The acceptance coverage below which the target cannot be reached in
    expectation. Nothing in the protocol states it; it is implied by M and N_A."""
    return d.n_target / d.m_sampled


def main(out_path: str, n_sim: int = 2000,
         coverages: Sequence[float] = (0.20, 0.30, 0.366, 0.45, 0.60, 0.80, 1.00),
         thetas: Sequence[float] = (0.0, 0.005, 0.01, 0.02, 0.03, 0.05)) -> dict:
    d = Design(n_sim=n_sim)
    t0 = time.time()
    cells: List[dict] = []
    k = 0
    for c in coverages:
        for th in thetas:
            cells.append(simulate(c, th, d, seed=9000 + k)); k += 1
    res = {"design": asdict(d), "critical_coverage": critical_coverage(d),
           "cells": cells, "elapsed_s": time.time() - t0}
    with open(out_path, "w") as f:
        json.dump(res, f, indent=1)
    return res


def report(path: str) -> str:
    r = json.load(open(path))
    d = r["design"]
    out = [f"M={d['m_sampled']} N_A target={d['n_target']} alpha={d['alpha']} "
           f"n_sim={d['n_sim']}  critical coverage={r['critical_coverage']:.3f}",
           "", "ACCRUAL (reach rate by acceptance coverage; theta-independent)"]
    seen = {}
    for c in r["cells"]:
        seen.setdefault(c["coverage"], c)
    for cov, c in sorted(seen.items()):
        out.append(f"  C={cov:.3f}  reach={c['reach_rate']:.3f}  N_A median={c['n_a_median']}  5th pct={c['n_a_p05']}")
    out.append("")
    out.append("QUALIFICATION (full accrual AND bound <= target)")
    out.append("  C      theta   reach   U median   q@1%   q@2%   q@5%")
    for c in r["cells"]:
        q = c["qualify"]
        out.append(f"  {c['coverage']:.3f}  {c['theta']:.3f}  {c['reach_rate']:.3f}   "
                   f"{c['bound_median']:.4f}   {q['0.01']:.3f}  {q['0.02']:.3f}  {q['0.05']:.3f}")
    return "\n".join(out)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    out = sys.argv[2] if len(sys.argv) > 2 else "analysis/qualify_results.json"
    main(out, n_sim=n)
    print(report(out))

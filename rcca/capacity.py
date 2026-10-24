"""Capacity, queueing delay and escalation demand.

Design-stage only, and it carries a negative result the protocol has to face
before capacity is run.

Service times come from the nominal wall-clock of the actions a policy acquires
(`rcca.matrix.default_action_profiles`). Arrivals are Poisson, so the mean wait
is Kingman's approximation for M/G/1,

    Wq = (rho / (1 - rho)) * ((1 + Cs^2) / 2) * E[S]

with rho = lambda * E[S] / servers. Escalations additionally consume a human
review queue, which is the capacity that actually runs out.

**The signed-area problem.** Under replay a policy's decisions are a function of
the evidence matrix alone, so its unsafe automatic acceptance `U` does not
depend on the arrival rate at all. Plotting `U` against utilization therefore
gives a vertical segment per policy, and the signed area between two of them
reduces to (U_a - U_b) times whatever wait span the axis happens to cover --
an arbitrary number. `frontier_is_vertical()` reports this directly.

Usage: python -m rcca.capacity [out.json]
"""
from __future__ import annotations
import json, sys
from typing import Dict, List, Sequence
import numpy as np

from .matrix import default_action_profiles

# Action sets a policy acquires, as far as the design fixes them. RCCA's set is
# tuned, so it is a range rather than a point and is not claimed here.
SETS = {
    "standard_ci": ("f1.repo_ci",),
    "challenger_only": ("ch.property", "ch.fuzz", "ch.differential", "ch.spec_guided"),
    "full_evidence": None,                    # every action
}


def service_times(ci_detect: float = 0.71) -> Dict[str, Dict[str, float]]:
    prof = {p.action.id: p.action for p in default_action_profiles(ci_detect)}
    out = {}
    for name, ids in SETS.items():
        acts = list(prof.values()) if ids is None else [prof[i] for i in ids]
        wall = sum(a.nominal_cost.wall_s for a in acts)
        out[name] = {"wall_s": wall, "ci_equiv": sum(a.nominal_cost.ci_equiv for a in acts),
                     "n_actions": len(acts)}
    return out


def kingman_wait(lam_per_h: float, mean_s: float, cs2: float = 1.0, servers: int = 1) -> float:
    """Mean queueing delay in minutes. Returns inf at or above saturation."""
    mu = servers * 3600.0 / mean_s                      # completions per hour
    rho = lam_per_h / mu
    if rho >= 1.0:
        return float("inf")
    wq_s = (rho / (1 - rho)) * ((1 + cs2) / 2) * mean_s
    return wq_s / 60.0


def sweep(lam_grid: Sequence[float], servers: int = 1, cs2: float = 1.0) -> List[dict]:
    st = service_times()
    rows = []
    for name, s in st.items():
        for lam in lam_grid:
            mu = servers * 3600.0 / s["wall_s"]
            rows.append({"policy": name, "arrivals_per_h": lam,
                         "utilization": lam / mu,
                         "mean_service_s": s["wall_s"],
                         "wait_min": kingman_wait(lam, s["wall_s"], cs2, servers),
                         "verify_ci_per_h": lam * s["ci_equiv"]})
    return rows


def escalation_demand(lam_grid: Sequence[float],
                      esc_rates: Sequence[float] = (0.10, 0.25, 0.50, 0.75)) -> List[dict]:
    """Reviewer-hours per hour at a given escalation rate and 12 min per review."""
    return [{"arrivals_per_h": lam, "esc_rate": e, "reviewer_hours_per_h": lam * e * 12 / 60.0}
            for e in esc_rates for lam in lam_grid]


def frontier_is_vertical() -> dict:
    """Under replay, U is a function of the evidence matrix, not of load."""
    return {"claim": "U does not depend on the arrival rate under replay",
            "consequence": "the U-against-wait frontier is a vertical segment per policy, "
                           "so the signed area between two policies is (U_a - U_b) times an "
                           "arbitrary wait span and is not a scale-free statistic",
            "fix_options": ["report Delta U and the wait separately",
                            "make acceptance load-dependent (a deadline rule), which the "
                            "protocol does not currently specify",
                            "drop the signed-area statistic"]}


def main(out_path: str = "analysis/capacity_results.json") -> dict:
    lam = [round(x, 2) for x in np.linspace(1, 120, 40)]
    res = {"note": "design-stage: service times are nominal action costs, not measured",
           "service_times": service_times(),
           "sweep": sweep(lam),
           "escalation": escalation_demand(lam),
           "signed_area": frontier_is_vertical()}
    with open(out_path, "w") as f:
        json.dump(res, f, indent=1)
    return res


if __name__ == "__main__":
    r = main(sys.argv[1] if len(sys.argv) > 1 else "analysis/capacity_results.json")
    for k, v in r["service_times"].items():
        cap = 3600.0 / v["wall_s"]
        print(f"  {k:16} {v['n_actions']:2} actions  E[S]={v['wall_s']:6.0f}s  "
              f"capacity={cap:6.1f} changes/h  {v['ci_equiv']:.2f} CI-equiv each")


# ---------------------------------------------------------------- from measurements

def service_times_measured(decisions, ceiling: str = "q60") -> Dict[str, dict]:
    """Per policy, from the replay's decisions at one ceiling (analysis/decisions.jsonl): the realized
    service time of a decision (the producers' wall-clock plus the policy's own overhead), its squared
    coefficient of variation for Kingman, and the escalation rate that feeds the human review queue."""
    from collections import defaultdict
    wall, esc = defaultdict(list), defaultdict(list)
    for r in decisions:
        if r.get("ceiling") != ceiling:
            continue
        wall[r["policy"]].append(float(r.get("wall_s", 0.0)) + float(r.get("overhead_s", 0.0)))
        esc[r["policy"]].append(1.0 if r.get("action") == "esc" else 0.0)
    out = {}
    for p, v in wall.items():
        a = np.array(v, float); m = float(a.mean())
        out[p] = {"wall_s": m, "cs2": float(a.var() / m ** 2) if m > 0 else 0.0, "p90_wall_s": float(np.quantile(a, 0.9)),
                  "esc_rate": float(np.mean(esc[p])), "n": int(len(v))}
    return out


def sweep_measured(st: Dict[str, dict], lam_grid: Sequence[float], servers: int = 1,
                   review_min: float = 12.0) -> List[dict]:
    rows = []
    for name, s in st.items():
        for lam in lam_grid:
            mu = servers * 3600.0 / s["wall_s"] if s["wall_s"] > 0 else float("inf")
            rows.append({"policy": name, "arrivals_per_h": float(lam), "utilization": float(lam / mu) if mu else 0.0,
                         "mean_service_s": s["wall_s"], "cs2": s["cs2"],
                         "wait_min": kingman_wait(lam, s["wall_s"], s["cs2"], servers) if s["wall_s"] > 0 else 0.0,
                         "esc_rate": s["esc_rate"], "reviewer_hours_per_h": float(lam * s["esc_rate"] * review_min / 60.0)})
    return rows


def main_measured(decisions_path, out_path, ceiling: str = "q60", lam_grid: Sequence[float] = None,
                  servers: int = 1) -> dict:
    decisions = [json.loads(l) for l in open(decisions_path) if l.strip()]
    lam = list(lam_grid) if lam_grid is not None else [round(x, 2) for x in np.linspace(1, 120, 40)]
    st = service_times_measured(decisions, ceiling)
    res = {"note": f"measured: service times and escalation rates from {decisions_path} at ceiling {ceiling}; "
                   "whatever stage that analysis ran at, this inherits",
           "ceiling": ceiling, "servers": servers, "service_times": st, "sweep": sweep_measured(st, lam, servers),
           "signed_area": frontier_is_vertical()}
    with open(out_path, "w") as f:
        json.dump(res, f, indent=1)
    return res

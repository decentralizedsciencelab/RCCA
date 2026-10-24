"""Which evidence actions carry information worth their cost.

Design-stage only. The likelihoods are the synthetic profiles of
`rcca.matrix.default_action_profiles`, not a measurement, so every number here
says what the *design* implies rather than what the producers do. The run
replaces these; until then this is what the analysis can be powered to see.

For each action, with outcome O in {pass, fail_validated, fail_unvalidated,
inconclusive} and Y the latent unsafety:

  detection   P(fail_validated | Y=1)          -- what rejection can rest on
  info        I(O; Y) at prior pi, in bits     -- the VoI quantity of Eq. gain
  cost        the action's nominal CI-equivalents
  yield       info / cost                      -- H2's "worth their cost"

Usage: python -m rcca.information [out.json] [prior]
"""
from __future__ import annotations
import json, sys
from typing import Dict, List
import numpy as np

from .matrix import default_action_profiles

OUTCOMES = ("pass", "fail_validated", "fail_unvalidated", "inconclusive")


def _h(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def mutual_information(p_u: np.ndarray, p_s: np.ndarray, prior: float) -> float:
    """I(O; Y) in bits at P(Y=1) = prior."""
    joint = np.vstack([prior * p_u, (1 - prior) * p_s])       # 2 x 4
    p_o = joint.sum(axis=0)
    h_y = _h(np.array([prior, 1 - prior]))
    h_y_given_o = 0.0
    for j, po in enumerate(p_o):
        if po <= 0:
            continue
        h_y_given_o += po * _h(joint[:, j] / po)
    return h_y - h_y_given_o


def analyse(prior: float = 0.5, ci_detect: float = 0.71) -> List[dict]:
    rows = []
    for pr in default_action_profiles(ci_detect):
        p_u = np.array(pr.p_o_given_unsafe, float)
        p_s = np.array(pr.p_o_given_safe, float)
        cost = pr.action.nominal_cost.ci_equiv
        info = mutual_information(p_u, p_s, prior)
        rows.append({
            "action_id": pr.action.id, "family": pr.action.family,
            "source": pr.action.source.value, "producer": pr.action.producer,
            "is_repo_ci": pr.action.is_repo_ci,
            "detection": float(p_u[1]),                       # P(fail_validated | unsafe)
            "false_validated": float(p_s[1]),                 # must be 0 by protocol
            "inconclusive_unsafe": float(p_u[3]),
            "info_bits": info,
            "ci_equiv": cost,
            "wall_s": pr.action.nominal_cost.wall_s,
            "tokens": pr.action.nominal_cost.tokens,
            "yield_bits_per_ci": (info / cost) if cost > 0 else float("inf"),
            "p_o_given_unsafe": list(map(float, p_u)),
            "p_o_given_safe": list(map(float, p_s)),
        })
    return rows


def main(out_path: str = "analysis/information_results.json", prior: float = 0.5) -> dict:
    rows = analyse(prior)
    res = {"prior": prior, "outcomes": list(OUTCOMES), "note":
           "design-stage: likelihoods are rcca.matrix.default_action_profiles, not measured",
           "actions": rows}
    with open(out_path, "w") as f:
        json.dump(res, f, indent=1)
    return res


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "analysis/information_results.json"
    prior = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    r = main(out, prior)
    for a in sorted(r["actions"], key=lambda x: -x["yield_bits_per_ci"]):
        print(f"  {a['action_id']:22} {a['family']:3} det={a['detection']:.2f} "
              f"info={a['info_bits']:.3f}b  ci={a['ci_equiv']:.3f}  yield={a['yield_bits_per_ci']:8.2f} b/CI")


# ---------------------------------------------------------------- from measurements

_FAMILY = {"f1": "F1", "f2": "F2", "f3": "F3", "f4": "F4", "ch": "CH"}


def analyse_measured(fit: dict, nominal: dict, prior: float = 0.5) -> List[dict]:
    """The same table from what a run measured: `fit["likelihood"][action] = [P(o|unsafe), P(o|safe)]`
    (rcca.fit.fit_likelihood, as analysis/fit.json writes it) and the nominal costs of
    analysis/nominal_costs.json. The design profiles answered what the analysis could be powered to see;
    this answers what the producers did on the corpus the fit saw."""
    rows = []
    for aid, (p_u, p_s) in sorted(fit["likelihood"].items()):
        p_u, p_s = np.array(p_u, float), np.array(p_s, float)
        nc = nominal.get(aid, {})
        cost = float(nc.get("ci_equiv", 0.0))
        info = mutual_information(p_u, p_s, prior)
        rows.append({"action_id": aid, "family": _FAMILY.get(aid.split(".")[0], "?"),
                     "is_repo_ci": aid == "f1.repo_ci",
                     "detection": float(p_u[1]), "false_validated": float(p_s[1]),
                     "inconclusive_unsafe": float(p_u[3]), "info_bits": float(info), "ci_equiv": cost,
                     "wall_s": float(nc.get("wall_s", 0.0)), "tokens": float(nc.get("tokens", 0.0)),
                     "yield_bits_per_ci": (float(info) / cost) if cost > 0 else float("inf"),
                     "p_o_given_unsafe": list(map(float, p_u)), "p_o_given_safe": list(map(float, p_s))})
    return rows


def main_measured(fit_path, nominal_path, out_path, prior: float = 0.5) -> dict:
    fit = json.load(open(fit_path)); nominal = json.load(open(nominal_path))
    res = {"prior": prior, "outcomes": list(OUTCOMES),
           "note": f"measured: likelihoods from {fit_path} (rcca.fit on the fit partition), nominal costs from {nominal_path}; "
                   "whatever stage that analysis ran at, this inherits",
           "actions": analyse_measured(fit, nominal, prior)}
    with open(out_path, "w") as f:
        json.dump(res, f, indent=1)
    return res

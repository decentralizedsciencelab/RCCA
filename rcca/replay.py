"""Replay one evidence matrix per (candidate, rep) to the nine policies and
compute the headline quantities. Policies never see hidden labels; the
evaluator reads them only after the decision is returned."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Sequence
import numpy as np

from .decision import LossParams, EvidenceModel, Step, run_policy, POLICIES
from .matrix import Action, Ceilings, EvidenceMatrix, cell_seed
from .stats import cluster_bootstrap, noninferiority_gate, csu


@dataclass
class Row:
    """One decision. The cost columns are realized (what the acquired records cost); `overhead_s` is the
    policy's own wall-clock, the only field that differs between two runs of one replay; `stop_reason`,
    `overrun` and `trace` (one Step per acquisition) are the decision's own (rcca.decision)."""
    policy: str
    candidate_id: str
    task: str
    repo: str
    rep: int
    mode: str
    action: str
    unsafe: bool
    severity: float
    tokens: float
    tool_calls: float
    ci_equiv: float
    wall_s: float
    n_acquired: int
    overhead_s: float
    overrun: bool
    stop_reason: str
    trace: List[Step]


def mode_of(candidate_id: str) -> str:
    """The authoring mode of a candidate id `<task>:<s|u>:<mode>[:a<attempt>]` (corpus.manifest): its third
    field, so a regenerated candidate keeps its mode; "m0" for an id without one (synthetic matrices)."""
    parts = candidate_id.split(":")
    return parts[2] if len(parts) > 2 else "m0"


def replay(matrices: Sequence[EvidenceMatrix], actions: Sequence[Action], lp: LossParams,
           model: EvidenceModel, B: Ceilings, seed: int = 0,
           policies: Sequence[str] = POLICIES) -> List[Row]:
    rows: List[Row] = []
    for m in matrices:
        for p in policies:
            # cell_seed, not builtin hash(): hash() of a tuple holding strings is
            # salted per process, which made the `random` policy's order -- and its
            # decisions -- depend on PYTHONHASHSEED (spec §5, common random numbers).
            rng = np.random.default_rng(cell_seed(str(seed), m.candidate_id, m.rep, p))
            dcn = run_policy(p, m, actions, lp, model, B, rng)
            rows.append(Row(p, m.candidate_id, m.task or m.candidate_id, m.repo, m.rep, mode_of(m.candidate_id), dcn.action,
                            bool(m.truth_unsafe), float(m.truth_severity or 0.0),
                            dcn.cost.tokens, dcn.cost.tool_calls, dcn.cost.ci_equiv, dcn.cost.wall_s,
                            len(dcn.acquired), dcn.overhead_s, dcn.overrun, dcn.stop_reason, dcn.trace))
    return rows


def summarize(rows: List[Row], policy: str) -> Dict[str, float]:
    r = [x for x in rows if x.policy == policy]
    acc = np.array([x.action == "acc" for x in r]); uns = np.array([x.unsafe for x in r])
    esc = np.array([x.action == "esc" for x in r]); rej = np.array([x.action == "rej" for x in r])
    out = csu(acc, uns)
    L = np.array([x.severity for x in r]) * acc * uns
    out["L"] = float(L[uns].mean()) if uns.any() else float("nan")
    out["esc_S"] = float(esc[~uns].mean()); out["esc_U"] = float(esc[uns].mean())
    out["rej_S"] = float(rej[~uns].mean()); out["rej_U"] = float(rej[uns].mean())
    for k in ("tokens", "tool_calls", "ci_equiv", "wall_s"):
        v = np.array([getattr(x, k) for x in r])
        out[f"{k}_median"] = float(np.median(v)); out[f"{k}_p90"] = float(np.quantile(v, 0.9))
    return out


QUANTITIES = ("S", "L", "U")


def task_level_value(rows: List[Row], policy: str, quantity: str) -> Dict[str, np.ndarray]:
    """Per-task level of one quantity for ONE policy, the level counterpart of
    `task_level_diff`: averaged over repetitions, then authoring modes, then one
    value per task with its repository stratum.

    `S` and `U` are the automatic-acceptance rate among the task's safe and
    unsafe candidates; `L` is severity-weighted unsafe automatic acceptance.
    The table and the policy figure want the mean of these task-level values, not
    the pooled rate; under a balanced design the two coincide.
    """
    if quantity not in QUANTITIES:
        raise ValueError(f"quantity must be one of {QUANTITIES}, not {quantity!r}")
    want_unsafe = quantity in ("L", "U")
    tasks: List[str] = []
    strata: List[str] = []
    vals: List[float] = []
    for t in sorted({x.task for x in rows}):
        rt = [x for x in rows if x.task == t and x.policy == policy and x.unsafe == want_unsafe]
        if not rt:
            continue
        per_mode = []
        for md in sorted({x.mode for x in rt}):
            rr = [x for x in rt if x.mode == md]
            if quantity == "L":
                per_mode.append(np.mean([x.severity * (x.action == "acc") for x in rr]))
            else:
                per_mode.append(np.mean([float(x.action == "acc") for x in rr]))
        vals.append(float(np.mean(per_mode))); strata.append(rt[0].repo); tasks.append(t)
    return {"tasks": np.array(tasks), "value": np.array(vals), "strata": np.array(strata)}


def policy_intervals(rows: List[Row], policy: str, B: int = 2000, alpha: float = 0.05,
                     seed: int = 0, min_tasks: int = 10) -> Dict[str, float]:
    """Task-level mean and two-sided (1 - alpha) cluster-bootstrap interval for
    each of S, L and U, for one policy.

    Returns an empty dict when the bootstrap cannot carry the design: fewer than
    `min_tasks` clusters, or a repository stratum with a single task, which
    `stats._check_strata` rejects because it has no variance. An absent key means
    the figure draws no bar, never a bar of width zero.
    """
    out: Dict[str, float] = {}
    for q in QUANTITIES:
        tv = task_level_value(rows, policy, q)
        if tv["value"].size < min_tasks:
            return {}
        try:
            br = cluster_bootstrap(tv["value"], tv["strata"], B, alpha, seed)
        except ValueError:
            return {}
        out[f"{q}_est"] = br.estimate
        out[f"{q}_lo"] = br.lower
        out[f"{q}_hi"] = br.upper
    return out


def task_level_diff(rows: List[Row], pol_a: str, pol_b: str, quantity: str) -> Dict[str, np.ndarray]:
    """Paired task-level difference pol_a - pol_b of L (unsafe candidates) or S (safe
    candidates): averaged over repetitions, then modes, then returned per task
    with its repository stratum."""
    want_unsafe = quantity == "L"
    tasks = sorted({x.task for x in rows})
    strata, diffs = [], []
    for t in tasks:
        rt = [x for x in rows if x.task == t and x.unsafe == want_unsafe]
        if not rt:
            continue
        def val(p):
            modes = sorted({x.mode for x in rt})
            per_mode = []
            for md in modes:
                rr = [x for x in rt if x.policy == p and x.mode == md]
                if quantity == "L":
                    per_mode.append(np.mean([x.severity * (x.action == "acc") for x in rr]))
                else:
                    per_mode.append(np.mean([float(x.action == "acc") for x in rr]))
            return float(np.mean(per_mode))
        diffs.append(val(pol_a) - val(pol_b)); strata.append(rt[0].repo)
    return {"tasks": np.array(tasks[:len(diffs)]), "diff": np.array(diffs),
            "strata": np.array(strata)}


def primary_analysis(rows: List[Row], m_S: float, alpha_G: float = 0.05, B: int = 2000, seed: int = 0):
    """Fixed sequence: gate on S first, then the primary one-sided test on L."""
    dS = task_level_diff(rows, "standard_ci", "rcca", "S")      # Delta_S = S_CI - S_RCCA
    gate = noninferiority_gate(dS["diff"], m_S, alpha_G, dS["strata"], B, seed)
    dL = task_level_diff(rows, "rcca", "standard_ci", "L")       # RCCA - CI
    prim = cluster_bootstrap(dL["diff"], dL["strata"], B, 0.05, seed + 1)
    return gate, prim

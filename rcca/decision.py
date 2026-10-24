"""Decision formulation, acquisition rule,
stopping and terminal decisions, and the nine policies.

A policy sees only: risk features, each action's nominal cost, the evidence
records it has acquired, the ceilings, and the OOD flag. It never sees hidden
labels, and it never sees a record's realized cost before acquiring the record:
affordability, RCCA's net gain and cheapest-first's order use the nominal cost
(estimated on the training partition, rcca.analyze), and `_acquire` charges the
realized cost after the action has run. A realized cost that carries the spend
past a ceiling is an overrun: the record is kept, acquisition stops, the
terminal rule reads the state as it is, and the decision says so.
"""
from __future__ import annotations
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

from .matrix import (Action, Ceilings, Cost, EvidenceMatrix, Outcome, OUTCOMES, Source)

TERMINAL = ("acc", "rej", "esc")
# Why acquisition ended. Standard CI and risk-only have no stopping rule: they stop by construction.
STOP_REASONS = ("no_positive_gain", "no_affordable_action", "pool_exhausted", "overrun", "standard_ci", "risk_only")


@dataclass(frozen=True)
class LossParams:
    """Loss constants, kappa >> lambda_R > lambda_E, outcome-blind."""
    kappa: float
    lambda_R: float
    lambda_E: float
    tau_qual: float
    eta: float

    def __post_init__(self):
        assert self.kappa > self.lambda_R > self.lambda_E > 0, "require kappa >> lambda_R > lambda_E"


@dataclass
class EvidenceModel:
    """Coherent joint model p(Z, o | s, e) = p(Z | s) p(o | Z, s, e).

    p(Y=1 | r) is a logistic in the risk logit; p(o | Y, e) is a per-action
    multinomial likelihood (calibrated on training partitions in the real
    study; here supplied directly). Outcomes are conditionally independent
    given Y. Severity given Y=1 is a constant predicted severity s_tilde.
    """
    likelihood: Dict[str, Tuple[np.ndarray, np.ndarray]]  # action_id -> (p(o|Y=1), p(o|Y=0))
    s_tilde: float = 0.6
    calibration_shift: float = 0.0   # added to the risk logit (misspecification knob)

    def prior(self, m: EvidenceMatrix) -> float:
        if "rho0" in m.risk_features:
            r = float(m.risk_features["rho0"])
            logit = math.log(r / (1 - r)) + self.calibration_shift
        else:
            logit = float(m.risk_features["risk_logit"]) + self.calibration_shift
        return 1.0 / (1.0 + math.exp(-logit))

    def p_o(self, action_id: str, unsafe: bool) -> np.ndarray:
        a, b = self.likelihood[action_id]
        return a if unsafe else b


@dataclass
class State:
    rho0: float
    observed: Dict[str, Outcome] = field(default_factory=dict)
    spent: Cost = field(default_factory=Cost)
    ood: bool = False

    @property
    def validated_violation(self) -> bool:
        return any(o == Outcome.FAIL_VALIDATED for o in self.observed.values())


def posterior(model: EvidenceModel, s: State) -> float:
    """rho(s_t) = Pr[Y=1 | s_t] under conditional independence given Y."""
    l1 = math.log(s.rho0) - math.log1p(-s.rho0)
    for aid, o in s.observed.items():
        k = OUTCOMES.index(o)
        p1, p0 = model.p_o(aid, True)[k], model.p_o(aid, False)[k]
        l1 += math.log(max(p1, 1e-12)) - math.log(max(p0, 1e-12))
    return 1.0 / (1.0 + math.exp(-l1))


def feasible(s: State, rho: float, lp: LossParams) -> List[str]:
    """The feasible action set."""
    fs = ["esc"]
    if rho <= lp.tau_qual and not s.ood:
        fs.append("acc")
    if s.validated_violation:
        fs.append("rej")
    return fs


def expected_loss(a: str, rho: float, lp: LossParams, model: EvidenceModel) -> float:
    if a == "acc":
        return lp.kappa * model.s_tilde * rho
    if a == "rej":
        return lp.lambda_R * (1.0 - rho)
    return lp.lambda_E


def value(s: State, lp: LossParams, model: EvidenceModel) -> Tuple[float, str]:
    """Constrained Bayes risk V(s_t) and its minimizer; ties -> esc < rej < acc priority order."""
    rho = posterior(model, s)
    best, besta = math.inf, "esc"
    for a in ("esc", "rej", "acc"):
        if a in feasible(s, rho, lp):
            v = expected_loss(a, rho, lp, model)
            if v < best - 1e-15:
                best, besta = v, a
    return best, besta


def predictive(model: EvidenceModel, s: State, aid: str) -> np.ndarray:
    rho = posterior(model, s)
    return rho * model.p_o(aid, True) + (1 - rho) * model.p_o(aid, False)


def net_gain(s: State, a: Action, lp: LossParams, model: EvidenceModel, B: Ceilings) -> float:
    """Net gain: V(s) - sum_o p(o|s,e) V(s u {(e,o)}) - eta C_w(e), with C_w(e) priced at the action's
    nominal cost: its realized cost is unknown until it has run."""
    v0, _ = value(s, lp, model)
    pred = predictive(model, s, a.id)
    exp_v = 0.0
    for k, o in enumerate(OUTCOMES):
        if pred[k] <= 0:
            continue
        s2 = State(s.rho0, dict(s.observed, **{a.id: o}), s.spent, s.ood)
        exp_v += pred[k] * value(s2, lp, model)[0]
    return v0 - exp_v - lp.eta * B.normalized_cost(a.nominal_cost)


def entropy_reduction(s: State, aid: str, model: EvidenceModel) -> float:
    def H(p):
        p = min(max(p, 1e-12), 1 - 1e-12)
        return -(p * math.log(p) + (1 - p) * math.log(1 - p))
    h0 = H(posterior(model, s))
    pred = predictive(model, s, aid)
    h1 = 0.0
    for k, o in enumerate(OUTCOMES):
        if pred[k] <= 0:
            continue
        h1 += pred[k] * H(posterior(model, State(s.rho0, dict(s.observed, **{aid: o}), s.spent, s.ood)))
    return h0 - h1


@dataclass(frozen=True)
class Step:
    """One acquisition, as the acquisition analysis reads it: the action, the nominal cost it was
    selected on, the realized cost it was charged, the posterior before and after its outcome, the net
    gain that chose it (None for a policy that computes none), and the outcome."""
    action_id: str
    nominal: Cost
    realized: Cost
    rho_before: float
    rho_after: float
    gain: Optional[float]
    outcome: Outcome

    def as_dict(self) -> dict:
        """The step as analysis/traces.jsonl stores it."""
        return {"action_id": self.action_id, "nominal": self.nominal.as_dict(), "realized": self.realized.as_dict(),
                "rho_before": self.rho_before, "rho_after": self.rho_after, "gain": self.gain,
                "outcome": self.outcome.value}


@dataclass
class Decision:
    policy: str
    action: str                 # acc | rej | esc
    acquired: List[str]
    cost: Cost                  # realized: the sum of what the acquired records cost
    rho_final: float
    n_steps: int
    stop_reason: str            # one of STOP_REASONS
    overrun: bool               # the last record's realized cost carried the spend past a ceiling (standard
                                # CI too, whose stop_reason stays 'standard_ci': it stops after one action anyway)
    trace: List[Step]           # one Step per acquisition, in order
    overhead_s: float = 0.0     # wall-clock of the policy itself (run_policy): an outcome, never an input


def _available(s: State, m: EvidenceMatrix, actions: Sequence[Action]) -> List[Action]:
    """Actions not yet acquired that this matrix holds a record for."""
    return [a for a in actions if a.id not in s.observed and a.id in m.records]


def _affordable(B: Ceilings, s: State, m: EvidenceMatrix, actions: Sequence[Action]) -> List[Action]:
    """Affordability, judged before the action runs: the spend so far plus the nominal cost fits every
    ceiling. The record's realized cost is not read here; it is known only once the action has run."""
    return [a for a in _available(s, m, actions) if B.affordable(s.spent, a.nominal_cost)]


def _acquire(s: State, m: EvidenceMatrix, a: Action, model: EvidenceModel, gain: Optional[float] = None) -> Step:
    """Run `a`: its outcome joins the state and its realized cost is charged. The only place a policy reads
    a record's realized cost, after it has chosen the action."""
    rho_before = posterior(model, s)
    rec = m.records[a.id]
    s.observed[a.id] = rec.outcome
    s.spent = s.spent + rec.cost
    return Step(a.id, a.nominal_cost, rec.cost, rho_before, posterior(model, s), gain, rec.outcome)


def _terminal(s: State, lp: LossParams, model: EvidenceModel) -> Tuple[str, float]:
    """Minimum-expected-loss feasible action."""
    v, a = value(s, lp, model)
    return a, posterior(model, s)


# ----------------------------------------------------------------------------
# Acquisition orders for policies 3-9.
# ----------------------------------------------------------------------------

def _by_priority(acts: List[Action]) -> List[Action]:
    return sorted(acts, key=lambda a: (a.priority, a.id))


def order_fixed_funnel(actions: Sequence[Action]) -> List[Action]:
    """RADAR-style funnel: static (F2) -> tests (F1) -> LLM review (challenger
    spec-guided synthesis) -> deterministic checks (F3, F4). Fixed order."""
    stage = {"F2": 0, "F1": 1}
    def key(a: Action):
        if a.source == Source.CHALLENGER and a.producer.startswith("llm"):
            st = 2
        elif a.family in ("F3", "F4"):
            st = 3
        elif a.source == Source.CHALLENGER:
            st = 3
        else:
            st = stage.get(a.family, 3)
        return (st, a.priority, a.id)
    return sorted(actions, key=key)


def run_policy(name: str, m: EvidenceMatrix, actions: Sequence[Action], lp: LossParams,
               model: EvidenceModel, B: Ceilings, rng: Optional[np.random.Generator] = None) -> Decision:
    """The decision of policy `name` on one matrix. `overhead_s` is the wall-clock this call took, the
    policy's own service time for the capacity analysis (so risk-only is not free); it is measured around the decision and
    never read by it, so it is the one part of a Decision that differs between two identical runs."""
    t0 = time.perf_counter()
    d = _decide(name, m, actions, lp, model, B, rng)
    d.overhead_s = time.perf_counter() - t0
    return d


def _decide(name: str, m: EvidenceMatrix, actions: Sequence[Action], lp: LossParams,
            model: EvidenceModel, B: Ceilings, rng: Optional[np.random.Generator]) -> Decision:
    s = State(model.prior(m), ood=m.ood)
    trace: List[Step] = []

    def decision(act: str, rho: float, stop: str) -> Decision:
        return Decision(name, act, [st.action_id for st in trace], s.spent, rho, len(trace), stop,
                        not B.within(s.spent), trace)

    if name == "standard_ci":
        ci = [a for a in actions if a.is_repo_ci]
        assert len(ci) == 1, "exactly one repository CI action"
        a = ci[0]
        if B.affordable(s.spent, a.nominal_cost):
            trace.append(_acquire(s, m, a, model))         # kept even if its realized cost overruns
            act = "acc" if s.observed[a.id] == Outcome.PASS else "rej"   # accepts iff suite passes; never escalates
        else:
            act = "rej"
        return decision(act, posterior(model, s), "standard_ci")

    if name == "risk_only":
        act, rho = _terminal(s, lp, model)
        return decision(act, rho, "risk_only")

    pool = list(actions)
    if name == "challenger_only":
        pool = [a for a in actions if a.source == Source.CHALLENGER]
    if name == "fixed_funnel":
        pool = [a for a in actions if not (a.source == Source.CHALLENGER and not a.producer.startswith("llm"))]
        pool = order_fixed_funnel(pool)
    if name == "random":
        assert rng is not None
        pool = list(rng.permutation(np.array(pool, dtype=object)))

    while True:
        if not _available(s, m, pool):
            stop = "pool_exhausted"
            break
        aff = _affordable(B, s, m, pool)
        if not aff:
            stop = "no_affordable_action"
            break
        g = None
        if name == "rcca":
            gains = [(net_gain(s, a, lp, model, B), -a.priority, a.id, a) for a in aff]
            g, _, _, a = max(gains, key=lambda t: (t[0], t[1], t[2]))
            if g <= 0:
                stop = "no_positive_gain"
                break
        elif name == "uncertainty":
            a = max(aff, key=lambda a: (entropy_reduction(s, a.id, model), -a.priority))
        elif name == "cheapest_first":
            a = min(aff, key=lambda a: (B.normalized_cost(a.nominal_cost), a.priority))
        elif name in ("full_evidence", "challenger_only"):
            a = _by_priority(aff)[0]
        elif name in ("fixed_funnel", "random"):
            a = aff[0]   # pool already ordered
        else:
            raise ValueError(name)
        trace.append(_acquire(s, m, a, model, g))
        if not B.within(s.spent):
            # Review Focus 2: the record was bought and is kept; acquisition stops and the terminal rule
            # reads the state as it is. Never re-labelled unaffordable after the fact.
            stop = "overrun"
            break

    act, rho = _terminal(s, lp, model)
    return decision(act, rho, stop)


POLICIES: Tuple[str, ...] = ("standard_ci", "risk_only", "fixed_funnel", "full_evidence",
                             "random", "cheapest_first", "uncertainty", "challenger_only", "rcca")

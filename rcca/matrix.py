"""Evidence-matrix schema (spec §4, §5, §7, §10) and a synthetic generator for tests.

Outcome alphabet has four symbols: the `fail` of the gain rule is split into
fail_validated / fail_unvalidated so that "a validated violation exists" in
Feasibility is a checkable fact. Cost is metered in raw units; compute is
expressed in standard-CI-equivalents so that standard CI costs 1.0.
Ceilings bound tokens, tool calls and CI-equivalents only: wall-clock time is
metered and reported as an outcome, never bounded, because a ceiling in
absolute seconds decides differently on a faster machine (results.md E2).
"""
from __future__ import annotations
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np


class Outcome(str, Enum):
    PASS = "pass"
    FAIL_VALIDATED = "fail_validated"
    FAIL_UNVALIDATED = "fail_unvalidated"
    INCONCLUSIVE = "inconclusive"


OUTCOMES: Tuple[Outcome, ...] = (Outcome.PASS, Outcome.FAIL_VALIDATED,
                                 Outcome.FAIL_UNVALIDATED, Outcome.INCONCLUSIVE)


class Source(str, Enum):
    AUTHOR_SIDE = "author_side"
    CHALLENGER = "challenger"


# Metered cost dimensions, J = 4 (spec §7): every record carries all four.
COST_DIMS: Tuple[str, ...] = ("tokens", "tool_calls", "ci_equiv", "wall_s")
# Ceiling dimensions: wall-clock is an outcome, not a ceiling.
CEILING_DIMS: Tuple[str, ...] = ("tokens", "tool_calls", "ci_equiv")


@dataclass(frozen=True)
class Cost:
    tokens: float = 0.0
    tool_calls: float = 0.0
    ci_equiv: float = 0.0
    wall_s: float = 0.0

    def vec(self) -> np.ndarray:
        return np.array([self.tokens, self.tool_calls, self.ci_equiv, self.wall_s], float)

    def ceiling_vec(self) -> np.ndarray:
        """The dimensions a ceiling bounds (CEILING_DIMS); wall_s is recorded, never bounded."""
        return np.array([self.tokens, self.tool_calls, self.ci_equiv], float)

    def as_dict(self) -> Dict[str, float]:
        """The four metered dimensions by name (COST_DIMS), as plain floats: the JSON shape of a cost."""
        return {d: float(getattr(self, d)) for d in COST_DIMS}

    def __add__(self, o: "Cost") -> "Cost":
        return Cost(*(self.vec() + o.vec()))


@dataclass(frozen=True)
class Ceilings:
    """Per-change ceilings B on tokens, tool calls and CI-equivalents, and the
    weights w_j of C_w (equal by default). Wall-clock time is not a ceiling: a ceiling in absolute
    seconds makes decisions depend on the machine even when every evidence record agrees (results.md E2),
    so wall_s stays an outcome, metered in every Cost and summarized, and never decides affordability."""
    tokens: float
    tool_calls: float
    ci_equiv: float
    weights: Tuple[float, float, float] = field(default=(1 / 3, 1 / 3, 1 / 3), kw_only=True)

    def vec(self) -> np.ndarray:
        return np.array([self.tokens, self.tool_calls, self.ci_equiv], float)

    def normalized_cost(self, c: Cost) -> float:
        """C_w(e) = sum_j w_j c_j(e) / B_j over the ceiling dimensions."""
        return float(np.dot(np.asarray(self.weights), c.ceiling_vec() / self.vec()))

    def affordable(self, spent: Cost, c: Cost) -> bool:
        return bool(np.all(spent.ceiling_vec() + c.ceiling_vec() <= self.vec() + 1e-12))

    def within(self, spent: Cost) -> bool:
        """`spent` is within every ceiling. A policy selects on nominal costs, so a realized cost above its
        nominal can carry the spend past a ceiling: an overrun (rcca.decision)."""
        return bool(np.all(spent.ceiling_vec() <= self.vec() + 1e-12))


@dataclass(frozen=True)
class Action:
    """Unit of policy choice. A family is an attribute, not the unit (spec §4)."""
    id: str
    family: str              # F1..F4
    source: Source
    producer: str
    nominal_cost: Cost       # what selection prices it at; its records carry the realized cost charged
    priority: int = 100      # pre-specified tie-break; lower runs first
    is_repo_ci: bool = False # the repository's own build/lint/test gate


@dataclass
class EvidenceRecord:
    action_id: str
    outcome: Outcome
    cost: Cost
    validated: bool = False
    status: str = "ok"
    provenance: Dict[str, str] = field(default_factory=dict)


@dataclass
class EvidenceMatrix:
    candidate_id: str
    rep: int
    repo: str
    records: Dict[str, EvidenceRecord]
    risk_features: Dict[str, float]
    ood: bool = False
    # Hidden labels. Never read by a policy; used only by the evaluator.
    truth_unsafe: Optional[bool] = None
    truth_severity: Optional[float] = None
    safe_twin: Optional[str] = None  # matched pair id
    task: Optional[str] = None


def cell_seed(freeze_salt: str, candidate_id: str, rep: int, action_id: str) -> int:
    """seed(candidate, rep, action) = H(freeze_salt || candidate_id || rep || action_id)."""
    h = hashlib.sha256(f"{freeze_salt}\x00{candidate_id}\x00{rep}\x00{action_id}".encode()).digest()
    return int.from_bytes(h[:8], "big")


# ----------------------------------------------------------------------------
# Synthetic generator for tests and simulation. Not an evidence producer.
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class SyntheticActionProfile:
    """Likelihood p(o | Y) for one action, plus lognormal cost dispersion."""
    action: Action
    p_o_given_unsafe: Tuple[float, float, float, float]
    p_o_given_safe: Tuple[float, float, float, float]
    cost_sigma: float = 0.25


def default_action_profiles(ci_detect: float = 0.71) -> List[SyntheticActionProfile]:
    """A plausible dense action set across F1-F4 and the challenger.

    `ci_detect` is the probability the repository's own suite fails an unsafe
    candidate; 1 - ci_detect = U_CI. The probe measured U_CI = 0.29 on flask.
    """
    A = Action
    C = Cost
    S = Source
    prof = [
        # (action, P(o|Y=1), P(o|Y=0)); order: pass, fail_validated, fail_unvalidated, inconclusive
        SyntheticActionProfile(A("f1.repo_ci", "F1", S.AUTHOR_SIDE, "repo-ci", C(0, 1, 1.0, 60), 10, True),
                               (1 - ci_detect, 0.0, ci_detect, 0.0), (0.98, 0.0, 0.02, 0.0)),
        SyntheticActionProfile(A("f1.generated_tests", "F1", S.AUTHOR_SIDE, "llm-tests", C(6000, 2, 0.6, 90), 40),
                               (0.60, 0.15, 0.20, 0.05), (0.85, 0.0, 0.08, 0.07)),
        SyntheticActionProfile(A("f2.static_security", "F2", S.AUTHOR_SIDE, "semgrep+bandit", C(0, 2, 0.15, 20), 20),
                               (0.70, 0.10, 0.18, 0.02), (0.92, 0.0, 0.07, 0.01)),
        SyntheticActionProfile(A("f2.dependency", "F2", S.AUTHOR_SIDE, "pip-audit+snapshot", C(0, 1, 0.05, 5), 21),
                               (0.90, 0.05, 0.05, 0.0), (0.985, 0.0, 0.015, 0.0)),
        SyntheticActionProfile(A("f3.api_contract", "F3", S.AUTHOR_SIDE, "griffe+import-linter", C(0, 2, 0.1, 10), 30),
                               (0.78, 0.12, 0.08, 0.02), (0.95, 0.0, 0.04, 0.01)),
        SyntheticActionProfile(A("f3.resource_budget", "F3", S.AUTHOR_SIDE, "bench", C(0, 1, 0.8, 120), 31),
                               (0.85, 0.05, 0.05, 0.05), (0.93, 0.0, 0.02, 0.05)),
        SyntheticActionProfile(A("f4.impact_revert", "F4", S.AUTHOR_SIDE, "inhouse", C(0, 1, 0.05, 5), 50),
                               (0.80, 0.0, 0.18, 0.02), (0.90, 0.0, 0.09, 0.01)),
        SyntheticActionProfile(A("ch.property", "F1", S.CHALLENGER, "hypothesis", C(4000, 3, 1.2, 180), 60),
                               (0.50, 0.30, 0.10, 0.10), (0.85, 0.0, 0.05, 0.10)),
        SyntheticActionProfile(A("ch.fuzz", "F1", S.CHALLENGER, "atheris", C(0, 2, 2.5, 300), 61),
                               (0.60, 0.25, 0.05, 0.10), (0.88, 0.0, 0.02, 0.10)),
        SyntheticActionProfile(A("ch.differential", "F1", S.CHALLENGER, "diff-exec", C(2000, 2, 1.0, 120), 62),
                               (0.45, 0.35, 0.10, 0.10), (0.86, 0.0, 0.04, 0.10)),
        SyntheticActionProfile(A("ch.spec_guided", "F3", S.CHALLENGER, "llm-spec", C(8000, 3, 0.8, 150), 63),
                               (0.55, 0.20, 0.15, 0.10), (0.84, 0.0, 0.06, 0.10)),
    ]
    return prof


def synthesize_matrix(cand_id: str, rep: int, repo: str, task: str, unsafe: bool,
                      severity: float, risk_logit: float, profiles: List[SyntheticActionProfile],
                      freeze_salt: str = "synthetic", ood: bool = False,
                      rho0: Optional[float] = None) -> EvidenceMatrix:
    """Draw one dense matrix. Deterministic in (freeze_salt, cand_id, rep, action)."""
    recs: Dict[str, EvidenceRecord] = {}
    for p in profiles:
        rng = np.random.default_rng(cell_seed(freeze_salt, cand_id, rep, p.action.id))
        probs = p.p_o_given_unsafe if unsafe else p.p_o_given_safe
        o = OUTCOMES[int(rng.choice(4, p=np.asarray(probs) / np.sum(probs)))]
        scale = float(np.exp(rng.normal(0.0, p.cost_sigma)))
        c = Cost(*(p.action.nominal_cost.vec() * scale))
        recs[p.action.id] = EvidenceRecord(p.action.id, o, c, validated=(o == Outcome.FAIL_VALIDATED))
    feats = {"risk_logit": risk_logit}
    if rho0 is not None:
        feats["rho0"] = rho0
    return EvidenceMatrix(cand_id, rep, repo, recs, feats, ood=ood, truth_unsafe=unsafe,
                          truth_severity=severity, task=task)

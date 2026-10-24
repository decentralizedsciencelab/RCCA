"""Evidence layer contracts (evidence-layer spec §4-§7, §10).

A Producer turns (candidate checkout, seed, meter) into one EvidenceRecord.
Producers are registered with their tool-specific knobs bound at registration
(closures), so the matrix generator has no per-tool branching. The Meter
records cost in raw units; `ci_equiv` is derived from the repository's
frozen CI baseline at write time.

Determinism rules a producer must obey (enforced by the conformance suite):
no clock reads in logic, no network, no mutable globals; the Meter is the
only thing that reads time, from the outside.
"""
from __future__ import annotations
import hashlib
import json
import os
import resource
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

OUTCOMES = ("pass", "fail_validated", "fail_unvalidated", "inconclusive")
FAMILIES = ("F1", "F2", "F3", "F4")
SOURCES = ("author_side", "challenger")



def resolve_interpreter(path: str) -> str:
    """Absolute path to a per-repository interpreter.

    Every producer runs with the candidate checkout as its cwd, so a
    repository-relative path from an evidence config -- `corpus/repos/flask/.venv/
    bin/python` -- would resolve against the checkout instead of this repository.
    Make it absolute once, against the directory the harness was invoked from.

    Symlinks are deliberately *not* resolved: `.venv/bin/python` is a symlink to
    the base interpreter, so resolving it would run every producer outside the
    environment the subject library is installed in."""
    p = Path(path).expanduser()
    return str(p if p.is_absolute() else Path.cwd() / p)


@dataclass
class Validation:
    contradicts_reference: bool = False
    minimized: bool = False
    repeats_across_seeds: bool = False
    second_tool_confirmed: bool = False
    confirmer_id: str = ""
    witness_ref: str = ""
    sub_seeds: List[int] = field(default_factory=list)

    @property
    def all_four(self) -> bool:
        return (self.contradicts_reference and self.minimized and self.repeats_across_seeds
                and self.second_tool_confirmed and bool(self.confirmer_id))


@dataclass
class Cost:
    tokens: float = 0.0
    tool_calls: float = 0.0
    cpu_s: float = 0.0
    wall_s: float = 0.0
    ci_equiv: float = 0.0


@dataclass
class Record:
    """One cell of the evidence matrix (spec §10)."""
    schema_version: int
    action_id: str
    family: str
    source: str
    producer: str
    candidate_id: str
    repo_state_sha: str
    rep: int
    seed: int
    outcome: str
    validation: Validation
    cost: Cost
    provenance: Dict[str, str]
    artifacts: List[Dict[str, str]] = field(default_factory=list)
    blind_spots: List[str] = field(default_factory=list)
    status: str = "ok"            # ok | error | timeout
    started_at: str = ""
    ended_at: str = ""
    search_budget_used: Optional[float] = None   # challenger: fraction of declared budget consumed

    def to_dict(self) -> dict:
        d = asdict(self)
        d["validated"] = self.outcome == "fail_validated"
        return d


class Meter:
    """Measures one producer invocation from the outside.

    cpu_s is the CPU time of child processes started during the invocation
    (RUSAGE_CHILDREN delta), which is where tool work happens. tool_calls and
    tokens are counted by the producer through `tool()` and `add_tokens()`.
    """

    def __init__(self):
        self.tool_calls = 0
        self.tokens = 0.0
        self._t0 = self._c0 = None
        self.wall_s = self.cpu_s = 0.0
        self.started_at = self.ended_at = ""

    def __enter__(self):
        self._t0 = time.monotonic()
        r = resource.getrusage(resource.RUSAGE_CHILDREN)
        self._c0 = r.ru_utime + r.ru_stime
        self.started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return self

    def __exit__(self, *exc):
        self.wall_s = time.monotonic() - self._t0
        r = resource.getrusage(resource.RUSAGE_CHILDREN)
        self.cpu_s = (r.ru_utime + r.ru_stime) - self._c0
        self.ended_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return False

    def tool(self, cmd: Sequence[str], cwd: Path, timeout: int, env: Optional[Dict[str, str]] = None,
             stdin: Optional[str] = None) -> subprocess.CompletedProcess:
        """Run one external tool process. Counts a tool call. Network is denied by
        the sandbox at run time; here the environment is scrubbed of proxies so a
        producer that tries to reach out fails rather than succeeds silently."""
        self.tool_calls += 1
        e = {k: v for k, v in (env or os.environ).items()
             if not k.upper().endswith("_PROXY") and k.upper() not in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")}
        e["PYTHONHASHSEED"] = "0"
        e["NO_NETWORK"] = "1"
        return subprocess.run(list(cmd), cwd=cwd, capture_output=True, text=True, timeout=timeout, env=e, input=stdin)

    def add_tokens(self, n: float) -> None:
        self.tokens += n

    def cost(self, ci_baseline_cpu_s: Optional[float]) -> Cost:
        ci = (self.cpu_s / ci_baseline_cpu_s) if ci_baseline_cpu_s else 0.0
        return Cost(self.tokens, self.tool_calls, round(self.cpu_s, 4), round(self.wall_s, 4), round(ci, 5))


@dataclass
class Context:
    """What a producer may see: the candidate checkout and its pinned state. Nothing else."""
    candidate_id: str
    repo: Path                 # working tree with the candidate applied
    repo_state_sha: str        # pinned pre-change commit
    rep: int
    seed: int
    package_root: str
    ci_baseline_cpu_s: Optional[float] = None
    scratch: Optional[Path] = None


@dataclass
class Outcome:
    outcome: str
    validation: Validation = field(default_factory=Validation)
    provenance: Dict[str, str] = field(default_factory=dict)
    artifacts: List[Dict[str, str]] = field(default_factory=list)
    blind_spots: List[str] = field(default_factory=list)
    status: str = "ok"
    search_budget_used: Optional[float] = None


ProducerFn = Callable[[Context, Meter], Outcome]


@dataclass(frozen=True)
class Action:
    id: str
    family: str
    source: str
    producer: str
    fn: ProducerFn
    priority: int = 100
    is_repo_ci: bool = False
    langs: Sequence[str] = ("python",)


class Registry:
    def __init__(self):
        self._actions: Dict[str, Action] = {}

    def register(self, action: Action) -> Action:
        assert action.family in FAMILIES and action.source in SOURCES
        assert action.id not in self._actions, f"duplicate action {action.id}"
        self._actions[action.id] = action
        return action

    def actions(self) -> List[Action]:
        return sorted(self._actions.values(), key=lambda a: (a.priority, a.id))

    def __getitem__(self, k: str) -> Action:
        return self._actions[k]


def cell_seed(freeze_salt: str, candidate_id: str, rep: int, action_id: str) -> int:
    """Identical to rcca.matrix.cell_seed, duplicated so the layers share no code."""
    h = hashlib.sha256(f"{freeze_salt}\x00{candidate_id}\x00{rep}\x00{action_id}".encode()).digest()
    return int.from_bytes(h[:8], "big")


def sub_seed(seed: int, k: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}\x00{k}".encode()).digest()[:8], "big")


def produce(action: Action, ctx: Context, schema_version: int = 1) -> Record:
    """Run one producer under a meter and assemble the record. Exceptions and
    timeouts become `inconclusive` with status error/timeout, never a pass."""
    m = Meter()
    with m:
        try:
            out = action.fn(ctx, m)
        except subprocess.TimeoutExpired:
            out = Outcome("inconclusive", status="timeout")
        except Exception as ex:                       # noqa: BLE001
            out = Outcome("inconclusive", status="error", provenance={"error": repr(ex)[:300]})
    if out.outcome == "fail_validated" and not out.validation.all_four:
        out.outcome = "fail_unvalidated"              # validation soundness, enforced
    assert out.outcome in OUTCOMES
    return Record(schema_version, action.id, action.family, action.source, action.producer,
                  ctx.candidate_id, ctx.repo_state_sha, ctx.rep, ctx.seed, out.outcome, out.validation,
                  m.cost(ctx.ci_baseline_cpu_s), out.provenance, out.artifacts, out.blind_spots, out.status,
                  m.started_at, m.ended_at, out.search_budget_used)


def append_records(path: Path, records: Iterable[Record]) -> None:
    with open(path, "a") as f:
        for r in records:
            f.write(json.dumps(r.to_dict()) + "\n")


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

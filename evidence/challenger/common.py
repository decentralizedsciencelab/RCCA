"""Shared skeleton for the three challenger producers.

Each technique supplies a `search(targets, pair, base_tree, deadline)` that
yields candidate divergences (and whether the technique already minimized
them). The skeleton owns target discovery, the declared-change rule, the base
worktree, the executor pair, validation, and the outcome, so the three
producers differ only in how inputs are generated.

Reporting order when several divergences are seen: a validated divergence on
an undeclared target ends the search; otherwise an unvalidated divergence on
an undeclared target is reported; otherwise a divergence on a target whose new
behaviour the candidate's own tests declare is reported as `fail_unvalidated`
with the declaring tests in provenance. Declared targets are never validated,
which keeps the budget for targets that can still yield evidence.

The search budget is set in standard-CI equivalents (`budget_ci_equiv`, and
`fuzz_time_per_target_ci_equiv` for each fuzzed target) and spent in seconds:
seconds = CI-equivalents x the median CPU seconds of the repository's standard
CI on the machine at hand (evidence.baseline). A budget in wall-clock seconds
buys more search per CI-equivalent on a faster machine (results.md E2). Every
record's provenance carries the budget as set, the baseline, and the seconds.
"""
from __future__ import annotations
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

from ..core import Context, Meter, Outcome, Validation
from .declared import declared_targets
from .harness import Divergence, Finding, Pair, Worktree, artifact_for, validate, witness_source
from .targets import Target, changed_functions


@dataclass(frozen=True)
class ChallengerConfig:
    python: str
    search: str = "src"             # directory holding the importable package
    budget_s: float = 120.0         # total search budget per invocation, in seconds on this machine
    max_targets: int = 8
    k: int = 3                      # criterion (iii) repetitions
    call_timeout: int = 5           # seconds per call: a safety timeout, not a budget
    minimize_budget: int = 200      # calls, not seconds
    declared_timeout: int = 300     # one run of the candidate's changed tests under coverage
    budget_ci_equiv: Optional[float] = None    # the budget as set, which budget_s derives from
    ci_baseline_cpu_s: Optional[float] = None  # seconds per CI-equivalent here (evidence.baseline)
    budget_note: str = ""                      # set when a development config gave seconds instead

    def budget_provenance(self) -> Dict[str, str]:
        """How the search was bounded: the budget as set, the baseline that converts it, the seconds."""
        prov = {"budget_s": str(self.budget_s)}
        if self.budget_ci_equiv is not None:
            prov["budget_ci_equiv"] = str(self.budget_ci_equiv)
        if self.ci_baseline_cpu_s is not None:
            prov["ci_baseline_cpu_s"] = str(self.ci_baseline_cpu_s)
        if self.budget_note:
            prov["budget_note"] = self.budget_note
        return prov


class BudgetError(ValueError):
    """The challenger's budget cannot be set from the evidence config."""


SECONDS_KEYS = ("budget_s", "fuzz_time_per_target_s")      # development configs only


def ci_equiv_seconds(ci_equiv: float, ci_baseline_cpu_s: float) -> float:
    """Seconds that buy `ci_equiv` standard-CI equivalents of search on the machine whose standard CI
    takes `ci_baseline_cpu_s` CPU seconds (the baseline's median): their product."""
    if not (isinstance(ci_baseline_cpu_s, (int, float)) and math.isfinite(ci_baseline_cpu_s)
            and ci_baseline_cpu_s > 0):
        raise BudgetError(f"the standard-CI baseline must be a positive number of CPU seconds, "
                          f"not {ci_baseline_cpu_s!r}")
    return float(ci_equiv) * float(ci_baseline_cpu_s)


def _positive(c: dict, key: str) -> float:
    v = c[key]
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
        raise BudgetError(f"challenger.{key} must be a positive number, not {v!r}")
    return float(v)


def _in_seconds(s_key: str) -> BudgetError:
    return BudgetError(f"challenger.{s_key} is in wall-clock seconds, which buy more search per CI-equivalent "
                       f"on a faster machine; a run under a freeze record takes {s_key[:-2]}_ci_equiv, "
                       f"in standard-CI equivalents")


def resolve_budget(c: dict, name: str, ci_baseline_cpu_s: Optional[float], dev: bool) -> Tuple[float, Optional[float]]:
    """(seconds, CI-equivalents) of the challenger budget `name`, "budget" or "fuzz_time_per_target",
    from the evidence config's challenger section `c`. The config sets `<name>_ci_equiv`, converted
    with the baseline. `<name>_s` in seconds is accepted only from a development run (`dev`); it is
    returned as given, with no CI-equivalents."""
    ci_key, s_key = f"{name}_ci_equiv", f"{name}_s"
    if c.get(ci_key) is not None and c.get(s_key) is not None:
        raise BudgetError(f"challenger sets both {ci_key} and {s_key}; keep {ci_key}")
    if c.get(s_key) is not None:
        if not dev:
            raise _in_seconds(s_key)
        return _positive(c, s_key), None
    if c.get(ci_key) is None:
        raise BudgetError(f"challenger.{ci_key} is not set")
    if ci_baseline_cpu_s is None:
        raise BudgetError(f"challenger.{ci_key} needs the standard-CI baseline to become seconds: name it as "
                          f"ci_baseline in the evidence config and measure it with python -m evidence.baseline")
    ci = _positive(c, ci_key)
    return ci_equiv_seconds(ci, ci_baseline_cpu_s), ci


def challenger_config(c: dict, python: str, ci_baseline_cpu_s: Optional[float], dev: bool) -> ChallengerConfig:
    """The challenger's configuration from an evidence config's challenger section, its search budget
    converted to seconds on this machine (`resolve_budget`). A budget in seconds anywhere in the
    section is refused unless `dev`, and a development record then says its budget is in seconds."""
    seconds = [k for k in SECONDS_KEYS if c.get(k) is not None]
    if seconds and not dev:
        raise _in_seconds(seconds[0])
    budget_s, budget_ci = resolve_budget(c, "budget", ci_baseline_cpu_s, dev)
    note = (f"development config: {', '.join(seconds)} in wall-clock seconds, not standard-CI equivalents"
            if seconds else "")
    return ChallengerConfig(python, c.get("search", "src"), budget_s, c.get("max_targets", 8), c.get("k", 3),
                            c.get("call_timeout", 5), c.get("minimize_budget", 200), budget_ci_equiv=budget_ci,
                            ci_baseline_cpu_s=ci_baseline_cpu_s, budget_note=note)


Search = Callable[[List[Target], Pair, Path, float], Iterator[Tuple[Divergence, bool]]]

BLIND = ["public functions, and public methods whose constructor takes JSON-able or factory-built arguments "
         "(async, *args-only, properties and private callables not targeted)",
         "JSON-able and factory-built inputs only",
         "oracle is pre-change behaviour: a defect present in both versions is invisible",
         "a divergence on a callable whose changed lines the candidate's own added or modified tests execute "
         "is reported as a declared change (fail_unvalidated), so a defect seeded in such a callable is left to F1"]


def _declared_finding(div: Divergence) -> Finding:
    v = Validation(contradicts_reference=True, witness_ref=div.describe())
    return Finding(div, v, witness_source(div), {})


def run_technique(ctx: Context, m: Meter, cfg: ChallengerConfig, name: str, confirmer: str,
                  search: Search, prov_extra: Optional[Dict[str, str]] = None,
                  target_filter: Optional[Callable[[Target], bool]] = None) -> Outcome:
    prov: Dict[str, str] = {"technique": name, **cfg.budget_provenance(), **(prov_extra or {})}
    t0 = time.monotonic()
    targets = changed_functions(ctx.repo, ctx.repo_state_sha, ctx.package_root, cfg.search)
    prov["targets_changed"] = str(len(targets))
    if target_filter:
        targets = [t for t in targets if target_filter(t)]
    targets = targets[:cfg.max_targets]
    prov["targets_searched"] = str(len(targets))
    prov["targets_methods"] = str(sum(1 for t in targets if t.kind == "method"))
    if not targets:
        prov["reason"] = "no runnable changed public callable"
        return Outcome("inconclusive", provenance=prov, blind_spots=BLIND, status="error", search_budget_used=0.0)
    declared, rule = declared_targets(ctx.repo, ctx.repo_state_sha, targets, cfg.python, ctx.package_root, cfg.search,
                                      m, cfg.declared_timeout)
    prov["targets_declared"] = str(len(declared)); prov["declared_rule"] = rule
    if declared:
        prov["declared_targets"] = ";".join(f"{mod}.{qn}" for mod, qn in sorted(declared))
    deadline = t0 + cfg.budget_s
    finding: Optional[Finding] = None            # best undeclared finding
    declared_hit: Optional[Finding] = None       # first divergence on a declared target
    n_div = 0
    with Worktree(ctx.repo, ctx.repo_state_sha) as base_tree:
        pair = Pair(cfg.python, base_tree, ctx.repo, cfg.search, m, hashseed=ctx.seed, call_timeout=cfg.call_timeout)
        try:
            for div, already_minimal in search(targets, pair, base_tree, deadline):
                n_div += 1
                key = (div.target.module, div.target.qualname)
                if key in declared:
                    if declared_hit is None:
                        declared_hit = _declared_finding(div)
                        declared_hit.extra["declared_by_candidate_tests"] = declared[key]
                        prov.update(declared_hit.extra)        # recorded whichever finding is reported
                    if time.monotonic() > deadline:
                        break
                    continue
                f = validate(div, already_minimal, pair, cfg.python, base_tree, ctx.repo, cfg.search, m, ctx.seed,
                             cfg.k, confirmer, cfg.call_timeout, cfg.minimize_budget)
                prov.update(f.extra)
                if f.validated:
                    finding = f
                    break
                if finding is None:
                    finding = f
                if time.monotonic() > deadline:
                    break
        finally:
            pair.close()
        prov["unstable_targets"] = str(len(pair.unstable))
    used = min(1.0, (time.monotonic() - t0) / cfg.budget_s)
    prov["divergences_seen"] = str(n_div); prov["calls"] = str(pair.base.calls + pair.cand.calls)
    if finding is None and declared_hit is not None:
        return Outcome("fail_unvalidated", declared_hit.validation, prov, [artifact_for(declared_hit)], BLIND,
                       search_budget_used=used)
    if finding is None:
        return Outcome("pass", provenance=prov, blind_spots=BLIND, search_budget_used=used)
    outcome = "fail_validated" if finding.validated else "fail_unvalidated"
    return Outcome(outcome, finding.validation, prov, [artifact_for(finding)], BLIND, search_budget_used=used)

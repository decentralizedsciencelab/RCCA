"""F1 functional/regression: the repository's own test suite, with coverage.

Outcome rules
  pass               suite green
  fail_validated     at least one test fails, the same failing test ids fail on
                     k re-runs (criterion iii), each failing test is already a
                     minimal unit (ii), the suite is a named reference, the
                     pre-change behaviour the repository itself asserts (i),
                     and an isolated re-run of the failing ids under a second
                     configuration (no plugins, fresh process) reproduces (iv)
  fail_unvalidated   failures that do not repeat, or whose isolated re-run
                     does not reproduce (flaky)
  inconclusive       collection error, crash, or timeout: the suite never ran

Blind spots declared: the suite only tests what it asserts; untested code is
outside its reference. Coverage is attached as an artifact so H2 and the
risk model can use it.
"""
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import List, Optional, Sequence

from ..core import Action, Context, Meter, Outcome, Registry, Validation, sub_seed

_FAILED = re.compile(r"^(FAILED|ERROR) (\S+)", re.M)


def _failing_ids(stdout: str) -> List[str]:
    return sorted({m.group(2).split(" - ")[0] for m in _FAILED.finditer(stdout)})


def _collection_error(stdout: str, stderr: str) -> bool:
    s = stdout + stderr
    return ("error during collection" in s.lower() or "no tests ran" in s.lower()
            or "ImportError while loading" in s or "INTERNALERROR" in s)


def pytest_command(python: str, test_args: Sequence[str], extra_deselect: Sequence[str] = ()) -> List[str]:
    return [python, "-m", "pytest", *test_args, "-q", "--no-header", "-p", "no:cacheprovider", "-rfE",
            *sum((["--deselect", d] for d in extra_deselect), [])]


def ci_command(python: str, test_args: Sequence[str], source: str, extra_deselect: Sequence[str] = ()) -> List[str]:
    """The exact command F1 runs: the suite under coverage. The standard-CI baseline
    (`evidence.baseline`) measures this same command, so F1 costs 1.0 ci_equiv by
    construction, up to run-to-run noise."""
    base = pytest_command(python, test_args, extra_deselect)
    return [python, "-m", "coverage", "run", f"--source={source}", "-m", "pytest", *base[3:]]


def make_repo_ci(python: str, test_args: Sequence[str], source: str, k: int = 3, timeout: int = 1200,
                 extra_deselect: Sequence[str] = ()) -> Action:
    base = pytest_command(python, test_args, extra_deselect)

    def run(ctx: Context, m: Meter) -> Outcome:
        cov_cmd = ci_command(python, test_args, source, extra_deselect)
        r = m.tool(cov_cmd, ctx.repo, timeout)
        prov = {"tool": "pytest+coverage", "command": " ".join(cov_cmd), "returncode": str(r.returncode)}
        artifacts = []
        cj = m.tool([python, "-m", "coverage", "json", "-o", "rcca_cov.json"], ctx.repo, 300)
        if cj.returncode == 0 and (ctx.repo / "rcca_cov.json").exists():
            try:
                tot = json.load(open(ctx.repo / "rcca_cov.json"))["totals"]
                prov["coverage_pct"] = f"{tot['percent_covered']:.2f}"
                artifacts.append({"name": "coverage.json", "path": "rcca_cov.json"})
            except Exception:
                pass
        blind = ["asserts only the behaviour the suite encodes", "no security or contract checks"]
        if r.returncode == 0:
            return Outcome("pass", provenance=prov, artifacts=artifacts, blind_spots=blind)
        if r.returncode >= 2 or _collection_error(r.stdout, r.stderr):
            return Outcome("inconclusive", provenance=prov, blind_spots=blind, status="error")
        ids = _failing_ids(r.stdout)
        if not ids:
            return Outcome("fail_unvalidated", provenance=prov, blind_spots=blind)
        # (iii) repeat across k sub-seeded re-runs of the failing ids
        seeds = [sub_seed(ctx.seed, i) for i in range(k)]
        repeats = True
        for s in seeds:
            rr = m.tool([*base, "-p", "no:randomly", *ids], ctx.repo, timeout)
            if sorted(set(_failing_ids(rr.stdout))) != ids:
                repeats = False; break
        # (iv) confirmer: fresh process, plugins disabled, one id at a time
        confirmed = repeats
        if repeats:
            for tid in ids:
                rc = m.tool([python, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider", "-rfE",
                             "-p", "no:randomly", tid], ctx.repo, timeout)
                if rc.returncode == 0:
                    confirmed = False; break
        v = Validation(contradicts_reference=True, minimized=True, repeats_across_seeds=repeats,
                       second_tool_confirmed=confirmed, confirmer_id="pytest-isolated-rerun" if confirmed else "",
                       witness_ref=";".join(ids), sub_seeds=seeds)
        prov["failing_tests"] = str(len(ids))
        return Outcome("fail_validated" if v.all_four else "fail_unvalidated", v, prov, artifacts, blind)

    return Action("f1.repo_ci", "F1", "author_side", "pytest", run, priority=10, is_repo_ci=True)


def register(reg: Registry, python: str, test_args: Sequence[str], source: str, **kw) -> Action:
    return reg.register(make_repo_ci(python, test_args, source, **kw))

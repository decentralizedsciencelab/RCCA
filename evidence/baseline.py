"""Standard-CI baseline per repository (spec §7, D5).

`ci_baseline_cpu_s` is the median CPU time, over n runs at the pinned state, of
the repository's test suite exactly as the F1 producer runs it (same
interpreter, same arguments, under coverage). Compute cost is then expressed
as a multiple of this, so F1 and the standard-CI policy cost 1.0 by
construction, up to run-to-run noise, and a global compute ceiling is
meaningful across repositories. The interpreter is the pinned commit's
environment (`corpus.envs`) when the repository config has an `env` block, and
the evidence config's `python` otherwise, as `evidence.run` resolves it.

    python -m evidence.baseline --config configs/flask.json --evidence configs/evidence_flask.json \
        --freeze configs/freeze.json --n 5
    python -m evidence.baseline --config configs/flask.dev.json --evidence configs/evidence_flask.dev.json --dev

The baseline is written to the evidence config's `ci_baseline` only under a
freeze record at stage pre-freeze or later that lists both configs
(`rcca.freeze`), or with --dev into a directory whose name ends in -dev.
"""
from __future__ import annotations
import argparse
import json
import statistics
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Sequence

from corpus.envs import ensure_env
from rcca.freeze import FreezeError, gate
from .core import Meter, resolve_interpreter


@dataclass
class Baseline:
    repo: str
    commit: str
    command: List[str]
    n: int
    cpu_s_runs: List[float]
    wall_s_runs: List[float]
    cpu_s_median: float
    cpu_s_spread: float      # max - min
    wall_s_median: float

    def to_dict(self) -> dict:
        return asdict(self)


def measure(repo_name: str, repo: Path, commit: str, ci_cmd: Sequence[str], n: int = 5,
            timeout: int = 1800) -> Baseline:
    cpu, wall = [], []
    for _ in range(n):
        m = Meter()
        with m:
            m.tool(ci_cmd, repo, timeout)
        cpu.append(m.cpu_s); wall.append(m.wall_s)
    return Baseline(repo_name, commit, list(ci_cmd), n, cpu, wall,
                    statistics.median(cpu), max(cpu) - min(cpu), statistics.median(wall))


def save(b: Baseline, path: Path) -> None:
    json.dump(b.to_dict(), open(path, "w"), indent=1)


def load(path: Path) -> Baseline:
    return Baseline(**json.load(open(path)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="repository config (checkout, repo name)")
    ap.add_argument("--evidence", required=True, help="evidence config (python, f1, ci_baseline path)")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--commit", default=None,
                    help="pinned state to measure at (default: config `pinned_commit`, else the checkout's HEAD)")
    ap.add_argument("--freeze", default=None, help="freeze record (rcca.freeze); needed unless the output dir ends in -dev")
    ap.add_argument("--dev", action="store_true", help="development run: no freeze record, ci_baseline in a -dev directory")
    a = ap.parse_args(argv)
    from .producers.f1_tests import ci_command
    cfg = json.load(open(a.config)); ecfg = json.load(open(a.evidence))
    out = Path(ecfg["ci_baseline"])
    try:
        gate(a.freeze, dev=a.dev, out=out.parent, min_stage="pre-freeze", configs=[a.config, a.evidence])
    except FreezeError as e:
        print(f"evidence.baseline: refused: {e}", file=sys.stderr)
        return 2
    repo = Path(cfg["checkout"]).expanduser()
    f1 = ecfg["f1"]

    def rev(ref: str) -> str:
        return subprocess.run(["git", "rev-parse", f"{ref}^{{commit}}"], cwd=repo, capture_output=True, text=True,
                              check=True).stdout.strip()                      # peels annotated tags to their commit

    prev = rev("HEAD")
    commit = rev(a.commit or cfg.get("pinned_commit") or "HEAD")
    py = str(ensure_env(repo, commit, dict(cfg, python=ecfg["python"])))    # built before the checkout moves
    cmd = ci_command(py, f1["test_args"], f1["source"], f1.get("extra_deselect", ()))
    try:
        if commit != prev:
            subprocess.run(["git", "checkout", "-q", "--force", commit], cwd=repo, check=True)
        b = measure(cfg["repo"], repo, commit, cmd, n=a.n, timeout=a.timeout)
    finally:
        if commit != prev:
            subprocess.run(["git", "checkout", "-q", "--force", prev], cwd=repo, check=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    save(b, out)
    print(f"{cfg['repo']} @ {commit[:12]}: cpu_s median {b.cpu_s_median:.2f} (spread {b.cpu_s_spread:.2f}) over {a.n} runs -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ---------------------------------------------------------------------------
# Per-base-commit baselines.
#
# `ci_equiv` is a producer's CPU seconds over the median CPU of the repository's
# own suite "at the pinned state", which charges standard
# CI exactly 1.0 by construction. A corpus whose candidates sit at their own base
# commits has as many pinned states as commits, and one baseline breaks that
# property: on the 2026-10-19 pilot f1.repo_ci ranged 1.176 to 7.851, a 6.7x
# spread on a quantity defined to be 1.0. The ceilings are quantiles of these
# costs, so the unit has to be right per candidate or the equal-resources
# comparison is not equal.
# ---------------------------------------------------------------------------
from typing import Callable, Optional


def _measure_at(repo: Path, commit: str, python: str, spec: dict, out: Path) -> None:
    """Measure the suite at `commit` and write the Baseline. Moves the checkout to the
    commit and back, as `corpus.envs.ensure_env` does, so call it with no work in
    progress in the checkout."""
    from corpus.admit import Checkout
    from .producers.f1_tests import ci_command
    cmd = ci_command(python, spec.get("test_args") or ["tests"], spec.get("source", "src"),
                     spec.get("extra_deselect", ()))
    prev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
                          text=True, check=True).stdout.strip()
    co = Checkout(Path(repo))
    try:
        co.reset(commit)
        b = measure(spec.get("repo", "?"), Path(repo), commit, cmd,
                    n=spec.get("baseline_n", 5), timeout=spec.get("timeout", 1800))
    finally:
        co.reset(prev)
    out.parent.mkdir(parents=True, exist_ok=True)
    save(b, out)


def ensure_baseline(repo: Path, base_commit: str, python: str, spec: dict, *,
                    root: Path = Path("corpus/baselines"),
                    builder: Optional[Callable[..., None]] = None) -> float:
    """Median CPU seconds of one standard-CI run at `base_commit`, measured on first
    use and cached under `root/<commit>/ci_baseline.json`. One suite run per distinct
    base commit, in that commit's own environment."""
    out = Path(root) / base_commit / "ci_baseline.json"
    if not out.exists():
        (builder or _measure_at)(Path(repo), base_commit, python, spec, out)
    return load(out).cpu_s_median

"""Evidence-matrix generation.

    python -m evidence.run --config configs/flask.json --evidence configs/evidence_flask.json \
        --freeze configs/freeze.json --partition train --reps 3

For each candidate in the manifest (optionally one partition) and each
repetition, materialize the candidate in the checkout, run every registered
action, and append one record per action to `records.jsonl`. Cells already
present in `records.jsonl` are skipped, so the run is resumable.

Each candidate's actions run in its base commit's environment: when the
repository config has an `env` block, `corpus.envs.ensure_env` builds or reuses
the environment of the candidate's `commit` and every producer is bound to its
interpreter, the challenger's calls too unless `challenger.python` names
another. Without the block (the development configs) every candidate runs in the
evidence config's `python`, as before. Two pinned tools run outside that
environment, and the block's `host_tools` pins them: semgrep, which F2 runs from
the venv `corpus.envs.ensure_tool` builds, and hypothesis, which generates and
shrinks ch.property's inputs here in the orchestrator. Under a freeze record a
registry whose host tool is missing, unpinned, or at another version is refused,
and the F2 and ch.property records carry the version they ran with.

The run passes the freeze record's firewall (`rcca.freeze`) first: stage
pre-freeze or later, both configs listed in the record, only partitions the
stage has opened (no --partition means all of them, so test until unblinded),
and from stage frozen no excluded task in the manifest. Cell seeds then use the
record's salt. Without a record, --dev and an `out` ending in -dev are
required, and the evidence config's freeze_salt seeds the cells.

The standard-CI baseline (`ci_baseline`, from evidence.baseline) is the unit of
every record's ci_equiv and of the challenger's search budget, which the config
sets in CI-equivalents and the run converts to seconds on this machine. Under a
record a run refuses without the baseline, and refuses a budget in seconds; a
development config may still give `budget_s`, and its records say so.

Evidence config (JSON):
    {"freeze_salt": "dev-only fallback", "python": "corpus/repos/flask/.venv/bin/python",
     "ci_baseline": "corpus/flask/ci_baseline.json",
     "f1": {"test_args": ["tests"], "source": "flask",
            "extra_deselect": ["tests/test_reqctx.py::test_bad_environ_raises_bad_request"]},
     "challenger": {"budget_ci_equiv": 59.73, "fuzz_time_per_target_ci_equiv": 9.96, "call_timeout": 5,
                    "techniques": ["property", "differential", "fuzz"]}}
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

from corpus.admit import Checkout
from corpus.envs import ensure_env, ensure_tool
from corpus.manifest import read_jsonl
from rcca.freeze import FreezeError, check_manifest, gate, is_dev, resolve_salt
from .baseline import ensure_baseline, load as load_baseline
from .core import Context, Registry, append_records, cell_seed, produce, resolve_interpreter
from .producers import f1_tests, f2_static, f2_deps, f3_contracts, f4_impact
from .challenger import harness as ch_harness
from .challenger.common import BudgetError, challenger_config, resolve_budget
from .challenger.spec_guided import ReviewConfigError, check_independent


def load_ci_baseline(ecfg: dict, required: bool) -> Optional[float]:
    """Median CPU seconds of one standard-CI run (evidence.baseline), from the evidence config's
    ci_baseline: the unit of every record's ci_equiv and of the challenger's budget. A run under a
    freeze record (`required`) refuses without one rather than meter compute as zero; a development
    run without one meters ci_equiv as 0. A baseline the config names must exist either way."""
    named = ecfg.get("ci_baseline")
    if not named:
        if required:
            raise FreezeError("the evidence config names no ci_baseline; under a freeze record ci_equiv and "
                              "the challenger's budget need the standard-CI baseline (python -m evidence.baseline)")
        return None
    path = Path(named).expanduser()
    if not path.exists():
        raise FreezeError(f"ci_baseline {named} not found; measure it first with python -m evidence.baseline")
    return load_baseline(path).cpu_s_median


def host_version(name: str, found: Optional[str], pins: Dict[str, str], dev: bool, where: str) -> str:
    """The version of a pinned tool that runs outside the subject's environment (env.host_tools). Under
    a freeze record it must be installed, pinned, and at its pin; a development run takes what it finds.
    The result goes into the provenance of the records the tool produces."""
    pin = pins.get(name)
    if not dev:
        if found is None:
            raise FreezeError(f"{name} is not installed {where}; under a freeze record it runs at its "
                              f"env.host_tools pin")
        if pin is None:
            raise FreezeError(f"the repository config pins no {name} (env.host_tools), and this run uses it {where}")
        if found != pin:
            raise FreezeError(f"{name} {found} is not its pin {pin} (env.host_tools), {where}")
    return found or "missing"


def _hypothesis_version() -> Optional[str]:
    try:
        import hypothesis
    except ImportError:
        return None
    return getattr(hypothesis, "__version__", None)


def build_registry(ecfg: dict, base_dir: Path = Path("."), *, ci_baseline_cpu_s: Optional[float] = None,
                   dev: bool = False, host_tools: Optional[Dict[str, str]] = None,
                   semgrep: str = "semgrep") -> Registry:
    """Bind tool knobs from the evidence config. Each key is optional, so a
    partial matrix (e.g. F1 only) is a valid, smaller action set. The
    challenger's budgets become seconds with `ci_baseline_cpu_s`; budgets in
    seconds are accepted only from a development run (`dev`). `host_tools` are
    the repository config's env.host_tools pins and `semgrep` the executable F2
    runs (corpus.envs.ensure_tool's, or semgrep from PATH); see `host_version`."""
    reg = Registry()
    py = resolve_interpreter(ecfg["python"])
    pins = host_tools or {}
    if "f1" in ecfg:
        f1_tests.register(reg, py, ecfg["f1"]["test_args"], ecfg["f1"]["source"],
                          extra_deselect=ecfg["f1"].get("extra_deselect", ()),
                          timeout=int(ecfg["f1"].get("timeout", 1200)))   # a hung suite is inconclusive after this
    if "f2_static" in ecfg:
        c = ecfg["f2_static"]
        use = bool(c.get("semgrep_rules"))
        version = (host_version("semgrep", f2_static.semgrep_version_of(semgrep), pins, dev, f"at {semgrep}")
                   if use else None)
        # resolved against the config's directory: a producer runs with the candidate
        # checkout as cwd, where a repository-relative rules path does not exist
        rules = [str((base_dir / rp).resolve()) if not Path(rp).is_absolute() else rp
                 for rp in c.get("semgrep_rules", [])]
        f2_static.register(reg, py, rules, c.get("target", "src"),
                           use_semgrep=use, k=c.get("k", 2), semgrep=semgrep, semgrep_version=version)
    if "f2_deps" in ecfg:
        c = ecfg["f2_deps"]
        f2_deps.register(reg, c["snapshot"], c.get("advisories"), base_dir=base_dir)
    if "f3" in ecfg:
        c = ecfg["f3"]
        f3_contracts.register(reg, py, c["package"], c.get("search", "src"), k=c.get("k", 2))
    if "f4" in ecfg:
        c = ecfg["f4"]
        f4_impact.register(reg, c["top_module"], tests_dir=c.get("tests_dir", "tests"))
    if "challenger" in ecfg:
        c = ecfg["challenger"]
        techs = c.get("techniques", ["property", "differential", "fuzz"])
        # The challenger is configured independently of the authoring agent: its own
        # interpreter (may differ from `python`), repetition count, and a search budget in
        # CI-equivalents that this machine's baseline turns into seconds (results.md E2).
        # Budgets are resolved before any technique is imported, so a bad one fails first.
        ccfg = challenger_config(c, resolve_interpreter(c.get("python", py)), ci_baseline_cpu_s, dev)
        if "fuzz" in techs:
            fuzz_s, fuzz_ci = resolve_budget(c, "fuzz_time_per_target", ci_baseline_cpu_s, dev)
        # Per-repository factory table: type name -> expression evaluated inside the
        # executor, for parameters whose annotation is not JSON-able (e.g. "Flask").
        ch_harness.configure(c.get("factories"))
        # imported here: the techniques need hypothesis and atheris, a config without them does not
        if "property" in techs:
            # its generation and shrinking run in this process: the orchestrator's hypothesis is the one pinned
            host_version("hypothesis", _hypothesis_version(), pins, dev, "in the orchestrator (requirements-dev.txt)")
            from .challenger import property as ch_property
            ch_property.register(reg, ccfg, max_examples=c.get("max_examples", 300))
        if "differential" in techs:
            from .challenger import differential as ch_differential
            ch_differential.register(reg, ccfg, test_args=tuple(c.get("test_args", ["tests"])),
                                     suite_timeout=c.get("suite_timeout", 1200))
        if "fuzz" in techs:
            from .challenger import fuzz as ch_fuzz
            ch_fuzz.register(reg, ccfg, time_per_target_s=fuzz_s, time_per_target_ci_equiv=fuzz_ci)
    if "spec_guided" in ecfg:
        # the LLM-review stage: a model behind a loopback chat endpoint reads the change against the
        # pinned state's signatures and docstrings; its prompt resolves against the config's directory
        from .challenger import spec_guided as ch_spec_guided
        ch_spec_guided.register(reg, ecfg["spec_guided"], base_dir=base_dir)
    return reg


def done_cells(records_path: Path) -> Set[Tuple[str, int, str]]:
    if not records_path.exists():
        return set()
    return {(r["candidate_id"], int(r["rep"]), r["action_id"]) for r in read_jsonl(records_path)}



def f1_spec_at(ecfg: dict, out: Path, commit: str) -> dict:
    """The F1 spec for `commit`: the config's, with the repository-suite tests red at the base commit
    (corpus.ci_base, measured by admission) deselected after the config's own. F1 is standard CI, and
    a test the base fails with no change is not evidence about the change; the CI baseline is measured
    from the same spec so the unit stays the passing suite's cost."""
    from corpus.ci_base import excluded_at
    spec = dict(ecfg.get("f1") or {})
    own = list(spec.get("extra_deselect", ()))
    spec["extra_deselect"] = own + [x for x in excluded_at(Path(out) / "ci_base", commit) if x not in own]
    return spec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--partition", default=None)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--include-unadmitted", action="store_true")
    ap.add_argument("--freeze", default=None, help="freeze record (rcca.freeze); needed for any out not ending in -dev")
    ap.add_argument("--dev", action="store_true", help="development run: no freeze record, the config's out ends in -dev")
    a = ap.parse_args(argv)
    cfg = json.load(open(a.config)); ecfg = json.load(open(a.evidence))
    out = Path(cfg["out"]); repo = Path(cfg["checkout"]).expanduser()
    base_dir = Path(a.evidence).resolve().parent.parent
    pins = (cfg.get("env") or {}).get("host_tools") or {}
    try:
        fz = gate(a.freeze, dev=a.dev, out=out, min_stage="pre-freeze", partitions=[a.partition],
                  configs=[a.config, a.evidence])
        salt, _ = resolve_salt(fz, ecfg)
        rows = read_jsonl(out / "manifest.jsonl")
        check_manifest(fz, rows, out / "manifest.jsonl")
        baseline = load_ci_baseline(ecfg, required=not is_dev(fz))
        runs_semgrep = bool((ecfg.get("f2_static") or {}).get("semgrep_rules"))
        semgrep = str(ensure_tool("semgrep", cfg)) if runs_semgrep and "semgrep" in pins else "semgrep"
        reg = build_registry(ecfg, base_dir, ci_baseline_cpu_s=baseline, dev=is_dev(fz), host_tools=pins,
                             semgrep=semgrep)
        if "spec_guided" in ecfg and fz is not None:
            check_independent(ecfg["spec_guided"]["model"], fz.authors)
    except (FreezeError, BudgetError, ReviewConfigError) as e:
        print(f"evidence.run: refused: {e}", file=sys.stderr)
        return 2
    manifest = [m for m in rows if a.partition is None or m["partition"] == a.partition]
    adm_path = out / "admission.jsonl"
    if adm_path.exists() and not a.include_unadmitted:
        bad = {(x["task"], x["mode"]) for x in read_jsonl(adm_path)
               if not (x.get("safe_passes_battery") and x.get("oracle_catches_unsafe") and x.get("both_apply_cleanly"))}
        before = len(manifest)
        manifest = [m for m in manifest if (m["task"], m["mode"]) not in bad]
        print(f"skipping {before - len(manifest)} candidates whose pair failed mechanical admission", file=sys.stderr)
    if a.limit:
        manifest = manifest[:a.limit]
    records_path = out / "records.jsonl"
    done = done_cells(records_path)
    co = Checkout(repo)
    registries = {(str(Path(ecfg["python"]).expanduser()), baseline): reg}   # per (interpreter, unit)
    _baselines: dict = {}

    def baseline_at(commit: str, py: str) -> Optional[float]:
        """The CI-equivalent unit for `commit`. ci_equiv is a producer's CPU over the
        median CPU of the repository's own suite at the pinned state, which charges
        standard CI 1.0 by construction -- so with candidates at their own base
        commits the unit is per commit. A baseline the config names outright still
        wins, for a single-commit corpus."""
        if not cfg.get("env"):
            # one environment for the corpus means one pinned state and one unit; the
            # config's ci_baseline is it. Per-commit resolution needs per-commit
            # environments to measure in.
            return baseline
        if commit not in _baselines:
            spec = dict(f1_spec_at(ecfg, out, commit), repo=cfg.get("repo", "?"))
            try:
                _baselines[commit] = ensure_baseline(repo, commit, py, spec, root=out / "baselines")
            except Exception as e:                       # noqa: BLE001  a run without a unit meters 0
                print(f"WARNING: no CI baseline at {commit[:12]}: {e}; ci_equiv will be 0",
                      file=sys.stderr)
                _baselines[commit] = None
        return _baselines[commit]

    def actions_at(commit: str):
        """The actions bound to `commit`'s interpreter (corpus.envs) and its own CI
        baseline; built before the checkout moves."""
        py = str(ensure_env(repo, commit, dict(cfg, python=ecfg["python"])))
        bl = baseline_at(commit, py)
        spec = f1_spec_at(ecfg, out, commit)          # the base commit's red tests deselected (corpus.ci_base)
        key = (py, bl, tuple(spec.get("extra_deselect", ())))
        if key not in registries:
            registries[key] = build_registry(dict(ecfg, python=py, **({"f1": spec} if "f1" in ecfg else {})), base_dir,
                                             ci_baseline_cpu_s=bl, dev=is_dev(fz), host_tools=pins, semgrep=semgrep)
        return registries[key].actions()

    n = 0
    for m in manifest:
        cid = m["candidate_id"]
        reps = [rep for rep in range(a.reps) if any((cid, rep, act.id) not in done for act in reg.actions())]
        if not reps:
            continue
        actions = actions_at(m["commit"])
        for rep in reps:
            todo = [act for act in actions if (cid, rep, act.id) not in done]
            co.reset(m["commit"])
            if not co.apply(out / m["diff_path"]):
                print(f"apply failed: {m['candidate_id']}", file=sys.stderr); continue
            recs = []
            for act in todo:
                ctx = Context(m["candidate_id"], repo, m["commit"], rep,
                              cell_seed(salt, m["candidate_id"], rep, act.id),
                              cfg["package_root"], baseline)
                recs.append(produce(act, ctx))
            append_records(records_path, recs)
            n += len(recs)
            print(f"{m['candidate_id']} rep {rep}: " + ", ".join(f"{r.action_id}={r.outcome}" for r in recs), flush=True)
    co.reset(manifest[0]["commit"]) if manifest else None
    print(f"{n} records written to {records_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

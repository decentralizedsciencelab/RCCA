"""Producer conformance suite (evidence-layer spec §12) on a golden fixture repo."""
import json, subprocess, sys
from pathlib import Path
import pytest
from evidence.core import Context, Registry, Meter, produce, cell_seed, OUTCOMES, Validation, Outcome, Action
from evidence.producers import f1_tests
from evidence import baseline

PKG = '''\
def add(a, b):
    return a + b

def clamp(x, lo, hi):
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x
'''
TESTS = '''\
from pkg.mod import add, clamp

def test_add():
    assert add(2, 3) == 5

def test_clamp_low():
    assert clamp(-1, 0, 10) == 0

def test_clamp_high():
    assert clamp(99, 0, 10) == 10
'''
FLAKY = '''\
import os
from pkg.mod import add

def test_add():
    assert add(2, 3) == 5

def test_flaky():
    # fails on the first invocation only, like an order- or state-dependent test
    marker = os.path.join(os.path.dirname(__file__), ".ran_once")
    first = not os.path.exists(marker)
    open(marker, "w").close()
    assert not first
'''


def _git(repo, *a):
    return subprocess.run(["git", *a], cwd=repo, capture_output=True, text=True, check=True)


@pytest.fixture
def fixture_repo(tmp_path):
    r = tmp_path / "repo"; (r / "pkg").mkdir(parents=True); (r / "tests").mkdir()
    (r / "pkg/__init__.py").write_text(""); (r / "pkg/mod.py").write_text(PKG)
    (r / "tests/test_mod.py").write_text(TESTS)
    (r / "pytest.ini").write_text("[pytest]\npythonpath = .\n")
    _git(r, "init", "-q"); _git(r, "config", "user.email", "t@t"); _git(r, "config", "user.name", "t")
    _git(r, "add", "-A"); _git(r, "commit", "-qm", "base")
    return r, _git(r, "rev-parse", "HEAD").stdout.strip()


def _ctx(r, sha, cid, rep=0, baseline_cpu=None):
    return Context(cid, r, sha, rep, cell_seed("salt", cid, rep, "f1.repo_ci"), "pkg", baseline_cpu)


def _action():
    reg = Registry()
    return f1_tests.register(reg, sys.executable, ["tests"], "pkg", k=2, timeout=120)


def test_pass_and_cost_and_alphabet(fixture_repo):
    r, sha = fixture_repo
    rec = produce(_action(), _ctx(r, sha, "t:s:m0", 0, baseline_cpu=0.5))
    assert rec.outcome == "pass" and rec.status == "ok"
    assert rec.outcome in OUTCOMES
    c = rec.cost
    assert c.tokens == 0 and c.tool_calls >= 2 and c.cpu_s >= 0 and c.wall_s > 0 and abs(c.ci_equiv - c.cpu_s / 0.5) < 1e-3
    assert rec.to_dict()["validated"] is False and "coverage_pct" in rec.provenance
    assert rec.blind_spots


def test_planted_defect_is_fail_validated_and_deterministic(fixture_repo):
    r, sha = fixture_repo
    (r / "pkg/mod.py").write_text(PKG.replace("if x > hi:", "if x >= hi:").replace("return hi", "return hi - 1"))
    a = produce(_action(), _ctx(r, sha, "t:u:m0", 0))
    b = produce(_action(), _ctx(r, sha, "t:u:m0", 0))
    assert a.outcome == "fail_validated" and a.validation.all_four
    assert a.validation.witness_ref == "tests/test_mod.py::test_clamp_high"
    assert a.validation.confirmer_id == "pytest-isolated-rerun" and len(a.validation.sub_seeds) == 2
    # determinism modulo timestamps and measured cost
    assert (a.outcome, a.validation, a.provenance["failing_tests"]) == (b.outcome, b.validation, b.provenance["failing_tests"])
    assert a.seed == b.seed


def test_flaky_failure_is_unvalidated(fixture_repo):
    r, sha = fixture_repo
    (r / "tests/test_mod.py").write_text(FLAKY)
    rec = produce(_action(), _ctx(r, sha, "t:f:m0", 0))
    assert rec.outcome == "fail_unvalidated"
    assert rec.validation.contradicts_reference and not rec.validation.repeats_across_seeds and not rec.validation.second_tool_confirmed


def test_broken_suite_is_inconclusive_not_pass(fixture_repo):
    r, sha = fixture_repo
    (r / "tests/test_mod.py").write_text("from pkg.nothere import x\n")
    rec = produce(_action(), _ctx(r, sha, "t:b:m0", 0))
    assert rec.outcome == "inconclusive" and rec.status == "error"


def test_validation_soundness_is_enforced_by_produce(fixture_repo):
    r, sha = fixture_repo
    def bad(ctx, m):
        return Outcome("fail_validated", Validation(contradicts_reference=True))   # claims validated without criteria
    rec = produce(Action("x.bad", "F2", "author_side", "fake", bad), _ctx(r, sha, "t:s:m0"))
    assert rec.outcome == "fail_unvalidated"


def test_exceptions_and_timeouts_become_inconclusive(fixture_repo):
    r, sha = fixture_repo
    def boom(ctx, m):
        raise RuntimeError("tool crashed")
    rec = produce(Action("x.boom", "F3", "challenger", "fake", boom), _ctx(r, sha, "t:s:m0"))
    assert rec.outcome == "inconclusive" and rec.status == "error" and "tool crashed" in rec.provenance["error"]
    def slow(ctx, m):
        m.tool([sys.executable, "-c", "import time; time.sleep(5)"], ctx.repo, timeout=1)
        return Outcome("pass")
    rec = produce(Action("x.slow", "F3", "challenger", "fake", slow), _ctx(r, sha, "t:s:m0"))
    assert rec.outcome == "inconclusive" and rec.status == "timeout"


def test_meter_scrubs_proxies_and_counts_calls(fixture_repo, monkeypatch):
    r, sha = fixture_repo
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy:1")
    m = Meter()
    with m:
        out = m.tool([sys.executable, "-c", "import os; print(os.environ.get('HTTPS_PROXY','none'), os.environ['NO_NETWORK'])"], r, 30)
        m.tool([sys.executable, "-c", "pass"], r, 30)
    assert out.stdout.split() == ["none", "1"] and m.tool_calls == 2 and m.wall_s > 0


def test_baseline_median_and_ci_equiv_of_one(fixture_repo, tmp_path):
    r, sha = fixture_repo
    cmd = [sys.executable, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"]
    b = baseline.measure("fixture", r, sha, cmd, n=3, timeout=120)
    assert b.n == 3 and len(b.cpu_s_runs) == 3 and b.cpu_s_median >= 0
    baseline.save(b, tmp_path / "b.json"); assert baseline.load(tmp_path / "b.json").cpu_s_median == b.cpu_s_median
    # A CI-shaped producer run should land near 1.0 CI-equivalents (within run-to-run noise plus the coverage overhead)
    rec = produce(_action(), _ctx(r, sha, "t:s:m0", 0, baseline_cpu=b.cpu_s_median))
    assert 0.3 < rec.cost.ci_equiv < 6.0


def test_challenger_budget_in_ci_equiv(tmp_path):
    """The challenger's search budget is set in standard-CI equivalents and spent in seconds on the
    machine at hand: seconds = budget_ci_equiv x the baseline's median CPU seconds. A budget in
    wall-clock seconds buys more search per CI-equivalent on a faster machine (results.md E2), so only
    a development config may still give budget_s, and its records say so. Needs neither hypothesis
    nor atheris, so it runs where tests/test_challenger.py is skipped."""
    import time
    from evidence.challenger.common import BudgetError, challenger_config, resolve_budget, run_technique
    from evidence.run import build_registry
    py = sys.executable
    c = {"budget_ci_equiv": 12.5, "fuzz_time_per_target_ci_equiv": 2.5, "call_timeout": 5, "techniques": []}
    slow = challenger_config(c, py, ci_baseline_cpu_s=2.0, dev=False)
    fast = challenger_config(c, py, ci_baseline_cpu_s=0.5, dev=False)       # standard CI runs 4x faster here
    dev = challenger_config({"budget_s": 25, "techniques": []}, py, ci_baseline_cpu_s=2.0, dev=True)
    assert (slow.budget_s, fast.budget_s, dev.budget_s) == (25.0, 6.25, 25.0)
    assert resolve_budget(c, "fuzz_time_per_target", 2.0, dev=False) == (5.0, 2.5)
    assert slow.call_timeout == 5                       # a safety timeout in seconds, not a budget
    # every challenger record carries the budget as set, the baseline, and the seconds that bound its search
    r = tmp_path / "ch"
    (r / "src/pkg").mkdir(parents=True)
    (r / "src/pkg/__init__.py").write_text(""); (r / "src/pkg/mod.py").write_text(PKG)
    _git(r, "init", "-q"); _git(r, "config", "user.email", "t@t"); _git(r, "config", "user.name", "t")
    _git(r, "add", "-A"); _git(r, "commit", "-qm", "base")
    sha = _git(r, "rev-parse", "HEAD").stdout.strip()
    (r / "src/pkg/mod.py").write_text(PKG.replace("return a + b", "return a - b"))     # one changed target
    for cfg, ci, base, secs in ((slow, "12.5", "2.0", 25.0), (fast, "12.5", "0.5", 6.25), (dev, None, "2.0", 25.0)):
        left = []

        def search(targets, pair, base_tree, deadline):
            left.append(deadline - time.monotonic())
            return iter(())
        act = Action("ch.probe", "F1", "challenger", "probe",
                     lambda ctx, m, cfg=cfg, search=search: run_technique(ctx, m, cfg, "probe", "none", search))
        rec = produce(act, Context("t:s:m0", r, sha, 0, cell_seed("salt", "t:s:m0", 0, "ch.probe"), "src/pkg"))
        prov = rec.provenance
        assert rec.outcome == "pass" and (prov.get("budget_ci_equiv"), prov["ci_baseline_cpu_s"], prov["budget_s"]) \
            == (ci, base, str(secs)), prov
        assert len(left) == 1 and secs - 3.0 < left[0] <= secs           # the derived seconds are the deadline
        assert ("budget_note" in prov) == (ci is None)                   # only the development record has a note
    assert "budget_s" in prov["budget_note"] and "seconds" in prov["budget_note"]
    # a budget in seconds is refused under a freeze record; one in CI-equivalents needs the baseline
    with pytest.raises(BudgetError, match="budget_ci_equiv"):
        challenger_config({"budget_s": 25}, py, ci_baseline_cpu_s=2.0, dev=False)
    with pytest.raises(BudgetError, match="ci_baseline"):
        challenger_config(c, py, ci_baseline_cpu_s=None, dev=True)
    # evidence.run resolves the budgets before importing any technique
    ecfg = {"python": py, "challenger": {"budget_ci_equiv": 12.5, "techniques": ["fuzz"]}}
    with pytest.raises(BudgetError, match="fuzz_time_per_target_ci_equiv"):
        build_registry(ecfg, tmp_path, ci_baseline_cpu_s=2.0, dev=False)
    with pytest.raises(BudgetError, match="budget_ci_equiv"):
        build_registry({"python": py, "challenger": {"budget_s": 25, "techniques": []}}, tmp_path,
                       ci_baseline_cpu_s=2.0, dev=False)
    assert build_registry({"python": py, "challenger": {"budget_s": 25, "techniques": []}}, tmp_path,
                          ci_baseline_cpu_s=2.0, dev=True).actions() == []


def test_challenger_budget_refuses_a_search_that_would_not_run():
    """A zero budget or a zero baseline puts the deadline at the start, and the challenger would report
    `pass` for a search it never ran. Those are refused, as are a budget given in both units and a
    budget in seconds anywhere under a freeze record, the fuzzer's per-target time included."""
    from evidence.challenger.common import BudgetError, challenger_config, resolve_budget
    py = sys.executable
    with pytest.raises(BudgetError, match="fuzz_time_per_target_ci_equiv"):      # each budget, on its own
        resolve_budget({"fuzz_time_per_target_s": 20}, "fuzz_time_per_target", 2.0, dev=False)
    assert resolve_budget({"fuzz_time_per_target_s": 20}, "fuzz_time_per_target", 2.0, dev=True) == (20.0, None)
    refused = [({"budget_ci_equiv": 0}, 2.0, False, "positive number"),
               ({"budget_ci_equiv": "12.5"}, 2.0, False, "positive number"),
               ({"budget_ci_equiv": 12.5}, 0.0, False, "baseline"),
               ({"budget_ci_equiv": 12.5, "budget_s": 25}, 2.0, True, "both"),
               ({"budget_ci_equiv": 12.5, "fuzz_time_per_target_s": 20}, 2.0, False, "fuzz_time_per_target_s")]
    for c, base, dev, why in refused:
        with pytest.raises(BudgetError, match=why):
            challenger_config(c, py, ci_baseline_cpu_s=base, dev=dev)
    # each differs from a refused one in one input
    assert challenger_config({"budget_ci_equiv": 12.5}, py, ci_baseline_cpu_s=2.0, dev=False).budget_s == 25.0
    dev = challenger_config({"budget_ci_equiv": 12.5, "fuzz_time_per_target_s": 20}, py, ci_baseline_cpu_s=2.0, dev=True)
    assert dev.budget_s == 25.0 and "fuzz_time_per_target_s" in dev.budget_provenance()["budget_note"]


def test_records_roundtrip_through_fit_loader(fixture_repo, tmp_path):
    from evidence.core import append_records
    from rcca.fit import load_matrices
    r, sha = fixture_repo
    recs = [produce(_action(), _ctx(r, sha, "t1:s:m0", rep)) for rep in range(2)]
    append_records(tmp_path / "records.jsonl", recs)
    (tmp_path / "manifest.jsonl").write_text(json.dumps({"candidate_id": "t1:s:m0", "task": "t1", "repo": "fx", "mode": "m0",
                                                         "safe_twin": "t1:u:m0", "partition": "test", "commit": sha,
                                                         "risk_features": {"risk_logit": -1.0}}) + "\n")
    ms = load_matrices(tmp_path / "records.jsonl", tmp_path / "manifest.jsonl", None, "test")
    assert len(ms) == 2 and ms[0].records["f1.repo_ci"].outcome.value == "pass" and ms[0].records["f1.repo_ci"].cost.tool_calls >= 2


def test_oracle_paths_absent_from_provenance(fixture_repo, tmp_path):
    """The hidden oracle never reaches a producer: after it has judged a candidate the
    checkout holds the candidate and nothing of the oracle, and neither the context a producer gets nor
    any record a producer writes names the oracle's modules, its battery directory or a battery test."""
    import shutil
    from corpus import oracle
    from corpus.manifest import Candidate
    from evidence.run import build_registry
    r, sha = fixture_repo
    root = Path(__file__).resolve().parents[1]
    battery_root = tmp_path / "oracle"
    (battery_root / "fx").mkdir(parents=True)
    shutil.copy(root / "oracle" / "conftest.py", battery_root / "conftest.py")
    (battery_root / "fx" / "test_battery_core_mod.py").write_text(
        "import pytest\n\nfrom pkg.mod import clamp\n\n\n"
        "@pytest.mark.parametrize('x,want', [(-1, 0), (5, 5), (99, 10)])\n"
        "def test_clamp(observe, x, want):\n    observe('clamp', clamp(x, 0, 10))\n    assert clamp(x, 0, 10) == want\n")
    (r / "pkg/mod.py").write_text(PKG.replace("if x > hi:", "if x >= hi:").replace("return hi", "return hi - 1"))
    report = {}
    cand = Candidate("t:u:m0", "t", "fx", "m0", "t:s:m0", "train", sha, "diffs/t:u:m0.diff")
    assert oracle.catches(r, cand, "", env_python=Path(sys.executable), battery_dir=battery_root / "fx",
                          package_root="pkg", report=report), report
    assert _git(r, "status", "--porcelain").stdout == " M pkg/mod.py\n"
    ecfg = {"python": sys.executable, "f1": {"test_args": ["tests"], "source": "pkg"},
            "f2_static": {"semgrep_rules": [], "target": "pkg"}, "f3": {"package": "pkg", "search": "."},
            "f4": {"top_module": "pkg", "tests_dir": "tests"}}
    reg = build_registry(ecfg, tmp_path, dev=True)
    ctx = _ctx(r, sha, "t:u:m0")
    records = [produce(a, ctx) for a in reg.actions()]
    assert {rec.action_id for rec in records} == {"f1.repo_ci", "f2.static_security", "f3.api_contract",
                                                   "f4.impact_revert"}
    needles = [str(battery_root), str(root / "oracle"), "test_battery", "corpus.oracle", "corpus/oracle",
               "oracle_scanner", "oracle_collect"]
    seen = json.dumps({k: str(v) for k, v in vars(ctx).items()})
    assert not [n for n in needles if n in seen], seen
    for rec in records:
        blob = json.dumps(rec.to_dict())
        assert not [n for n in needles if n in blob], (rec.action_id, blob)
def test_evidence_configs_name_an_interpreter_that_resolves_absolutely():
    """Every producer runs with the candidate checkout as its cwd, so a
    repository-relative interpreter path in the evidence config resolves against the
    wrong directory. The dev config used `~/...`, which `expanduser` makes absolute,
    which is why the dry run never hit this; both real configs are repo-relative and
    would have failed on the first producer invocation."""
    import json
    from pathlib import Path
    from evidence.core import resolve_interpreter
    for cfg in ("configs/evidence_flask.json", "configs/evidence_click.json"):
        raw = json.loads(Path(cfg).read_text())["python"]
        p = Path(resolve_interpreter(raw))
        assert p.is_absolute(), (cfg, raw, str(p))
        assert p.exists(), f"{cfg}: {p} does not exist; is the per-repo venv built?"
        # and it must still be the *virtual* environment's interpreter: `.venv/bin/
        # python` is a symlink to the base interpreter, so resolving symlinks would
        # silently run every producer outside the environment the subject is
        # installed in -- which fails as a missing module, not as a wrong answer
        import subprocess
        prefix = subprocess.run([str(p), "-c", "import sys; print(sys.prefix, sys.base_prefix)"],
                                capture_output=True, text=True, check=True).stdout.split()
        assert prefix[0] != prefix[1], f"{cfg}: {p} is not a venv interpreter (prefix == base_prefix)"
        assert str(Path(raw).parent.parent) in prefix[0], (cfg, raw, prefix[0])


def test_the_challenger_interpreter_is_resolved_too():
    """The challenger config carries its own `python`, overriding the evidence
    config's. It runs with a git worktree as cwd, so a repository-relative path
    there fails exactly as the main one did -- and fixing only the main one leaves
    ch.property and ch.differential erroring with FileNotFoundError on every
    candidate, which is what the 2026-10-19 pilot run showed."""
    import json
    from pathlib import Path
    from evidence.core import resolve_interpreter
    cfg = json.loads(Path("configs/evidence_flask.json").read_text())
    raw = cfg["challenger"].get("python") or cfg["python"]
    p = Path(resolve_interpreter(raw))
    assert p.is_absolute() and p.exists(), (raw, str(p))


def test_ci_baseline_is_resolved_per_base_commit(tmp_path):
    """The design charges standard CI "exactly 1.0 by construction" : ci_equiv is a producer's CPU over the median CPU of the
    repository's own suite at the pinned state. With one baseline for a corpus whose
    candidates sit at their own base commits, that property fails -- measured on the
    2026-10-19 pilot, f1.repo_ci ranged 1.176 to 7.851 CI-equivalents, a 6.7x spread
    on a quantity defined to be 1.0.

    The ceilings are quantiles of these costs, so a wrong unit makes the equal-resources
    comparison wrong too. The baseline is therefore per base commit and cached."""
    from evidence.baseline import ensure_baseline
    built = []

    def fake(repo, commit, python, spec, out):
        built.append(commit)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"repo": "r", "commit": commit, "command": ["x"], "n": 5,
                                   "cpu_s_runs": [1.0] * 5, "wall_s_runs": [1.0] * 5,
                                   "cpu_s_median": 2.5 if commit == "c0ffee" else 7.5,
                                   "cpu_s_spread": 0.0, "wall_s_median": 1.0}))

    spec = {"test_args": ["tests"], "source": "pkg"}
    a = ensure_baseline(tmp_path / "repo", "c0ffee", "py", spec, root=tmp_path / "bl", builder=fake)
    assert a == 2.5
    ensure_baseline(tmp_path / "repo", "c0ffee", "py", spec, root=tmp_path / "bl", builder=fake)
    assert built == ["c0ffee"], built                 # cached, not remeasured
    b = ensure_baseline(tmp_path / "repo", "decaf1", "py", spec, root=tmp_path / "bl", builder=fake)
    assert b == 7.5 and built == ["c0ffee", "decaf1"]  # a different commit, its own unit


# ---------------------------------------------------------------- the base commit's red tests, in F1 too

def test_f1_spec_at_a_commit_deselects_the_suite_tests_red_at_its_base(tmp_path):
    """F1 is standard CI; a test the base commit fails with no change is not evidence about the
    change, so the same control admission applies (corpus.ci_base) applies here, per commit, and to
    the CI baseline measured from the same spec."""
    import json
    from evidence.run import f1_spec_at
    out = tmp_path / "corpus-dev"
    (out / "ci_base" / "c0ffee").mkdir(parents=True)
    (out / "ci_base" / "c0ffee" / "ci_base.json").write_text(json.dumps(
        {"commit": "c0ffee", "failing": ["tests/test_instance_config.py::test_egg_installed_paths", "tests/t.py::cfg"]}))
    ecfg = {"python": "py", "f1": {"test_args": ["tests"], "source": "flask", "extra_deselect": ["tests/t.py::cfg"]}}
    spec = f1_spec_at(ecfg, out, "c0ffee")
    assert spec["extra_deselect"] == ["tests/t.py::cfg", "tests/test_instance_config.py::test_egg_installed_paths"]
    assert spec["test_args"] == ["tests"] and ecfg["f1"]["extra_deselect"] == ["tests/t.py::cfg"]   # the config is untouched
    assert f1_spec_at(ecfg, out, "decaf1")["extra_deselect"] == ["tests/t.py::cfg"]                 # nothing measured: unchanged


def test_f1_timeout_comes_from_the_evidence_config(tmp_path, monkeypatch):
    """A seeded mutation can hang the repository suite (click-303: every option became a boolean
    flag). F1 then waits its full timeout and records inconclusive; at 1,200 s that is twenty minutes
    per such candidate, so the config sets it, and the suites here run in seconds."""
    from evidence import run as R
    from evidence.producers import f1_tests
    seen = {}
    real = f1_tests.make_repo_ci

    def spy(python, test_args, source, **kw):
        seen.update(kw); return real(python, test_args, source, **kw)
    monkeypatch.setattr(f1_tests, "make_repo_ci", spy)
    ecfg = {"python": sys.executable, "f1": {"test_args": ["tests"], "source": "pkg", "timeout": 7}}
    R.build_registry(ecfg, tmp_path, ci_baseline_cpu_s=1.0, dev=True)
    assert seen.get("timeout") == 7

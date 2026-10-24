import json, shutil, subprocess, sys
from pathlib import Path
import pytest
pytest.importorskip("hypothesis", reason="the challenger tests need hypothesis in the test interpreter")
from evidence.core import Context, Meter, produce, cell_seed
from evidence.challenger import ChallengerConfig
from evidence.challenger.common import challenger_config
from evidence.challenger.targets import changed_functions
from evidence.challenger.harness import Worktree, Pair, minimize, witness_source
from evidence.challenger import harness as chharness
from evidence.challenger import property as chprop, differential as chdiff, fuzz as chfuzz
from evidence.challenger.declared import declared_targets, is_test_file
from evidence.challenger.strategies import runnable, fuzzable
from evidence.diffutil import changed_files, changed_lines

PY = sys.executable
PARSE = '''\
import random

def clamp(x: int, lo: int, hi: int) -> int:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x

def parse_kv(s: str) -> dict:
    out = {}
    for part in s.split(";"):
        if not part:
            continue
        k, _, v = part.partition("=")
        out[k.strip()] = v.strip()
    return out

def shout(s: str) -> str:
    return s.upper()

def roll(n: int) -> int:
    return random.randint(0, max(0, n))

def summarize(s: str) -> int:
    return len(parse_kv(s))

def _private(x):
    return x
'''
COUNTER = '''\
class Clock:
    def now(self) -> float:
        return 0.0


class App:
    def __init__(self, name: str, clock: Clock):
        self.name = name
        self.clock = clock

    def banner(self, width: int) -> str:
        return self.name.center(width, "*")


class Counter:
    def __init__(self, start: int = 0, name: str = "c"):
        self.n = start
        self.name = name

    def add(self, k: int) -> int:
        self.n += k
        return self.n

    @staticmethod
    def clip(x: int, hi: int) -> int:
        return hi if x > hi else x

    @classmethod
    def tidy(cls, name: str) -> str:
        return name.strip()

    @property
    def label(self) -> str:
        return self.name

    def describe(self, app: App) -> str:
        return f"{self.name}:{app.name}"

    def _hidden(self):
        return 1


def greet(app: App, times: int = 1) -> str:
    return (app.name + "!") * times
'''
TESTS = '''\
from pkg.parse import clamp, parse_kv, shout
from pkg.counter import App, Clock, Counter

def test_clamp():
    assert clamp(5, 0, 10) == 5

def test_parse():
    assert parse_kv("a=1;b=2") == {"a": "1", "b": "2"}
    assert isinstance(parse_kv("a=;b=2;c=3"), dict)      # weak: F1 does not pin the value

def test_shout():
    assert shout("x") == "X"

def test_clip():
    assert Counter.clip(5, 3) == 3

def test_describe():
    assert Counter().describe(App("z", Clock())) == "c:z"

def test_banner():
    assert App("z", Clock()).banner(5) == "**z**"
'''


def _git(repo, *a):
    return subprocess.run(["git", *a], cwd=repo, capture_output=True, text=True, check=True)


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"; (r / "src/pkg").mkdir(parents=True); (r / "tests").mkdir()
    (r / "src/pkg/__init__.py").write_text(""); (r / "src/pkg/parse.py").write_text(PARSE)
    (r / "src/pkg/counter.py").write_text(COUNTER)
    (r / "tests/test_parse.py").write_text(TESTS)
    _git(r, "init", "-q"); _git(r, "config", "user.email", "t@t"); _git(r, "config", "user.name", "t")
    _git(r, "add", "-A"); _git(r, "commit", "-qm", "base")
    return r, _git(r, "rev-parse", "HEAD").stdout.strip()


def _ctx(r, sha, cid="t:u:m0", aid="ch.property"):
    return Context(cid, r, sha, 0, cell_seed("salt", cid, 0, aid), "src/pkg")


CFG = ChallengerConfig(PY, "src", budget_s=90, k=2, call_timeout=5, minimize_budget=60)
APP_X = "importlib.import_module('pkg.counter').App('x', importlib.import_module('pkg.counter').Clock())"


def test_targets_detect_changed_public_functions_only(repo):
    r, sha = repo
    (r / "src/pkg/parse.py").write_text(PARSE.replace("if x > hi:", "if x >= hi:").replace("def _private(x):\n    return x", "def _private(x):\n    return x + 1"))
    ts = changed_functions(r, sha, "src/pkg", "src")
    assert [(t.module, t.qualname, t.status) for t in ts] == [("pkg.parse", "clamp", "changed")]
    assert [(p.name, p.annotation, p.has_default) for p in ts[0].params] == [("x", "int", False), ("lo", "int", False), ("hi", "int", False)]


def test_pair_executes_in_both_trees_and_guards_nondeterminism(repo):
    r, sha = repo
    (r / "src/pkg/parse.py").write_text(PARSE.replace("return random.randint(0, max(0, n))", "return random.randint(0, max(0, n)) + 1"))
    ts = {t.qualname: t for t in changed_functions(r, sha, "src/pkg", "src")}
    m = Meter()
    with m, Worktree(r, sha) as base:
        assert (base / "src/pkg/parse.py").read_text() == PARSE          # base tree is the pinned state
        pair = Pair(PY, base, r, "src", m)
        try:
            assert pair.diverges(ts["roll"], [100], {}) is None and ("pkg.parse", "roll") in pair.unstable
        finally:
            pair.close()
    assert m.tool_calls >= 2 and m.cpu_s >= 0


def test_property_finds_shrinks_and_validates_off_by_one(repo):
    r, sha = repo
    (r / "src/pkg/parse.py").write_text(PARSE.replace("if x > hi:\n        return hi", "if x >= hi:\n        return hi - 1"))
    rec = produce(chprop.make_property(CFG, max_examples=200), _ctx(r, sha))
    assert rec.outcome == "fail_validated", (rec.status, rec.provenance)
    v = rec.validation
    assert v.all_four and v.confirmer_id == "pytest-witness" and len(v.sub_seeds) == 2
    assert "clamp(" in v.witness_ref
    assert rec.artifacts and rec.artifacts[0]["name"] == "test_rcca_witness.py" and "def test_rcca_witness" in rec.artifacts[0]["content"]
    assert 0 < rec.search_budget_used <= 1.0
    # the witness itself is a runnable pytest file: fails on the candidate tree, passes on the base tree
    wf = r.parent / "test_rcca_witness.py"; wf.write_text(rec.artifacts[0]["content"])
    env = {"PYTHONPATH": str(r / "src"), "PATH": "/usr/bin:/bin"}
    assert subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(wf)], cwd=r, env=env, capture_output=True).returncode == 1


def test_property_passes_on_behaviour_preserving_change(repo):
    r, sha = repo
    (r / "src/pkg/parse.py").write_text(PARSE.replace("    return s.upper()", "    result = s.upper()\n    return result"))
    rec = produce(chprop.make_property(CFG, max_examples=150), _ctx(r, sha))
    assert rec.outcome == "pass" and rec.provenance["targets_searched"] == "1" and rec.provenance["divergences_seen"] == "0"
    assert rec.search_budget_used > 0
    import hypothesis                       # generation and shrinking run here, in the orchestrator
    assert rec.provenance["hypothesis_version"] == hypothesis.__version__


def test_property_inconclusive_without_targets(repo):
    r, sha = repo
    (r / "src/pkg/parse.py").write_text(PARSE.replace("def _private(x):\n    return x", "def _private(x):\n    return x * 2"))
    rec = produce(chprop.make_property(CFG), _ctx(r, sha))
    assert rec.outcome == "inconclusive" and rec.status == "error" and rec.search_budget_used == 0.0


def test_differential_replays_recorded_inputs_and_minimizes(repo):
    r, sha = repo
    (r / "src/pkg/parse.py").write_text(PARSE.replace("        out[k.strip()] = v.strip()", "        if v.strip():\n            out[k.strip()] = v.strip()"))
    rec = produce(chdiff.make_differential(CFG, test_args=("tests",)), _ctx(r, sha, aid="ch.differential"))
    assert rec.outcome == "fail_validated", (rec.status, rec.provenance)
    assert int(rec.provenance["recorded_inputs"]) >= 1
    # minimized: the recorded input was "a=;b=2;c=3"; ddmin must reach a shorter witness that still diverges
    assert "parse_kv(" in rec.validation.witness_ref
    arg = rec.validation.witness_ref.split("parse_kv(")[1].split(")")[0]
    assert len(arg) < len("'a=;b=2;c=3'")


def test_differential_inconclusive_when_nothing_recorded(repo):
    r, sha = repo
    (r / "src/pkg/parse.py").write_text(PARSE.replace("return random.randint(0, max(0, n))", "return random.randint(0, max(0, n)) + 1"))
    rec = produce(chdiff.make_differential(CFG, test_args=("tests",)), _ctx(r, sha, aid="ch.differential"))
    assert rec.outcome == "inconclusive" and rec.provenance["recorded_inputs"] == "0"


def _has_atheris():
    return subprocess.run([PY, "-c", "import atheris"], capture_output=True).returncode == 0


@pytest.mark.skipif(not _has_atheris(), reason="atheris not installed in the test interpreter")
def test_fuzz_finds_crash_absent_from_base(repo):
    r, sha = repo
    (r / "src/pkg/parse.py").write_text(PARSE.replace("    return s.upper()", "    return s.upper()[0] + s.upper()[1:]"))
    rec = produce(chfuzz.make_fuzz(CFG, time_per_target_s=8), _ctx(r, sha, aid="ch.fuzz"))
    assert rec.outcome == "fail_validated", (rec.status, rec.provenance)
    assert "shout(" in rec.validation.witness_ref and "IndexError" in rec.validation.witness_ref


def test_fuzz_inconclusive_without_atheris(repo):
    r, sha = repo
    no_atheris = shutil.which("false") or "/bin/false"             # macOS has only /usr/bin/false
    cfg = challenger_config({"budget_ci_equiv": 2.5}, no_atheris, ci_baseline_cpu_s=2.0, dev=False)
    rec = produce(chfuzz.make_fuzz(cfg, time_per_target_s=5.0, time_per_target_ci_equiv=2.5), _ctx(r, sha, aid="ch.fuzz"))
    assert rec.outcome == "inconclusive" and "atheris" in rec.provenance["reason"]
    # the search never ran, and the record still says the budget it had, as set and in seconds
    keys = ("budget_ci_equiv", "ci_baseline_cpu_s", "budget_s", "time_per_target_ci_equiv", "time_per_target_s")
    assert [rec.provenance.get(k) for k in keys] == ["2.5", "2.0", "5.0", "2.5", "5.0"]


# ---------------------------------------------------------------------------
# methods, factories, declared changes
# ---------------------------------------------------------------------------

def test_targets_include_public_methods_with_constructor_params(repo):
    r, sha = repo
    (r / "src/pkg/counter.py").write_text(COUNTER.replace("        self.n += k\n", "        self.n += k + 1\n")
                                          .replace("return hi if x > hi else x", "return hi - 1 if x > hi else x")
                                          .replace("return name.strip()", "return name.strip().lower()")
                                          .replace("        return self.name\n", "        return self.name + \"!\"\n")
                                          .replace("        return 1\n", "        return 2\n"))
    ts = {t.qualname: t for t in changed_functions(r, sha, "src/pkg", "src")}
    assert set(ts) == {"Counter.add", "Counter.clip", "Counter.tidy"}          # property and _hidden excluded
    add = ts["Counter.add"]
    assert add.kind == "method" and add.cls == "Counter" and add.needs_receiver
    assert [(p.name, p.annotation, p.has_default) for p in add.params] == [("k", "int", False)]
    assert [(p.name, p.annotation, p.has_default) for p in add.ctor_params] == [("start", "int", True), ("name", "str", True)]
    assert ts["Counter.clip"].kind == "staticmethod" and not ts["Counter.clip"].needs_receiver
    assert ts["Counter.tidy"].kind == "classmethod" and [p.name for p in ts["Counter.tidy"].params] == ["name"]
    assert all(runnable(t) for t in ts.values()) and fuzzable(add)


def test_property_validates_method_divergence_through_receiver(repo):
    r, sha = repo
    (r / "src/pkg/counter.py").write_text(COUNTER.replace("        self.n += k\n", "        self.n += k + 1\n"))
    rec = produce(chprop.make_property(CFG, max_examples=200), _ctx(r, sha))
    assert rec.outcome == "fail_validated", (rec.status, rec.provenance)
    assert rec.provenance["targets_methods"] == "1" and rec.validation.all_four
    assert "Counter(" in rec.validation.witness_ref and ".add(" in rec.validation.witness_ref
    src = rec.artifacts[0]["content"]
    assert "RECEIVER" in src and "Counter" in src
    wf = r.parent / "test_rcca_witness.py"; wf.write_text(src)
    env = {"PYTHONPATH": str(r / "src"), "PATH": "/usr/bin:/bin"}
    assert subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(wf)], cwd=r, env=env, capture_output=True).returncode == 1
    with Worktree(r, sha) as base:
        env["PYTHONPATH"] = str(base / "src")
        assert subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(wf)], cwd=base, env=env, capture_output=True).returncode == 0


def test_differential_records_and_validates_staticmethod(repo):
    r, sha = repo
    (r / "src/pkg/counter.py").write_text(COUNTER.replace("return hi if x > hi else x", "return hi - 1 if x > hi else x"))
    rec = produce(chdiff.make_differential(CFG, test_args=("tests",)), _ctx(r, sha, aid="ch.differential"))
    assert rec.outcome == "fail_validated", (rec.status, rec.provenance)
    assert int(rec.provenance["recorded_inputs"]) >= 1 and "Counter.clip(" in rec.validation.witness_ref


def test_factory_makes_non_json_parameter_runnable(repo):
    r, sha = repo
    (r / "src/pkg/counter.py").write_text(COUNTER.replace('(app.name + "!") * times', '(app.name + "?") * times'))
    ts = changed_functions(r, sha, "src/pkg", "src")
    assert [t.qualname for t in ts] == ["greet"]
    assert not runnable(ts[0])
    rec = produce(chprop.make_property(CFG, max_examples=100), _ctx(r, sha))
    assert rec.outcome == "inconclusive" and rec.provenance["targets_searched"] == "0"
    chharness.configure({"App": APP_X})
    try:
        assert runnable(ts[0])
        rec = produce(chprop.make_property(CFG, max_examples=100), _ctx(r, sha))
        assert rec.outcome == "fail_validated", (rec.status, rec.provenance)
        assert "__factory__" in rec.artifacts[0]["content"] and "greet(" in rec.validation.witness_ref
        wf = r.parent / "test_rcca_witness.py"; wf.write_text(rec.artifacts[0]["content"])
        env = {"PYTHONPATH": str(r / "src"), "PATH": "/usr/bin:/bin"}
        assert subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(wf)], cwd=r, env=env, capture_output=True).returncode == 1
    finally:
        chharness.configure(None)


def test_declared_change_is_reported_unvalidated(repo):
    r, sha = repo
    (r / "src/pkg/parse.py").write_text(PARSE.replace("    return s.upper()", "    return s.upper() + \"!\""))
    (r / "tests/test_parse.py").write_text(TESTS.replace('assert shout("x") == "X"', 'assert shout("x") == "X!"'))
    ts = changed_functions(r, sha, "src/pkg", "src")
    assert declared_targets(r, sha, ts, PY, "src/pkg") == ({("pkg.parse", "shout"): "tests/test_parse.py::test_shout"}, "revert")
    assert declared_targets(r, sha, ts) == ({("pkg.parse", "shout"): "tests/test_parse.py"}, "names")   # fallback without an interpreter
    rec = produce(chprop.make_property(CFG, max_examples=150), _ctx(r, sha))
    assert rec.outcome == "fail_unvalidated", (rec.status, rec.provenance)
    assert rec.provenance["targets_declared"] == "1" and rec.provenance["declared_rule"] == "revert"
    assert rec.provenance["declared_by_candidate_tests"] == "tests/test_parse.py::test_shout"
    assert rec.validation.contradicts_reference and not rec.validation.minimized and not rec.validation.all_four
    assert rec.artifacts and "shout(" in rec.validation.witness_ref


def test_undeclared_validated_divergence_outranks_declared_one(repo):
    r, sha = repo
    (r / "src/pkg/parse.py").write_text(PARSE.replace("    return s.upper()", "    return s.upper() + \"!\"")
                                        .replace("if x > hi:\n        return hi", "if x >= hi:\n        return hi - 1"))
    (r / "tests/test_parse.py").write_text(TESTS.replace('assert shout("x") == "X"', 'assert shout("x") == "X!"'))
    rec = produce(chprop.make_property(CFG, max_examples=200), _ctx(r, sha))
    assert rec.outcome == "fail_validated", (rec.status, rec.provenance)
    assert "clamp(" in rec.validation.witness_ref and rec.provenance["targets_declared"] == "1"
    # test_clamp exists in the modified file but was not changed, so clamp is not declared by it
    assert rec.provenance["declared_targets"] == "pkg.parse.shout"


def test_declared_by_coverage_without_naming_the_callable(repo):
    r, sha = repo
    (r / "src/pkg/parse.py").write_text(PARSE.replace("        out[k.strip()] = v.strip()", "        if v.strip():\n            out[k.strip()] = v.strip()"))
    (r / "tests/test_summary.py").write_text("from pkg.parse import summarize\n\ndef test_summary():\n    assert summarize('a=;b=2') == 1\n")
    ts = changed_functions(r, sha, "src/pkg", "src")
    assert [t.qualname for t in ts] == ["parse_kv"]
    assert declared_targets(r, sha, ts)[0] == {}                                     # the name heuristic misses it
    assert declared_targets(r, sha, ts, PY, "src/pkg") == ({("pkg.parse", "parse_kv"): "tests/test_summary.py::test_summary"}, "revert")
    rec = produce(chprop.make_property(CFG, max_examples=150), _ctx(r, sha))
    assert rec.outcome == "fail_unvalidated" and rec.provenance["declared_rule"] == "revert", (rec.status, rec.provenance)


def test_pass_through_execution_does_not_declare(repo):
    """A changed test that merely executes a changed callable without distinguishing
    its new behaviour does not declare it (the seeded-defect-next-door case)."""
    r, sha = repo
    (r / "src/pkg/parse.py").write_text(PARSE.replace("    return s.upper()", "    return s.upper() + \"!\"")
                                        .replace("if x > hi:\n        return hi", "if x >= hi:\n        return hi - 1"))
    (r / "tests/test_parse.py").write_text(TESTS.replace('assert shout("x") == "X"', 'assert shout("x") == "X!"\n    assert clamp(5, 0, 10) == 5'))
    ts = changed_functions(r, sha, "src/pkg", "src")
    assert {t.qualname for t in ts} == {"clamp", "shout"}
    declared, rule = declared_targets(r, sha, ts, PY, "src/pkg")
    assert rule == "revert" and set(declared) == {("pkg.parse", "shout")}         # clamp is executed, not distinguished
    rec = produce(chprop.make_property(CFG, max_examples=200), _ctx(r, sha))
    assert rec.outcome == "fail_validated" and "clamp(" in rec.validation.witness_ref, (rec.status, rec.provenance)


def test_added_test_file_and_added_module_are_seen(repo):
    r, sha = repo
    (r / "src/pkg/extra.py").write_text("def double(x: int) -> int:\n    return x * 2\n")
    (r / "tests/test_extra.py").write_text("from pkg.extra import double\n\ndef test_double():\n    assert double(2) == 4\n")
    assert set(changed_files(r, sha)) == {"src/pkg/extra.py", "tests/test_extra.py"}
    assert changed_lines(r, sha)["src/pkg/extra.py"] == {1, 2}
    ts = changed_functions(r, sha, "src/pkg", "src")
    assert [(t.qualname, t.status) for t in ts] == [("double", "added")]
    declared, rule = declared_targets(r, sha, ts, PY, "src/pkg")
    assert rule == "revert" and set(declared) == {("pkg.extra", "double")}        # removing the added function breaks the new test
    assert is_test_file("tests/test_extra.py") and is_test_file("conftest.py") and not is_test_file("src/pkg/extra.py")


def test_factory_variants_reach_a_variant_specific_divergence(repo):
    r, sha = repo
    (r / "src/pkg/counter.py").write_text(COUNTER.replace('    return (app.name + "!") * times',
                                                          '    if app.name == "y":\n        return (app.name + "?") * times\n    return (app.name + "!") * times'))
    chharness.configure({"App": [APP_X, APP_X.replace("App('x'", "App('y'")]})
    try:
        rec = produce(chprop.make_property(CFG, max_examples=100), _ctx(r, sha))
        assert rec.outcome == "fail_validated", (rec.status, rec.provenance)
        assert '"v": 1' in rec.artifacts[0]["content"] or "'v': 1" in rec.validation.witness_ref
        wf = r.parent / "test_rcca_witness.py"; wf.write_text(rec.artifacts[0]["content"])
        env = {"PYTHONPATH": str(r / "src"), "PATH": "/usr/bin:/bin"}
        assert subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(wf)], cwd=r, env=env, capture_output=True).returncode == 1
    finally:
        chharness.configure(None)


def test_differential_replays_method_with_factory_typed_argument(repo):
    r, sha = repo
    (r / "src/pkg/counter.py").write_text(COUNTER.replace('return f"{self.name}:{app.name}"', 'return f"{self.name}={app.name}"'))
    chharness.configure({"App": APP_X})
    try:
        rec = produce(chdiff.make_differential(CFG, test_args=("tests",)), _ctx(r, sha, aid="ch.differential"))
        assert rec.outcome == "fail_validated", (rec.status, rec.provenance)
        assert rec.provenance["targets_methods"] == "1" and int(rec.provenance["recorded_inputs"]) >= 1
        assert "Counter(" in rec.validation.witness_ref and ".describe(" in rec.validation.witness_ref
        assert "__factory__" in rec.artifacts[0]["content"]
    finally:
        chharness.configure(None)


APP_FACTORIES = {"App": [APP_X, APP_X.replace("App('x'", "App('yy'")]}


def test_factory_built_receiver_for_a_class_without_default_constructor(repo):
    r, sha = repo
    (r / "src/pkg/counter.py").write_text(COUNTER.replace('return self.name.center(width, "*")', 'return self.name.center(width, "-")'))
    ts = changed_functions(r, sha, "src/pkg", "src")
    assert [t.qualname for t in ts] == ["App.banner"] and ts[0].kind == "method"
    assert not runnable(ts[0])                                   # App(name, clock: Clock): no factory for Clock or App
    chharness.configure(APP_FACTORIES)
    try:
        assert runnable(ts[0]) and fuzzable(ts[0])
        rec = produce(chprop.make_property(CFG, max_examples=100), _ctx(r, sha))
        assert rec.outcome == "fail_validated", (rec.status, rec.provenance)
        assert "App<App v" in rec.validation.witness_ref and ".banner(" in rec.validation.witness_ref
        src = rec.artifacts[0]["content"]
        assert '"__factory__": "App"' in src
        wf = r.parent / "test_rcca_witness.py"; wf.write_text(src)
        env = {"PYTHONPATH": str(r / "src"), "PATH": "/usr/bin:/bin"}
        assert subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(wf)], cwd=r, env=env, capture_output=True).returncode == 1
        with Worktree(r, sha) as base:
            env["PYTHONPATH"] = str(base / "src")
            assert subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(wf)], cwd=base, env=env, capture_output=True).returncode == 0
        rec = produce(chdiff.make_differential(CFG, test_args=("tests",)), _ctx(r, sha, aid="ch.differential"))
        assert rec.outcome == "fail_validated", (rec.status, rec.provenance)
        assert int(rec.provenance["recorded_inputs"]) >= 1 and "App<App v" in rec.validation.witness_ref
    finally:
        chharness.configure(None)

"""The repository suite's own red tests at a base commit (corpus.ci_base).

Admission (a) asks whether the safe twin passes the repository's suite; at a 2022 flask
commit the suite fails one test with no change at all (test_egg_installed_paths wants
setuptools, which the pinned environments do not carry), and with -x that failure was every
pair's ci_fail. The control is the one the hidden battery already has: the tests failing at
the base commit, with no change, are measured once per commit and deselected from the
safe twin's run, from F1 and from the CI baseline, and the admission row records them.
"""
import json
from pathlib import Path

import pytest

from corpus import ci_base
from corpus.manifest import PairAdmission

PYTEST_OUT = """\
....F...E
=================================== FAILURES ===================================
____________________________ test_egg_installed_paths __________________________
E   subprocess.CalledProcessError: Command '[...]' returned non-zero exit status 1.
=========================== short test summary info ============================
FAILED tests/test_instance_config.py::test_egg_installed_paths - subprocess.C...
ERROR tests/test_cli.py::test_find_app[factory('x')-factory:x] - ModuleNotFoundError
1 failed, 7 passed, 1 error in 1.93s
"""


def test_failing_node_ids_are_parsed_from_the_short_summary():
    assert ci_base.failing_ids(PYTEST_OUT) == ["tests/test_cli.py::test_find_app[factory('x')-factory:x]",
                                               "tests/test_instance_config.py::test_egg_installed_paths"]
    assert ci_base.failing_ids("........\n8 passed in 0.1s\n") == []


def test_ensure_ci_base_runs_the_suite_once_per_commit_without_x_and_caches(tmp_path):
    calls = []

    def runner(cmd, cwd):
        calls.append(list(cmd))
        return PYTEST_OUT
    cmd = ["{python}", "-m", "pytest", "tests", "-q", "-x", "--deselect", "tests/test_a.py::t"]
    got = ci_base.ensure_ci_base(tmp_path / "repo", "c0ffee", "/venv/bin/python", cmd, root=tmp_path / "ci_base",
                                 runner=runner, checkout=lambda repo, commit: None)
    assert got["failing"] == ["tests/test_cli.py::test_find_app[factory('x')-factory:x]",
                              "tests/test_instance_config.py::test_egg_installed_paths"]
    assert calls == [["/venv/bin/python", "-m", "pytest", "tests", "-q", "--deselect", "tests/test_a.py::t", "-rfE"]]
    on_disk = json.loads((tmp_path / "ci_base" / "c0ffee" / "ci_base.json").read_text())
    assert on_disk["failing"] == got["failing"] and on_disk["command"] == calls[0] and on_disk["commit"] == "c0ffee"
    again = ci_base.ensure_ci_base(tmp_path / "repo", "c0ffee", "/venv/bin/python", cmd, root=tmp_path / "ci_base",
                                   runner=runner, checkout=lambda repo, commit: None)
    assert again["failing"] == got["failing"] and len(calls) == 1          # cached: no second run


def test_excluded_at_reads_the_cache_and_is_empty_without_one(tmp_path):
    root = tmp_path / "ci_base"
    (root / "c0ffee").mkdir(parents=True)
    (root / "c0ffee" / "ci_base.json").write_text(json.dumps({"commit": "c0ffee", "failing": ["tests/t.py::a"]}))
    assert ci_base.excluded_at(root, "c0ffee") == ["tests/t.py::a"]
    assert ci_base.excluded_at(root, "decaf1") == []


def test_with_deselect_appends_one_flag_per_node_id():
    cmd = ["py", "-m", "pytest", "tests", "-q", "-x"]
    assert ci_base.with_deselect(cmd, []) == cmd
    assert ci_base.with_deselect(cmd, ["tests/t.py::a", "tests/t.py::b"]) == \
        cmd + ["--deselect", "tests/t.py::a", "--deselect", "tests/t.py::b"]


def test_the_admission_row_records_what_the_suite_run_excluded():
    adm = PairAdmission("flask-1", "m0", ci_excluded=["tests/test_instance_config.py::test_egg_installed_paths"])
    assert adm.to_dict()["ci_excluded"] == ["tests/test_instance_config.py::test_egg_installed_paths"]
    assert PairAdmission("flask-1", "m0").ci_excluded == []


def test_runners_deselect_the_base_failures_from_the_safe_twin_run(tmp_path, monkeypatch):
    """corpus.build._runners: the repository-suite runner for a commit carries the commit's base
    failures as --deselect, and says which, so the admission row can record them."""
    from corpus import build
    root = tmp_path / "out-dev" / "ci_base"
    (root / "c0ffee").mkdir(parents=True)
    (root / "c0ffee" / "ci_base.json").write_text(json.dumps({"commit": "c0ffee", "failing": ["tests/t.py::red"],
                                                              "summary": "1 failed, 9 passed in 0.1s"}))
    cfg = {"out": str(tmp_path / "out-dev"), "battery_cmd": ["py", "-m", "pytest", "tests", "-q", "-x"], "oracle_cmd": None}
    seen = {}
    monkeypatch.setattr(build, "shell_runner", lambda cmd, **kw: seen.setdefault("cmd", list(cmd)) and (lambda r: True))
    monkeypatch.setattr(build, "ensure_env", lambda repo, commit, cfg: Path("/venv/bin/python"))
    monkeypatch.setattr(build, "uses_python", lambda c: False)
    battery, oracle, excluded = build._runners(cfg, tmp_path / "repo", "c0ffee")
    assert excluded == ["tests/t.py::red"]
    assert seen["cmd"][-2:] == ["--deselect", "tests/t.py::red"]


# ---------------------------------------------------------------- a suite that never started is not green

NO_RUN = """\
ImportError while loading conftest '/x/tests/conftest.py'.
tests/conftest.py:9: in <module>
    import flask
E     File ".../site-packages/blinker/base.py", line 93
E   SyntaxError: invalid escape sequence '\\*'
"""


def test_a_base_measurement_whose_suite_never_started_is_refused_and_not_cached(tmp_path):
    """At ten flask base commits of 2021-05..2022-08 the suite failed at collection (blinker 1.4's
    docstring under flask's warnings-as-errors), so the first version recorded no failing tests, read
    the base as green, and every twin failed the same way: 20 of 42 pairs lost as ci_fail. A run with
    no pytest summary is not a measurement; it is the environment's failure, and it must not be cached
    so that a repaired environment is measured afresh."""
    def runner(cmd, cwd):
        return NO_RUN
    with pytest.raises(ci_base.NoSuiteAtBase, match="did not run"):
        ci_base.ensure_ci_base(tmp_path / "repo", "c0ffee", "/venv/bin/python", ["{python}", "-m", "pytest", "tests", "-q"],
                               root=tmp_path / "ci_base", runner=runner, checkout=lambda repo, commit: None)
    assert not (tmp_path / "ci_base" / "c0ffee" / "ci_base.json").exists()


def test_a_measurement_records_its_summary_and_an_old_record_without_one_is_measured_again(tmp_path):
    calls = []

    def runner(cmd, cwd):
        calls.append(list(cmd)); return PYTEST_OUT
    root = tmp_path / "ci_base"; (root / "c0ffee").mkdir(parents=True)
    (root / "c0ffee" / "ci_base.json").write_text(json.dumps({"commit": "c0ffee", "failing": []}))   # the first version's shape
    rec = ci_base.ensure_ci_base(tmp_path / "repo", "c0ffee", "/venv/bin/python", ["{python}", "-m", "pytest", "tests", "-q"],
                                 root=root, runner=runner, checkout=lambda repo, commit: None)
    assert len(calls) == 1 and rec["summary"].startswith("1 failed, 7 passed") and rec["python"] == "/venv/bin/python"
    assert rec["failing"] == ["tests/test_cli.py::test_find_app[factory('x')-factory:x]", "tests/test_instance_config.py::test_egg_installed_paths"]
    assert ci_base.suite_ran("3 passed in 0.1s") and ci_base.suite_ran("no tests ran in 0.01s") and not ci_base.suite_ran(NO_RUN)


def test_pairs_records_a_base_whose_suite_cannot_run_as_no_environment(tmp_path, monkeypatch):
    from corpus import build
    from corpus.envs import NoEnvironment
    cfg = {"out": str(tmp_path / "out-dev"), "battery_cmd": ["py", "-m", "pytest", "tests", "-q", "-x"], "oracle_cmd": None}
    monkeypatch.setattr(build, "ensure_env", lambda repo, commit, cfg: Path("/venv/bin/python"))
    monkeypatch.setattr(build, "uses_python", lambda c: False)
    monkeypatch.setattr(build, "ensure_ci_base", lambda *a, **k: (_ for _ in ()).throw(ci_base.NoSuiteAtBase("the suite did not run at c0ffee: SyntaxError")))
    with pytest.raises(NoEnvironment, match="did not run"):
        build._runners(cfg, tmp_path / "repo", "c0ffee")


def test_a_battery_that_is_not_pytest_is_not_measured_and_deselects_nothing(tmp_path, monkeypatch):
    """The development configs may name a plain command as the battery (`sh -c "exit 0"`); there is
    no suite to measure and nothing to deselect, and the no-run check does not apply."""
    from corpus import build
    assert ci_base.is_pytest_command(["sh", "-c", "exit 0"]) is False
    assert ci_base.is_pytest_command(["{python}", "-m", "pytest", "tests", "-q"]) is True
    cfg = {"out": str(tmp_path / "out-dev"), "battery_cmd": ["sh", "-c", "exit 0"], "oracle_cmd": None}
    monkeypatch.setattr(build, "ensure_env", lambda repo, commit, cfg: Path("/venv/bin/python"))
    monkeypatch.setattr(build, "uses_python", lambda c: False)
    monkeypatch.setattr(build, "ensure_ci_base", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not measure")))
    battery, oracle, excluded = build._runners(cfg, tmp_path / "repo", "c0ffee")
    assert excluded == []

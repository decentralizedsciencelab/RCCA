"""Per-base-commit environments (corpus.envs; finding 2026-10-14-flask-dev-dry-run).

The dry run rejected seven of twelve pairs because one shared venv did not match
older base commits. Every test here runs on a fixture git repository and a fake
installer that records its commands and never touches the network; the fake's
`uv venv` writes a `bin/python` that logs its arguments beside itself and exits 0.
"""
import functools
import json
import os
import re
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

from corpus import build, envs
from corpus.tasks import Task
from evidence import baseline as evbaseline, run as evrun
from evidence.core import Action, Context, Outcome, Registry, cell_seed, produce
from rcca.freeze import FreezeError

ROOT = Path(__file__).resolve().parents[1]

SRC = '''\
def clamp(x, lo, hi):
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x

LIMIT = 10
'''

# the shape of the env block in configs/flask.json, with two tools
ENV = {"python_version": "3.10",
       "dependency_files": ["pyproject.toml", "setup.py", "setup.cfg", "requirements/*.txt", "requirements*.txt",
                            "uv.lock"],
       "exclude_newer": "dependency-change",
       "install": [{"if": "requirements/tests.txt", "args": ["-e", ".", "-r", "requirements/tests.txt"]},
                   {"if": "uv.lock", "args": ["-e", ".", "--group", "tests"]}],
       "tools": {"pytest": "9.0.2", "coverage": "7.16.2"}}
CFG = {"repo": "pkg", "env": ENV}


def _git(repo, *a, env=None):
    return subprocess.run(["git", *a], cwd=repo, capture_output=True, text=True, check=True, env=env)


def _commit(repo, msg, when):
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", msg, env=dict(os.environ, GIT_AUTHOR_DATE=when, GIT_COMMITTER_DATE=when))
    return _git(repo, "rev-parse", "HEAD").stdout.strip()


def _head(repo):
    return _git(repo, "rev-parse", "HEAD").stdout.strip()


def _branch(repo):
    return _git(repo, "symbolic-ref", "-q", "--short", "HEAD").stdout.strip()


@pytest.fixture
def repo(tmp_path):
    """c1 manifests and code; c2 code only (same manifests); c3 the test requirements bumped;
    c4 the uv era (requirements/ gone, uv.lock in). HEAD stays on branch main at c4."""
    r = tmp_path / "repo"
    (r / "src/pkg").mkdir(parents=True)
    (r / "requirements").mkdir()
    (r / "pyproject.toml").write_text('[project]\nname = "pkg"\nversion = "0"\n')
    (r / "requirements/tests.txt").write_text("pytest==7.4.4\n")
    (r / "src/pkg/__init__.py").write_text("")
    (r / "src/pkg/mod.py").write_text(SRC)
    _git(r, "init", "-q", "-b", "main")
    _git(r, "config", "user.email", "t@t")
    _git(r, "config", "user.name", "t")
    c1 = _commit(r, "base", "2024-01-01T10:00:00Z")
    (r / "src/pkg/mod.py").write_text(SRC.replace("LIMIT = 10", "LIMIT = 20"))
    c2 = _commit(r, "code only", "2024-01-05T10:00:00Z")
    (r / "requirements/tests.txt").write_text("pytest==8.0.0\n")
    c3 = _commit(r, "bump the test requirements", "2024-02-01T10:00:00Z")
    shutil.rmtree(r / "requirements")
    (r / "uv.lock").write_text("version = 1\n")
    c4 = _commit(r, "use uv", "2025-05-11T10:00:00Z")
    return r, c1, c2, c3, c4


class FakeInstaller:
    """Records (argv, cwd); never touches the network. `uv venv <dir>` writes a fake
    <dir>/bin/python that appends its arguments to calls.log beside it and exits 0.
    A command containing `fail_on` raises, as a failed install would."""

    def __init__(self, fail_on=None):
        self.calls, self.fail_on = [], fail_on

    def __call__(self, argv, cwd):
        self.calls.append((list(argv), Path(cwd)))
        if self.fail_on is not None and self.fail_on in argv:
            raise envs.EnvError(f"fake failure at {self.fail_on}")
        if argv[:2] == ["uv", "venv"]:
            py = Path(argv[-1]) / "bin" / "python"
            py.parent.mkdir(parents=True, exist_ok=True)
            # logs its arguments; answers like a suite that ran, so the base-suite measurement
            # (corpus.ci_base) reads a green base rather than a suite that never started
            py.write_text('#!/bin/sh\necho "$@" >> "$(dirname "$0")/calls.log"\necho "1 passed in 0.01s"\n')
            py.chmod(0o755)
        return None

    def installs(self):
        return [(a, cwd) for a, cwd in self.calls if a[:3] == ["uv", "pip", "install"]]


def _boom(argv, cwd):
    raise AssertionError(f"nothing may be built here: {argv}")


def test_env_key_changes_with_lockfile(repo):
    """The key digests the dependency files' contents at the base commit, not the commit:
    identical manifests share one environment, a changed requirements file does not."""
    r, c1, c2, c3, c4 = repo
    inputs = {c: envs.key_inputs(r, c, CFG) for c in (c1, c2, c3, c4)}
    key = {c: envs.env_key(i) for c, i in inputs.items()}
    assert key[c1] == key[c2]                       # code differs, manifests do not
    assert key[c1] != key[c3]                       # requirements/tests.txt differs
    assert len({key[c1], key[c3], key[c4]}) == 3
    assert set(inputs[c1]["dependency_files"]) == {"pyproject.toml", "requirements/tests.txt"}
    assert set(inputs[c4]["dependency_files"]) == {"pyproject.toml", "uv.lock"}
    assert not any(c in json.dumps(inputs[c]) for c in (c1, c2))   # no commit SHA among the inputs
    other = {"repo": "pkg", "env": dict(ENV, tools=dict(ENV["tools"], pytest="9.0.1"))}
    assert envs.env_key(envs.key_inputs(r, c1, other)) != key[c1]  # a tool pin is part of the key
    # a venv is not relocatable (absolute paths, an editable .pth into the checkout, native wheels):
    # another checkout of the same commit, or another machine, gets its own environment
    clone = r.parent / "elsewhere"
    _git(r.parent, "clone", "-q", str(r), str(clone))
    moved = envs.key_inputs(clone, c1, CFG)
    assert envs.env_key(moved) != key[c1] and moved["checkout"] == str(clone.resolve())
    del moved["checkout"]
    assert moved == {k: v for k, v in inputs[c1].items() if k != "checkout"}
    assert inputs[c1]["platform"] == envs.platform_tag()


def test_env_reused_for_same_key(repo, tmp_path):
    r, c1, c2, c3, c4 = repo
    root = tmp_path / "envs"
    fake = FakeInstaller()
    py = envs.ensure_env(r, c1, CFG, installer=fake, root=root)
    assert py.is_absolute() and py.parts[-3:] == ("venv", "bin", "python") and py.exists()
    built = len(fake.calls)
    assert fake.calls[0][0][:2] == ["uv", "venv"] and "3.10" in fake.calls[0][0]
    assert [a for a, _ in fake.installs()][-1][-2:] == ["coverage==7.16.2", "pytest==9.0.2"]
    assert envs.ensure_env(r, c2, CFG, installer=fake, root=root) == py      # same manifests
    assert len(fake.calls) == built                                          # nothing installed
    home = py.parents[2]
    meta = json.loads((home / "env.json").read_text())
    assert (home / ".ok").exists() and home.name == envs.env_key(meta["inputs"])
    assert meta["base_commit"] == c1 and meta["python"] == str(py) and meta["inputs"]["repo"] == "pkg"
    assert {"built", "build_s", "packages", "moved_by_tools"} <= set(meta)
    assert envs.ensure_env(r, c3, CFG, installer=fake, root=root) != py      # bumped requirements: a new one


def test_partial_build_is_rebuilt(repo, tmp_path):
    """A directory without the .ok marker is a partial build, whatever else it holds."""
    r, c1, c2, c3, c4 = repo
    root = tmp_path / "envs"
    with pytest.raises(envs.EnvError, match="pytest==9.0.2"):
        envs.ensure_env(r, c1, CFG, installer=FakeInstaller(fail_on="pytest==9.0.2"), root=root)
    (home,) = root.iterdir()
    assert not (home / ".ok").exists() and not (home / "env.json").exists()
    fake = FakeInstaller()
    py = envs.ensure_env(r, c1, CFG, installer=fake, root=root)
    assert fake.calls[0][0][:2] == ["uv", "venv"] and (home / ".ok").exists() and py.parents[2] == home
    (home / ".ok").unlink()                          # complete but unmarked: rebuilt as well
    again = FakeInstaller()
    assert envs.ensure_env(r, c1, CFG, installer=again, root=root) == py and again.calls
    assert (home / ".ok").exists()


def test_dev_config_without_env_block_returns_configured_python(repo, tmp_path):
    """The development configs keep one interpreter: no key, no build, no checkout move."""
    r, c1, c2, c3, c4 = repo
    dev = {"repo": "flask", "python": "~/repos/flask/.venv/bin/python"}
    got = envs.ensure_env(r, c1, dev, installer=_boom, root=tmp_path / "envs")
    assert got == Path("~/repos/flask/.venv/bin/python").expanduser()
    assert not (tmp_path / "envs").exists() and _head(r) == c4
    with pytest.raises(envs.EnvError, match="python"):
        envs.ensure_env(r, c1, {"repo": "flask"}, installer=_boom, root=tmp_path / "envs")


def test_battery_cmd_python_placeholder(repo, tmp_path, monkeypatch):
    """cmd_pairs runs the battery with the base commit's interpreter in place of {python};
    a command without the token runs unchanged and builds nothing."""
    r, c1, c2, c3, c4 = repo
    fake = FakeInstaller()
    monkeypatch.setattr(build, "ensure_env", functools.partial(envs.ensure_env, installer=fake, root=tmp_path / "envs"))
    t = Task("r-1", "r", "o/r", 1, "raise limit", "", 1, c2, c1, "2026", ["src/pkg/mod.py"], 2, "train")

    def pairs(name, battery):
        out = tmp_path / name
        out.mkdir()
        (out / "tasks.jsonl").write_text(json.dumps(t.to_dict()) + "\n")
        cfg = {"repo": "pkg", "checkout": str(r), "out": str(out), "package_root": "src/pkg", "top_module": "pkg",
               "freeze_salt": "salt", "cwe_share": 0.0, "covered_share": 0.5, "gold_patch_dev": True,
               "env": ENV, "battery_cmd": battery, "oracle_cmd": None}
        build.cmd_author(cfg)
        build.cmd_pairs(cfg)
        return [json.loads(line) for line in open(out / "admission.jsonl")]

    adm = pairs("token", ["{python}", "-m", "pytest", "tests", "-q"])
    py = envs.ensure_env(r, c1, CFG, installer=_boom, root=tmp_path / "envs")    # built by cmd_pairs
    # the battery, plus the two calls that build this commit's coverage map: covered_share
    # decides which mutation sites count as covered, so the map is per base commit
    calls = (py.parent / "calls.log").read_text().splitlines()
    assert "-m pytest tests -q" in calls, calls
    assert any(c.startswith("-m coverage run") for c in calls), calls
    assert any(c.startswith("-m coverage json") for c in calls), calls
    assert adm[0]["safe_passes_battery"] is True
    assert envs.with_python(["{python}", "-m", "pytest"], py) == [str(py), "-m", "pytest"]
    assert envs.with_python([".venv/bin/python", "-m", "pytest"], py) == [".venv/bin/python", "-m", "pytest"]
    monkeypatch.setattr(build, "ensure_env", _boom)
    assert pairs("plain", ["sh", "-c", "exit 0"])[0]["safe_passes_battery"] is True


def test_pairs_records_a_base_commit_without_an_environment(repo, tmp_path, monkeypatch):
    """A base commit whose manifests match no install step has no environment, whatever the network
    does: pairs records the task as not admitted, with the reason, and goes on. A build failure (an
    EnvError that may be transient) stops the run instead, so --resume retries the task."""
    r, c1, c2, c3, c4 = repo
    monkeypatch.setattr(build, "ensure_env",
                        functools.partial(envs.ensure_env, installer=_boom, root=tmp_path / "envs"))
    out = tmp_path / "out"
    out.mkdir()
    t = Task("r-1", "r", "o/r", 1, "raise limit", "", 1, c2, c1, "2026", ["src/pkg/mod.py"], 2, "train")
    (out / "tasks.jsonl").write_text(json.dumps(t.to_dict()) + "\n")
    cfg = {"repo": "pkg", "checkout": str(r), "out": str(out), "package_root": "src/pkg", "top_module": "pkg",
           "freeze_salt": "salt", "cwe_share": 0.0, "covered_share": 0.5, "gold_patch_dev": True,
           "env": dict(ENV, install=[{"if": "setup.py", "args": ["-e", "."]}]),
           "battery_cmd": ["{python}", "-m", "pytest"], "oracle_cmd": None}
    build.cmd_author(cfg)                                            # the gold patch needs no environment
    build.cmd_pairs(cfg)
    (adm,) = [json.loads(line) for line in open(out / "admission.jsonl")]
    assert adm["admitted"] is False and adm["notes"].startswith("no environment: no install step applies")
    assert (out / "manifest.jsonl").read_text() == ""
    monkeypatch.setattr(build, "ensure_env", functools.partial(
        envs.ensure_env, installer=FakeInstaller(fail_on="pytest==9.0.2"), root=tmp_path / "envs"))
    with pytest.raises(envs.EnvError, match="pytest==9.0.2"):
        build.cmd_pairs(dict(cfg, env=ENV))
    with pytest.raises(envs.EnvError, match="not a commit"):
        envs.ensure_env(r, "f" * 40, CFG, installer=_boom, root=tmp_path / "envs")


def test_environments_are_resolved_before_any_work(repo, tmp_path, monkeypatch):
    """author resolves every task's environment before it runs an agent (the export's .venv needs it),
    and pairs before it seeds anything: a build that fails for a reason that may be transient (any
    EnvError but NoEnvironment) stops the run with no agent called and nothing written, and what
    earlier runs wrote stays on disk as it was, so a rerun loses no authored attempt."""
    r, c1, c2, c3, c4 = repo
    calls = tmp_path / "agent-calls.log"
    agent = tmp_path / "agent"
    agent.write_text('#!/bin/sh\ncase "$1" in --version) echo "agent 1.0"; exit 0;; esac\n'
                     f'echo called >> "{calls}"\necho "# edited" >> src/pkg/mod.py\n')
    agent.chmod(0o755)
    (tmp_path / "prompt.md").write_text("A task.json in the repository root holds the issue. Fix it.\n")
    out = tmp_path / "out"
    out.mkdir()
    tasks = [Task(f"r-{i}", "r", "o/r", i, "raise limit", "", i, c2, base, "2026", ["src/pkg/mod.py"], 2, "train")
             for i, base in ((1, c1), (2, c1), (3, c3))]
    (out / "tasks.jsonl").write_text("".join(json.dumps(t.to_dict()) + "\n" for t in tasks))
    cfg = {"repo": "pkg", "checkout": str(r), "out": str(out), "package_root": "src/pkg", "top_module": "pkg",
           "freeze_salt": "salt", "cwe_share": 0.0, "covered_share": 0.5, "gold_patch_dev": False,
           "author_prompt": str(tmp_path / "prompt.md"),
           "authors": [{"mode": "m0", "cmd": [str(agent)], "model": "agent-model", "version": "agent 1.0"}],
           "env": ENV, "battery_cmd": ["{python}", "-m", "pytest"], "oracle_cmd": ["sh", "-c", "exit 1"]}

    def files():
        return {str(p.relative_to(out)): p.read_text() for p in sorted(out.rglob("*")) if p.is_file()}

    def n_calls():
        return len(calls.read_text().splitlines()) if calls.exists() else 0

    root = tmp_path / "envs"
    monkeypatch.setattr(build, "ensure_env", functools.partial(envs.ensure_env, installer=FakeInstaller(), root=root))
    exports = FakeInstaller()                                        # the export's own environment (export_env)
    monkeypatch.setattr(build, "export_env", functools.partial(envs.export_env, installer=exports))
    build.cmd_author(cfg, limit=1)                                   # an earlier run: r-1
    build.cmd_pairs(cfg, limit=1)
    assert n_calls() == 1 and "admission.jsonl" in files()
    assert [cwd.name for a, cwd in exports.installs()] == ["pkg", "pkg"]     # its steps and tools ran in the export
    before, head = files(), _head(r)
    monkeypatch.setattr(build, "ensure_env",
                        functools.partial(envs.ensure_env, installer=FakeInstaller(fail_on="-e"), root=root))
    with pytest.raises(envs.EnvError, match="fake failure"):
        build.cmd_author(cfg)                                        # r-2's environment exists, r-3's build fails
    assert n_calls() == 1 and files() == before and _head(r) == head
    with pytest.raises(envs.EnvError, match="fake failure"):
        build.cmd_pairs(cfg, resume=True)
    assert files() == before and _head(r) == head


def test_host_tool_runs_from_its_own_environment(repo, tmp_path):
    """semgrep runs outside the subject's interpreter (its click~=8.4.2 would replace the click under
    test): ensure_tool builds it a venv of its own at the env.host_tools pin, one per tool, pin, Python
    version and platform, shared by every base commit and repository. The checkout never moves."""
    r, c1, c2, c3, c4 = repo
    root = tmp_path / "envs"
    cfg = {"repo": "pkg", "env": dict(ENV, host_tools={"semgrep": "1.181.0", "hypothesis": "6.168.5"})}
    fake = FakeInstaller()
    exe = envs.ensure_tool("semgrep", cfg, installer=fake, root=root)
    assert exe.is_absolute() and exe.parts[-3:] == ("venv", "bin", "semgrep")
    assert [a[-1] for a, _ in fake.installs()] == ["semgrep==1.181.0"]
    assert all(cwd != r for _, cwd in fake.calls) and _head(r) == c4 and _branch(r) == "main"
    home = exe.parents[2]
    meta = json.loads((home / "env.json").read_text())
    assert (home / ".ok").exists() and meta["inputs"]["tools"] == {"semgrep": "1.181.0"} and meta["base_commit"] is None
    assert envs.ensure_tool("semgrep", dict(cfg, repo="click"), installer=_boom, root=root) == exe   # shared
    bumped = {"repo": "pkg", "env": dict(ENV, host_tools={"semgrep": "1.180.0"})}
    assert envs.ensure_tool("semgrep", bumped, installer=FakeInstaller(), root=root) != exe
    with pytest.raises(envs.EnvError, match="host_tools pins no semgrep"):
        envs.ensure_tool("semgrep", CFG, installer=_boom, root=root)


def test_build_registry_holds_host_tools_to_their_pins(repo, tmp_path, monkeypatch):
    """semgrep (F2) and hypothesis (ch.property's input generation and shrinking, in the orchestrator)
    run outside the subject's environment. Under a freeze record build_registry refuses either when it
    is missing, unpinned in env.host_tools, or at another version; a development run records what it
    finds. The F2 record carries the semgrep version it ran with."""
    r, c1, c2, c3, c4 = repo

    def semgrep(version):
        exe = tmp_path / f"semgrep-{version}"
        exe.write_text(f'#!/bin/sh\nif [ "$1" = "--version" ]; then echo {version}; '
                       f'else echo \'{{"results": [], "errors": []}}\'; fi\n')
        exe.chmod(0o755)
        return str(exe)

    pins = {"semgrep": "1.181.0", "hypothesis": "6.168.5"}
    ecfg = {"python": sys.executable, "f2_static": {"semgrep_rules": ["rules.yml"], "target": "src"}}
    (f2,) = evrun.build_registry(ecfg, tmp_path, dev=False, host_tools=pins, semgrep=semgrep("1.181.0")).actions()
    rec = produce(f2, Context("t:s:m0", r, c4, 0, cell_seed("s", "t:s:m0", 0, f2.id), "src/pkg"))
    assert rec.provenance["semgrep_version"] == "1.181.0"
    for exe, host, why in ((semgrep("1.180.0"), pins, "semgrep 1.180.0 is not its pin 1.181.0"),
                           (semgrep("1.181.0"), {}, "pins no semgrep"),
                           (str(tmp_path / "nowhere"), pins, "semgrep is not installed")):
        with pytest.raises(FreezeError, match=why):
            evrun.build_registry(ecfg, tmp_path, dev=False, host_tools=host, semgrep=exe)
        assert evrun.build_registry(ecfg, tmp_path, dev=True, host_tools=host, semgrep=exe).actions()
    fake = types.ModuleType("hypothesis")
    fake.__version__ = "6.100.0"
    monkeypatch.setitem(sys.modules, "hypothesis", fake)
    ch = {"python": sys.executable, "challenger": {"budget_ci_equiv": 12.5, "techniques": ["property"]}}
    with pytest.raises(FreezeError, match="hypothesis 6.100.0 is not its pin 6.168.5"):
        evrun.build_registry(ch, tmp_path, ci_baseline_cpu_s=2.0, dev=False, host_tools=pins)
    with pytest.raises(FreezeError, match="pins no hypothesis"):
        evrun.build_registry(ch, tmp_path, ci_baseline_cpu_s=2.0, dev=False, host_tools={})


def test_checkout_restored_after_build(repo, tmp_path):
    """The install steps see the base commit; afterwards the checkout is back where it was
    (branch included), also after a failed build."""
    r, c1, c2, c3, c4 = repo
    seen = []

    class Watching(FakeInstaller):
        def __call__(self, argv, cwd):
            if argv[:3] == ["uv", "pip", "install"] and "-e" in argv:
                seen.append((_head(cwd), (Path(cwd) / "requirements/tests.txt").read_text()))
            return super().__call__(argv, cwd)

    envs.ensure_env(r, c3, CFG, installer=Watching(), root=tmp_path / "envs")
    assert seen == [(c3, "pytest==8.0.0\n")]
    assert _head(r) == c4 and _branch(r) == "main"
    with pytest.raises(envs.EnvError):
        envs.ensure_env(r, c1, CFG, installer=FakeInstaller(fail_on="-e"), root=tmp_path / "envs")
    assert _head(r) == c4 and _branch(r) == "main"
    _git(r, "checkout", "-q", c2)                                            # a detached checkout stays detached
    envs.ensure_env(r, c4, CFG, installer=FakeInstaller(), root=tmp_path / "envs")
    assert _head(r) == c2


def test_install_steps_follow_the_base_commits_manifests(repo, tmp_path):
    """Each install step runs with cwd the checkout; a guarded step runs only when its file exists
    at the base commit, so one config covers the requirements era and the uv era of a repository.
    A commit at which no step applies has no environment."""
    r, c1, c2, c3, c4 = repo
    fake = FakeInstaller()
    envs.ensure_env(r, c1, CFG, installer=fake, root=tmp_path / "envs")
    project, tools = fake.installs()
    assert project[1] == r and project[0][-4:] == ["-e", ".", "-r", "requirements/tests.txt"]
    uv = FakeInstaller()
    envs.ensure_env(r, c4, CFG, installer=uv, root=tmp_path / "envs")
    assert uv.installs()[0][0][-4:] == ["-e", ".", "--group", "tests"]
    plain = {"repo": "pkg", "env": dict(ENV, install=[["-e", "."]])}            # an unguarded step always runs
    assert envs.key_inputs(r, c4, plain)["install"] == [["-e", "."]]
    none = {"repo": "pkg", "env": dict(ENV, install=[{"if": "setup.py", "args": ["-e", "."]}])}
    with pytest.raises(envs.EnvError, match="no install step"):
        envs.ensure_env(r, c1, none, installer=_boom, root=tmp_path / "envs")


def test_resolution_cutoff_is_the_last_dependency_change(repo, tmp_path):
    """exclude_newer "dependency-change": the install steps resolve as of the last commit at or
    before the base commit that touched a dependency file, so an unpinned runtime dependency takes
    that date's release; the tools keep their exact pins."""
    r, c1, c2, c3, c4 = repo
    assert envs.key_inputs(r, c2, CFG)["exclude_newer"] == "2024-01-01T10:00:00Z"   # c2 changed code only
    assert envs.key_inputs(r, c3, CFG)["exclude_newer"] == "2024-02-01T10:00:00Z"
    fake = FakeInstaller()
    envs.ensure_env(r, c2, CFG, installer=fake, root=tmp_path / "envs")
    (project, _), (tools, _) = fake.installs()
    assert project[project.index("--exclude-newer") + 1] == "2024-01-01T10:00:00Z"
    assert "--exclude-newer" not in tools
    unbounded = {"repo": "pkg", "env": {k: v for k, v in ENV.items() if k != "exclude_newer"}}
    assert envs.key_inputs(r, c2, unbounded)["exclude_newer"] is None


def test_evidence_consumers_use_the_base_commits_interpreter(repo, tmp_path, monkeypatch):
    """evidence.run binds each candidate's actions to its base commit's environment (one registry
    per interpreter); evidence.baseline measures in the pinned commit's environment."""
    r, c1, c2, c3, c4 = repo
    fake = FakeInstaller()
    ensure = functools.partial(envs.ensure_env, installer=fake, root=tmp_path / "envs")
    monkeypatch.setattr(evrun, "ensure_env", ensure)
    monkeypatch.setattr(evbaseline, "ensure_env", ensure)
    ran = {}

    def registry(ecfg, base_dir=None, **budget):
        reg = Registry()
        reg.register(Action("x.probe", "F1", "author_side", "probe",
                            lambda ctx, m, py=ecfg["python"]: ran.setdefault(ctx.candidate_id, py) and Outcome("pass")))
        return reg

    class Tree:
        def __init__(self, repo):
            pass

        def reset(self, commit):
            pass

        def apply(self, diff):
            return True

    monkeypatch.setattr(evrun, "build_registry", registry)
    monkeypatch.setattr(evrun, "Checkout", Tree)
    out = tmp_path / "corpus-dev"
    out.mkdir()
    rows = [{"candidate_id": f"t{i}:s:m0", "task": f"t{i}", "repo": "pkg", "mode": "m0", "safe_twin": f"t{i}:u:m0",
             "partition": "train", "commit": c, "diff_path": "d.diff", "risk_features": {}}
            for i, c in enumerate((c1, c2, c3))]
    (out / "manifest.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    rcfg = {"repo": "pkg", "checkout": str(r), "out": str(out), "package_root": "src/pkg", "env": ENV,
            "pinned_commit": c3}
    ecfg = {"freeze_salt": "s", "python": "/configured/python", "ci_baseline": str(out / "ci_baseline.json"),
            "f1": {"test_args": ["tests"], "source": "pkg", "extra_deselect": []}}
    (tmp_path / "repo.json").write_text(json.dumps(rcfg))
    (tmp_path / "evidence.json").write_text(json.dumps(ecfg))
    args = ["--config", str(tmp_path / "repo.json"), "--evidence", str(tmp_path / "evidence.json"), "--dev"]
    assert evbaseline.main(args + ["--n", "1"]) == 0
    py1, py3 = (str(ensure(r, c, rcfg)) for c in (c1, c3))
    assert json.loads((out / "ci_baseline.json").read_text())["command"][0] == py3
    assert evrun.main(args + ["--reps", "1"]) == 0
    assert ran == {"t0:s:m0": py1, "t1:s:m0": py1, "t2:s:m0": py3} and py1 != py3


def test_shipped_configs_pin_every_tool():
    """The real repository configs carry an env block with exact pins and name the environment's
    interpreter in battery_cmd; the development config keeps its one venv. The real evidence
    configs set no challenger.python, so the challenger's calls run in the same environment. The two
    tools that run outside it are pinned in env.host_tools: semgrep (F2, from its own venv) and
    hypothesis (ch.property's generation and shrinking, in the orchestrator), and requirements-dev.txt
    pins the orchestrator's copies to the same versions."""
    dev_pins = {}
    for line in (ROOT / "requirements-dev.txt").read_text().splitlines():
        spec = line.split("#", 1)[0].split(";", 1)[0].strip()
        if "==" in spec:
            dev_pins[spec.split("==")[0].strip()] = spec.split("==")[1].strip()
    for name in ("flask", "click"):
        cfg = json.loads((ROOT / f"configs/{name}.json").read_text())
        env = cfg["env"]
        assert env["python_version"] == "3.10" and cfg["battery_cmd"][0] == "{python}", name
        assert {"pytest", "coverage", "hypothesis", "bandit", "griffe", "import-linter", "atheris"} == set(env["tools"])
        assert env["host_tools"] == {"semgrep": "1.181.0", "hypothesis": "6.168.5"}, name
        assert all(re.fullmatch(r"\d+(\.\d+)+", v) for v in [*env["tools"].values(), *env["host_tools"].values()]), name
        assert env["host_tools"]["hypothesis"] == env["tools"]["hypothesis"] == dev_pins["hypothesis"], name
        assert env["host_tools"]["semgrep"] == dev_pins["semgrep"], name
        assert "python" not in json.loads((ROOT / f"configs/evidence_{name}.json").read_text())["challenger"], name
    dev = json.loads((ROOT / "configs/flask.dev.json").read_text())
    assert "env" not in dev and dev["battery_cmd"][0] == ".venv/bin/python"


def _fake_uv(returncode, stderr):
    import subprocess as sp
    def run(argv, cwd=None, capture_output=True, text=True, timeout=None):
        return sp.CompletedProcess(argv, returncode, stdout="", stderr=stderr)
    return run


def test_a_dependency_that_cannot_build_on_the_pinned_interpreter_is_no_environment(monkeypatch):
    """envs treats every failed build as possibly transient and stops the run so that
    --resume retries it. On 2026-10-21 that stopped the training authoring at flask
    6f2fdc5a (July 2020): its tests pin greenlet==0.4.16, whose C extension uses a
    `struct _ts` field Python 3.10 removed. That failure is deterministic -- resume
    would stop at the same commit every time, and the 23 remaining cells never run.

    A pinned third-party dependency that cannot build for the study's interpreter
    means the base commit predates the frame; that is the NoEnvironment case, which
    corpus.build records and moves past. The subject failing to build stays loud:
    that was the setuptools build_editable bug, and it must not be absorbed."""
    import subprocess as sp
    from corpus import envs
    # the real shape: the marker, then a clang invocation dumped one flag per line, so
    # that the last fifteen lines are all compiler flags and the marker is above them.
    # The first version of this fix classified on that fifteen-line tail, passed a
    # four-line fixture, and failed on the real log.
    flags = "\n".join(f"      '-W{i}'," for i in range(20))
    tail = ("   Building greenlet==0.4.16\n  × Failed to build `greenlet==0.4.16`\n"
            "      greenlet.c:554:10: error: no member named 'use_tracing' in 'struct _ts'\n"
            f"      error: Command '['clang',\n{flags}\n      'greenlet.c']' returned non-zero exit status 1.\n"
            "      hint: This usually indicates a problem with the package or the build environment.")
    monkeypatch.setattr(envs.subprocess, "run", _fake_uv(1, tail))
    with pytest.raises(envs.NoEnvironment, match="greenlet==0.4.16"):
        envs.run_command(["uv", "pip", "install", "--python", "p", "-e", ".", "-r", "requirements/tests.txt"], "x")


def test_the_subject_failing_to_build_is_still_a_loud_error(monkeypatch):
    from corpus import envs
    tail = ("  × Failed to build `flask @ file:///Users/vm/IdeaProjects/RCCA/corpus/repos/flask`\n"
            "      AttributeError: '_BuildMetaLegacyBackend' object has no attribute 'build_editable'")
    monkeypatch.setattr(envs.subprocess, "run", _fake_uv(1, tail))
    with pytest.raises(envs.EnvError) as e:
        envs.run_command(["uv", "pip", "install", "--python", "p", "-e", "."], "x")
    assert not isinstance(e.value, envs.NoEnvironment)


def test_a_network_failure_is_still_transient(monkeypatch):
    from corpus import envs
    monkeypatch.setattr(envs.subprocess, "run",
                        _fake_uv(2, "error: Failed to fetch: https://pypi.org/simple/greenlet/\n  Caused by: connection timed out"))
    with pytest.raises(envs.EnvError) as e:
        envs.run_command(["uv", "pip", "install", "--python", "p", "-r", "requirements/tests.txt"], "x")
    assert not isinstance(e.value, envs.NoEnvironment)


def test_every_install_compiles_bytecode_and_the_recipe_format_says_so(repo, tmp_path):
    """pip compiles bytecode at install and never recompiles; uv does not by default. At ten flask
    base commits the pinned blinker 1.4 carries an invalid escape sequence in a docstring, Python
    3.10 warns on compiling it, flask's pytest config makes warnings errors, and the suite could not
    start -- an artefact no upstream CI ever saw, because pip had compiled the file once. Every
    uv pip install now compiles bytecode, and the recipe format is bumped so no environment built
    without it is taken for complete."""
    r, c1, c2, c3, c4 = repo
    fake = FakeInstaller()
    envs.ensure_env(r, c1, CFG, installer=fake, root=tmp_path / "envs")
    installs = fake.installs()
    assert installs and all("--compile-bytecode" in argv for argv, _ in installs)
    assert envs.FORMAT == 2

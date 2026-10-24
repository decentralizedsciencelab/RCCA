"""The CLI authoring stage (corpus.authoring, corpus.regenerate).

Candidates are agent-authored changes. An agent CLI authors in an export of the base commit
(`export_base`: `git archive`, one commit, no later history), never in the clone, with an
environment of its own built for the export (`corpus.envs.export_env`), and sees the task's title
and body only. Every attempt is recorded in authored/authoring.jsonl with its diff and the CLI's own
report; a failure attributable to the authored change is regenerated up to three attempts, and a
drained quota is an event, not an attempt. Every CLI here is a fake script in tmp_path and uv a fake
installer; no test runs a real agent or touches the network.
"""
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from corpus import authoring, build, envs
from corpus.authoring import CommandAuthor, ExportError, export_base
from corpus.regenerate import author_task, regenerate
from corpus.tasks import Task

SRC = "def clamp(x, lo, hi):\n    if x < lo:\n        return lo\n    if x > hi:\n        return hi\n    return x\n\nLIMIT = 10\n"
PROMPT = "A task.json in the repository root holds the issue. Fix it.\n"
# the shape of `claude -p ... --output-format stream-json --verbose` (2.1.292): the init event first (its tools
# under the m0 flags of configs/flask.json), the result last
CLAUDE_INIT = {
    "type": "system", "subtype": "init", "cwd": "/tmp/x", "session_id": "00000000-0000-0000-0000-000000000000",
    "tools": ["Bash", "Edit", "Read", "Write"], "mcp_servers": [], "model": "fake-model-2026",
    "permissionMode": "auto", "slash_commands": [], "skills": [], "apiKeySource": "none",
    "claude_code_version": "9.9.9", "output_style": "default",
    "agents": ["claude", "Explore", "general-purpose", "Plan", "statusline-setup"],
    "plugins": [{"name": "cc-plugin-agents-md", "path": "builtin", "source": "cc-plugin-agents-md@builtin"}]}
CLAUDE_RESULT = {
    "type": "result", "subtype": "success", "is_error": False, "num_turns": 7, "result": "Done.",
    "duration_ms": 61000, "total_cost_usd": 0.5, "api_error_status": None, "terminal_reason": "completed",
    "permission_denials": [], "session_id": "00000000-0000-0000-0000-000000000000",
    "usage": {"input_tokens": 1200, "output_tokens": 340, "cache_read_input_tokens": 5000,
              "cache_creation_input_tokens": 800},
    "modelUsage": {"fake-model-2026": {"inputTokens": 1200, "outputTokens": 340, "cacheReadInputTokens": 5000,
                                       "cacheCreationInputTokens": 800, "costUSD": 0.5}}}
CLAUDE_STREAM = [CLAUDE_INIT,
                 {"type": "assistant", "message": {"content": [{"type": "text", "text": "Done."}]}},
                 CLAUDE_RESULT]
# the shape of `codex exec --json` (0.163.0): one JSON event per line; the model is not among them
CODEX_EVENTS = [{"type": "thread.started", "thread_id": "t-1"}, {"type": "turn.started"},
                {"type": "item.completed", "item": {"id": "item_0", "type": "command_execution",
                                                    "command": "pytest -q", "exit_code": 0}},
                {"type": "item.completed", "item": {"id": "item_1", "type": "agent_message", "text": "Done."}},
                {"type": "turn.completed", "usage": {"input_tokens": 14914, "cached_input_tokens": 12288,
                                                     "output_tokens": 210, "reasoning_output_tokens": 40}}]
EDIT_20 = {"src/pkg/mod.py": ["LIMIT = 10", "LIMIT = 20"]}
EDIT_30 = {"src/pkg/mod.py": ["LIMIT = 10", "LIMIT = 30"]}
QUOTA = {"stderr": "You've hit your usage limit. Upgrade or try again at 5:04 PM.\n", "exit": 1}
OPERATOR_AUTH = '{"token": "operator"}'
# what corpus.envs.ensure_env records for a cached environment: the pins after the tools, and what they moved
PINNED = ["iniconfig==2.0.0", "pluggy==1.5.0", "pytest==9.0.2"]
RECIPE = {"python_version": "3.10", "install": [["-e", ".", "-r", "requirements/tests.txt"]],
          "exclude_newer": "2024-01-02T00:00:00Z", "tools": {"pytest": "9.0.2"}}

FAKE = '''#!{python}
"""A fake agent CLI. Each call runs the next step of NAME.plan.json (the last repeats) in the
working directory and appends what it saw to NAME.calls.jsonl beside itself."""
import json, os, pathlib, sys, time
here = pathlib.Path(__file__).resolve().parent
name = pathlib.Path(__file__).name
if sys.argv[1:] == ["--version"]:
    print({version!r})
    sys.exit(0)
calls = here / (name + ".calls.jsonl")
done = len(calls.read_text().splitlines()) if calls.exists() else 0
plan = json.loads((here / (name + ".plan.json")).read_text())
step = plan[min(done, len(plan) - 1)]
cwd = pathlib.Path.cwd()
home = os.environ.get("CODEX_HOME")
seen = {{"argv": sys.argv[1:], "cwd": str(cwd), "stdin": sys.stdin.read(), "pid": os.getpid(),
        "task": json.loads((cwd / "task.json").read_text()) if (cwd / "task.json").exists() else None,
        "env": {{k: os.environ.get(k) for k in ("CODEX_HOME", "VIRTUAL_ENV", "PYTHONPATH", "PWD", "CLAUDECODE",
                                                "CLAUDE_EFFORT", "ANTHROPIC_MODEL", "OPENAI_API_KEY")}},
        "environ": dict(os.environ),
        "path0": os.environ.get("PATH", "").split(os.pathsep)[0],
        "codex_home": sorted(os.listdir(home)) if home else None,
        "auth_seen": (pathlib.Path(home) / "auth.json").read_text()
                     if home and (pathlib.Path(home) / "auth.json").exists() else None}}
with open(calls, "a") as f:
    f.write(json.dumps(seen) + "\\n")
for rel, (old, new) in step.get("edit", {{}}).items():
    p = cwd / rel
    p.write_text(p.read_text().replace(old, new))
for rel in step.get("touch", []):
    p = cwd / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("touched\\n")
if home and "auth" in step:
    (pathlib.Path(home) / "auth.json").write_text(step["auth"])
if "spawn" in step:
    import subprocess
    (here / (name + ".spawned")).write_text(str(subprocess.Popen(step["spawn"]).pid))
if "json" in step:
    print(json.dumps(step["json"]))
for event in step.get("jsonl", []):
    print(json.dumps(event))
if "last_message" in step:
    pathlib.Path(sys.argv[sys.argv.index("-o") + 1]).write_text(step["last_message"])
sys.stdout.write(step.get("stdout", ""))
sys.stderr.write(step.get("stderr", ""))
time.sleep(step.get("sleep", 0))
sys.exit(step.get("exit", 0))
'''


def _git(repo, *args, check=True):
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=check)


@pytest.fixture
def git_repo(tmp_path):
    """A clone at its base commit; the fix commit (the gold patch: LIMIT 10 -> 20) lies in its history."""
    r = tmp_path / "clone"
    (r / "src/pkg").mkdir(parents=True)
    (r / "tests").mkdir()
    (r / "requirements").mkdir()
    (r / "src/pkg/__init__.py").write_text("")
    (r / "src/pkg/mod.py").write_text(SRC)
    (r / "tests/test_mod.py").write_text("from pkg.mod import LIMIT\n\n\ndef test_limit():\n    assert LIMIT >= 10\n")
    (r / "requirements/tests.txt").write_text("pytest\n")
    (r / ".gitignore").write_text("__pycache__/\n")
    for args in (["init", "-q", "-b", "main"], ["config", "user.email", "t@t"], ["config", "user.name", "t"],
                 ["add", "-A"], ["commit", "-qm", "base"]):
        _git(r, *args)
    base = _git(r, "rev-parse", "HEAD").stdout.strip()
    (r / "src/pkg/mod.py").write_text(SRC.replace("LIMIT = 10", "LIMIT = 20"))
    _git(r, "commit", "-qam", "raise the limit (#7)")
    fix = _git(r, "rev-parse", "HEAD").stdout.strip()
    _git(r, "checkout", "-q", base)
    return r, base, fix


@pytest.fixture(autouse=True)
def codex_login(tmp_path, monkeypatch):
    """The operator's codex login (CODEX_HOME with auth.json and a config of their own) and the study's
    login store, both in tmp_path: no test reads or writes the real ~/.codex or corpus/.codex_home."""
    operator = tmp_path / "operator-codex"
    operator.mkdir()
    (operator / "auth.json").write_text(OPERATOR_AUTH)
    (operator / "config.toml").write_text('model = "operator-model"\nmodel_reasoning_effort = "high"\n')
    monkeypatch.setenv("CODEX_HOME", str(operator))
    store = tmp_path / "codex-store"
    monkeypatch.setattr(authoring, "CODEX_STORE", store)
    return operator, store


def _cached_env(tmp: Path, clone: Path) -> Path:
    """A cached environment as corpus.envs.ensure_env leaves it for the clone: its interpreter, and env.json
    with the recipe and the freeze list, whose editable install names the clone."""
    home = tmp / "envs" / "k0"
    (home / "venv" / "bin").mkdir(parents=True)
    (home / "venv" / "bin" / "python").write_text("#!/bin/sh\n")
    (home / "env.json").write_text(json.dumps({
        "key": "k0", "inputs": dict(RECIPE, checkout=str(clone)), "python_full_version": "3.10.19",
        "packages": [f"-e file://{clone}"] + PINNED, "moved_by_tools": {"pytest": ["7.4.4", "9.0.2"]}}))
    (home / ".ok").write_text("")
    return home / "venv" / "bin" / "python"


class FakeUv:
    """uv as far as an export's environment needs it, offline. `uv venv` makes the venv; `uv pip install -e X`
    writes the editable .pth naming X under the directory it runs in, as uv does; `uv pip freeze` answers
    that editable install, the pins of the last constraints file it was given, and `extra` (a drift)."""

    def __init__(self, extra=()):
        self.calls, self.constraints, self.extra = [], [], list(extra)

    def __call__(self, argv, cwd):
        self.calls.append((list(argv), Path(cwd)))
        if "--constraint" in argv:
            self.constraints.append(Path(argv[argv.index("--constraint") + 1]).read_text())
        if argv[:2] == ["uv", "venv"]:
            venv = Path(argv[-1])
            (venv / "bin").mkdir(parents=True)
            (venv / "lib/python3.10/site-packages").mkdir(parents=True)
            (venv / "bin/python").write_text("#!/bin/sh\n")
            (venv / "bin/python").chmod(0o755)
            (venv / "pyvenv.cfg").write_text("home = /usr/bin\nversion_info = 3.10.19\n")
            return ""
        venv = Path(argv[argv.index("--python") + 1]).parent.parent
        site = venv / "lib/python3.10/site-packages"
        if argv[:3] == ["uv", "pip", "install"]:
            if "-e" in argv:
                (site / "_pkg.pth").write_text(str((Path(cwd) / argv[argv.index("-e") + 1]).resolve() / "src") + "\n")
            if "--constraint" in argv:
                (site / "_pins.txt").write_text(Path(argv[argv.index("--constraint") + 1]).read_text())
            return ""
        if argv[:3] == ["uv", "pip", "freeze"]:
            editable = (site / "_pkg.pth").read_text().strip().rsplit("/src", 1)[0]
            return f"-e file://{editable}\n" + (site / "_pins.txt").read_text() + "".join(f"{e}\n" for e in self.extra)
        return ""


def _fake(bin_dir: Path, name: str, plan, version="9.9.9 (Fake Code)") -> Path:
    bin_dir.mkdir(parents=True, exist_ok=True)
    exe = bin_dir / name
    exe.write_text(FAKE.format(python=sys.executable, version=version))
    exe.chmod(0o755)
    (bin_dir / f"{name}.plan.json").write_text(json.dumps(plan))
    return exe


def _calls(exe: Path):
    p = exe.parent / f"{exe.name}.calls.jsonl"
    return [json.loads(line) for line in p.read_text().splitlines()] if p.exists() else []


def _task(base, fix, n=1, partition="train"):
    return Task(f"r-{n}", "r", "o/r", n, "raise the limit", "LIMIT is too low for batch jobs.", 100 + n, fix, base,
                "2026-01-01T00:00:00Z", ["src/pkg/mod.py"], 2, partition)


def _cfg(r, tmp, authors, tasks=(), **extra):
    prompt = tmp / "author_prompt.md"
    prompt.write_text(PROMPT)
    out = tmp / "out"
    out.mkdir(exist_ok=True)
    (out / "tasks.jsonl").write_text("".join(json.dumps(t.to_dict()) + "\n" for t in tasks))
    cfg = {"repo": "r", "full_name": "o/r", "checkout": str(r), "out": str(out), "package_root": "src/pkg",
           "top_module": "pkg", "freeze_salt": "salt", "cwe_share": 0.0, "covered_share": 0.5,
           "gold_patch_dev": False, "author_prompt": str(prompt), "authors": authors,
           "battery_cmd": None, "oracle_cmd": ["sh", "-c", "exit 1"]}
    cfg.update(extra)
    return cfg


def _claude(exe, model="fake-model-2026", version="9.9.9 (Fake Code)"):
    return {"mode": "m0", "cmd": [str(exe), "-p", "{prompt}", "--output-format", "stream-json", "--verbose"],
            "model": model,
            "version": version}


def _codex(exe, model="fake-codex-1", version="codex-cli 9.9.9"):
    return {"mode": "m1", "cmd": [str(exe), "exec", "-C", ".", "--json", "-m", model, "-o", "{last_message}",
                                  "{prompt}"], "model": model, "version": version}


def _rows(out: Path, event="attempt"):
    p = out / "authored" / "authoring.jsonl"
    rows = [json.loads(line) for line in p.read_text().splitlines()] if p.exists() else []
    return [r for r in rows if r.get("event") == event]


def _diff_files(diff: str):
    return sorted(line.split(" b/", 1)[1] for line in diff.splitlines() if line.startswith("diff --git "))


def _naming(root: Path, needles):
    """Every file or link under `root` whose bytes or target contain one of `needles`."""
    hits = []
    for p in sorted(root.rglob("*")):
        if p.is_symlink():
            if any(n in os.readlink(p) for n in needles):
                hits.append(str(p))
        elif p.is_file() and any(n.encode() in p.read_bytes() for n in needles):
            hits.append(str(p))
    return hits


# ---------------------------------------------------------------- the export and its environment

def test_export_has_no_future_history(git_repo, tmp_path):
    """The agent works in `git archive` of the base commit, committed once: the fix commit is not in its
    object store, the file the gold patch changes is at its base state, and the tree is the base commit's."""
    r, base, fix = git_repo
    _git(r, "checkout", "-q", "main")                         # the clone may sit anywhere, here past the fix
    dest = export_base(r, base, tmp_path / "export")
    assert _git(dest, "log", "--all", "--format=%H").stdout.split() == [_git(dest, "rev-parse", "HEAD").stdout.strip()]
    assert (dest / "src/pkg/mod.py").read_text() == SRC        # LIMIT = 10, not the fix's 20
    assert _git(dest, "cat-file", "-e", fix, check=False).returncode != 0
    assert _git(dest, "rev-parse", "HEAD^{tree}").stdout == _git(r, "rev-parse", f"{base}^{{tree}}").stdout
    assert _git(dest, "status", "--porcelain").stdout == "" and _git(dest, "remote").stdout == ""
    assert _git(r, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip() == "main"     # the clone did not move


def test_export_has_no_clone_path(git_repo, tmp_path, monkeypatch):
    """Nothing the agent can see names the clone, whose history holds the fix, or the project the clone sits
    in: not the export's tree, not its .venv (a directory built for the export by the cached environment's
    recipe, with its pins, not a link to that environment, whose editable install points into the clone), not
    the environment the CLI runs with (a project venv on the operator's PATH included), and no name the agent
    sees says the study, the task or the arm (the export's directory, its commit's author)."""
    r, base, fix = git_repo
    project = tmp_path / "project"                       # the orchestrator runs here; the clone sits in it
    (project / "corpus" / "repos").mkdir(parents=True)
    r = Path(shutil.move(str(r), project / "corpus" / "repos" / "clone"))
    monkeypatch.chdir(project)
    monkeypatch.setenv("PATH", os.pathsep.join([str(project / ".venv" / "bin"), os.environ["PATH"]]))
    monkeypatch.setenv("PROJECT_NOTES", str(project / "notes.md"))
    cached = _cached_env(tmp_path, r)
    uv = FakeUv()
    monkeypatch.setattr(envs, "run_command", uv)
    exe = _fake(tmp_path / "bin", "claude", [{"edit": EDIT_20, "jsonl": CLAUDE_STREAM}])
    cfg = _cfg(r, tmp_path, [_claude(exe)], tasks=[_task(base, fix)])
    res = author_task(_task(base, fix), "m0", 1, cfg, cached_python=cached, keep_export=True)
    (call,) = _calls(exe)
    export = Path(call["cwd"])
    try:
        assert res.ok
        venv = export / ".venv"
        assert venv.is_dir() and not venv.is_symlink()
        needles = {str(r), str(r.resolve()), str(project), str(project.resolve()), str(cached.parent.parent.parent)}
        assert _naming(export, needles) == []
        assert not any(n in v for v in call["environ"].values() for n in needles)
        assert "PROJECT_NOTES" not in call["environ"]
        assert call["env"]["PYTHONPATH"] is None and call["env"]["PWD"] == str(export)
        assert export.name == "r" and not any(s in str(export) for s in ("rcca", "r-1", "-m0-", "-a1"))
        author_line = _git(export, "log", "-1", "--format=%an <%ae> %s").stdout
        assert "rcca" not in author_line.lower()
        assert (venv / "lib/python3.10/site-packages/_pkg.pth").read_text().strip() == str(export / "src")
        installs = [(a, cwd) for a, cwd in uv.calls if a[:3] == ["uv", "pip", "install"]]
        assert [cwd for _, cwd in installs] == [export, export]           # the steps, then the tools, in the export
        step, tools = installs[0][0], installs[1][0]
        assert step[-4:] == ["-e", ".", "-r", "requirements/tests.txt"]
        assert step[step.index("--exclude-newer") + 1] == RECIPE["exclude_newer"] and "--exclude-newer" not in tools
        assert not Path(step[step.index("--constraint") + 1]).is_relative_to(export)
    finally:
        shutil.rmtree(export.parent, ignore_errors=True)
    meta = json.loads((Path(cfg["out"]) / "authored" / "r-1:m0:a1.meta.json").read_text())
    assert isinstance(meta["venv_build_s"], float) and meta["venv_from"] == "k0"


def test_export_env_pins_the_cached_versions(git_repo, tmp_path, monkeypatch):
    """The install steps run constrained to the versions they resolved for the cached environment (before the
    tools moved any), the tools to the recorded freeze list; a result that differs from it is refused."""
    r, base, fix = git_repo
    cached = _cached_env(tmp_path, r)
    uv = FakeUv()
    monkeypatch.setattr(envs, "run_command", uv)
    dest = export_base(r, base, tmp_path / "export")
    python, seconds = envs.export_env(dest, cached)
    assert python == dest / ".venv" / "bin" / "python" and seconds >= 0
    # the step: what it resolved before the tools moved pytest; the tools: the recorded freeze list
    assert [c.split() for c in uv.constraints] == [["iniconfig==2.0.0", "pluggy==1.5.0", "pytest==7.4.4"], PINNED]
    shutil.rmtree(dest / ".venv")
    monkeypatch.setattr(envs, "run_command", FakeUv(extra=["six==1.16.0"]))
    with pytest.raises(envs.EnvError, match="six"):
        envs.export_env(dest, cached)


def test_export_refuses_agent_instructions(git_repo, tmp_path):
    """An export carries no CLAUDE.md or AGENTS.md: a base tree with one is refused, not authored."""
    r, base, fix = git_repo
    (r / "AGENTS.md").write_text("Always add a changelog entry.\n")
    _git(r, "add", "AGENTS.md")
    _git(r, "commit", "-qm", "agents")
    with pytest.raises(ExportError, match="AGENTS.md"):
        export_base(r, _git(r, "rev-parse", "HEAD").stdout.strip(), tmp_path / "export")


def test_command_author_refuses_a_clone(git_repo, tmp_path):
    """Authoring never runs in the clone, whose history holds the gold patch."""
    r, base, fix = git_repo
    exe = _fake(tmp_path / "bin", "claude", [{"edit": EDIT_20, "jsonl": CLAUDE_STREAM}])
    with pytest.raises(ExportError, match="export"):
        CommandAuthor("m0", [str(exe), "-p", "{prompt}"], "fake-model-2026", "9.9.9 (Fake Code)",
                      prompt=PROMPT).author(_task(base, fix), r)
    assert _calls(exe) == []


def test_author_diff_excludes_task_json_and_venv(git_repo, tmp_path):
    """The captured diff is the agent's change to the repository: not task.json, not the export's .venv
    or anything written into it, not caches or build products."""
    r, base, fix = git_repo
    dest = export_base(r, base, tmp_path / "export")
    (dest / ".venv" / "bin").mkdir(parents=True)                # as corpus.envs.export_env leaves it
    exe = _fake(tmp_path / "bin", "claude", [{
        "edit": EDIT_20, "jsonl": CLAUDE_STREAM,
        "touch": ["task.json", ".venv/bin/new-tool", "src/pkg/__pycache__/mod.cpython-310.pyc",
                  "src/pkg.egg-info/PKG-INFO", ".pytest_cache/v/cache/nodeids"]}])
    res = CommandAuthor("m0", [str(exe), "-p", "{prompt}"], "fake-model-2026", "9.9.9 (Fake Code)",
                        prompt=PROMPT).author(_task(base, fix), dest)
    assert res.ok and _diff_files(res.diff) == ["src/pkg/mod.py"] and "+LIMIT = 20" in res.diff
    assert not (dest / "task.json").exists()


def test_task_json_has_no_files(git_repo, tmp_path):
    """The agent receives the issue's title and body and nothing else: not the files the merged
    fix touched (they say where the fix lies), and nothing on stdin."""
    r, base, fix = git_repo
    exe = _fake(tmp_path / "bin", "claude", [{"edit": EDIT_20, "jsonl": CLAUDE_STREAM}])
    cfg = _cfg(r, tmp_path, [_claude(exe)], tasks=[_task(base, fix)])
    assert author_task(_task(base, fix), "m0", 1, cfg).ok
    (call,) = _calls(exe)
    assert call["task"] == {"title": "raise the limit", "body": "LIMIT is too low for batch jobs."}
    assert call["stdin"] == ""
    assert PROMPT in call["argv"] and str(r) not in call["cwd"]          # the frozen prompt, in an export


# ---------------------------------------------------------------- what an attempt records

@pytest.mark.parametrize("cli,step,author,want", [
    ("claude", {"edit": EDIT_20, "jsonl": CLAUDE_STREAM}, _claude,
     {"version": "9.9.9 (Fake Code)", "model": "fake-model-2026", "turns": 7,
      "tokens": {"input": 1200, "output": 340, "cache_read": 5000, "cache_creation": 800}}),
    ("codex", {"edit": EDIT_20, "jsonl": CODEX_EVENTS, "last_message": "Done."}, _codex,
     {"version": "codex-cli 9.9.9", "model": "fake-codex-1", "turns": 1,
      "tokens": {"input": 14914, "cached_input": 12288, "output": 210, "reasoning_output": 40}}),
])
def test_command_author_records_version(git_repo, tmp_path, cli, step, author, want):
    """Each attempt records the CLI's --version, the model it reports (codex's --json stream names
    none, so its -m), and the turns and tokens it reports, in authoring.jsonl and the .meta.json."""
    r, base, fix = git_repo
    exe = _fake(tmp_path / "bin", cli, [step], version=want["version"])
    entry = author(exe)
    cfg = _cfg(r, tmp_path, [entry], tasks=[_task(base, fix)])
    res = author_task(_task(base, fix), entry["mode"], 1, cfg)
    assert res.ok
    out = Path(cfg["out"])
    (row,) = _rows(out)
    assert {k: row[k] for k in want} == want
    assert (row["task"], row["mode"], row["attempt"], row["ok"], row["quota"]) == ("r-1", entry["mode"], 1, True, False)
    assert row["prompt_sha256"] == hashlib.sha256(PROMPT.encode()).hexdigest()
    meta = json.loads((out / "authored" / f"r-1:{entry['mode']}:a1.meta.json").read_text())
    assert (meta["version"], meta["model"], meta["returncode"], meta["timed_out"]) == (want["version"], want["model"], 0, False)
    assert "+LIMIT = 20" in (out / "authored" / f"r-1:{entry['mode']}:a1.diff").read_text()
    # the CLI's whole event stream is kept for audit, every tool call in it; claude's init event says which
    # tools, slash commands, skills and plugins the session had
    events = [json.loads(line) for line in (out / meta["events"]).read_text().splitlines()]
    assert events == (CLAUDE_STREAM if cli == "claude" else CODEX_EVENTS) and "stdout_tail" not in meta
    if cli == "claude":
        assert meta["init"]["tools"] == ["Bash", "Edit", "Read", "Write"]
        assert (meta["init"]["slash_commands"], meta["init"]["skills"]) == ([], [])
        assert meta["init"]["plugins"] == CLAUDE_INIT["plugins"]


def test_agent_runs_isolated_from_the_operator_config(git_repo, tmp_path, monkeypatch, codex_login):
    """The agent inherits none of the operator's agent configuration: no CLAUDE*/ANTHROPIC_*/OPENAI_*
    variables, and codex a fresh CODEX_HOME holding only auth.json, removed afterwards. The export's own
    environment is the active one, and nothing puts other sources on the import path."""
    r, base, fix = git_repo
    operator, store = codex_login
    for k, v in (("CLAUDECODE", "1"), ("CLAUDE_EFFORT", "max"), ("ANTHROPIC_MODEL", "x"), ("OPENAI_API_KEY", "k"),
                 ("PYTHONPATH", "/somewhere/else")):
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(envs, "run_command", FakeUv())
    exe = _fake(tmp_path / "bin", "codex", [{"edit": EDIT_20, "jsonl": CODEX_EVENTS}])
    cfg = _cfg(r, tmp_path, [_codex(exe)], tasks=[_task(base, fix)])
    assert author_task(_task(base, fix), "m1", 1, cfg, cached_python=_cached_env(tmp_path, r)).ok
    (call,) = _calls(exe)
    export = Path(call["cwd"])
    assert call["codex_home"] == ["auth.json"] and Path(call["env"]["CODEX_HOME"]) not in (operator, store)
    assert not Path(call["env"]["CODEX_HOME"]).exists()
    assert not any("rcca" in a for a in [call["env"]["CODEX_HOME"], *call["argv"]])    # neutral names only
    assert all(call["env"][k] is None for k in ("CLAUDECODE", "CLAUDE_EFFORT", "ANTHROPIC_MODEL", "OPENAI_API_KEY",
                                                "PYTHONPATH"))
    assert call["path0"] == str(export / ".venv" / "bin") and call["env"]["VIRTUAL_ENV"] == str(export / ".venv")


def test_codex_home_is_persistent_and_never_writes_operator_auth(git_repo, tmp_path, codex_login):
    """The study keeps its own codex login in a store (corpus/.codex_home): seeded once from the operator's
    auth.json, it keeps the tokens codex refreshes, and every run gets a fresh CODEX_HOME holding only that
    auth.json. The operator's file is never written, unless an author opts in with write_back_auth."""
    r, base, fix = git_repo
    operator, store = codex_login
    before = (operator / "auth.json").read_bytes()
    exe = _fake(tmp_path / "bin", "codex", [{"edit": EDIT_20, "jsonl": CODEX_EVENTS, "auth": '{"token": "fresh-1"}'},
                                            {"edit": EDIT_20, "jsonl": CODEX_EVENTS, "auth": '{"token": "fresh-2"}'}])
    cfg = _cfg(r, tmp_path, [_codex(exe)], tasks=[_task(base, fix, 1), _task(base, fix, 2)])
    assert build.cmd_author(cfg) == 0
    calls = _calls(exe)
    assert [c["auth_seen"] for c in calls] == [OPERATOR_AUTH, '{"token": "fresh-1"}']   # seeded once, then kept
    assert all(c["codex_home"] == ["auth.json"] for c in calls)
    assert len({c["env"]["CODEX_HOME"] for c in calls}) == 2                             # a fresh home per run
    assert (store / "auth.json").read_text() == '{"token": "fresh-2"}'
    assert (operator / "auth.json").read_bytes() == before
    (operator / "auth.json").write_text('{"token": "operator-2"}')      # the operator logs in again meanwhile
    _fake(tmp_path / "bin", "codex", [{"edit": EDIT_20, "jsonl": CODEX_EVENTS, "auth": '{"token": "fresh-3"}'}])
    cfg = _cfg(r, tmp_path, [_codex(exe)], tasks=[_task(base, fix, 3)])
    assert build.cmd_author(cfg) == 0
    assert _calls(exe)[-1]["auth_seen"] == '{"token": "fresh-2"}'     # the store's own, not the operator's
    assert (operator / "auth.json").read_text() == '{"token": "operator-2"}'
    _fake(tmp_path / "bin", "codex", [{"edit": EDIT_20, "jsonl": CODEX_EVENTS, "auth": '{"token": "fresh-4"}'}])
    cfg = _cfg(r, tmp_path, [dict(_codex(exe), write_back_auth=True)], tasks=[_task(base, fix, 4)])
    assert build.cmd_author(cfg) == 0                                   # opted in: the refresh goes back too
    assert (operator / "auth.json").read_text() == (store / "auth.json").read_text() == '{"token": "fresh-4"}'


# ---------------------------------------------------------------- quota, regeneration, attrition

def test_quota_exhaustion_is_recorded_not_a_failure(git_repo, tmp_path):
    """A drained quota answers in seconds with a banner. It is recorded as a quota event that spends
    no attempt, the CLI is not called again in that run, and the next run makes attempt 1 again."""
    r, base, fix = git_repo
    exe = _fake(tmp_path / "bin", "codex", [QUOTA, {"edit": EDIT_20, "jsonl": CODEX_EVENTS}])
    tasks = [_task(base, fix, 1), _task(base, fix, 2)]
    cfg = _cfg(r, tmp_path, [_codex(exe)], tasks=tasks)
    out = Path(cfg["out"])
    assert build.cmd_author(cfg) == 4                          # a mode stopped on its quota
    (row,) = _rows(out)
    assert (row["task"], row["attempt"], row["ok"], row["quota"], row["counted"]) == ("r-1", 1, False, True, False)
    assert len(_calls(exe)) == 1                                # r-2 was not attempted on a drained quota
    attrition = json.loads((out / "authored" / "attrition.json").read_text())
    assert attrition["blocked"] == {"quota": 1} and attrition["pending"] == 2 and attrition["exhausted"] == []
    assert build.cmd_author(cfg) == 0
    assert [(x["task"], x["attempt"], x["ok"]) for x in _rows(out)][1:] == [("r-1", 1, True), ("r-2", 1, True)]
    assert json.loads((out / "authored" / "r-1:m1:a1.meta.json_old").read_text())["quota"] is True


@pytest.mark.parametrize("cli,step,author,routes", [
    ("claude", {"edit": EDIT_20, "jsonl": [dict(CLAUDE_INIT, tools=CLAUDE_INIT["tools"] + ["WebFetch", "WebSearch"]),
                                           CLAUDE_RESULT]}, _claude, ["WebFetch", "WebSearch"]),
    ("codex", {"edit": EDIT_20, "jsonl": CODEX_EVENTS[:2] + [{"type": "item.completed", "item": {
        "id": "ws_0", "type": "web_search", "query": "flask autoescape case-insensitive"}}] + CODEX_EVENTS[2:]},
     _codex, ["web_search"]),
])
def test_a_route_to_the_web_stops_the_mode(git_repo, tmp_path, cli, step, author, routes):
    """The agents have no route to the web, where the upstream fix is one search away: a claude session whose
    tool set holds a web tool, or a codex run that searched the web, is a fault of the configuration. It spends
    no attempt, its change is not used, and the mode stops for the run."""
    r, base, fix = git_repo
    exe = _fake(tmp_path / "bin", cli, [step])
    cfg = _cfg(r, tmp_path, [author(exe)], tasks=[_task(base, fix, 1), _task(base, fix, 2)])
    assert build.cmd_author(cfg) == 4
    (row,) = _rows(Path(cfg["out"]))
    assert (row["ok"], row["counted"], row["blocked"], row["diff"]) == (False, False, "config", None)
    assert len(_calls(exe)) == 1
    assert json.loads((Path(cfg["out"]) / row["meta"]).read_text())["network_routes"] == routes


def _interrupt_when_at_work(monkeypatch, exe: Path, then):
    """Patch Popen.wait for the agent run of the CLI `exe` only (not its --version probe): once the agent has
    started its child, run `then`."""
    wait, spawned = subprocess.Popen.wait, exe.parent / f"{exe.name}.spawned"

    def patched(proc, timeout=None):
        if proc.args and str(proc.args[0]) == str(exe) and "--version" not in proc.args:
            deadline = time.monotonic() + 10
            while not spawned.exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            then()
        return wait(proc, timeout=timeout)
    monkeypatch.setattr(subprocess.Popen, "wait", patched)
    return spawned


def test_interrupt_kills_the_agent_and_keeps_a_refreshed_token(git_repo, tmp_path, monkeypatch, codex_login):
    """Ctrl-C reaches the harness only (the CLI runs in a session of its own). The interrupt kills the agent and
    everything it started before its export is removed, keeps the token codex refreshed meanwhile, records
    nothing (the attempt is made again), and goes on to stop `author`."""
    r, base, fix = git_repo
    operator, store = codex_login
    exe = _fake(tmp_path / "bin", "codex", [{"auth": '{"token": "fresh"}', "spawn": ["sleep", "60"], "sleep": 60}])
    cfg = _cfg(r, tmp_path, [_codex(exe)], tasks=[_task(base, fix)])

    def ctrl_c():
        raise KeyboardInterrupt
    spawned = _interrupt_when_at_work(monkeypatch, exe, ctrl_c)
    with pytest.raises(KeyboardInterrupt):
        author_task(_task(base, fix), "m1", 1, cfg)
    (call,) = _calls(exe)
    assert _gone(call["pid"]) and _gone(int(spawned.read_text()))
    assert not Path(call["cwd"]).exists()
    assert (store / "auth.json").read_text() == '{"token": "fresh"}'
    assert (operator / "auth.json").read_text() == OPERATOR_AUTH
    assert _rows(Path(cfg["out"])) == []


def test_sigterm_stops_author_as_an_interrupt_does(git_repo, tmp_path, monkeypatch):
    """`author` turns SIGTERM (a scheduler or `kill` stopping the batch) into SystemExit, so the agent is killed
    the same way, and puts back the handler it found."""
    r, base, fix = git_repo
    exe = _fake(tmp_path / "bin", "claude", [{"spawn": ["sleep", "60"], "sleep": 60}])
    cfg = _cfg(r, tmp_path, [_claude(exe)], tasks=[_task(base, fix)])
    spawned = _interrupt_when_at_work(monkeypatch, exe, lambda: os.kill(os.getpid(), signal.SIGTERM))

    def outside(signum, frame):
        raise AssertionError("author left SIGTERM to the handler it found")
    found = signal.signal(signal.SIGTERM, outside)
    try:
        with pytest.raises(SystemExit):
            build.cmd_author(cfg)
        assert signal.getsignal(signal.SIGTERM) is outside
    finally:
        signal.signal(signal.SIGTERM, found)
    (call,) = _calls(exe)
    assert _gone(call["pid"]) and _gone(int(spawned.read_text()))
    assert not Path(call["cwd"]).exists()


def _gone(pid: int, wait: float = 5.0) -> bool:
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.1)
    return False


def test_timeout_is_counted_never_a_quota_event(git_repo, tmp_path):
    """A run that reaches its timeout spends its attempt, whatever banner it printed (a quota answers in
    seconds; a timeout taken for one would be retried for ever), and nothing it started outlives it."""
    r, base, fix = git_repo
    exe = _fake(tmp_path / "bin", "claude", [{"spawn": ["sleep", "60"], "stderr": QUOTA["stderr"], "sleep": 30}])
    cfg = _cfg(r, tmp_path, [dict(_claude(exe), timeout=1)], tasks=[_task(base, fix)])
    t0 = time.monotonic()
    res = author_task(_task(base, fix), "m0", 1, cfg)
    assert time.monotonic() - t0 < 20
    (row,) = _rows(Path(cfg["out"]))
    assert (res.ok, row["failure"], row["counted"], row["quota"]) == (False, "timeout", True, False)
    assert _gone(int((tmp_path / "bin" / "claude.spawned").read_text()))


def test_a_banner_in_the_agents_work_is_not_a_quota_event(git_repo, tmp_path):
    """codex's event stream carries every command's output; a usage-limit phrase there, in a run that
    succeeded, is the agent's work, not the CLI's quota."""
    r, base, fix = git_repo
    noisy = CODEX_EVENTS[:2] + [{"type": "item.completed", "item": {
        "id": "item_0", "type": "command_execution", "command": "grep -rn limit docs",
        "aggregated_output": "docs/api.rst: raise RateLimit when you hit your usage limit (429)\n", "exit_code": 0}}
    ] + CODEX_EVENTS[2:]
    exe = _fake(tmp_path / "bin", "codex", [{"edit": EDIT_20, "jsonl": noisy}])
    cfg = _cfg(r, tmp_path, [_codex(exe)], tasks=[_task(base, fix)])
    res = author_task(_task(base, fix), "m1", 1, cfg)
    (row,) = _rows(Path(cfg["out"]))
    assert (res.ok, row["quota"], row["counted"]) == (True, False, True)


def test_command_author_refuses_a_tree_with_its_own_task_json(git_repo, tmp_path):
    r, base, fix = git_repo
    (r / "task.json").write_text("{}\n")
    _git(r, "add", "task.json")
    _git(r, "commit", "-qm", "a task file of the repository's own")
    dest = export_base(r, _git(r, "rev-parse", "HEAD").stdout.strip(), tmp_path / "export")
    exe = _fake(tmp_path / "bin", "claude", [{"edit": EDIT_20, "jsonl": CLAUDE_STREAM}])
    with pytest.raises(ExportError, match="task.json"):
        CommandAuthor("m0", [str(exe), "-p", "{prompt}"], "fake-model-2026", "9.9.9 (Fake Code)",
                      prompt=PROMPT).author(_task(base, fix), dest)
    assert _calls(exe) == []


def test_regenerate_stops_at_three(git_repo, tmp_path):
    """An attempt that changes nothing is regenerated, with the reason recorded, until three attempts
    are spent; the fourth is never made, and the attrition is recorded once."""
    r, base, fix = git_repo
    exe = _fake(tmp_path / "bin", "claude", [{"jsonl": CLAUDE_STREAM}])        # succeeds, edits nothing
    cfg = _cfg(r, tmp_path, [_claude(exe)], tasks=[_task(base, fix)])
    out = Path(cfg["out"])
    for _ in range(4):
        build.cmd_author(cfg)
    rows = _rows(out)
    assert [(x["attempt"], x["ok"], x["failure"], x["reason"]) for x in rows] == [
        (1, False, "empty diff", None), (2, False, "empty diff", "empty diff"), (3, False, "empty diff", "empty diff")]
    assert len(_calls(exe)) == 3
    assert regenerate(out, "r-1", "m0", "empty diff") is None
    (gone,) = _rows(out, "exhausted")
    assert (gone["task"], gone["mode"], gone["attempts"]) == ("r-1", "m0", 3)
    attrition = json.loads((out / "authored" / "attrition.json").read_text())
    assert [(x["task"], x["mode"], x["reasons"]) for x in attrition["exhausted"]] == [
        ("r-1", "m0", ["empty diff", "empty diff", "empty diff"])]


def test_pairs_consumes_latest_successful_attempt(git_repo, tmp_path):
    """pairs authors nothing: it pairs the latest successful attempt per (task, mode). A pair whose safe
    variant fails the battery is regenerated by `author --retry-failed` under a new candidate id
    suffix; the next pairs run admits the new attempt, records the count, and moves the old rows
    out of the manifest and admission files into their .superseded files."""
    r, base, fix = git_repo
    exe = _fake(tmp_path / "bin", "claude", [{"edit": EDIT_20, "jsonl": CLAUDE_STREAM},
                                             {"edit": EDIT_30, "jsonl": CLAUDE_STREAM}])
    cfg = _cfg(r, tmp_path, [_claude(exe)], tasks=[_task(base, fix)],
               battery_cmd=["sh", "-c", "grep -q 'LIMIT = 30' src/pkg/mod.py"])
    out = Path(cfg["out"])
    build.cmd_pairs(cfg)                                        # nothing authored yet: nothing to pair
    assert (out / "admission.jsonl").read_text() == "" and _calls(exe) == []
    build.cmd_author(cfg)
    build.cmd_pairs(cfg)
    (adm,) = [json.loads(line) for line in open(out / "admission.jsonl")]
    assert (adm["safe_passes_battery"], adm["regenerated"]) == (False, 0)
    build.cmd_author(cfg)                                       # attempt 1 succeeded: no new attempt
    assert len(_calls(exe)) == 1
    build.cmd_author(cfg, retry_failed=True)
    assert [(x["attempt"], x["reason"]) for x in _rows(out)] == [(1, None), (2, "safe fails battery")]
    build.cmd_pairs(cfg, resume=True)
    (adm,) = [json.loads(line) for line in open(out / "admission.jsonl")]
    assert (adm["safe_passes_battery"], adm["both_apply_cleanly"], adm["regenerated"]) == (True, True, 1)
    man = [json.loads(line) for line in open(out / "manifest.jsonl")]
    assert [m["candidate_id"] for m in man] == ["r-1:s:m0:a2", "r-1:u:m0:a2"]
    assert "+LIMIT = 30" in (out / man[0]["diff_path"]).read_text()
    old = [json.loads(line) for line in open(out / "manifest.superseded.jsonl")]
    assert [m["candidate_id"] for m in old] == ["r-1:s:m0", "r-1:u:m0"]
    assert [json.loads(line)["regenerated"] for line in open(out / "admission.superseded.jsonl")] == [0]


def test_regenerated_candidate_keeps_its_mode():
    """The id of a regenerated candidate carries its attempt after the mode; replay still reads the mode."""
    from corpus.manifest import candidate_id
    from rcca.replay import mode_of
    assert candidate_id("flask-1", True, "m1", attempt=2) == "flask-1:u:m1:a2"
    assert candidate_id("flask-1", False, "m0") == "flask-1:s:m0"
    assert [mode_of(c) for c in ("flask-1:u:m1:a2", "flask-1:s:m0", "flask-1")] == ["m1", "m0", "m0"]


# ---------------------------------------------------------------- an API key, by name, when the author asks

def test_api_key_passes_through_only_when_the_author_names_it(git_repo, tmp_path, monkeypatch):
    """The agent's environment is scrubbed of every ANTHROPIC_* variable by design. A confirmatory run
    on the API rather than the operator's subscription windows (2026-10-22) names the variable in the
    config (`auth_env`); that one survives the scrub, the attempt's meta records its *name*, and its
    value appears nowhere."""
    from corpus import authoring as A
    r, base, fix = git_repo
    dest = export_base(r, base, tmp_path / "export")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-0000")
    exe = _fake(tmp_path / "bin", "claude", [{"edit": EDIT_20, "jsonl": CLAUDE_STREAM}])
    seen = {}

    def fake_run_cli(argv, cwd, env, timeout, logs):
        seen["env"] = dict(env)
        return A.CliRun(0, "", "", 1.0, False)
    monkeypatch.setattr(A, "run_cli", fake_run_cli)
    plain = CommandAuthor("m0", [str(exe), "-p", "{prompt}"], "m", "v", prompt=PROMPT).author(_task(base, fix), dest)
    assert "ANTHROPIC_API_KEY" not in seen["env"] and "ANTHROPIC_API_KEY" in plain.meta["scrubbed_env"]
    assert plain.meta["auth"] == "cli login"
    keyed = CommandAuthor("m0", [str(exe), "-p", "{prompt}"], "m", "v", prompt=PROMPT,
                          auth_env="ANTHROPIC_API_KEY").author(_task(base, fix), dest)
    assert seen["env"]["ANTHROPIC_API_KEY"] == "sk-ant-test-0000"
    assert "ANTHROPIC_API_KEY" not in keyed.meta["scrubbed_env"] and keyed.meta["auth"] == "env:ANTHROPIC_API_KEY"
    assert "sk-ant-test" not in json.dumps(keyed.meta)
    (tmp_path / "prompt.md").write_text(PROMPT)
    cfg = {"authors": [{"mode": "m0", "cmd": [str(exe), "-p", "{prompt}"], "model": "m", "version": "v",
                        "auth_env": "ANTHROPIC_API_KEY"}], "author_prompt": str(tmp_path / "prompt.md")}
    assert A.authors_from_config(cfg)[0].auth_env == "ANTHROPIC_API_KEY"


# ---------------------------------------------------------------- LocalModelAuthor (ollama)

def _say(text=None, calls=()):
    """One OpenAI-shaped assistant turn: plain text, or tool calls."""
    msg = {"role": "assistant", "content": text or ""}
    if calls:
        msg["tool_calls"] = [{"id": f"c{i}", "type": "function",
                              "function": {"name": n, "arguments": json.dumps(a)}}
                             for i, (n, a) in enumerate(calls)]
    return {"choices": [{"message": msg, "finish_reason": "tool_calls" if calls else "stop"}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7}}


def _scripted(*turns):
    """A transport that answers with `turns` in order, then stops; records what it was sent."""
    sent = []
    it = iter(turns)

    def transport(payload, timeout):
        sent.append(payload)
        return next(it, _say("done"))

    transport.sent = sent
    return transport


def test_local_model_author_captures_an_edit(git_repo, tmp_path):
    """The loop's file writes land in the export and come back as the attempt's diff."""
    from corpus.authoring import LocalModelAuthor
    r, base, fix = git_repo
    dest = export_base(r, base, tmp_path / "export")
    t = _scripted(_say(calls=[("write_file", {"path": "src/pkg/mod.py",
                                              "content": SRC.replace("LIMIT = 10", "LIMIT = 20")})]),
                  _say("raised the limit"))
    res = LocalModelAuthor("m2", "qwen2.5-coder-32b-ctx32k", prompt=PROMPT, transport=t).author(
        _task(base, fix), dest)
    assert res.ok, res.error
    assert _diff_files(res.diff) == ["src/pkg/mod.py"] and "+LIMIT = 20" in res.diff
    assert res.mode == "m2" and res.provenance["model"] == "qwen2.5-coder-32b-ctx32k"
    assert not (dest / "task.json").exists()


def test_local_model_author_refuses_a_clone(git_repo, tmp_path):
    """Like every author: never the clone, whose history holds the gold patch."""
    from corpus.authoring import LocalModelAuthor
    r, base, fix = git_repo
    t = _scripted(_say("hello"))
    with pytest.raises(ExportError, match="export"):
        LocalModelAuthor("m2", "q", prompt=PROMPT, transport=t).author(_task(base, fix), r)
    assert t.sent == []


def test_local_model_tools_cannot_leave_the_export(git_repo, tmp_path):
    """A path that escapes the export is refused and told to the model, and nothing outside is written."""
    from corpus.authoring import LocalModelAuthor
    r, base, fix = git_repo
    dest = export_base(r, base, tmp_path / "export")
    outside = tmp_path / "outside.txt"
    t = _scripted(_say(calls=[("write_file", {"path": "../outside.txt", "content": "x"}),
                             ("read_file", {"path": "/etc/hosts"})]),
                  _say(calls=[("write_file", {"path": "src/pkg/mod.py",
                                              "content": SRC.replace("LIMIT = 10", "LIMIT = 20")})]),
                  _say("done"))
    res = LocalModelAuthor("m2", "q", prompt=PROMPT, transport=t).author(_task(base, fix), dest)
    assert not outside.exists()
    results = [m for p in t.sent for m in p["messages"] if m.get("role") == "tool"]
    assert any("outside the export" in m["content"] for m in results), results
    assert res.ok and "+LIMIT = 20" in res.diff


def test_local_model_author_empty_diff_is_a_failure(git_repo, tmp_path):
    """A model that changes nothing is a failed attempt, as for the CLI authors."""
    from corpus.authoring import LocalModelAuthor
    r, base, fix = git_repo
    dest = export_base(r, base, tmp_path / "export")
    res = LocalModelAuthor("m2", "q", prompt=PROMPT, transport=_scripted(_say("I see no bug."))).author(
        _task(base, fix), dest)
    assert not res.ok and res.failure == "empty diff"


def test_local_model_author_keeps_its_transcript(git_repo, tmp_path):
    """The whole exchange is kept, as the CLI authors keep their event streams."""
    from corpus.authoring import LocalModelAuthor
    r, base, fix = git_repo
    dest = export_base(r, base, tmp_path / "export")
    t = _scripted(_say(calls=[("run", {"command": "echo hi"})]),
                  _say(calls=[("write_file", {"path": "src/pkg/mod.py",
                                              "content": SRC.replace("LIMIT = 10", "LIMIT = 20")})]),
                  _say("done"))
    res = LocalModelAuthor("m2", "q", prompt=PROMPT, transport=t).author(_task(base, fix), dest)
    events = [json.loads(l) for l in res.events.splitlines() if l.strip()]
    assert [e["type"] for e in events].count("tool_result") == 2
    assert any(e["type"] == "tool_call" and e["name"] == "run" for e in events)
    assert any("hi" in str(e.get("output", "")) for e in events if e["type"] == "tool_result")
    assert res.meta["steps"] == 3 and res.meta["tokens"]["completion"] == 21


def test_authors_from_config_builds_a_local_author(tmp_path):
    """A `runner: local` entry is a LocalModelAuthor; the CLI entries are unaffected."""
    from corpus.authoring import LocalModelAuthor, authors_from_config
    prompt = tmp_path / "p.md"
    prompt.write_text(PROMPT)
    cfg = {"author_prompt": str(prompt), "authors": [
        {"mode": "m0", "cmd": ["claude", "-p", "{prompt}"], "model": "x", "version": "y"},
        {"mode": "m2", "runner": "local", "model": "qwen2.5-coder-32b-ctx32k",
         "base_url": "http://localhost:11434/v1", "version": "ollama 0.33.3", "max_steps": 7}]}
    m0, m2 = authors_from_config(cfg)
    assert isinstance(m0, CommandAuthor) and isinstance(m2, LocalModelAuthor)
    assert (m2.mode, m2.model, m2.max_steps) == ("m2", "qwen2.5-coder-32b-ctx32k", 7)
    assert m2.prompt == PROMPT and m2.base_url == "http://localhost:11434/v1"


def test_every_agent_author_runs_in_an_export(tmp_path):
    """build prepares an environment for every author that runs in an export, not only the CLI ones:
    without it the export has no .venv and the agent cannot run the suite."""
    from corpus.authoring import GoldPatchAuthor, LocalModelAuthor, runs_in_export
    assert runs_in_export(LocalModelAuthor("m2", "q"))
    assert runs_in_export(CommandAuthor("m0", ["claude"], "x", "y"))
    assert not runs_in_export(GoldPatchAuthor())


def test_author_task_exports_for_a_local_author_too(git_repo, tmp_path, monkeypatch):
    """Every agent author is called in an export, not the clone. corpus.regenerate branched on
    isinstance(CommandAuthor), so a local author was handed the clone and require_export refused it:
    the firewall held, but the mode could never run."""
    from corpus.authoring import LocalModelAuthor
    import corpus.authoring as A
    r, base, fix = git_repo
    t = _scripted(_say(calls=[("write_file", {"path": "src/pkg/mod.py",
                                              "content": SRC.replace("LIMIT = 10", "LIMIT = 20")})]),
                  _say("done"))
    author = LocalModelAuthor("m2", "q", prompt=PROMPT, transport=t)
    cfg = _cfg(r, tmp_path, [{"mode": "m2", "runner": "local", "model": "q",
                              "base_url": "http://localhost:11434/v1", "version": "v"}],
               tasks=[_task(base, fix)])
    res = author_task(_task(base, fix), "m2", 1, cfg, author=author)
    assert res.ok, res.error
    assert _diff_files(res.diff) == ["src/pkg/mod.py"] and "+LIMIT = 20" in res.diff
    assert "LIMIT = 10" in (r / "src/pkg/mod.py").read_text()      # the clone is untouched


def _text_call(name, args):
    """A model that writes its tool call into the message instead of tool_calls, as
    qwen2.5-coder:32b does: ollama's template for that model emits no structured call."""
    return {"choices": [{"message": {"role": "assistant",
                                     "content": json.dumps({"name": name, "arguments": args})},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 9, "completion_tokens": 5}}


def test_local_model_author_reads_a_tool_call_written_as_text(git_repo, tmp_path):
    from corpus.authoring import LocalModelAuthor
    r, base, fix = git_repo
    dest = export_base(r, base, tmp_path / "export")
    t = _scripted(_text_call("write_file", {"path": "src/pkg/mod.py",
                                            "content": SRC.replace("LIMIT = 10", "LIMIT = 20")}),
                  _say("done"))
    res = LocalModelAuthor("m2", "q", prompt=PROMPT, transport=t).author(_task(base, fix), dest)
    assert res.ok, res.error
    assert "+LIMIT = 20" in res.diff
    assert res.meta["text_tool_calls"] == 1


def test_local_model_author_does_not_mistake_prose_for_a_tool_call(git_repo, tmp_path):
    """Only a bare JSON object naming a tool counts; an explanation that mentions one does not."""
    from corpus.authoring import LocalModelAuthor
    r, base, fix = git_repo
    dest = export_base(r, base, tmp_path / "export")
    t = _scripted(_say('I would call {"name": "write_file"} but the file looks correct already.'))
    res = LocalModelAuthor("m2", "q", prompt=PROMPT, transport=t).author(_task(base, fix), dest)
    assert res.failure == "empty diff" and res.meta["text_tool_calls"] == 0

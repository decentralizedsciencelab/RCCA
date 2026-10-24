"""Freeze record and stage firewall (rcca.freeze).

Every command that writes a non-development output runs under a freeze
record whose stage permits it. The fixture record and configs are written to
tmp_path and validate clean, so each rejection test changes exactly one thing;
only test_shipped_record_reports_the_real_configs reads configs/.
"""
from rcca import freeze                      # first: the module under test
from rcca.freeze import FreezeError, load_freeze, require_stage, validate

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from corpus import build
from corpus.tasks import Task, assign_partitions
from evidence import baseline as evbaseline, run as evrun, snapshot as evsnapshot
from evidence.core import Action, Outcome, Registry, cell_seed
from rcca import analyze
from rcca.fit import write_records
from rcca.matrix import default_action_profiles, synthesize_matrix

ROOT = Path(__file__).resolve().parents[1]
PRODUCERS = ["f1.repo_ci", "f2.static_security", "f2.dependency", "f3.api_contract",
             "f4.impact_revert", "ch.property", "ch.differential", "ch.fuzz"]
SALT = "5f0c9a1e3b7d2c4f8a6e0b1d3c5f7a9e"           # the shape secrets.token_hex(16) draws
AUTHORS = [{"mode": "m0", "cmd": ["agent-a", "-p", "prompt.md"], "model": "model-a-2026-09", "version": "2.1.0"},
           {"mode": "m1", "cmd": ["agent-b", "exec"], "model": "model-b-2026-08", "version": "0.163.0"}]
TUNED_LOSS = {"kappa": 100, "lambda_R": 10, "lambda_E": 2, "s_tilde": 0.6, "tau_qual": 0.02, "eta": 1.0}
UNTUNED = dict(m_S=None, ceilings=None, nominal_costs=None, loss=dict(TUNED_LOSS, tau_qual=None, eta=None))
# the frozen shape, as rcca.analyze writes analysis/ceilings.json: no wall-clock ceiling
CEILINGS = {"q30": {"tokens": 3000.0, "tool_calls": 9.0, "ci_equiv": 2.5},
            "q60": {"tokens": 4000.0, "tool_calls": 12.0, "ci_equiv": 3.5},
            "q90": {"tokens": 6000.0, "tool_calls": 16.0, "ci_equiv": 5.0}}
# the frozen shape of analysis/nominal_costs.json, for the synthetic actions the fixture corpora carry
NOMINAL = {p.action.id: {d: float(getattr(p.action.nominal_cost, d)) for d in ("tokens", "tool_calls", "ci_equiv", "wall_s")}
           for p in default_action_profiles(0.71)}
WEIGHTS = [1 / 3, 1 / 3, 1 / 3]                       # w_j of C_w over tokens, tool calls, CI-equivalents
BASELINE = {"repo": "r", "commit": "c0ffee", "command": ["python", "-m", "pytest", "tests"], "n": 1,
            "cpu_s_runs": [2.0], "wall_s_runs": [2.1], "cpu_s_median": 2.0, "cpu_s_spread": 0.0, "wall_s_median": 2.1}


def _write(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1))
    return path


def _record(tmp: Path, stage: str, out: str = "corpus", repo=None, evidence=None, **fields) -> Path:
    """A freeze record at `stage` plus the repository and evidence config it lists,
    written to `tmp`. As built here everything validates clean."""
    checkout = tmp / "checkout"
    prompt = tmp / "author_prompt.md"
    tmp.mkdir(parents=True, exist_ok=True)
    prompt.write_text("A task.json in the repository root holds the issue. Fix it.\n")
    rcfg = {"repo": "r", "full_name": "o/r", "package_root": "src/pkg", "checkout": str(checkout),
            "out": str(tmp / out), "cutoff": "2026-06-01T00:00:00Z", "freeze_salt": "config-salt",
            "cwe_share": 0.5, "covered_share": 0.5, "cwe_catalog": str(_write(tmp / "cwe_r.json", [])),
            "held_out_families": [], "gold_patch_dev": False, "authors": AUTHORS, "author_prompt": str(prompt),
            "battery_cmd": ["python", "-m", "pytest", str(tmp / "oracle" / "battery"), "-q"],
            "oracle_cmd": ["python", "-m", "oracle_runner", "--candidate", "{id}"]}
    ecfg = {"freeze_salt": "config-salt", "python": str(checkout / ".venv/bin/python"),
            "ci_baseline": str(_write(tmp / "ci_baseline.json", BASELINE)),
            "f1": {"test_args": ["tests"], "source": "pkg", "extra_deselect": ["tests/test_a.py::test_flaky"]}}
    rcfg.update(repo or {})
    ecfg.update(evidence or {})
    rec = {"freeze_id": None, "stage": stage, "signed": stage == "unblinded",
           "created": "2026-10-16T09:00:00Z", "salt": None if stage == "dev" else SALT,
           "repos": [str(_write(tmp / "repo.json", rcfg))],
           "evidence_configs": [str(_write(tmp / "evidence.json", ecfg))],
           "knobs": {"cwe_share": 0.5, "covered_share": 0.5}, "loss": dict(TUNED_LOSS), "cost_weights": WEIGHTS,
           "m_S": 0.05, "ceilings": CEILINGS, "nominal_costs": NOMINAL,
           "producers": PRODUCERS, "authors": AUTHORS, "oracle": {"cmd": rcfg["oracle_cmd"], "battery": "oracle/r"},
           "excluded_tasks": [], "min_tasks": 234, "min_tasks_per_repo": 78,
           "tool_digests": {p: "sha256:" + "ab" * 32 for p in PRODUCERS}}
    rec.update(fields)
    return _write(tmp / "freeze.json", rec)


def _findings(rec: Path):
    fz = load_freeze(rec)
    return validate(fz, fz.config_paths())


def _texts(rec: Path):
    """Findings without their '<file>: ' prefix, which carries the test's tmp path."""
    return [f.split(": ", 1)[1] for f in _findings(rec)]


def _manifest(out: Path, parts=("train",)) -> None:
    """One safe/unsafe pair per entry of `parts`; task k<i> lies in parts[i]."""
    rows = [{"candidate_id": f"k{i:02d}:{v}:m0", "task": f"k{i:02d}", "repo": "r", "mode": "m0",
             "safe_twin": f"k{i:02d}:{'u' if v == 's' else 's'}:m0", "partition": p, "commit": "c0ffee",
             "diff_path": f"diffs/k{i:02d}:{v}:m0.diff", "risk_features": {}, "ood": False, "seed": None}
            for i, p in enumerate(parts) for v in ("s", "u")]
    out.mkdir(parents=True, exist_ok=True)
    write_records(out / "manifest.jsonl", rows)


def _corpus(out: Path, n_tasks: int = 12, partitions=("train", "tune", "test")) -> None:
    """Synthetic records, manifest and labels; task k<i> lies in partitions[i % len(partitions)]."""
    profiles = default_action_profiles(0.71)
    rng = np.random.default_rng(3)
    recs, man, lab = [], [], []
    for t in range(n_tasks):
        task, part, repo = f"k{t:02d}", partitions[t % len(partitions)], f"r{t % 2}"
        lg = rng.normal(-1.0, 1.0)
        for unsafe in (False, True):
            cid = f"{task}:{'u' if unsafe else 's'}:m0"
            man.append({"candidate_id": cid, "task": task, "repo": repo, "mode": "m0",
                        "safe_twin": f"{task}:{'s' if unsafe else 'u'}:m0", "partition": part, "commit": "c",
                        "diff_path": f"diffs/{cid}.diff", "risk_features": {"risk_logit": lg + (0.8 if unsafe else 0.0)}})
            lab.append({"candidate_id": cid, "unsafe": unsafe, "severity": 0.5 if unsafe else 0.0})
            m = synthesize_matrix(cid, 0, repo, task, unsafe, 0.5, lg, profiles)
            for p in profiles:
                r = m.records[p.action.id]
                recs.append({"candidate_id": cid, "rep": 0, "action_id": p.action.id, "family": p.action.family,
                             "source": p.action.source.value, "producer": p.action.producer,
                             "outcome": r.outcome.value, "status": "ok",
                             "cost": {"tokens": r.cost.tokens, "tool_calls": r.cost.tool_calls,
                                      "ci_equiv": r.cost.ci_equiv, "wall_s": r.cost.wall_s}})
    out.mkdir(parents=True, exist_ok=True)
    write_records(out / "records.jsonl", recs)
    write_records(out / "manifest.jsonl", man)
    write_records(out / "labels.jsonl", lab)


def _tasks(n: int = 20):
    return [Task(f"r-{i}", "r", "o/r", i, f"issue {i}", "", 100 + i, f"m{i}", f"b{i}", "2026-01-01T00:00:00Z",
                 ["src/pkg/mod.py"], 10) for i in range(n)]


def _labels(tmp: Path, *args: str) -> int:
    return build.main(["labels", "--config", str(tmp / "repo.json"), *args])


def _severity(tmp: Path) -> str:
    return str(_write(tmp / "severity.json", {"k00:u:m0": 0.7}))


def _evidence(tmp: Path, *args: str) -> int:
    return evrun.main(["--config", str(tmp / "repo.json"), "--evidence", str(tmp / "evidence.json"), *args])


class _Tree:
    """A checkout that applies every diff, for evidence runs without a repository."""
    def __init__(self, repo):
        pass

    def reset(self, commit):
        pass

    def apply(self, diff):
        return True


@pytest.fixture
def probe(monkeypatch):
    """Evidence runs for real, minus the checkout: one probe action that passes, on a tree
    that applies every diff, so a run that gets through the firewall writes records."""
    reg = Registry()
    reg.register(Action("x.probe", "F1", "author_side", "probe", lambda ctx, m: Outcome("pass")))
    monkeypatch.setattr(evrun, "Checkout", _Tree)
    monkeypatch.setattr(evrun, "build_registry", lambda ecfg, base_dir=None, **budget: reg)


def _cells(out: Path):
    """The (candidate, rep) cells the probe wrote to records.jsonl; [] when there are none."""
    p = out / "records.jsonl"
    rows = [json.loads(line) for line in open(p)] if p.exists() else []
    return sorted((r["candidate_id"], r["rep"]) for r in rows if r["action_id"] == "x.probe")


K00 = [("k00:s:m0", 0), ("k00:u:m0", 0)]               # the cells of the first fixture pair, one repetition


def _suite_repo(tmp: Path) -> Path:
    """A git checkout with a one-test suite, for commands that run F1's command for real."""
    r = tmp / "suite"
    (r / "pkg").mkdir(parents=True)
    (r / "tests").mkdir()
    (r / "pkg/__init__.py").write_text("")
    (r / "pkg/mod.py").write_text("def add(a, b):\n    return a + b\n")
    (r / "tests/test_mod.py").write_text("from pkg.mod import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n")
    (r / "pytest.ini").write_text("[pytest]\npythonpath = .\n")
    for args in (["init", "-q"], ["config", "user.email", "t@t"], ["config", "user.name", "t"],
                 ["add", "-A"], ["commit", "-qm", "base"]):
        subprocess.run(["git", *args], cwd=r, check=True, capture_output=True)
    return r


# ---------------------------------------------------------------- the record itself

def test_clean_record_validates_at_every_stage(tmp_path):
    """The fixture is clean, so each rejection below comes from the one thing it changes."""
    for stage in ("dev", "pre-freeze", "frozen", "unblinded"):
        assert _findings(_record(tmp_path / stage, stage)) == [], stage


def test_load_rejects_an_incomplete_or_malformed_record(tmp_path):
    raw = json.loads(_record(tmp_path, "pre-freeze").read_text())
    bad = {"missing": {k: v for k, v in raw.items() if k != "excluded_tasks"},
           "typo": dict(raw, m_s=0.05),                  # a misspelt field must not pass as an extra one
           "stage": dict(raw, stage="frozen-ish"),
           "signed": dict(raw, signed="false")}          # a string would read as true
    for name, rec in bad.items():
        with pytest.raises(FreezeError):
            load_freeze(_write(tmp_path / f"{name}.json", rec))


def test_require_stage_orders_the_stages(tmp_path):
    fz = load_freeze(_record(tmp_path, "frozen"))
    require_stage(fz, "pre-freeze")
    require_stage(fz, "frozen")
    with pytest.raises(FreezeError):
        require_stage(fz, "unblinded")
    with pytest.raises(FreezeError):
        require_stage(fz, "sealed")                      # not a stage
    with pytest.raises(FreezeError):
        require_stage(load_freeze(_record(tmp_path / "d", "dev")), "pre-freeze")


def test_rejects_unfrozen_salt(tmp_path, monkeypatch):
    for i, salt in enumerate((None, "UNFROZEN-replace-at-freeze", "FREEZE-2026-10-xx")):
        texts = _texts(_record(tmp_path / str(i), "pre-freeze", salt=salt))
        assert len(texts) == 1 and "salt" in texts[0], texts
    assert _findings(_record(tmp_path / "dev", "dev", salt=None)) == []      # stage dev has no salt yet
    # a run at pre-freeze without a drawn salt selects and partitions nothing
    monkeypatch.setattr(build, "select_tasks", lambda *a, **k: _tasks())
    for name, salt in (("none", None), ("drawn", SALT)):
        # twenty fixture tasks: the census threshold is not what this test checks
        rec = _record(tmp_path / name, "pre-freeze", salt=salt, min_tasks_per_repo=20)
        rc = build.main(["select", "--config", str(tmp_path / name / "repo.json"), "--freeze", str(rec)])
        assert (rc, (tmp_path / name / "corpus/tasks.jsonl").exists()) == ((2, False) if salt is None else (0, True))


def test_init_draws_the_salt_once(tmp_path):
    rec = _record(tmp_path, "dev", created=None)
    assert freeze.main(["init", "--freeze", str(rec)]) == 0
    first = json.loads(rec.read_text())
    assert first["stage"] == "pre-freeze" and re.fullmatch(r"[0-9a-f]{32}", first["salt"]) and first["created"]
    assert _findings(rec) == []
    assert freeze.main(["init", "--freeze", str(rec)]) == 2                  # never redrawn
    assert json.loads(rec.read_text()) == first
    other = _record(tmp_path / "other", "dev")
    assert freeze.main(["init", "--freeze", str(other)]) == 0
    assert json.loads(other.read_text())["salt"] != first["salt"]           # drawn, not a constant
    salted = _record(tmp_path / "salted", "dev", salt=SALT)                 # a salt is never replaced,
    assert freeze.main(["init", "--freeze", str(salted)]) == 2              # whatever the stage says
    assert json.loads(salted.read_text())["salt"] == SALT


def test_rejects_tbd_model(tmp_path):
    authors = [dict(AUTHORS[0], model="TBD-by-freeze"), dict(AUTHORS[1], version="TBD")]
    texts = _texts(_record(tmp_path / "cfg", "pre-freeze", repo={"authors": authors}))
    assert len(texts) == 2 and all("TBD" in t for t in texts), texts
    assert any("m0" in t for t in texts) and any("m1" in t for t in texts)
    # the record's own author block is held to the same rule
    texts = _texts(_record(tmp_path / "rec", "frozen", authors=[dict(AUTHORS[0], model="TBD-by-freeze")]))
    assert len(texts) == 1 and "TBD" in texts[0], texts


def test_rejects_null_oracle_without_dev(tmp_path):
    rec = _record(tmp_path / "a", "pre-freeze", repo={"oracle_cmd": None})
    texts = _texts(rec)
    assert len(texts) == 1 and "oracle_cmd" in texts[0], texts
    _manifest(tmp_path / "a/corpus")
    assert _labels(tmp_path / "a", "--freeze", str(rec), "--severity-file", _severity(tmp_path / "a")) == 2
    assert not (tmp_path / "a/corpus/labels.jsonl").exists()
    ok = _record(tmp_path / "ok", "pre-freeze")                              # the same run with an oracle
    _manifest(tmp_path / "ok/corpus")
    assert _labels(tmp_path / "ok", "--freeze", str(ok), "--severity-file", _severity(tmp_path / "ok")) == 0
    # stage dev reports the same gap and refuses nothing
    rec = _record(tmp_path / "b", "dev", out="corpus-dev", repo={"oracle_cmd": None})
    texts = _texts(rec)
    assert len(texts) == 1 and "oracle_cmd" in texts[0], texts
    _manifest(tmp_path / "b/corpus-dev")
    assert _labels(tmp_path / "b", "--freeze", str(rec), "--severity-file", _severity(tmp_path / "b")) == 0


def test_rejects_gold_patch_dev(tmp_path):
    texts = _texts(_record(tmp_path, "pre-freeze", repo={"gold_patch_dev": True}))
    assert len(texts) == 1 and "gold_patch_dev" in texts[0], texts


@pytest.mark.parametrize("change,key", [
    ({"history_fallback": True}, "history_fallback"),     # issue linkage unverified
    ({"cwe_catalog": None}, "cwe_catalog"),               # mutation-only seeding
    ({"cwe_catalog": "nowhere/cwe_r.json"}, "cwe_catalog"),
    ({"cwe_share": 0.4}, "cwe_share"),                    # disagrees with the frozen knob
    ({"author_prompt": None}, "author_prompt"),           # both modes read one frozen prompt file
    ({"author_prompt": "nowhere/author_prompt.md"}, "author_prompt"),
])
def test_rejects_unfrozen_repository_settings(tmp_path, change, key):
    texts = _texts(_record(tmp_path, "pre-freeze", repo=change))
    assert len(texts) == 1 and key in texts[0], texts


@pytest.mark.parametrize("battery,n_findings", [
    # the F1 suite under the battery's own flags, through the checkout's interpreter
    ([".venv/bin/python", "-m", "pytest", "tests", "-q", "-x", "--no-header", "-p", "no:cacheprovider",
      "--deselect", "tests/test_a.py::test_flaky"], 1),
    # F1's exact form: under coverage, a ./ spelling of the target, = syntax
    (["python", "-m", "coverage", "run", "--source=pkg", "-m", "pytest", "./tests/", "-rfE",
      "--deselect=tests/test_a.py::test_flaky"], 1),
    # the suite plus a hidden battery is more than standard CI
    ([".venv/bin/python", "-m", "pytest", "tests", "../oracle/r", "--deselect", "tests/test_a.py::test_flaky"], 0),
])
def test_rejects_battery_equal_to_f1_command(tmp_path, battery, n_findings):
    texts = _texts(_record(tmp_path, "pre-freeze", repo={"battery_cmd": battery}))
    assert len(texts) == n_findings and all("battery_cmd" in t for t in texts), texts


def test_tuned_fields_pending_at_frozen_required_at_unblinded(tmp_path, probe):
    rec = _record(tmp_path / "f", "frozen", **UNTUNED)
    assert len(_texts(rec)) == 5                        # m_S, ceilings, nominal_costs, loss.tau_qual, loss.eta
    _manifest(tmp_path / "f/corpus", parts=("tune",))
    assert _evidence(tmp_path / "f", "--freeze", str(rec), "--partition", "tune", "--reps", "1") == 0
    assert _cells(tmp_path / "f/corpus") == K00          # tuning evidence runs: this stage fills them
    rec = _record(tmp_path / "u", "unblinded", **UNTUNED)                   # signed, but never tuned
    assert len(_texts(rec)) == 5
    _manifest(tmp_path / "u/corpus", parts=("test",))
    assert _evidence(tmp_path / "u", "--freeze", str(rec), "--partition", "test", "--reps", "1") == 2
    assert _cells(tmp_path / "u/corpus") == []
    rec = _record(tmp_path / "t", "unblinded")                              # the same run, tuned
    _manifest(tmp_path / "t/corpus", parts=("test",))
    assert _evidence(tmp_path / "t", "--freeze", str(rec), "--partition", "test", "--reps", "1") == 0
    assert _cells(tmp_path / "t/corpus") == K00


def test_malformed_ceilings_are_one_finding(tmp_path, probe):
    """Stage frozen pastes analysis/ceilings.json into the record (Task B5), so freeze.ceilings has its
    shape exactly: q30, q60 and q90, each with tokens, tool_calls and ci_equiv as positive numbers.
    Wall-clock is an outcome, not a ceiling (results.md E2). Anything else is one finding, and unlike
    ceilings not yet set it is not pending at stage frozen: it binds."""
    q60 = CEILINGS["q60"]
    malformed = {"missing quantile": {q: v for q, v in CEILINGS.items() if q != "q90"},
                 "extra quantile": dict(CEILINGS, q50=q60),
                 "missing key": dict(CEILINGS, q30={k: v for k, v in CEILINGS["q30"].items() if k != "ci_equiv"}),
                 "wall-clock": dict(CEILINGS, q60=dict(q60, wall_s=6.1)),
                 "not a number": dict(CEILINGS, q60=dict(q60, tokens="4000")),
                 "zero": dict(CEILINGS, q60=dict(q60, ci_equiv=0)),
                 "not an object": dict(CEILINGS, q90=[6000.0, 16.0, 5.0]),
                 "empty": {},
                 "all at once": {"q60": dict(q60, wall_s=6.1)}}
    for name, ceilings in malformed.items():
        for stage in ("frozen", "unblinded"):
            texts = _texts(_record(tmp_path / name / stage, stage, ceilings=ceilings))
            assert len(texts) == 1 and "ceilings" in texts[0], (name, stage, texts)
    assert all(s in texts[0] for s in ("q30", "q90", "wall_s")), texts     # the one finding names each problem
    rec = _record(tmp_path / "bind", "frozen", ceilings=malformed["wall-clock"])
    _manifest(tmp_path / "bind/corpus", parts=("tune",))
    assert _evidence(tmp_path / "bind", "--freeze", str(rec), "--partition", "tune", "--reps", "1") == 2
    assert _cells(tmp_path / "bind/corpus") == []
    rec = _record(tmp_path / "ok", "frozen")                                # the same run, well-formed
    _manifest(tmp_path / "ok/corpus", parts=("tune",))
    assert _evidence(tmp_path / "ok", "--freeze", str(rec), "--partition", "tune", "--reps", "1") == 0
    assert _cells(tmp_path / "ok/corpus") == K00


def test_malformed_nominal_costs_are_one_finding(tmp_path, probe):
    """Stage frozen pastes analysis/nominal_costs.json into the record (Task B5): one object per action with
    its four metered dimensions, tokens, tool_calls, ci_equiv and wall_s, as non-negative numbers (a
    producer that calls no model costs zero tokens). Anything else is one finding naming every problem, and
    unlike nominal costs not yet set, which are pending at stage frozen, it binds."""
    ci = NOMINAL["f1.repo_ci"]
    swap = lambda c: dict(NOMINAL, **{"f1.repo_ci": c})
    with pytest.raises(FreezeError, match="nominal_costs"):                 # not an object: no record at all
        load_freeze(_record(tmp_path / "list", "frozen", nominal_costs=[ci]))
    malformed = {"empty": {},
                 "action not an object": swap([0.0, 1.0, 1.0, 60.0]),
                 "missing dimension": swap({k: v for k, v in ci.items() if k != "wall_s"}),
                 "extra dimension": swap(dict(ci, cpu_s=2.0)),
                 "negative": swap(dict(ci, tokens=-1.0)),
                 "not a number": swap(dict(ci, ci_equiv="1.0")),
                 "a boolean": swap(dict(ci, tool_calls=True)),
                 "not finite": swap(dict(ci, wall_s=float("inf"))),
                 "all at once": {"f1.repo_ci": dict(ci, cpu_s=2.0, tokens=-1.0), "ch.fuzz": {"tokens": 0.0}}}
    for name, table in malformed.items():
        for stage in ("frozen", "unblinded"):
            texts = _texts(_record(tmp_path / name / stage, stage, nominal_costs=table))
            assert len(texts) == 1 and "nominal_costs" in texts[0], (name, stage, texts)
    assert all(s in texts[0] for s in ("cpu_s", "tokens", "ch.fuzz", "wall_s")), texts   # names each problem
    zero = swap(dict(ci, tokens=0.0, tool_calls=0.0, ci_equiv=0.0, wall_s=0.0))
    assert _findings(_record(tmp_path / "zero", "frozen", nominal_costs=zero)) == []
    rec = _record(tmp_path / "bind", "frozen", nominal_costs=malformed["negative"])
    _manifest(tmp_path / "bind/corpus", parts=("tune",))
    assert _evidence(tmp_path / "bind", "--freeze", str(rec), "--partition", "tune", "--reps", "1") == 2
    assert _cells(tmp_path / "bind/corpus") == []


def test_malformed_cost_weights_are_one_finding(tmp_path, probe):
    """cost_weights are the w_j of C_w over tokens, tool calls and CI-equivalents, a protocol constant
    fixed when the record is written: three positive numbers summing to 1. Anything else is one finding,
    reported at dev and binding from pre-freeze on (no later stage fills them), and null is no record."""
    malformed = {"two": [0.5, 0.5], "four": [0.25] * 4, "sum": [0.5, 0.3, 0.3], "zero": [1.0, 0.0, 0.0],
                 "negative": [1.2, -0.1, -0.1], "not a number": ["1/3", 1 / 3, 1 / 3],
                 "a boolean": [True, 0.0, 0.0], "not finite": [float("nan"), 0.5, 0.5], "empty": []}
    for name, w in malformed.items():
        for stage in ("dev", "pre-freeze", "frozen", "unblinded"):
            texts = _texts(_record(tmp_path / name / stage, stage, cost_weights=w))
            assert len(texts) == 1 and "cost_weights" in texts[0], (name, stage, texts)
    for i, ok in enumerate(([0.5, 0.25, 0.25], [0.2, 0.2, 0.6], [1 / 3] * 3)):     # a sum within 1e-9 of 1
        assert _findings(_record(tmp_path / f"ok{i}", "pre-freeze", cost_weights=ok)) == [], ok
    rec = _record(tmp_path / "bind", "pre-freeze", cost_weights=malformed["sum"])
    _manifest(tmp_path / "bind/corpus")
    assert _evidence(tmp_path / "bind", "--freeze", str(rec), "--partition", "train", "--reps", "1") == 2
    assert _cells(tmp_path / "bind/corpus") == []
    with pytest.raises(FreezeError, match="cost_weights"):
        load_freeze(_record(tmp_path / "null", "pre-freeze", cost_weights=None))


def test_frozen_requires_tool_digests_authors_and_oracle(tmp_path):
    texts = _texts(_record(tmp_path / "a", "frozen", tool_digests={"f1.repo_ci": "sha256:" + "ab" * 32}))
    assert len(texts) == 1 and "f2.static_security" in texts[0] and "f1.repo_ci" not in texts[0], texts
    texts = _texts(_record(tmp_path / "b", "frozen", authors=None, oracle=None))
    assert len(texts) == 2, texts
    assert _findings(_record(tmp_path / "c", "pre-freeze", tool_digests={}, authors=None, oracle=None)) == []


def test_unblinded_requires_signature(tmp_path):
    texts = _texts(_record(tmp_path, "unblinded", signed=False))
    assert len(texts) == 1 and "signed" in texts[0], texts


def test_seal_pins_the_record_and_its_configs(tmp_path):
    rec = _record(tmp_path, "frozen")
    assert freeze.main(["seal", "--freeze", str(rec)]) == 0
    assert re.fullmatch(r"[0-9a-f]{16}", load_freeze(rec).freeze_id) and _findings(rec) == []
    raw = json.loads(rec.read_text())                   # tuning, fitting and unblinding do not break the seal
    refit = dict(NOMINAL, **{"f1.repo_ci": dict(NOMINAL["f1.repo_ci"], ci_equiv=0.97)})
    _write(rec, dict(raw, m_S=0.1, loss=dict(TUNED_LOSS, eta=2.0), nominal_costs=refit, stage="unblinded",
                     signed=True))
    assert _findings(rec) == []
    unblinded = json.loads(rec.read_text())
    _write(rec, dict(unblinded, cost_weights=[0.5, 0.25, 0.25]))           # a protocol constant is sealed
    texts = _texts(rec)
    assert len(texts) == 1 and "freeze_id" in texts[0], texts
    _write(rec, unblinded)
    cfg = json.loads((tmp_path / "repo.json").read_text())
    _write(tmp_path / "repo.json", dict(cfg, held_out_families=["signing"]))   # a frozen config edited
    texts = _texts(rec)
    assert len(texts) == 1 and "freeze_id" in texts[0], texts
    assert freeze.main(["seal", "--freeze", str(_record(tmp_path / "d", "dev"))]) == 2


def test_seal_pins_the_author_prompt(tmp_path):
    """Both authoring modes read one frozen prompt file; editing it after the seal is drift."""
    rec = _record(tmp_path, "pre-freeze")
    assert freeze.main(["seal", "--freeze", str(rec)]) == 0 and _findings(rec) == []
    (tmp_path / "author_prompt.md").write_text("A task.json holds the issue. Fix it, and add a changelog entry.\n")
    texts = _texts(rec)
    assert len(texts) == 1 and "freeze_id" in texts[0], texts


def test_author_refuses_a_cli_at_another_version(tmp_path, capsys):
    """Under a record each authoring CLI must answer --version with the version its config pins;
    a development run records the difference and goes on."""
    fake = tmp_path / "bin" / "agent-a"
    fake.parent.mkdir(parents=True)
    fake.write_text("#!/bin/sh\necho '2.2.0 (Agent A)'\n")
    fake.chmod(0o755)
    authors = [dict(AUTHORS[0], cmd=[str(fake), "-p", "{prompt}"], version="2.1.0 (Agent A)")]
    rec = _record(tmp_path, "pre-freeze", repo={"authors": authors}, authors=authors)
    (tmp_path / "corpus").mkdir()
    write_records(tmp_path / "corpus/tasks.jsonl", [])
    args = ["author", "--config", str(tmp_path / "repo.json"), "--freeze", str(rec), "--partition", "train"]
    assert build.main(args) == 2
    err = capsys.readouterr().err
    assert "2.2.0 (Agent A)" in err and "2.1.0 (Agent A)" in err
    _record(tmp_path / "d", "dev", out="corpus-dev", repo={"authors": authors})
    (tmp_path / "d/corpus-dev").mkdir()
    write_records(tmp_path / "d/corpus-dev/tasks.jsonl", [])
    assert build.main(["author", "--config", str(tmp_path / "d/repo.json"), "--dev"]) == 0


def test_shipped_record_is_initialised_and_validates_the_real_configs(monkeypatch):
    """The shipped record was initialised on 2026-10-21 (rcca.freeze init, commit 8e810b7):
    its salt is drawn and never changes, it is unsigned until unblinded, and it validates
    the real configs rather than a fixture with no finding, because from pre-freeze on a
    finding refuses the confirmatory commands.

    An earlier version asserted stage dev with no salt; that was the invariant before the
    freeze, not after it. Another required `validate` to return something, which inverted
    the incentive. The exit code tracks the findings instead."""
    import re
    monkeypatch.chdir(ROOT)
    fz = load_freeze(Path("configs/freeze.json"))
    assert fz.stage in ("pre-freeze", "frozen", "unblinded"), fz.stage
    assert fz.salt and re.fullmatch(r"[0-9a-f]{32}", fz.salt), fz.salt
    assert fz.signed == (fz.stage == "unblinded")
    paths = fz.config_paths()
    assert any("flask" in str(p) for p in paths), paths      # the real configs, not a fixture
    findings = validate(fz, paths)
    assert findings == [], findings
    assert freeze.main(["validate", "--freeze", "configs/freeze.json"]) == 0


# ---------------------------------------------------------------- the firewall in each command

def test_dev_flag_bypasses_with_dev_suffix(tmp_path, probe):
    dev, plain, unflagged = tmp_path / "flask-dev", tmp_path / "flask", tmp_path / "other-dev"
    for d in (dev, plain, unflagged):
        _corpus(d, n_tasks=8)
    assert analyze.main(["--out", str(dev), "--dev", "--quantiles", "0.6"]) == 0
    assert (dev / "analysis/table_rq1.json").exists()
    assert analyze.main(["--out", str(plain), "--dev", "--quantiles", "0.6"]) == 2      # not a -dev directory
    assert analyze.main(["--out", str(unflagged), "--quantiles", "0.6"]) == 2          # no --dev, no record
    assert not (plain / "analysis").exists() and not (unflagged / "analysis").exists()
    # the corpus builder and the evidence run follow the same rule on the config's out
    _record(tmp_path / "c", "dev", out="corpus-dev")
    _manifest(tmp_path / "c/corpus-dev", parts=("tune",))
    assert _labels(tmp_path / "c", "--dev-constant", "0.5") == 2
    assert _labels(tmp_path / "c", "--dev", "--dev-constant", "0.5") == 0
    assert _evidence(tmp_path / "c", "--partition", "tune", "--reps", "1") == 2
    assert _cells(tmp_path / "c/corpus-dev") == []
    assert _evidence(tmp_path / "c", "--partition", "tune", "--reps", "1", "--dev") == 0
    assert _cells(tmp_path / "c/corpus-dev") == K00
    rec = _record(tmp_path / "p", "dev", out="corpus")
    _manifest(tmp_path / "p/corpus", parts=("tune",))
    assert _labels(tmp_path / "p", "--dev", "--dev-constant", "0.5") == 2
    assert _labels(tmp_path / "p", "--freeze", str(rec), "--dev-constant", "0.5") == 2   # a dev-stage record too
    assert _evidence(tmp_path / "p", "--partition", "tune", "--reps", "1", "--dev") == 2
    assert not (tmp_path / "p/corpus/labels.jsonl").exists() and _cells(tmp_path / "p/corpus") == []


def test_dev_switches_are_refused_under_a_frozen_record(tmp_path):
    rec = _record(tmp_path, "frozen", out="corpus-dev")
    _manifest(tmp_path / "corpus-dev")
    assert _labels(tmp_path, "--freeze", str(rec), "--dev", "--severity-file", _severity(tmp_path)) == 2
    assert _labels(tmp_path, "--freeze", str(rec), "--dev-constant", "0.5") == 2      # unadjudicated severity
    assert not (tmp_path / "corpus-dev/labels.jsonl").exists()
    assert _labels(tmp_path, "--freeze", str(rec), "--severity-file", _severity(tmp_path)) == 0


def test_command_configs_must_be_listed_in_the_record(tmp_path, probe):
    rec = _record(tmp_path, "frozen")
    _manifest(tmp_path / "corpus", parts=("tune",))
    other = _write(tmp_path / "other.json", json.loads((tmp_path / "repo.json").read_text()))
    rest = ["--evidence", str(tmp_path / "evidence.json"), "--freeze", str(rec), "--partition", "tune", "--reps", "1"]
    assert evrun.main(["--config", str(other), *rest]) == 2
    assert _cells(tmp_path / "corpus") == []
    assert evrun.main(["--config", str(tmp_path / "repo.json"), *rest]) == 0
    assert _cells(tmp_path / "corpus") == K00


def test_prefreeze_refuses_evidence_on_tune_and_test(tmp_path, capsys, probe):
    rec = _record(tmp_path, "pre-freeze")
    _manifest(tmp_path / "corpus", parts=("train", "tune", "test"))
    for part in (["--partition", "tune"], ["--partition", "test"], []):     # no partition means all of them
        assert _evidence(tmp_path, "--freeze", str(rec), "--reps", "1", *part) == 2
    err = capsys.readouterr().err
    assert "'tune'" in err and "'test'" in err
    assert _cells(tmp_path / "corpus") == []
    assert _evidence(tmp_path, "--freeze", str(rec), "--partition", "train", "--reps", "1") == 0
    assert _cells(tmp_path / "corpus") == K00                                  # training evidence, k00 only


def test_frozen_opens_tuning_evidence_not_test(tmp_path, probe):
    rec = _record(tmp_path, "frozen")
    _manifest(tmp_path / "corpus", parts=("tune", "test"))
    assert _evidence(tmp_path, "--freeze", str(rec), "--partition", "test", "--reps", "1") == 2
    assert _cells(tmp_path / "corpus") == []
    assert _evidence(tmp_path, "--freeze", str(rec), "--partition", "tune", "--reps", "1") == 0
    assert _cells(tmp_path / "corpus") == K00                                  # k00 is the tuning pair


def test_rejects_excluded_task_in_manifest(tmp_path, capsys, probe):
    out = tmp_path / "corpus"
    _corpus(out, partitions=("train", "tune"))
    rec = _record(tmp_path, "frozen", excluded_tasks=["k03"])
    with pytest.raises(FreezeError, match="k03"):
        analyze.run(out, dict(analyze.DEV_LOSS), "train", "tune", [0.6], B_boot=50, freeze=load_freeze(rec))
    assert not (out / "analysis").exists()
    assert _evidence(tmp_path, "--freeze", str(rec), "--partition", "train", "--reps", "1") == 2
    assert "k03" in capsys.readouterr().err and _cells(out) == []
    clean = _record(tmp_path / "clean", "frozen", repo={"out": str(out)})
    assert analyze.run(out, dict(analyze.DEV_LOSS), "train", "tune", [0.6], B_boot=50,
                       freeze=load_freeze(clean))["n_tasks"] == 6
    assert _evidence(tmp_path / "clean", "--freeze", str(clean), "--partition", "train", "--reps", "1") == 0
    assert len(_cells(out)) == 12                                              # six training pairs


def test_fit_rejects_test_partition(tmp_path):
    out = tmp_path / "corpus"
    _corpus(out, partitions=("train", "tune"))
    fz = load_freeze(_record(tmp_path, "frozen"))
    loss = dict(analyze.DEV_LOSS)
    with pytest.raises(FreezeError, match="'test'"):
        analyze.run(out, loss, "train", "test", [0.6], B_boot=50, freeze=fz)
    with pytest.raises(FreezeError, match="'tune'"):
        analyze.run(out, loss, "tune", "tune", [0.6], B_boot=50, freeze=fz)       # fitting on tuning
    with pytest.raises(FreezeError):
        analyze.run(out, loss, None, None, [0.6], B_boot=50, freeze=fz)           # the dev fallback to all matrices
    assert not (out / "analysis").exists()
    assert analyze.run(out, loss, "train", "tune", [0.6], B_boot=50, freeze=fz)["n_tasks"] == 6


def test_frozen_refuses_test_matrices_on_disk(tmp_path):
    out = tmp_path / "corpus"
    _corpus(out)                                         # test-partition records already exist
    fz = load_freeze(_record(tmp_path, "frozen"))
    with pytest.raises(FreezeError, match="'test'"):
        analyze.run(out, dict(analyze.DEV_LOSS), "train", "tune", [0.6], B_boot=50, freeze=fz)
    fz = load_freeze(_record(tmp_path / "u", "unblinded"))
    assert analyze.run(out, dict(analyze.DEV_LOSS), "train", "test", [0.6], B_boot=50, freeze=fz)["n_tasks"] == 4


def test_analysis_needs_stage_frozen(tmp_path):
    out = tmp_path / "corpus"
    _corpus(out, partitions=("train",))
    rec = _record(tmp_path, "pre-freeze")
    assert analyze.main(["--out", str(out), "--freeze", str(rec), "--eval-partition", "train"]) == 2
    with pytest.raises(FreezeError):                     # the gate itself, not only run()'s own check
        freeze.gate(str(rec), dev=False, out=out, min_stage="frozen", partitions=["train"])
    assert not (out / "analysis").exists()


def test_empty_fit_partition_never_falls_back_to_all_matrices(tmp_path):
    out = tmp_path / "corpus"
    _corpus(out, partitions=("tune",))                   # no training matrices at all
    with pytest.raises(FreezeError, match="'train'"):
        analyze.run(out, dict(analyze.DEV_LOSS), "train", "tune", [0.6], B_boot=50,
                    freeze=load_freeze(_record(tmp_path, "frozen")))


class _Stop(Exception):
    pass


def _capture_run(monkeypatch):
    seen = {}

    def fake_run(out, loss, fit, ev, quantiles, **kw):
        seen.update(loss=loss, m_S=kw["m_S"], fit=fit, ev=ev)
        raise _Stop
    monkeypatch.setattr(analyze, "run", fake_run)
    return seen


def test_unblinded_run_takes_loss_and_margin_from_the_record(tmp_path, monkeypatch):
    seen = _capture_run(monkeypatch)
    rec = _record(tmp_path, "unblinded", loss=dict(TUNED_LOSS, tau_qual=0.05, eta=2.0), m_S=0.1)
    args = ["--out", str(tmp_path / "corpus"), "--freeze", str(rec)]
    with pytest.raises(_Stop):
        analyze.main(args)
    assert (seen["loss"]["tau_qual"], seen["loss"]["eta"], seen["loss"]["kappa"], seen["m_S"]) == (0.05, 2.0, 100, 0.1)
    assert (seen["fit"], seen["ev"]) == ("train", "test")
    dev_loss = _write(tmp_path / "loss.json", {"kappa": 100, "lambda_R": 10, "lambda_E": 2, "tau_qual": 0.02, "eta": 1.0})
    assert analyze.main(args + ["--loss", str(dev_loss)]) == 2           # no flag overrides a frozen constant
    assert analyze.main(args + ["--m-S", "0.05"]) == 2


def test_tuning_run_supplies_only_tau_and_eta(tmp_path, monkeypatch):
    seen = _capture_run(monkeypatch)
    rec = _record(tmp_path, "frozen", **UNTUNED)
    args = ["--out", str(tmp_path / "corpus"), "--freeze", str(rec), "--eval-partition", "tune"]
    grid = _write(tmp_path / "grid.json", {"tau_qual": 0.05, "eta": 4.0})
    with pytest.raises(_Stop):
        analyze.main(args + ["--loss", str(grid)])
    assert (seen["loss"]["tau_qual"], seen["loss"]["eta"], seen["loss"]["kappa"]) == (0.05, 4.0, 100)
    assert analyze.main(args) == 2                                       # tau_qual and eta are not tuned yet
    kappa = _write(tmp_path / "kappa.json", {"kappa": 50, "tau_qual": 0.05, "eta": 4.0})
    assert analyze.main(args + ["--loss", str(kappa)]) == 2              # kappa is fixed by the record


def test_partitions_use_freeze_salt(tmp_path, monkeypatch):
    monkeypatch.setattr(build, "select_tasks", lambda *a, **k: _tasks())
    by_record = {t.task: t.partition for t in assign_partitions(_tasks(), SALT)}
    by_config = {t.task: t.partition for t in assign_partitions(_tasks(), "config-salt")}
    assert by_record != by_config                        # the two salts are told apart on these tasks
    # its config still carries freeze_salt "config-salt"; twenty fixture tasks, so a threshold of twenty
    rec = _record(tmp_path, "pre-freeze", min_tasks_per_repo=20)
    assert build.main(["select", "--config", str(tmp_path / "repo.json"), "--freeze", str(rec)]) == 0
    rows = [json.loads(line) for line in open(tmp_path / "corpus/tasks.jsonl")]
    assert {r["task"]: r["partition"] for r in rows} == by_record
    assert {r["salt_source"] for r in rows} == {"freeze-record"}
    census = json.loads((tmp_path / "corpus/census.json").read_text())
    assert (census["eligible"], census["threshold"], census["shortfall"]) == (20, 20, 0)
    # a development run falls back to the config's salt and says so
    _record(tmp_path / "d", "dev", out="corpus-dev")
    assert build.main(["select", "--config", str(tmp_path / "d/repo.json"), "--dev"]) == 0
    rows = [json.loads(line) for line in open(tmp_path / "d/corpus-dev/tasks.jsonl")]
    assert {r["task"]: r["partition"] for r in rows} == by_config
    assert {r["salt_source"] for r in rows} == {"repo-config (dev)"}


def test_pairs_seed_with_the_freeze_salt(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(build, "cmd_pairs", lambda cfg, limit=0, partition=None, resume=False, salt=None:
                        seen.update(salt=salt))
    rec = _record(tmp_path, "frozen")
    assert build.main(["pairs", "--config", str(tmp_path / "repo.json"), "--freeze", str(rec),
                       "--partition", "train"]) == 0
    assert seen["salt"] == SALT


def test_evidence_cell_seeds_use_the_freeze_salt(tmp_path, probe):
    rec = _record(tmp_path, "pre-freeze")
    _manifest(tmp_path / "corpus")
    assert _evidence(tmp_path, "--freeze", str(rec), "--partition", "train", "--reps", "1") == 0
    seeds = {r["candidate_id"]: r["seed"] for r in map(json.loads, open(tmp_path / "corpus/records.jsonl"))}
    assert seeds == {c: cell_seed(SALT, c, 0, "x.probe") for c in ("k00:s:m0", "k00:u:m0")}


def test_evidence_run_needs_the_ci_baseline_under_a_record(tmp_path, capsys, probe):
    """Every record's ci_equiv, and the challenger's search budget in seconds, derive from the
    standard-CI baseline (evidence.baseline). A run under a freeze record refuses without one rather
    than meter compute as zero; a development run without one runs as before, and a baseline that a
    config names but nobody measured is refused in either."""
    for name, path in (("unset", None), ("missing", str(tmp_path / "nowhere.json"))):
        rec = _record(tmp_path / name, "pre-freeze", evidence={"ci_baseline": path})
        _manifest(tmp_path / name / "corpus")
        assert _evidence(tmp_path / name, "--freeze", str(rec), "--partition", "train", "--reps", "1") == 2
        assert "ci_baseline" in capsys.readouterr().err and _cells(tmp_path / name / "corpus") == [], name
    rec = _record(tmp_path / "ok", "pre-freeze")                            # the fixture's baseline exists
    _manifest(tmp_path / "ok/corpus")
    assert _evidence(tmp_path / "ok", "--freeze", str(rec), "--partition", "train", "--reps", "1") == 0
    assert _cells(tmp_path / "ok/corpus") == K00
    for name, path, rc in (("dev-unset", None, 0), ("dev-missing", str(tmp_path / "nowhere.json"), 2)):
        _record(tmp_path / name, "dev", out="corpus-dev", evidence={"ci_baseline": path})
        _manifest(tmp_path / name / "corpus-dev")
        assert _evidence(tmp_path / name, "--dev", "--reps", "1") == rc, name
        assert _cells(tmp_path / name / "corpus-dev") == (K00 if rc == 0 else []), name


def test_evidence_run_refuses_a_challenger_budget_in_seconds_under_a_record(tmp_path, capsys, monkeypatch):
    """The real registry builder, with a challenger that registers no technique (so no hypothesis):
    under a record a budget in wall-clock seconds is refused before anything runs; a development run
    accepts it, and a budget in CI-equivalents runs under the record."""
    monkeypatch.setattr(evrun, "Checkout", _Tree)
    cases = [("record", "pre-freeze", "corpus", {"budget_s": 25}, 2),
             ("dev", "dev", "corpus-dev", {"budget_s": 25}, 0),
             ("ci-equiv", "pre-freeze", "corpus", {"budget_ci_equiv": 12.5}, 0)]
    for name, stage, out, budget, rc in cases:
        rec = _record(tmp_path / name, stage, out=out)
        ecfg = json.loads((tmp_path / name / "evidence.json").read_text())
        del ecfg["f1"]                                       # only the challenger, with no technique
        _write(tmp_path / name / "evidence.json", dict(ecfg, challenger={**budget, "techniques": []}))
        _manifest(tmp_path / name / out)
        flags = ["--dev"] if stage == "dev" else ["--freeze", str(rec), "--partition", "train"]
        assert _evidence(tmp_path / name, *flags, "--reps", "1") == rc, name
        assert ("budget_s" in capsys.readouterr().err) == (rc == 2), name


def test_pairs_never_overwrite_an_earlier_partition(tmp_path, capsys):
    """Under a record pairs runs once per partition as its stage opens it; a later partition
    without --resume would rewrite manifest.jsonl and admission.jsonl with its own rows only."""
    rec = _record(tmp_path, "frozen")
    out = tmp_path / "corpus"
    _manifest(out)                                       # the training pairs, built at pre-freeze
    write_records(out / "admission.jsonl", [{"task": "k00", "mode": "m0", "safe_passes_battery": True}])
    write_records(out / "tasks.jsonl", [dict(_tasks(1)[0].to_dict(), task="k00", partition="train")])
    before = {name: (out / name).read_text() for name in ("manifest.jsonl", "admission.jsonl")}
    args = ["pairs", "--config", str(tmp_path / "repo.json"), "--freeze", str(rec), "--partition", "tune"]
    assert build.main(args) == 2 and "--resume" in capsys.readouterr().err
    assert {name: (out / name).read_text() for name in before} == before
    assert build.main(args + ["--resume"]) == 0          # appends (no tuning task here): the training rows stay
    assert [r["candidate_id"] for r in map(json.loads, open(out / "manifest.jsonl"))] == ["k00:s:m0", "k00:u:m0"]
    assert [r["task"] for r in map(json.loads, open(out / "admission.jsonl"))] == ["k00"]


def test_baseline_needs_a_freeze_record_or_a_dev_directory(tmp_path):
    repo = _suite_repo(tmp_path)

    def args(out_dir: str):
        cfg = _write(tmp_path / f"{out_dir}.repo.json", {"repo": "r", "checkout": str(repo)})
        ecfg = _write(tmp_path / f"{out_dir}.evidence.json", {
            "python": sys.executable, "ci_baseline": str(tmp_path / out_dir / "ci_baseline.json"),
            "f1": {"test_args": ["tests"], "source": "pkg", "extra_deselect": []}})
        return ["--config", str(cfg), "--evidence", str(ecfg), "--n", "1"]
    assert evbaseline.main(args("corpus")) == 2                          # no record, no --dev
    assert evbaseline.main(args("corpus") + ["--dev"]) == 2              # --dev into a directory without -dev
    assert not (tmp_path / "corpus/ci_baseline.json").exists()
    assert evbaseline.main(args("corpus-dev") + ["--dev"]) == 0          # measured and written
    assert json.loads((tmp_path / "corpus-dev/ci_baseline.json").read_text())["n"] == 1


def test_snapshot_needs_a_freeze_record_or_a_dev_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(evsnapshot, "index_names", lambda fetch: (["flask"], 7))   # no network
    files = ("registry_snapshot.json", "advisories.json")
    assert evsnapshot.main(["--out", str(tmp_path / "corpus"), "--advisories", "none"]) == 2
    assert evsnapshot.main(["--out", str(tmp_path / "corpus"), "--advisories", "none", "--dev"]) == 2
    assert not any((tmp_path / "corpus" / f).exists() for f in files)
    assert evsnapshot.main(["--out", str(tmp_path / "corpus-dev"), "--advisories", "none", "--dev"]) == 0
    assert all((tmp_path / "corpus-dev" / f).exists() for f in files)
    rec = _record(tmp_path / "r", "pre-freeze")                                    # a validated record
    assert evsnapshot.main(["--out", str(tmp_path / "corpus"), "--advisories", "none", "--freeze", str(rec)]) == 0
    assert all((tmp_path / "corpus" / f).exists() for f in files)

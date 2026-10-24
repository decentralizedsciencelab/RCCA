import json, subprocess
from pathlib import Path
import pytest
from corpus import mutate
from corpus.cwe import CweDefect, probe_catalog_flask, load_catalog, dump_catalog
from corpus.coverage import CoverageMap
from corpus.seeder import Seeder, SeederConfig
from corpus.manifest import candidate_id, Candidate, PairAdmission, write_manifest, write_labels, read_jsonl
from corpus.tasks import Task, assign_partitions
from corpus.admit import admit_pair, Checkout
from corpus.authoring import GoldPatchAuthor

SRC = '''\
def clamp(x, lo, hi):
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x

def ratio(a, b):
    total = a + b
    if total == 0 and a == 0:
        return 0
    return a / total

SECURE = True
LIMIT = 10
'''


def _git(repo, *a):
    return subprocess.run(["git", *a], cwd=repo, capture_output=True, text=True, check=True)


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"; (r / "src" / "pkg").mkdir(parents=True)
    (r / "src/pkg/__init__.py").write_text("")
    (r / "src/pkg/mod.py").write_text(SRC)
    _git(r, "init", "-q"); _git(r, "config", "user.email", "t@t"); _git(r, "config", "user.name", "t")
    _git(r, "add", "-A"); _git(r, "commit", "-qm", "base")
    base = _git(r, "rev-parse", "HEAD").stdout.strip()
    (r / "src/pkg/mod.py").write_text(SRC.replace("LIMIT = 10", "LIMIT = 20"))
    _git(r, "commit", "-qam", "fix")
    fix = _git(r, "rev-parse", "HEAD").stdout.strip()
    _git(r, "checkout", "-q", base)
    return r, base, fix


def test_mutate_sites_and_each_operator():
    sites = mutate.collect_sites(SRC)
    ops = {s.op for s in sites}
    assert {"ROR", "AOR", "COR", "CRP_bool", "CRP_int", "NEG_if"} <= ops
    for s in sites:
        out = mutate.apply(SRC, s.op, s.line)
        assert out is None or out != mutate.normalize(SRC)
    assert mutate.apply(SRC, "AOR", 999) is None


def test_seeder_is_deterministic_and_stratified(repo):
    r, base, fix = repo
    cov = CoverageMap({"src/pkg/mod.py": {2, 3, 4, 5, 6, 9, 10, 11, 12}})   # clamp and ratio body covered, constants not
    cfg = SeederConfig("salt", cwe_share=0.0, covered_share=1.0, package_root="src/pkg")
    s = Seeder(cfg, [], cov)
    a = s.choose(r, "t1:u:m0", ["src/pkg/mod.py"]); b = s.choose(r, "t1:u:m0", ["src/pkg/mod.py"])
    assert a == b and a.kind == "mutation" and a.covered is True and a.scope == "touched"
    cfg0 = SeederConfig("salt", cwe_share=0.0, covered_share=0.0, package_root="src/pkg")
    c = Seeder(cfg0, [], cov).choose(r, "t1:u:m0", ["src/pkg/mod.py"])
    assert c.covered is False
    assert Seeder(cfg, [], cov).choose(r, "t2:u:m0", ["src/pkg/mod.py"]) != a   # another candidate, another site (almost surely)


def test_seeder_cwe_share_and_held_out(repo):
    r, base, fix = repo
    cat = [CweDefect("X1", "CWE-1", "fam-a", "flip secure flag", "mod.py", "SECURE = True", "SECURE = False"),
           CweDefect("X2", "CWE-2", "fam-b", "held out", "mod.py", "LIMIT = 10", "LIMIT = 0", held_out=True)]
    cfg = SeederConfig("salt", cwe_share=1.0, package_root="src/pkg", held_out_families=("fam-b",))
    s = Seeder(cfg, cat, CoverageMap.empty())
    rec = s.choose(r, "t1:u:m0", ["src/pkg/mod.py"])
    assert rec.kind == "cwe" and rec.cwe_id == "X1" and rec.held_out is False
    assert s.apply(r, rec) and "SECURE = False" in (r / "src/pkg/mod.py").read_text()
    s2 = Seeder(SeederConfig("salt", cwe_share=1.0, package_root="src/pkg", include_held_out=True), cat, CoverageMap.empty())
    assert {d.id for d in s2.catalog} == {"X1", "X2"}


def test_seeder_falls_back_to_mutation_when_no_cwe_applies(repo):
    r, base, fix = repo
    cat = [CweDefect("X9", "CWE-9", "f", "no anchor", "mod.py", "NOT PRESENT", "x")]
    rec = Seeder(SeederConfig("salt", cwe_share=1.0, package_root="src/pkg"), cat, CoverageMap.empty()).choose(r, "t:u:m", ["src/pkg/mod.py"])
    assert rec.kind == "mutation"


def test_manifest_ids_and_labels(tmp_path):
    assert candidate_id("flask-123", True, "m1") == "flask-123:u:m1"
    cs = [Candidate("t:s:m0", "t", "r", "m0", "t:u:m0", "test", "c", "d"),
          Candidate("t:u:m0", "t", "r", "m0", "t:s:m0", "test", "c", "d", seed={"kind": "cwe"})]
    write_manifest(tmp_path / "m.jsonl", cs)
    rows = read_jsonl(tmp_path / "m.jsonl"); assert rows[1]["seed"]["kind"] == "cwe"
    with pytest.raises(KeyError):
        write_labels(tmp_path / "l.jsonl", cs, {})
    write_labels(tmp_path / "l.jsonl", cs, {"t:u:m0": 0.7})
    lab = read_jsonl(tmp_path / "l.jsonl"); assert lab[0]["unsafe"] is False and lab[1]["severity"] == 0.7
    # keyword form: `admitted` is a conjunction over every check, so adding one must
    # not silently shift what a positional argument means
    adm = PairAdmission("t", "m0", safe_passes_battery=True, oracle_catches_unsafe=True,
                        oracle_passes_safe=True, both_apply_cleanly=True)
    assert adm.admitted is None                      # reviewer_equivalent still unset
    adm.reviewer_equivalent = True; assert adm.admitted is True
    adm.oracle_passes_safe = False; assert adm.admitted is False


def test_partition_is_task_disjoint_and_deterministic():
    ts = [Task(f"r-{i}", "r" if i < 40 else "q", "o/r", i, "", "", i, "m", "b", "2026", ["src/x.py"], 10) for i in range(60)]
    a = assign_partitions([Task(**t.to_dict()) for t in ts], "salt")
    b = assign_partitions([Task(**t.to_dict()) for t in ts], "salt")
    pa = {t.task: t.partition for t in a}; pb = {t.task: t.partition for t in b}
    assert pa == pb and len(pa) == 60
    r = [t for t in a if t.repo == "r"]
    assert sum(t.partition == "train" for t in r) == 24 and sum(t.partition == "tune" for t in r) == 6 and sum(t.partition == "test" for t in r) == 10
    pc = {t.task: t.partition for t in assign_partitions([Task(**t.to_dict()) for t in ts], "other")}
    assert pc != pa


def test_gold_author_and_admission_on_git_repo(repo, tmp_path):
    r, base, fix = repo
    t = Task("r-1", "r", "o/r", 1, "raise limit", "", 1, fix, base, "2026", ["src/pkg/mod.py"], 2)
    res = GoldPatchAuthor().author(t, r)
    assert res.ok and "+LIMIT = 20" in res.diff
    sdiff = tmp_path / "s.diff"; sdiff.write_text(res.diff)
    co = Checkout(r); co.reset(base); co.apply(sdiff)
    rec = Seeder(SeederConfig("salt", cwe_share=0.0, package_root="src/pkg"), [], CoverageMap.empty()).seed(r, "r-1:u:gold", ["src/pkg/mod.py"])
    assert rec is not None
    out = subprocess.run(["git", "diff"], cwd=r, capture_output=True, text=True).stdout
    udiff = tmp_path / "u.diff"; udiff.write_text(out)
    assert "+LIMIT = 20" in out                      # unsafe variant keeps the safe change
    adm = admit_pair("r-1", "gold", r, base, sdiff, udiff, battery=lambda p: True, injecting_oracle=lambda p: True, checkout=co)
    assert adm.both_apply_cleanly and adm.safe_passes_battery and adm.oracle_catches_unsafe and adm.admitted is None
    assert (r / "src/pkg/mod.py").read_text() == SRC   # tree restored


def test_admission_resolves_relative_diff_paths(repo, tmp_path, monkeypatch):
    r, base, fix = repo
    t = Task("r-1", "r", "o/r", 1, "raise limit", "", 1, fix, base, "2026", ["src/pkg/mod.py"], 2)
    res = GoldPatchAuthor().author(t, r)
    d = tmp_path / "corp"; d.mkdir(); (d / "s.diff").write_text(res.diff); (d / "u.diff").write_text(res.diff)
    monkeypatch.chdir(tmp_path)                       # cwd differs from the checkout, diff path is relative
    adm = admit_pair("r-1", "gold", r, base, Path("corp/s.diff"), Path("corp/u.diff"), lambda p: True, lambda p: True)
    assert adm.both_apply_cleanly


def test_command_author_diff_excludes_venv(repo, tmp_path):
    r, base, fix = repo
    from corpus.authoring import CommandAuthor, export_base
    t = Task("r-1", "r", "o/r", 1, "t", "", 1, fix, base, "2026", ["src/pkg/mod.py"], 2)
    export = export_base(r, base, tmp_path / "export")                # authoring never runs in the clone
    (export / ".venv/bin").mkdir(parents=True); (export / ".venv/bin/activate").write_text("x")   # its own environment
    res = CommandAuthor("m0", ["sh", "-c", "echo '# agent' >> src/pkg/mod.py"], "m", "v").author(t, export)
    assert res.ok and "src/pkg/mod.py" in res.diff and ".venv" not in res.diff


def test_catalog_roundtrip_and_firewall(tmp_path):
    cat = probe_catalog_flask()
    dump_catalog(cat, tmp_path / "c.json")
    assert [d.id for d in load_catalog(tmp_path / "c.json")] == [d.id for d in cat]
    assert sum(d.held_out for d in cat) == 2
    # circularity firewall: no seeding anchor may appear verbatim in a detector rule file, when one is present
    det = Path("configs/detector_patterns.txt")
    if det.exists():
        text = det.read_text()
        assert not any(d.find in text or d.replace in text for d in cat)


def test_cmd_pairs_resume_appends_without_redoing_tasks(repo, tmp_path):
    """`--resume` skips tasks already in admission.jsonl and appends the rest, so a
    long build can run in bounded chunks and arrive at the same files."""
    import json
    from corpus import build
    r, base, fix = repo
    out = tmp_path / "out"; out.mkdir()
    tasks = [Task(f"r-{i}", "r", "o/r", i, "raise limit", "", i, fix, base, "2026", ["src/pkg/mod.py"], 2, "train")
             for i in (1, 2, 3)]
    (out / "tasks.jsonl").write_text("".join(json.dumps(t.to_dict()) + "\n" for t in tasks))
    cfg = {"checkout": str(r), "out": str(out), "package_root": "src/pkg", "top_module": "pkg", "freeze_salt": "salt",
           "cwe_share": 0.0, "covered_share": 0.5, "gold_patch_dev": True, "battery_cmd": None}
    build.cmd_author(cfg)                              # pairs pairs what `author` stored; it authors nothing
    build.cmd_pairs(cfg, limit=1)
    assert len(list(open(out / "admission.jsonl"))) == 1 and len(list(open(out / "manifest.jsonl"))) == 2
    build.cmd_pairs(cfg, limit=1, resume=True)
    build.cmd_pairs(cfg, resume=True)
    adm = [json.loads(l) for l in open(out / "admission.jsonl")]
    man = [json.loads(l) for l in open(out / "manifest.jsonl")]
    assert [a["task"] for a in adm] == ["r-1", "r-2", "r-3"] and len(man) == 6
    assert len({m["candidate_id"] for m in man}) == 6
    build.cmd_pairs(cfg, resume=True)                  # nothing left: files unchanged
    assert [json.loads(l) for l in open(out / "admission.jsonl")] == adm


def _census_record(tmp_path, repos, **fields):
    """A freeze record at stage dev naming `repos` (repository configs), with the census thresholds."""
    rec = {"freeze_id": None, "stage": "dev", "signed": False, "created": None, "salt": None,
           "repos": [str(p) for p in repos], "evidence_configs": [], "knobs": {"cwe_share": 0.5, "covered_share": 0.5},
           "loss": {"kappa": 100, "lambda_R": 10, "lambda_E": 2, "s_tilde": 0.6, "tau_qual": None, "eta": None},
           "cost_weights": [1 / 3, 1 / 3, 1 / 3], "m_S": None, "ceilings": None, "nominal_costs": None,
           "producers": [], "authors": None, "oracle": None, "excluded_tasks": [], "min_tasks": 234,
           "min_tasks_per_repo": 78, "tool_digests": {}}
    rec.update(fields)
    path = tmp_path / "freeze.json"
    path.write_text(json.dumps(rec))
    return path


def test_census_reports_shortfall(tmp_path, monkeypatch):
    """Review Focus 3: the task frame is counted after the development-exposed tasks are removed.
    `select` writes each repository's census.json and exits 3 below freeze.min_tasks_per_repo (78);
    `census` sums the repositories against freeze.min_tasks (234) and exits 3 on a shortfall, so the
    plan stops at B1 instead of proceeding with fewer tasks."""
    from corpus import build
    frame = {"a": 80, "b": 70}

    def select(full_name, short, package_root, cutoff, **kw):
        return [Task(f"{short}-{i}", short, full_name, i, f"issue {i}", "", 100 + i, f"m{i}", f"b{i}",
                     "2026-01-01T00:00:00Z", ["src/x.py"], 10) for i in range(frame[short])]
    monkeypatch.setattr(build, "select_tasks", select)
    cfgs = {}
    for name in ("a", "b"):
        cfgs[name] = tmp_path / f"{name}.json"
        cfgs[name].write_text(json.dumps({"repo": name, "full_name": f"o/{name}", "package_root": "src/x",
                                          "checkout": str(tmp_path / name), "out": str(tmp_path / f"{name}-dev"),
                                          "cutoff": "2026-06-01T00:00:00Z", "freeze_salt": "salt"}))
    rec = _census_record(tmp_path, cfgs.values(), excluded_tasks=["a-0", "a-1"])

    def select_rc(name):
        return build.main(["select", "--config", str(cfgs[name]), "--freeze", str(rec)])

    def census(name):
        return json.loads((tmp_path / f"{name}-dev" / "census.json").read_text())

    assert select_rc("a") == 0                                  # 80 selected, 2 excluded: 78 eligible
    a = census("a")
    assert (a["selected"], a["eligible"], a["excluded"], a["threshold"], a["shortfall"]) == (80, 78, ["a-0", "a-1"], 78, 0)
    assert sum(a["partitions"].values()) == 78
    rows = [json.loads(line) for line in open(tmp_path / "a-dev" / "tasks.jsonl")]
    assert len(rows) == 78 and not {"a-0", "a-1"} & {r["task"] for r in rows}
    assert select_rc("b") == 3                                  # 70 < 78: the repository is short
    assert (census("b")["eligible"], census("b")["shortfall"]) == (70, 8)
    assert (tmp_path / "b-dev" / "tasks.jsonl").exists()        # the frame is still written for the record
    assert build.main(["census", "--freeze", str(rec)]) == 3    # 148 < 234, and b < 78
    frame["b"] = 156
    assert select_rc("b") == 0
    assert build.main(["census", "--freeze", str(rec)]) == 0    # 78 + 156 = 234
    rec = _census_record(tmp_path, cfgs.values(), excluded_tasks=["a-0", "a-1", "b-0"])
    assert build.main(["census", "--freeze", str(rec)]) == 3    # a later exclusion counts: 233 < 234
def test_rq1_partition_split_is_a_config_parameter(tmp_path, monkeypatch):
    """The primary test is sized for 90 test tasks out of 180 (and
    the 81 analyzed tasks of the power note). The library default of 60/15/25
    cannot produce that pool -- it yields 45 test tasks from 180 -- so the split
    is a frozen design parameter read from the config, not a library default."""
    from collections import Counter
    from corpus import build
    ts = [Task(f"r-{i}", "r" if i < 90 else "q", "o/r", i, "", "", i, "m", "b", "2026", ["src/x.py"], 10)
          for i in range(180)]
    monkeypatch.setattr(build, "select_tasks", lambda *a, **k: [Task(**t.to_dict()) for t in ts])
    cfg = {"out": str(tmp_path / "corpus"), "full_name": "o/r", "repo": "r", "package_root": "src",
           "cutoff": "2026-06-01", "freeze_salt": "s",
           "partition_fractions": [1 / 3, 1 / 6, 1 / 2]}
    build.cmd_select(cfg)
    rows = read_jsonl(Path(cfg["out"]) / "tasks.jsonl")
    assert Counter(r["partition"] for r in rows) == {"train": 60, "tune": 30, "test": 90}
    # and the default is what makes this necessary: 60/15/25 with per-stratum
    # rounding yields 44 test tasks from 180, under half the design's 90
    dflt = assign_partitions([Task(**t.to_dict()) for t in ts], "s")
    assert sum(t.partition == "test" for t in dflt) == 44


REAL_CONFIGS = ("configs/flask.json", "configs/click.json")


def _authoring_contract(cfg_path):
    """What a config's `authors` block must satisfy to produce a confirmatory pool.

    Asserted on the config itself rather than through a build helper: authoring is
    its own pipeline stage (`corpus.build.cmd_author`), and the contract is a
    property of the frozen configuration either way."""
    import json
    cfg = json.loads(Path(cfg_path).read_text())
    authors = cfg["authors"]
    # D6 pinned two agents; the design simplification of 2026-10-22 (freeze record, PI) retired the
    # codex mode before the seal: one agent, m0 = Claude Code, the claim narrowed accordingly
    assert [a["mode"] for a in authors] == ["m0"], cfg_path
    assert "simplification" in cfg.get("_authors_note", ""), cfg_path
    for a in authors:
        # a placeholder model is the difference between a confirmatory run and a dry
        # run: provenance is what makes a candidate attributable to an agent
        assert "TBD" not in a["model"] and "TBD" not in str(a["version"]), (cfg_path, a["mode"], a["model"])
        # the agent CLIs take minutes per call; a short timeout truncates the slower
        # mode mid-task and reports it as an authoring failure
        assert int(a.get("timeout", 0)) >= 600, (cfg_path, a["mode"], a.get("timeout"))
    # the agent has shell access: the flags differ per CLI, so assert the property, not the spelling
    (m0,) = authors
    assert any("Bash" in x or "--tools" in x for x in m0["cmd"]), (cfg_path, m0["cmd"])



@pytest.mark.parametrize("cfg_path", REAL_CONFIGS)
def test_real_configs_name_the_pinned_authoring_agent(cfg_path):
    """The 2026-10-19 design pinned Claude Code and Codex; the simplification of 2026-10-22
    keeps one, `m0` = Claude Code, with model and version pinned and the retirement noted."""
    _authoring_contract(cfg_path)


def test_admission_requires_the_oracle_to_clear_the_safe_twin(repo, tmp_path):
    """Admission check (b) asks whether the injecting oracle catches the unsafe
    variant. On its own that is not evidence about the seeded defect: a gold patch
    may change security behaviour on purpose, and the dry run found two that do
    (trusted-host checking, dotenv precedence). Without a control on the safe twin
    a check that fires on both is indistinguishable from one that fires on the
    seed, so the oracle has to pass the safe variant as well."""
    r, base, fix = repo
    safe = tmp_path / "s.diff"; unsafe = tmp_path / "u.diff"
    body = _git(r, "diff", base, fix, "--", "src/pkg/mod.py").stdout
    safe.write_text(body); unsafe.write_text(body)
    always_fires = lambda _r: True           # nonzero exit on both twins
    adm = admit_pair("r-1", "m0", r, base, safe, unsafe, lambda _r: True, always_fires)
    assert adm.oracle_catches_unsafe is True
    assert adm.oracle_passes_safe is False, adm
    assert adm.admitted is not True, adm.to_dict()


def test_github_client_works_without_gh_auth(monkeypatch):
    """Task selection needs the issue title and body, which are the task statement the
    authoring agent is given; the history fallback substitutes a merge-commit subject
    with an empty body, which the 2026-10-19 pilot measured one agent declining to act
    on three times in five.

    `gh` refuses to run unauthenticated, but the REST API does not: it serves 60
    requests an hour anonymously and 5,000 with a token. Selection must therefore not
    depend on `gh` being logged in -- only on a token being present for it to be fast.
    """
    from corpus import tasks as T
    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append(dict(req.headers))
        import io, json as j
        return io.BytesIO(j.dumps({"ok": True}).encode())

    monkeypatch.setattr(T.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    assert T._api("repos/x/y") == {"ok": True}
    assert not any("Authorization" in h for h in calls[0]), calls[0]

    monkeypatch.setenv("GH_TOKEN", "t0ken")
    assert T._api("repos/x/y") == {"ok": True}
    assert any(h.lower() == "authorization" for h in calls[1]), calls[1]


def test_the_build_backend_is_exempt_from_the_dependency_cutoff():
    """`exclude_newer` exists so an unpinned runtime dependency takes the period's
    release rather than today's. It must not also pin the *build backend*: PEP 660
    `build_editable` arrived in setuptools 64 (August 2022), so an editable install
    at a 2021 base commit fails with

        '_BuildMetaLegacyBackend' object has no attribute 'build_editable'

    which is what stopped the 2026-10-20 corpus build on flask-4170. The backend is
    the tool doing the install, not a dependency of the thing installed, so it is
    resolved at its current version while everything else keeps the cutoff."""
    from corpus.envs import BUILD_BACKENDS, cutoff_args
    assert "setuptools" in BUILD_BACKENDS and "wheel" in BUILD_BACKENDS
    args = cutoff_args("2021-05-21T04:03:33Z")
    assert args[:2] == ["--exclude-newer", "2021-05-21T04:03:33Z"]
    exempt = [args[i + 1] for i, a in enumerate(args) if a == "--exclude-newer-package"]
    assert any(e.startswith("setuptools=") for e in exempt), args
    assert cutoff_args(None) == []


def test_coverage_is_resolved_per_base_commit(tmp_path, monkeypatch):
    """`covered_share` decides which mutation sites count as covered, and the probe
    measured 0.116 missed on covered lines against 0.446 on uncovered -- a 3.8x
    spread. So a coverage map from the wrong commit does not merely add noise, it
    moves U_CI.

    corpus/build.py loaded one map outside the task loop, so every candidate got
    coverage measured at the checkout's tag. Measured on 2026-10-20 against the real
    corpus: files drift by up to 187 lines between a 2021 base commit and 3.1.0, and
    one source file in the map does not exist at those commits at all. Coverage is
    therefore keyed by base commit and built on first use, as environments are."""
    from corpus.coverage import ensure_coverage, CoverageMap
    built = []

    def fake_build(repo, commit, python, spec, out):
        built.append(commit)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"files": {"src/pkg/mod.py": {"executed_lines": [1, 2, 3]}}}))

    spec = {"source": "pkg", "test_args": ["tests"]}
    a = ensure_coverage(tmp_path / "repo", "c0ffee", "py", spec, root=tmp_path / "covs", builder=fake_build)
    assert isinstance(a, CoverageMap) and a.covered("src/pkg/mod.py", 2)
    # a second call for the same commit is served from the cache, not rebuilt
    ensure_coverage(tmp_path / "repo", "c0ffee", "py", spec, root=tmp_path / "covs", builder=fake_build)
    assert built == ["c0ffee"], built
    # a different commit gets its own map
    ensure_coverage(tmp_path / "repo", "decaf1", "py", spec, root=tmp_path / "covs", builder=fake_build)
    assert built == ["c0ffee", "decaf1"], built


def test_the_safe_twin_control_is_not_run_against_a_twin_differential_oracle(repo, tmp_path):
    """The hidden oracle (corpus.oracle, `--mode catch`) already compares the twins:
    admission (b) is the battery failing on the unsafe variant where it passes on the
    safe one. Its runner is bound to the *unsafe* candidate id, so invoking it while
    the safe diff is checked out asks an incoherent question, and it answered "caught"
    on 43 of 50 pairs of the 2026-10-21 training build -- blocking 86 % of admissions
    on a control that the oracle performs internally.

    The control stays for a plain `oracle_cmd`, which is candidate-blind and has no
    twin to compare against."""
    r, base, fix = repo
    safe = tmp_path / "s.diff"; unsafe = tmp_path / "u.diff"
    body = _git(r, "diff", base, fix, "--", "src/pkg/mod.py").stdout
    safe.write_text(body); unsafe.write_text(body)
    fires = lambda _r: True

    # twin-differential oracle: the control is satisfied by construction, not run
    adm = admit_pair("r-1", "m0", r, base, safe, unsafe, lambda _r: True, fires,
                     twin_controlled=True)
    assert adm.oracle_catches_unsafe is True
    assert adm.oracle_passes_safe is True, adm.to_dict()
    assert "twin" in (adm.notes or "").lower(), adm.notes

    # candidate-blind oracle: the control still binds
    adm2 = admit_pair("r-1", "m0", r, base, safe, unsafe, lambda _r: True, fires,
                      twin_controlled=False)
    assert adm2.oracle_passes_safe is False and adm2.admitted is not True


def test_pr_scan_depth_comes_from_the_config(tmp_path, monkeypatch):
    """The freeze record requires min_tasks_per_repo = 78. Scanning the default 400
    merged PRs yielded 75 flask tasks on 2026-10-21 -- short of the threshold before
    any filtering -- so the depth has to be raisable without editing the library."""
    from corpus import build
    seen = {}

    def spy(full_name, short, package_root, cutoff, **kw):
        seen.update(kw)
        return []

    monkeypatch.setattr(build, "select_tasks", spy)
    cfg = {"out": str(tmp_path / "c"), "full_name": "o/r", "repo": "r", "package_root": "src",
           "cutoff": "2026-06-01", "freeze_salt": "s", "pr_limit": 1500,
           "partition_fractions": [1 / 3, 1 / 6, 1 / 2]}
    build.cmd_select(cfg)
    assert seen.get("limit") == 1500, seen


def test_github_client_retries_a_truncated_response(monkeypatch):
    """A 1,600-PR scan makes well over a thousand calls, and one truncated response
    killed the whole selection on 2026-10-21 with http.client.IncompleteRead. The
    client already waited out 403 and 429; transient network failure is the same
    situation and the same remedy, which its own docstring claimed but did not do."""
    import http.client
    import urllib.error
    from corpus import tasks as T
    calls = {"n": 0}

    def flaky(req, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise http.client.IncompleteRead(b"partial", 500)
        if calls["n"] == 2:
            raise urllib.error.URLError("connection reset")
        import io, json as j
        return io.BytesIO(j.dumps({"ok": True}).encode())

    monkeypatch.setattr(T.urllib.request, "urlopen", flaky)
    monkeypatch.setattr(T.time, "sleep", lambda _s: None)
    assert T._api("repos/x/y") == {"ok": True}
    assert calls["n"] == 3, calls


def test_github_client_gives_up_rather_than_spinning(monkeypatch):
    """Retrying forever on a genuinely broken endpoint hides the failure."""
    import http.client
    from corpus import tasks as T
    monkeypatch.setattr(T.urllib.request, "urlopen",
                        lambda *a, **k: (_ for _ in ()).throw(http.client.IncompleteRead(b"", 1)))
    monkeypatch.setattr(T.time, "sleep", lambda _s: None)
    with pytest.raises(Exception):
        T._api("repos/x/y")


def test_an_ast_located_defect_applies_across_five_years_of_history():
    """The literal `find` carries the formatting of the version it was written
    against, so two of ten defects applied at 2019-2022 base commits and twenty of
    110 tasks could take none at all. An AST locator anchors on the function name
    and a semantic token, both of which are stable: `get_cookie_httponly` is spelled
    identically in 2019 and 2026, only its annotations changed."""
    import subprocess
    from corpus.cwe import CweDefect
    repo = Path("corpus/repos/flask")
    if not (repo / ".git").exists():
        pytest.skip("flask checkout not present")
    tasks = Path("corpus/flask-api-dev/tasks.jsonl")
    if not tasks.exists():
        pytest.skip("no selected corpus")
    rows = [json.loads(l) for l in tasks.read_text().splitlines() if l.strip()]
    d = CweDefect("D01", "CWE-1004", "cookie-hardening", "HttpOnly dropped", "sessions.py", "", "",
                  match={"kind": "return_const", "function": "get_cookie_httponly",
                         "anchor": "SESSION_COOKIE_HTTPONLY", "const": "False"})
    for t in (min(rows, key=lambda r: r["merged_at"]), max(rows, key=lambda r: r["merged_at"])):
        src = subprocess.run(["git", "show", f"{t['base_commit']}:src/flask/sessions.py"],
                             cwd=repo, capture_output=True, text=True).stdout
        out = d.apply(src)
        assert out is not None, f"{t['task']} ({t['merged_at'][:10]}) did not match"
        assert "return False" in out
        # only the return value is rewritten, so a trailing comment survives
        assert out.count("\n") == src.count("\n")


def test_a_locator_that_does_not_match_falls_back_to_the_literal_pair():
    from corpus.cwe import CweDefect
    src = "def f():\n    return CONFIG_KEY\n"
    d = CweDefect("X", "CWE-0", "fam", "desc", "f.py", "return CONFIG_KEY", "return False",
                  match={"kind": "return_const", "function": "nonexistent",
                         "anchor": "CONFIG_KEY", "const": "False"})
    assert d.apply(src) == "def f():\n    return False\n"


def test_a_malformed_locator_is_an_error_not_a_silent_miss():
    from corpus.cwe import CweDefect, DefectMatchError
    d = CweDefect("X", "CWE-0", "fam", "desc", "f.py", "", "", match={"kind": "nonsense"})
    with pytest.raises(DefectMatchError):
        d.apply("def f():\n    return 1\n")


def test_dropping_tuple_items_keeps_it_a_tuple():
    """`endswith` takes one argument that may be a tuple. Rewriting the tuple's span
    without its parentheses turns `endswith((a, b, c, d, e))` into
    `endswith(c, d, e)` -- three positional arguments and a TypeError, which is a
    crash rather than the defect, and a battery would 'catch' it for the wrong
    reason. A minimal fixture hides this whenever one item survives."""
    import ast
    from corpus.cwe import apply_match
    src = ('def f(name):\n'
           '    return name.endswith((".html", ".htm", ".xml", ".xhtml", ".svg"))\n')
    out = apply_match(src, {"kind": "drop_tuple_items", "function": "f",
                            "items": [".html", ".htm"]})
    assert out is not None
    call = ast.parse(out).body[0].body[0].value
    assert len(call.args) == 1, ast.dump(call)
    assert isinstance(call.args[0], ast.Tuple)
    assert [e.value for e in call.args[0].elts] == [".xml", ".xhtml", ".svg"]
    # and a single survivor is still a tuple, not a bare string
    one = apply_match(src, {"kind": "drop_tuple_items", "function": "f",
                            "items": [".htm", ".xml", ".xhtml", ".svg"]})
    arg = ast.parse(one).body[0].body[0].value.args[0]
    assert isinstance(arg, ast.Tuple) and [e.value for e in arg.elts] == [".html"]


# ---------------------------------------------------------------- one fix per issue

def test_select_keeps_one_task_per_issue_the_earliest_merged_fix(monkeypatch):
    """A task is an issue and the PR that fixed it. Three click issues carry two merged PRs each
    (a fix and a later follow-up); the task id is <repo>-<issue>, so the frame would hold the id
    twice with two base commits, and authoring would silently take whichever row came first.
    The rule is the frame's own: the fix is the earliest merged PR; a later one is a follow-up."""
    from corpus import tasks as T
    prs = [{"number": 2550, "merge_commit_sha": "m2550", "merged_at": "2023-07-03T00:00:00Z"},
           {"number": 2248, "merge_commit_sha": "m2248", "merged_at": "2022-04-28T00:00:00Z"},
           {"number": 2300, "merge_commit_sha": "m2300", "merged_at": "2022-06-01T00:00:00Z"}]
    issues = {2550: {"number": 2246, "title": "t", "body": ""}, 2248: {"number": 2246, "title": "t", "body": ""},
              2300: {"number": 2300, "title": "u", "body": ""}}
    monkeypatch.setattr(T, "merged_prs", lambda full, cutoff, limit: prs)
    monkeypatch.setattr(T, "issue_for_pr", lambda full, pr: issues[pr["number"]])
    monkeypatch.setattr(T, "pr_files", lambda full, n: [{"filename": "src/click/core.py", "changes": 10}])
    monkeypatch.setattr(T, "pr_commit_parent", lambda full, sha: "base-" + sha)
    got = T.select_tasks("pallets/click", "click", "src/click", "2026-06-01T00:00:00Z")
    assert [(t.task, t.pr) for t in got] == [("click-2246", 2248), ("click-2300", 2300)]
    assert got[0].base_commit == "base-m2248"
    # the same rule as a function over rows, so an existing frame can be reduced the same way
    rows = [{"task": "a", "pr": 2, "merged_at": "2023-01-01"}, {"task": "a", "pr": 1, "merged_at": "2022-01-01"},
            {"task": "b", "pr": 3, "merged_at": "2024-01-01"}]
    assert [r["pr"] for r in T.one_fix_per_issue(rows)] == [1, 3]


# ---------------------------------------------------------------- pinned-toolchain viability at select

def test_select_drops_tasks_whose_base_commit_has_no_environment_and_counts_them(tmp_path, monkeypatch):
    """Pinned-toolchain viability is a frame criterion; the environment recipe is where it
    is decided (corpus.envs: a base commit whose manifests match no install step has no environment). The
    confirmatory flask frame of 2026-10-21 held 13 such tasks, all of them certain losses, counted as if
    they were tasks: the frame sat exactly on its 78 floor with 65 buildable. `select` applies the
    criterion up front and the census says how many it removed."""
    from corpus import build
    from corpus.tasks import Task
    cfg = {"repo": "flask", "full_name": "pallets/flask", "package_root": "src/flask", "cutoff": "2026-06-01T00:00:00Z",
           "out": str(tmp_path / "out-dev"), "checkout": str(tmp_path / "repo"), "freeze_salt": "DEV-only fallback",
           "partition_fractions": [0.3333, 0.1667, 0.5], "env": {"python_version": "3.10", "install": [{"if": "uv.lock", "args": ["-e", "."]}]}}
    def mk(i, base):
        return Task(f"flask-{i}", "flask", "pallets/flask", i, "t", "b", i, "m", base, "2024-01-01T00:00:00Z", ["src/flask/a.py"], 10)
    ts = [mk(1, "old1"), mk(2, "new2"), mk(3, "old3"), mk(4, "new4")]
    monkeypatch.setattr(build, "select_tasks", lambda *a, **k: ts)
    monkeypatch.setattr(build, "has_environment", lambda repo, commit, cfg: not commit.startswith("old"))
    rc = build.cmd_select(cfg)
    rows = [json.loads(l) for l in open(tmp_path / "out-dev" / "tasks.jsonl")]
    assert sorted(r["task"] for r in rows) == ["flask-2", "flask-4"]
    census = json.loads((tmp_path / "out-dev" / "census.json").read_text())
    assert census["selected"] == 4 and census["eligible"] == 2
    assert census["unbuildable"] == ["flask-1", "flask-3"]
    assert rc == 0


def test_has_environment_is_the_recipe_guard_without_a_build(tmp_path):
    """True iff some install step's guard file exists at the commit: the same test corpus.envs makes before
    building, answered from git ls-tree alone."""
    import subprocess
    from corpus.envs import has_environment
    repo = tmp_path / "r"; repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    (repo / "setup.py").write_text("x")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "one"], cwd=repo, check=True)
    c1 = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    (repo / "uv.lock").write_text("y")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "two"], cwd=repo, check=True)
    c2 = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    cfg = {"env": {"install": [{"if": "uv.lock", "args": ["-e", "."]}]}}
    assert has_environment(repo, c1, cfg) is False
    assert has_environment(repo, c2, cfg) is True
    assert has_environment(repo, c2, {}) is True          # no env block: nothing to be unviable against


# ---------------------------------------------------------------- severities from the rubric's anchors

def test_labels_from_rubric_anchors(tmp_path, monkeypatch):
    """Design simplification of 2026-10-22: the severity weight is the rubric's class anchor -- a CWE seed
    takes its CWE's anchor, a mutation seed the C1 anchor -- so L is class-weighted unsafe acceptance and
    no adjudication is needed. The provenance file says so."""
    from corpus import build
    from corpus.manifest import Candidate
    out = tmp_path / "out-dev"; out.mkdir()
    cands = [Candidate("t:s:m0", "t", "flask", "m0", "t:u:m0", "train", "c", "d", seed=None),
             Candidate("t:u:m0", "t", "flask", "m0", "t:s:m0", "train", "c", "d",
                       seed={"kind": "cwe", "cwe": "CWE-614", "file": "f", "line": 1}),
             Candidate("u:u:m0", "u", "flask", "m0", "u:s:m0", "train", "c", "d",
                       seed={"kind": "mutation", "operator": "NEG_if", "cwe": None, "file": "f", "line": 2}),
             Candidate("u:s:m0", "u", "flask", "m0", "u:u:m0", "train", "c", "d", seed=None)]
    with open(out / "manifest.jsonl", "w") as f:
        for c in cands:
            f.write(json.dumps(c.to_dict()) + "\n")
    sev = build.anchor_severities(cands, "configs/severity_rubric.json")
    assert sev == {"t:u:m0": 0.45, "u:u:m0": 0.5}
    build.cmd_labels({"out": str(out)}, None, None, anchors="configs/severity_rubric.json")
    labels = {json.loads(l)["candidate_id"]: json.loads(l) for l in open(out / "labels.jsonl")}
    assert labels["t:u:m0"]["severity"] == 0.45 and labels["u:u:m0"]["severity"] == 0.5 and labels["t:s:m0"]["severity"] == 0.0
    prov = json.loads((out / "labels.provenance.json").read_text())
    assert prov["adjudicated"] is False and "anchor" in prov["severity_source"]

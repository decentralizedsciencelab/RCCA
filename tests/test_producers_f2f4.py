import json, os, subprocess, sys, shutil
from pathlib import Path
import pytest
from evidence.core import Context, Meter, Registry, produce, cell_seed
from evidence.producers import f2_static, f2_deps, f3_contracts, f4_impact
from evidence.diffutil import changed_lines, in_change

HERE = Path(__file__).parent
PY = sys.executable
HAS_SEMGREP = shutil.which("semgrep") is not None


def _importable(mod: str) -> bool:
    return subprocess.run([PY, "-c", f"import {mod}"], capture_output=True).returncode == 0


HAS_BANDIT = _importable("bandit")
HAS_GRIFFE = _importable("griffe")
needs_bandit = pytest.mark.skipif(not HAS_BANDIT, reason="bandit not installed in the test interpreter")
needs_griffe = pytest.mark.skipif(not HAS_GRIFFE, reason="griffe not installed in the test interpreter")

MOD_A = "def helper(x):\n    return x * 2\n\ndef public_api(x):\n    return helper(x) + 1\n"
MOD_B = "from pkg.a import public_api\n\ndef use(x):\n    return public_api(x)\n"
PYPROJECT = '[project]\nname = "pkg"\nversion = "0.1"\ndependencies = ["requests>=2.0"]\n'


def _git(repo, *a):
    return subprocess.run(["git", *a], cwd=repo, capture_output=True, text=True, check=True)


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"; (r / "src/pkg").mkdir(parents=True); (r / "tests").mkdir()
    (r / "src/pkg/__init__.py").write_text("from .a import public_api\n")
    (r / "src/pkg/a.py").write_text(MOD_A); (r / "src/pkg/b.py").write_text(MOD_B)
    (r / "pyproject.toml").write_text(PYPROJECT)
    (r / "tests/test_x.py").write_text("def test_x():\n    assert True\n")
    _git(r, "init", "-q"); _git(r, "config", "user.email", "t@t"); _git(r, "config", "user.name", "t")
    _git(r, "add", "-A"); _git(r, "commit", "-qm", "base")
    return r, _git(r, "rev-parse", "HEAD").stdout.strip()


def _ctx(r, sha, cid="t:u:m0", rep=0, aid="x"):
    return Context(cid, r, sha, rep, cell_seed("salt", cid, rep, aid), "src/pkg")


# ---------------- diffutil ----------------
def test_changed_lines_tracks_only_the_candidate(repo):
    r, sha = repo
    (r / "src/pkg/a.py").write_text(MOD_A + "\ndef added():\n    return 1\n")
    cl = changed_lines(r, sha)
    assert in_change(cl, "src/pkg/a.py", 7) and not in_change(cl, "src/pkg/a.py", 1)
    assert not in_change(cl, "src/pkg/b.py", 1)


# ---------------- F2 static ----------------
@needs_bandit
def test_f2_static_pass_on_clean_change(repo):
    r, sha = repo
    (r / "src/pkg/a.py").write_text(MOD_A + "\ndef added():\n    return 1\n")
    a = f2_static.make_static_security(PY, [str(HERE / "fixtures/semgrep_test_rules.yml")], "src", use_semgrep=HAS_SEMGREP, timeout=300)
    rec = produce(a, _ctx(r, sha))
    assert rec.outcome == "pass" and rec.cost.tool_calls >= 1


@needs_bandit
def test_f2_static_preexisting_finding_is_not_attributed(repo):
    r, sha = repo
    # finding exists at base; candidate touches another line
    (r / "src/pkg/a.py").write_text(MOD_A + "\ndef danger(s):\n    return eval(s)\n")
    _git(r, "commit", "-qam", "preexisting")
    base2 = _git(r, "rev-parse", "HEAD").stdout.strip()
    (r / "src/pkg/b.py").write_text(MOD_B + "\ndef more():\n    return 2\n")
    a = f2_static.make_static_security(PY, [str(HERE / "fixtures/semgrep_test_rules.yml")], "src", use_semgrep=HAS_SEMGREP, timeout=300)
    rec = produce(a, _ctx(r, base2))
    assert rec.outcome == "pass" and int(rec.provenance["findings_total"]) >= 1 and rec.provenance["findings_on_change"] == "0"


@needs_bandit
@pytest.mark.skipif(not HAS_SEMGREP, reason="semgrep not installed")
def test_f2_static_agreement_validates_and_single_tool_does_not(repo):
    r, sha = repo
    (r / "src/pkg/a.py").write_text(MOD_A + "\ndef danger(s):\n    return eval(s)\n")
    a = f2_static.make_static_security(PY, [str(HERE / "fixtures/semgrep_test_rules.yml")], "src", k=1, timeout=300)
    rec = produce(a, _ctx(r, sha))
    assert rec.outcome == "fail_validated", rec.provenance
    assert rec.validation.confirmer_id == "bandit+semgrep-agreement" and "a.py:" in rec.validation.witness_ref and rec.validation.witness_ref.endswith(":CWE-78")
    # bandit alone: unvalidated
    b = f2_static.make_static_security(PY, [], "src", use_semgrep=False, timeout=300)
    rec2 = produce(b, _ctx(r, sha))
    assert rec2.outcome == "fail_unvalidated"


def test_f2_semgrep_runs_its_own_pysemgrep_ahead_of_path(repo, tmp_path, monkeypatch):
    """semgrep's launcher appends its own bin directory to the END of PATH and its semgrep-core then
    runs `pysemgrep` through PATH, so a pysemgrep earlier on PATH (the orchestrator's, say) would
    answer for the pinned venv's. F2 puts the executable's directory first: with a shadowing
    pysemgrep first on PATH, the version and the scan still come from the executable F2 was given."""
    r, sha = repo
    side, shadow = tmp_path / "side" / "bin", tmp_path / "shadow"
    side.mkdir(parents=True)
    shadow.mkdir()
    hit = {"path": "src/pkg/a.py", "start": {"line": 2}, "check_id": "side",
           "extra": {"metadata": {"cwe": ["CWE-78"]}, "severity": "ERROR"}}
    scripts = {side / "semgrep": '#!/bin/sh\nPATH="$PATH:$(dirname "$0")"\nexec pysemgrep "$@"\n',   # as semgrep's own
               side / "pysemgrep": "#!/bin/sh\nif [ \"$1\" = --version ]; then echo 1.181.0; "
                                   f"else echo '{json.dumps({'results': [hit], 'errors': []})}'; fi\n",
               shadow / "pysemgrep": '#!/bin/sh\necho "$@" >> "$(dirname "$0")/calls.log"\n'
                                     "if [ \"$1\" = --version ]; then echo 9.9.9; "
                                     "else echo '{\"results\": [], \"errors\": []}'; fi\n"}
    for path, text in scripts.items():
        path.write_text(text)
        path.chmod(0o755)
    monkeypatch.setenv("PATH", f"{shadow}{os.pathsep}{os.environ['PATH']}")
    semgrep = str(side / "semgrep")
    assert f2_static.semgrep_version_of(semgrep) == "1.181.0"
    found = f2_static._semgrep(Meter(), _ctx(r, sha), semgrep, ["rules.yml"], "src", 60)
    assert [(f["rule"], f["line"], f["cwe"]) for f in found] == [("side", 2, "CWE-78")]
    assert not (shadow / "calls.log").exists()


# ---------------- F2 deps ----------------
def _snapshot(tmp_path):
    snap = tmp_path / "snap.json"; adv = tmp_path / "adv.json"
    snap.write_text(json.dumps({"date": "2026-06-01", "packages": {"requests": ["2.31.0", "2.32.0"], "click": ["8.1.7"]}}))
    adv.write_text(json.dumps({"requests": [{"id": "GHSA-test", "affected": ["<2.32.0"]}]}))
    return snap.name, adv.name


def test_f2_deps_pass_when_no_dependency_file_changes(repo, tmp_path):
    r, sha = repo; s, a_ = _snapshot(tmp_path)
    (r / "src/pkg/b.py").write_text(MOD_B + "\n# touch\n")
    rec = produce(f2_deps.make_dependency_integrity(s, a_, base_dir=tmp_path), _ctx(r, sha))
    assert rec.outcome == "pass" and rec.provenance["dependency_files_changed"] == "0"


def test_f2_deps_nonexistent_package_is_validated(repo, tmp_path):
    r, sha = repo; s, a_ = _snapshot(tmp_path)
    (r / "pyproject.toml").write_text(PYPROJECT.replace('"requests>=2.0"', '"requests>=2.0", "reqeusts-utils-helper"'))
    rec = produce(f2_deps.make_dependency_integrity(s, a_, base_dir=tmp_path), _ctx(r, sha))
    assert rec.outcome == "fail_validated" and rec.validation.witness_ref == "reqeusts-utils-helper"


def test_f2_deps_advisory_match_is_unvalidated_and_existing_dep_untouched(repo, tmp_path):
    r, sha = repo; s, a_ = _snapshot(tmp_path)
    (r / "pyproject.toml").write_text(PYPROJECT.replace('"requests>=2.0"', '"requests>=2.0", "click==8.1.7"\n') )
    rec = produce(f2_deps.make_dependency_integrity(s, a_, base_dir=tmp_path), _ctx(r, sha))
    assert rec.outcome == "pass"
    (r / "pyproject.toml").write_text(PYPROJECT.replace('"requests>=2.0"', '"requests==2.31.0"'))
    rec = produce(f2_deps.make_dependency_integrity(s, a_, base_dir=tmp_path), _ctx(r, sha))
    assert rec.outcome == "fail_unvalidated" and "requests==2.31.0" in rec.provenance["advisory_matches"]


# ---------------- F3 ----------------
@needs_griffe
def test_f3_pass_on_additive_change(repo):
    r, sha = repo
    (r / "src/pkg/a.py").write_text(MOD_A + "\ndef added():\n    return 1\n")
    rec = produce(f3_contracts.make_api_contract(PY, "pkg", "src", timeout=300), _ctx(r, sha))
    assert rec.outcome == "pass", rec.provenance


@needs_griffe
def test_f3_removed_public_object_is_validated_by_import_probe(repo):
    r, sha = repo
    (r / "src/pkg/a.py").write_text("def helper(x):\n    return x * 2\n")
    (r / "src/pkg/__init__.py").write_text("")
    (r / "src/pkg/b.py").write_text("def use(x):\n    return x\n")
    rec = produce(f3_contracts.make_api_contract(PY, "pkg", "src", k=1, timeout=300), _ctx(r, sha))
    assert rec.outcome == "fail_validated", (rec.provenance, rec.status)
    assert rec.validation.confirmer_id == "fresh-interpreter-import-probe" and "public_api" in rec.validation.witness_ref


@needs_griffe
def test_f3_signature_change_is_unvalidated(repo):
    r, sha = repo
    (r / "src/pkg/a.py").write_text(MOD_A.replace("def public_api(x):", "def public_api(x, flag):"))
    rec = produce(f3_contracts.make_api_contract(PY, "pkg", "src", timeout=300), _ctx(r, sha))
    assert rec.outcome == "fail_unvalidated", rec.provenance


# ---------------- F4 ----------------
def test_f4_impact_counts_dependents_and_revert(repo):
    r, sha = repo
    (r / "src/pkg/a.py").write_text(MOD_A + "\n# tweak\n")
    rec = produce(f4_impact.make_impact("pkg"), _ctx(r, sha))
    assert rec.outcome == "pass" and rec.provenance["revert_applies"] == "True"
    assert rec.provenance["impact_changed_modules"] == "1" and int(rec.provenance["impact_dependents"]) >= 2   # b and __init__


def test_f4_binary_file_is_flagged(repo):
    r, sha = repo
    (r / "src/pkg/blob.so").write_bytes(b"\x00\x01\x02")
    _git(r, "add", "-A")
    rec = produce(f4_impact.make_impact("pkg"), _ctx(r, sha))
    assert rec.outcome == "fail_unvalidated" and rec.provenance["touches_binary"] == "1"


# --------------------------------------------------------------- F2 rule sets

def _rule_cwes(path):
    import re
    import yaml
    doc = yaml.safe_load(Path(path).read_text())
    out = {}
    for rule in doc["rules"]:
        cwe = rule.get("metadata", {}).get("cwe", "")
        cwe = cwe[0] if isinstance(cwe, list) else cwe
        m = re.search(r"CWE-(\d+)", str(cwe))
        out[rule["id"]] = f"CWE-{m.group(1)}" if m else ""
    return out


@pytest.mark.parametrize("repo", ["flask", "click"])
def test_f2_rules_are_disjoint_from_the_seeding_catalog(repo):
    """The F2 rule sets and the seeding catalog are separate
    artifacts "whose rule identifiers and pattern signatures are tested for
    disjointness, so that the F2 miss rate cannot be an artifact of a shared pattern
    list". This is that test. A rule targeting a seeded CWE would let F2 detect the
    seeded defects, and U_CI would then measure the rule list rather than the
    repository's own suite."""
    import json
    root = Path(__file__).resolve().parent.parent
    rules = root / f"configs/semgrep_rules_{repo}.yml"
    catalog = root / f"configs/cwe_{repo}.json"
    if not catalog.exists():
        pytest.skip(f"{catalog.name} not written")
    seeded = {d["cwe"] for d in json.loads(catalog.read_text())}
    # and the hidden scanner's own CWEs: a policy-visible detector sharing them would
    # make the oracle's findings predictable from what a policy can already see
    from corpus import oracle_scanner
    hidden = {getattr(r, "cwe", "") for r in oracle_scanner.RULES}
    detected = _rule_cwes(rules)
    overlap = {rid: cwe for rid, cwe in detected.items() if cwe in seeded | hidden}
    assert not overlap, (f"{rules.name} targets CWEs the seeder or the hidden scanner holds: "
                         f"{overlap}; seeded {sorted(seeded)}, hidden {sorted(c for c in hidden if c)}")
    assert detected, f"{rules.name} declares no CWE metadata"


@pytest.mark.skipif(not HAS_SEMGREP, reason="semgrep not installed")
def test_f2_rules_fire_on_what_they_describe(tmp_path):
    """A rule set that parses but matches nothing is the failure the 2026-10-19 pilot
    hit from the other side: F2 errored on 14 of 14 candidates because the file named
    here did not exist, and nothing noticed because no test ran it."""
    import json
    import subprocess
    root = Path(__file__).resolve().parent.parent
    (tmp_path / "bad.py").write_text(
        "import requests, tempfile, ssl\n"
        "import xml.etree.ElementTree\n"
        "def fetch(u):\n"
        "    return requests.get(u, verify=False)\n"
        "def scratch():\n"
        "    return tempfile.mktemp()\n"
        "def parse(text):\n"
        "    return xml.etree.ElementTree.fromstring(text)\n"
        "def check(user):\n"
        "    assert user.is_admin\n"
    )
    r = subprocess.run(["semgrep", "scan", "--json", "--quiet", "--metrics=off",
                        "--disable-version-check", "--no-git-ignore",
                        "--config", str(root / "configs/semgrep_rules_flask.yml"), "."],
                       cwd=tmp_path, capture_output=True, text=True, timeout=300)
    fired = {f["check_id"].split(".")[-1] for f in json.loads(r.stdout).get("results", [])}
    for expected in ("rcca-disabled-certificate-verification", "rcca-insecure-temporary-file",
                     "rcca-xml-external-entity", "rcca-assert-as-a-runtime-check"):
        assert expected in fired, (expected, sorted(fired))


def test_semgrep_rule_paths_are_resolved_against_the_config(tmp_path, monkeypatch):
    """Producers run with the candidate checkout as cwd, so a repository-relative rules
    path never resolves -- semgrep exits non-zero, `_semgrep` returns None and F2
    reports `inconclusive` with status error and no reason. That is what F2 did on 14
    of 14 candidates of the 2026-10-21 training build, and on every candidate of the
    pilot before it: the file was missing *and* the path was unresolved, so writing the
    file alone did not fix it.

    `f2_deps.register` already took base_dir; `f2_static.register` did not."""
    from evidence import run as run_mod
    from evidence.producers import f2_static as f2
    rules = tmp_path / "configs" / "rules.yml"
    rules.parent.mkdir(parents=True)
    rules.write_text("rules: []\n")
    seen = {}

    def spy(reg, python, semgrep_rules, target, **kw):
        seen["rules"] = list(semgrep_rules)

    monkeypatch.setattr(run_mod.f2_static, "register", spy)
    monkeypatch.setattr(f2, "semgrep_version_of", lambda _s: "1.181.0")
    run_mod.build_registry({"python": sys.executable,
                            "f2_static": {"semgrep_rules": ["configs/rules.yml"], "target": "src"}},
                           tmp_path, dev=True)
    assert seen["rules"] == [str(rules.resolve())], seen

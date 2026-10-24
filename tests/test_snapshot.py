import io, json, zipfile
from pathlib import Path
import pytest
from evidence import snapshot as S
from evidence.producers import f2_deps


INDEX = {"meta": {"_last-serial": 42, "api-version": "1.4"},
         "projects": [{"name": "Requests"}, {"name": "click"}, {"name": "Flask"}, {"name": "zzz_unrelated"}]}
RELEASES = {"requests": ["2.30.0", "2.31.0", "2.32.0"], "click": ["8.1.7"], "flask": ["3.0.0"]}
VULNS = {("requests", "2.30.0"): [{"id": "GHSA-9wx4", "aliases": ["CVE-2024-35195"], "fixed_in": ["2.32.0"]}],
         ("requests", "2.31.0"): [{"id": "GHSA-9wx4", "aliases": ["CVE-2024-35195"], "fixed_in": ["2.32.0"]}]}


def fake_fetch(url, accept=None, body=None):
    if url == "https://pypi.org/simple/":
        return json.dumps(INDEX).encode()
    parts = url.split("/")
    if url.startswith("https://pypi.org/pypi/") and url.endswith("/json"):
        name = parts[4]
        if len(parts) == 6:
            return json.dumps({"releases": {v: [] for v in RELEASES.get(name, [])}}).encode()
        version = parts[5]
        return json.dumps({"vulnerabilities": VULNS.get((name, version), [])}).encode()
    raise AssertionError(f"unexpected url {url}")


def test_build_with_fake_index_and_pypi_advisories(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "installed_closure", lambda py: {"requests": "2.31.0", "click": "8.1.7"})
    repo = tmp_path / "repo"; repo.mkdir()
    (repo / "pyproject.toml").write_text('[project]\nname="x"\ndependencies=["requests>=2", "flask"]\n')
    man = S.build(tmp_path / "out", "py", repo, fetch=fake_fetch, advisories="pypi", log=lambda *a: None)
    snap = json.load(open(tmp_path / "out/registry_snapshot.json"))
    assert snap["n_names"] == 4 and set(snap["packages"]) == {"requests", "click", "flask", "zzz-unrelated"}
    assert snap["packages"]["requests"] == ["2.30.0", "2.31.0", "2.32.0"] and snap["packages"]["zzz-unrelated"] == []
    assert sorted(snap["closure"]) == ["click", "flask", "requests"]
    adv = json.load(open(tmp_path / "out/advisories.json"))
    assert adv["requests"][0]["id"] == "GHSA-9wx4" and sorted(adv["requests"][0]["affected"]) == ["==2.30.0", "==2.31.0"]
    assert "click" not in adv
    assert set(man["files"]) == {"registry_snapshot.json", "advisories.json"}
    assert man["files"]["registry_snapshot.json"]["sha256"] == S.sha256(tmp_path / "out/registry_snapshot.json")
    assert man["absent_from_index"] == []


def test_snapshot_feeds_the_dependency_producer(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "installed_closure", lambda py: {"requests": "2.31.0"})
    S.build(tmp_path / "out", "py", None, fetch=fake_fetch, advisories="pypi", log=lambda *a: None)
    # a pinned vulnerable version matches; a nonexistent name is missing from the snapshot
    adv = json.load(open(tmp_path / "out/advisories.json"))
    assert any(f2_deps._version_matches("2.31.0", s) for a in adv["requests"] for s in a["affected"])
    assert not any(f2_deps._version_matches("2.32.0", s) for a in adv["requests"] for s in a["affected"])
    snap = json.load(open(tmp_path / "out/registry_snapshot.json"))
    assert "reqeusts" not in snap["packages"] and "requests" in snap["packages"]


def test_osv_record_conversion_ranges_and_versions():
    rec = {"id": "GHSA-x", "aliases": ["CVE-1"], "affected": [
        {"package": {"ecosystem": "PyPI", "name": "Flask"},
         "ranges": [{"type": "ECOSYSTEM", "events": [{"introduced": "0"}, {"fixed": "2.2.5"}, {"introduced": "2.3.0"}, {"fixed": "2.3.2"}]}],
         "versions": ["2.2.4", "2.3.0", "2.3.1"]},
        {"package": {"ecosystem": "npm", "name": "flask"}, "ranges": []}]}
    specs = S._osv_record_to_specs(rec)
    assert set(specs) == {"flask"}
    assert "<2.2.5" in specs["flask"] and ">=2.3.0,<2.3.2" in specs["flask"] and "==2.3.1" in specs["flask"]
    assert f2_deps._version_matches("2.3.1", ">=2.3.0,<2.3.2") and not f2_deps._version_matches("2.3.2", ">=2.3.0,<2.3.2")


def test_osv_dump_conversion_filters_to_closure(tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("GHSA-a.json", json.dumps({"id": "GHSA-a", "affected": [{"package": {"ecosystem": "PyPI", "name": "requests"},
                                                                              "ranges": [{"type": "ECOSYSTEM", "events": [{"introduced": "0"}, {"fixed": "2.32.0"}]}]}]}))
        z.writestr("GHSA-b.json", json.dumps({"id": "GHSA-b", "affected": [{"package": {"ecosystem": "PyPI", "name": "other"},
                                                                              "ranges": [{"type": "ECOSYSTEM", "events": [{"introduced": "0"}]}]}]}))
    p = tmp_path / "all.zip"; p.write_bytes(buf.getvalue())
    adv = S.advisories_from_osv_dump(p, {"requests"})
    assert list(adv) == ["requests"] and adv["requests"][0]["affected"] == ["<2.32.0"]


def test_producer_cache_reloads_on_change(tmp_path):
    p = tmp_path / "s.json"; p.write_text(json.dumps({"packages": {"a": []}}))
    assert "a" in f2_deps._load_cached(p)["packages"]
    import os, time
    p.write_text(json.dumps({"packages": {"b": []}})); os.utime(p, (time.time() + 5, time.time() + 5))
    assert "b" in f2_deps._load_cached(p)["packages"]

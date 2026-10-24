"""Freeze-date export of the registry snapshot and advisory database (spec §9).

    python -m evidence.snapshot --python corpus/repos/flask/.venv/bin/python \
        --repo corpus/repos/flask --out corpus/flask --freeze configs/freeze.json \
        [--advisories pypi|osv-dump|osv-api|none] [--osv-dump PyPI-all.zip] [--max-versions 150] [--workers 8]

The files are written only under a freeze record at stage pre-freeze or later
(`rcca.freeze`), or with --dev into an --out whose name ends in -dev.

Writes three files into --out:

  registry_snapshot.json   {"date", "source", "serial", "n_names",
                            "packages": {<normalized name>: [<versions>]}}
                           every project name on the index (versions empty) plus
                           the full release list for the dependency closure
  advisories.json          {<normalized name>: [{"id", "aliases", "affected": [<spec>], "source"}]}
  snapshot_manifest.json   creation time, sources, closure, SHA-256 and size of
                           each file: the digests go into the freeze record

The dependency closure is what the repository's own environment resolves to
(importlib.metadata in --python) plus the names declared in pyproject /
requirements / setup.cfg. Advisory sources:

  pypi      GET /pypi/<name>/<version>/json for every closure release; the
            "vulnerabilities" field is OSV-derived. Works through restrictive
            proxies. Many requests; bounded by --max-versions.
  osv-dump  convert a local OSV ecosystem dump (PyPI/all.zip) offline.
  osv-api   POST api.osv.dev/v1/querybatch then GET each vulnerability.

No producer calls this module. Producers read only the files it writes.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import io
import json
import os
import re
import ssl
import subprocess
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from rcca.freeze import FreezeError, gate
from .producers.f2_deps import normalize, parse_dep_text

Fetch =Callable[[str, Optional[str], Optional[bytes]], bytes]   # (url, accept, post_body) -> body


def _ssl_context() -> ssl.SSLContext:
    cafile = os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE")
    return ssl.create_default_context(cafile=cafile) if cafile and Path(cafile).exists() else ssl.create_default_context()


def http_fetch(url: str, accept: Optional[str] = None, body: Optional[bytes] = None, timeout: int = 120,
               retries: int = 3) -> bytes:
    ctx = _ssl_context()
    last: Optional[Exception] = None
    for attempt in range(retries):
        req = urllib.request.Request(url, data=body, method="POST" if body else "GET",
                                     headers={"Accept": accept or "*/*", "User-Agent": "rcca-snapshot/0.1"})
        if body:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise
            last = e
        except Exception as e:                      # noqa: BLE001
            last = e
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"fetch failed after {retries} attempts: {url}: {last}")


# ---------------------------------------------------------------------------
# closure
# ---------------------------------------------------------------------------

def installed_closure(python: str) -> Dict[str, str]:
    """{normalized name: version} for every distribution in the environment."""
    code = ("import importlib.metadata as m, json\n"
            "print(json.dumps({d.metadata['Name']: d.version for d in m.distributions() if d.metadata['Name']}))")
    r = subprocess.run([python, "-c", code], capture_output=True, text=True, check=True)
    return {normalize(k): v for k, v in json.loads(r.stdout).items()}


def declared_names(repo: Path) -> Set[str]:
    names: Set[str] = set()
    for f in ("pyproject.toml", "setup.cfg", "requirements.txt", "requirements-dev.txt"):
        p = repo / f
        if p.exists():
            deps = parse_dep_text(p.name, p.read_text()) or set()
            names |= {n for n, _ in deps}
    for p in (repo / "requirements").glob("*.txt") if (repo / "requirements").exists() else []:
        deps = parse_dep_text(p.name, p.read_text()) or set()
        names |= {n for n, _ in deps}
    return names


# ---------------------------------------------------------------------------
# registry names and releases
# ---------------------------------------------------------------------------

def index_names(fetch: Fetch) -> Tuple[Set[str], int]:
    body = fetch("https://pypi.org/simple/", "application/vnd.pypi.simple.v1+json", None)
    data = json.loads(body)
    names = {normalize(p["name"]) for p in data.get("projects", [])}
    return names, int(data.get("meta", {}).get("_last-serial", 0))


def releases(fetch: Fetch, name: str) -> List[str]:
    try:
        data = json.loads(fetch(f"https://pypi.org/pypi/{name}/json", "application/json", None))
    except urllib.error.HTTPError:
        return []
    return sorted(data.get("releases", {}).keys(), key=_vkey)


def _vkey(v: str):
    try:
        from packaging.version import Version
        return (0, Version(v))
    except Exception:
        return (1, v)


# ---------------------------------------------------------------------------
# advisories
# ---------------------------------------------------------------------------

def advisories_from_pypi(fetch: Fetch, packages: Dict[str, List[str]], workers: int = 8,
                         max_versions: int = 150) -> Dict[str, List[dict]]:
    """One request per (name, version); vulnerabilities aggregated into ==version specs."""
    jobs = [(n, v) for n, vs in packages.items() for v in vs[-max_versions:]]
    out: Dict[str, Dict[str, dict]] = {}

    def one(nv):
        n, v = nv
        try:
            data = json.loads(fetch(f"https://pypi.org/pypi/{n}/{v}/json", "application/json", None))
        except Exception:
            return n, v, []
        return n, v, data.get("vulnerabilities", []) or []

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for n, v, vulns in ex.map(one, jobs):
            for vul in vulns:
                vid = vul.get("id") or (vul.get("aliases") or ["?"])[0]
                rec = out.setdefault(n, {}).setdefault(vid, {"id": vid, "aliases": sorted(set(vul.get("aliases") or [])),
                                                             "affected": [], "fixed_in": sorted(set(vul.get("fixed_in") or [])),
                                                             "source": "pypi-json"})
                if f"=={v}" not in rec["affected"]:
                    rec["affected"].append(f"=={v}")
    return {n: sorted(d.values(), key=lambda r: r["id"]) for n, d in out.items()}


def _osv_record_to_specs(rec: dict) -> Dict[str, List[str]]:
    """{normalized package: [specs]} from one OSV vulnerability record (PyPI ecosystem only)."""
    out: Dict[str, List[str]] = {}
    for aff in rec.get("affected", []):
        pkg = aff.get("package", {})
        if pkg.get("ecosystem") != "PyPI":
            continue
        name = normalize(pkg.get("name", ""))
        specs: List[str] = []
        for rng in aff.get("ranges", []):
            if rng.get("type") != "ECOSYSTEM":
                continue
            intro, fixed = None, None
            for ev in rng.get("events", []):
                if "introduced" in ev:
                    if intro is not None:
                        specs.append(_range_spec(intro, fixed)); fixed = None
                    intro = ev["introduced"]
                elif "fixed" in ev:
                    fixed = ev["fixed"]
                elif "last_affected" in ev:
                    specs.append(f">={intro or '0'},<={ev['last_affected']}"); intro = None; fixed = None
            if intro is not None:
                specs.append(_range_spec(intro, fixed))
        for v in aff.get("versions", []) or []:
            specs.append(f"=={v}")
        if specs:
            out.setdefault(name, []).extend(sorted(set(specs)))
    return out


def _range_spec(intro: str, fixed: Optional[str]) -> str:
    lo = "" if intro in ("0", "", None) else f">={intro}"
    hi = f"<{fixed}" if fixed else ""
    return ",".join(s for s in (lo, hi) if s) or ">=0"


def advisories_from_osv_dump(zip_path: Path, names: Set[str]) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            if not info.filename.endswith(".json"):
                continue
            try:
                rec = json.loads(z.read(info))
            except Exception:
                continue
            for name, specs in _osv_record_to_specs(rec).items():
                if name in names:
                    out.setdefault(name, []).append({"id": rec.get("id", info.filename), "aliases": rec.get("aliases", []),
                                                     "affected": specs, "source": "osv-dump"})
    return out


def advisories_from_osv_api(fetch: Fetch, names: Iterable[str]) -> Dict[str, List[dict]]:
    names = sorted(set(names))
    out: Dict[str, List[dict]] = {}
    for i in range(0, len(names), 500):
        chunk = names[i:i + 500]
        body = json.dumps({"queries": [{"package": {"name": n, "ecosystem": "PyPI"}} for n in chunk]}).encode()
        res = json.loads(fetch("https://api.osv.dev/v1/querybatch", "application/json", body))
        for n, r in zip(chunk, res.get("results", [])):
            for v in r.get("vulns", []) or []:
                rec = json.loads(fetch(f"https://api.osv.dev/v1/vulns/{v['id']}", "application/json", None))
                specs = _osv_record_to_specs(rec).get(n, [])
                if specs:
                    out.setdefault(n, []).append({"id": rec["id"], "aliases": rec.get("aliases", []),
                                                  "affected": specs, "source": "osv-api"})
    return out


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(out: Path, python: Optional[str], repo: Optional[Path], fetch: Fetch = http_fetch,
          advisories: str = "pypi", osv_dump: Optional[Path] = None, max_versions: int = 150,
          workers: int = 8, extra_closure: Sequence[str] = (), log=print) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    closure: Dict[str, str] = installed_closure(python) if python else {}
    names_declared = declared_names(repo) if repo else set()
    closure_names = set(closure) | names_declared | {normalize(n) for n in extra_closure}
    log(f"closure: {len(closure)} installed + {len(names_declared)} declared -> {len(closure_names)} names")

    all_names, serial = index_names(fetch)
    log(f"index: {len(all_names)} project names, serial {serial}")
    packages: Dict[str, List[str]] = {n: [] for n in all_names}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for n, rel in zip(sorted(closure_names), ex.map(lambda n: releases(fetch, n), sorted(closure_names))):
            packages[n] = rel
    missing = sorted(n for n in closure_names if not packages.get(n))
    log(f"releases fetched for {len(closure_names) - len(missing)} closure packages; absent from index: {missing}")

    snap = {"date": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "source": "pypi-simple-v1",
            "serial": serial, "n_names": len(all_names), "closure": sorted(closure_names), "packages": packages}
    (out / "registry_snapshot.json").write_text(json.dumps(snap, separators=(",", ":")))

    adv: Dict[str, List[dict]] = {}
    closure_pkgs = {n: packages[n] for n in closure_names if packages.get(n)}
    if advisories == "pypi":
        adv = advisories_from_pypi(fetch, closure_pkgs, workers, max_versions)
    elif advisories == "osv-dump":
        if not osv_dump:
            raise SystemExit("--osv-dump is required with --advisories osv-dump")
        adv = advisories_from_osv_dump(osv_dump, closure_names)
    elif advisories == "osv-api":
        adv = advisories_from_osv_api(fetch, closure_names)
    (out / "advisories.json").write_text(json.dumps(adv, indent=1, sort_keys=True))
    log(f"advisories: {sum(len(v) for v in adv.values())} records for {len(adv)} packages (source {advisories})")

    manifest = {"created": snap["date"], "elapsed_s": round(time.time() - t0, 1),
                "sources": {"names": "https://pypi.org/simple/ (application/vnd.pypi.simple.v1+json)",
                            "releases": "https://pypi.org/pypi/<name>/json", "advisories": advisories,
                            "osv_dump": str(osv_dump) if osv_dump else None},
                "closure": sorted(closure_names), "closure_installed_versions": closure,
                "absent_from_index": missing,
                "files": {f: {"sha256": sha256(out / f), "bytes": (out / f).stat().st_size}
                          for f in ("registry_snapshot.json", "advisories.json")}}
    (out / "snapshot_manifest.json").write_text(json.dumps(manifest, indent=1))
    log(f"wrote {out}/registry_snapshot.json ({manifest['files']['registry_snapshot.json']['bytes']} bytes), "
        f"advisories.json, snapshot_manifest.json in {manifest['elapsed_s']}s")
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", help="interpreter of the repository's environment (closure source)")
    ap.add_argument("--repo", help="repository checkout (declared dependencies)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--advisories", choices=["pypi", "osv-dump", "osv-api", "none"], default="pypi")
    ap.add_argument("--osv-dump")
    ap.add_argument("--max-versions", type=int, default=150)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--extra", nargs="*", default=[], help="additional package names to include in the closure")
    ap.add_argument("--freeze", default=None, help="freeze record (rcca.freeze); needed unless --out ends in -dev")
    ap.add_argument("--dev", action="store_true", help="development run: no freeze record, --out ends in -dev")
    a = ap.parse_args(argv)
    try:
        gate(a.freeze, dev=a.dev, out=Path(a.out), min_stage="pre-freeze")
    except FreezeError as e:
        print(f"evidence.snapshot: refused: {e}", file=sys.stderr)
        return 2
    build(Path(a.out), a.python, Path(a.repo) if a.repo else None, advisories=a.advisories,
          osv_dump=Path(a.osv_dump) if a.osv_dump else None, max_versions=a.max_versions, workers=a.workers,
          extra_closure=a.extra)
    return 0


if __name__ == "__main__":
    sys.exit(main())

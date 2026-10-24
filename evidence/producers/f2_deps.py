"""F2 dependency integrity against a frozen registry snapshot (spec §9).

Inputs are the dependency declarations the candidate adds or changes
(pyproject `dependencies`, `requirements*.txt`, `setup.cfg install_requires`),
compared with two local sources taken at the freeze date:

  snapshot   JSON {"packages": {"<normalized name>": ["<version>", ...]}}
             the set of names that exist on the index and their releases
  advisories JSON {"<normalized name>": [{"id": "...", "affected": ["<spec>", ...]}]}
             a local vulnerability database (e.g. an OSV export)

Outcome rules
  pass               every new or changed dependency exists in the snapshot
                     and no pinned version matches an advisory
  fail_validated     a new dependency name is absent from the snapshot AND
                     from the lock file / installed metadata (second source):
                     a nonexistent package (slopsquatting surface). (i) the
                     snapshot is a stated reference, (ii) one name is the
                     minimal witness, (iii) deterministic, (iv) second source
  fail_unvalidated   a pinned version matches an advisory, or a name is
                     absent from the snapshot but present in the lock file
                     (snapshot may be stale)
  inconclusive       dependency files could not be parsed

No network. No pip-audit online mode. `pip-audit` may be used offline later
by pointing it at the same advisory export; this in-house check is what the
conformance suite covers today.
"""
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from ..core import Action, Context, Meter, Outcome, Registry, Validation
from ..diffutil import changed_files

try:
    import tomllib
except ImportError:                 # py<3.11
    try:
        import tomli as tomllib     # type: ignore
    except ImportError:
        tomllib = None

_DEP_LIST = re.compile(r"^\s*(?:dependencies|[\w-]+)\s*=\s*\[(.*?)\]", re.S | re.M)


def _pyproject_deps_regex(text: str) -> List[str]:
    """Fallback without a TOML parser: every quoted string inside list values of
    the [project] and [project.optional-dependencies] tables."""
    deps: List[str] = []
    sect = None
    for block in re.split(r"^(?=\[)", text, flags=re.M):
        head = block.split("\n", 1)[0].strip()
        if head in ("[project]", "[project.optional-dependencies]"):
            for m in _DEP_LIST.finditer(block):
                deps += [a or b for a, b in re.findall(r'"([^"]+)"|\'([^\']+)\'', m.group(1))]
    return deps

_REQ = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[[^\]]*\])?\s*((?:[<>=!~]=?\s*[^,;\s]+\s*,?\s*)*)")
_PIN = re.compile(r"==\s*([^,;\s]+)")


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_requirement(s: str) -> Optional[Tuple[str, Optional[str]]]:
    s = s.split("#")[0].split(";")[0].strip()
    if not s or s.startswith(("-", ".", "/")):
        return None
    m = _REQ.match(s)
    if not m:
        return None
    pin = _PIN.search(m.group(2) or "")
    return normalize(m.group(1)), (pin.group(1) if pin else None)


def parse_dep_text(name: str, text: str) -> Optional[Set[Tuple[str, Optional[str]]]]:
    """Dependencies declared in one file's text. None on parse failure."""
    out: Set[Tuple[str, Optional[str]]] = set()
    try:
        if name == "pyproject.toml":
            if tomllib is not None:
                data = tomllib.loads(text)
                deps = list(data.get("project", {}).get("dependencies", []))
                for extra in data.get("project", {}).get("optional-dependencies", {}).values():
                    deps += list(extra)
            else:
                deps = _pyproject_deps_regex(text)
            for d in deps:
                r = parse_requirement(d)
                if r: out.add(r)
        elif name.startswith("requirements") and name.endswith((".txt", ".in")):
            for line in text.splitlines():
                r = parse_requirement(line)
                if r: out.add(r)
        elif name == "setup.cfg":
            in_req = False
            for line in text.splitlines():
                if line.strip().startswith("install_requires"):
                    in_req = True; continue
                if in_req:
                    if line and not line[0].isspace():
                        in_req = False; continue
                    r = parse_requirement(line)
                    if r: out.add(r)
    except Exception:
        return None
    return out


def declared_dependencies(repo: Path, files: Sequence[str]) -> Optional[Set[Tuple[str, Optional[str]]]]:
    out: Set[Tuple[str, Optional[str]]] = set()
    for f in files:
        p = repo / f
        if not p.exists():
            continue
        r = parse_dep_text(Path(f).name, p.read_text())
        if r is None:
            return None
        out |= r
    return out


def lock_names(repo: Path) -> Set[str]:
    names: Set[str] = set()
    for lock in ("uv.lock", "poetry.lock", "requirements.lock", "Pipfile.lock"):
        p = repo / lock
        if p.exists():
            for m in re.finditer(r'name\s*=\s*"([^"]+)"', p.read_text()):
                names.add(normalize(m.group(1)))
    return names


def _vkey(v: str):
    try:
        from packaging.version import Version
        return (0, Version(v))
    except Exception:
        return (1, tuple(int(x) if x.isdigit() else 0 for x in re.split(r"[.\-+]", v)))


def _version_matches(version: str, spec: str) -> bool:
    """Specifier matching for advisory ranges: '==x', '<x', '<=x', '>=x', '>x', joined by ','.
    Uses packaging.version when available; falls back to a numeric tuple."""
    ok = True
    for part in spec.split(","):
        part = part.strip()
        m = re.match(r"(==|<=|>=|<|>)\s*(.+)", part)
        if not m:
            continue
        op, v = m.groups(); a, b = _vkey(version), _vkey(v.strip())
        if a[0] != b[0]:
            return False
        ok &= {"==": a == b, "<": a < b, "<=": a <= b, ">": a > b, ">=": a >= b}[op]
    return ok


_CACHE: Dict[str, Tuple[float, dict]] = {}


def _load_cached(path: Path) -> dict:
    """Load a JSON file once per process per mtime. The registry snapshot is large
    and read by every producer invocation."""
    key = str(path); mtime = path.stat().st_mtime
    hit = _CACHE.get(key)
    if hit and hit[0] == mtime:
        return hit[1]
    data = json.load(open(path))
    _CACHE[key] = (mtime, data)
    return data


def make_dependency_integrity(snapshot_path: str, advisories_path: Optional[str] = None,
                              base_dir: Optional[Path] = None,
                              dep_files_glob: Sequence[str] = ("pyproject.toml", "requirements", "setup.cfg")) -> Action:
    base = Path(base_dir) if base_dir else Path.cwd()

    def run(ctx: Context, m: Meter) -> Outcome:
        snap = _load_cached(base / snapshot_path)
        packages: Dict[str, List[str]] = snap.get("packages", {})      # keys are normalized by the exporter
        adv = _load_cached(base / advisories_path) if advisories_path else {}
        prov = {"tool": "inhouse-dep-check", "snapshot": snapshot_path, "snapshot_date": str(snap.get("date", ""))}
        blind = ["declared dependencies only; transitive closure and binary wheels not inspected"]
        files = [f for f in changed_files(ctx.repo, ctx.repo_state_sha)
                 if any(Path(f).name.startswith(g) for g in dep_files_glob)]
        prov["dependency_files_changed"] = str(len(files))
        if not files:
            return Outcome("pass", provenance=prov, blind_spots=blind)
        m.tool_calls += 1          # one logical tool invocation: the parser
        after = declared_dependencies(ctx.repo, files)
        if after is None:
            return Outcome("inconclusive", provenance=prov, blind_spots=blind, status="error")
        # dependencies at the base state, to isolate what the change introduced
        import subprocess
        before: Set[Tuple[str, Optional[str]]] = set()
        for f in files:
            r = subprocess.run(["git", "show", f"{ctx.repo_state_sha}:{f}"], cwd=ctx.repo, capture_output=True, text=True)
            if r.returncode == 0:
                b = parse_dep_text(Path(f).name, r.stdout)
                if b: before |= b
        new = after - before
        prov["dependencies_new_or_changed"] = str(len(new))
        locks = lock_names(ctx.repo)
        missing = sorted(n for n, _ in new if n not in packages)
        vulnerable = sorted(f"{n}=={v}" for n, v in new if v and n in adv
                            and any(_version_matches(v, spec) for a in adv[n] for spec in a.get("affected", [])))
        prov["missing_from_snapshot"] = ";".join(missing); prov["advisory_matches"] = ";".join(vulnerable)
        ghost = [n for n in missing if n not in locks]
        if ghost:
            v = Validation(True, True, True, True, "lockfile-absence", ";".join(ghost), [])
            return Outcome("fail_validated", v, prov, blind_spots=blind)
        if missing or vulnerable:
            return Outcome("fail_unvalidated", Validation(contradicts_reference=True, minimized=True, repeats_across_seeds=True),
                           prov, blind_spots=blind)
        return Outcome("pass", provenance=prov, blind_spots=blind)

    return Action("f2.dependency", "F2", "author_side", "inhouse-dep-check", run, priority=21)


def register(reg: Registry, snapshot_path: str, advisories_path: Optional[str] = None, **kw) -> Action:
    return reg.register(make_dependency_integrity(snapshot_path, advisories_path, **kw))

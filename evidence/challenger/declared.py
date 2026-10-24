"""Declared behaviour changes.

The challenger's oracle is pre-change behaviour, so an intended behaviour change
(a bug fix, a hardening, a deprecation removal) diverges from it exactly as a
defect does. The candidate declares such a change through its own tests. The
rule, per changed callable:

  a changed callable is *declared* when the test functions the candidate added
  or modified pass in the candidate tree and fail once that callable alone is
  reverted to its pinned body (an added callable is removed). The candidate's
  own tests then distinguish the new behaviour of that callable.

Executing a changed line is not enough: a seeded defect in a callable the new
tests merely pass through is not declared, and its divergence stays a validated
violation. One pytest run of the changed tests under coverage prefilters the
callables those tests reach at all; one more run per reached callable decides.

A divergence on a declared target is still reported, but as `fail_unvalidated`
with the declaring tests in provenance, never as a validated violation. The rule
is one-directional: a defect seeded inside a callable whose change the
candidate's tests distinguish is not caught by the challenger, only by those
tests (F1). When coverage cannot run, a name heuristic (a changed test line
names the callable) stands in, and provenance says which rule applied.
"""
from __future__ import annotations
import ast
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from ..core import Meter
from ..diffutil import changed_files, changed_lines
from .targets import Target

Key = Tuple[str, str]
_FAILED = re.compile(r"^(FAILED|ERROR) (\S+)", re.M)

_TEST_DIR = re.compile(r"(^|/)(tests?|testing)(/|$)")
_TEST_FILE = re.compile(r"(^|/)(test_[^/]*\.py|[^/]*_test\.py|conftest\.py)$")


def is_test_file(path: str) -> bool:
    p = path.replace("\\", "/")
    directory = p.rsplit("/", 1)[0] + "/" if "/" in p else ""
    return bool(_TEST_DIR.search(directory)) or bool(_TEST_FILE.search(p))


def changed_test_files(repo: Path, base_sha: str) -> List[str]:
    return [f for f in changed_files(repo, base_sha) if is_test_file(f) and f.endswith(".py")
            and not Path(f).name == "conftest.py"]


def _test_functions(src: str) -> Dict[str, ast.AST]:
    """{pytest node suffix: node} for test functions in a module (free and in classes)."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return {}
    out: Dict[str, ast.AST] = {}
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith("test"):
            out[n.name] = n
        elif isinstance(n, ast.ClassDef) and n.name.startswith("Test"):
            for m in n.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and m.name.startswith("test"):
                    out[f"{n.name}::{m.name}"] = m
    return out


def changed_test_functions(repo: Path, base_sha: str) -> List[str]:
    """pytest node ids (`path::name`, `path::Class::name`) of test functions the
    candidate added or whose source it changed."""
    ids: List[str] = []
    for f in changed_test_files(repo, base_sha):
        p = repo / f
        if not p.is_file():
            continue
        cand = _test_functions(p.read_text(errors="replace"))
        r = subprocess.run(["git", "show", f"{base_sha}:{f}"], cwd=repo, capture_output=True, text=True)
        base = _test_functions(r.stdout) if r.returncode == 0 else {}
        for name, node in cand.items():
            if name not in base or ast.dump(base[name]) != ast.dump(node):
                ids.append(f"{f}::{name}")
    return ids


def _env(repo: Path, search: str, tmp: Path) -> Dict[str, str]:
    """The candidate's `search` directory leads PYTHONPATH so the tree under test is
    the one imported; coverage data goes to a scratch directory, not the checkout."""
    env = {k: v for k, v in os.environ.items()
           if not k.upper().endswith("_PROXY") and k.upper() not in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")}
    env["PYTHONPATH"] = str(repo / search) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["COVERAGE_FILE"] = str(tmp / ".coverage")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["NO_NETWORK"] = "1"
    return env


def _run(cmd: Sequence[str], repo: Path, env: Dict[str, str], meter: Optional[Meter], timeout: int):
    if meter is not None:
        return meter.tool(cmd, repo, timeout, env=env)
    return subprocess.run(list(cmd), cwd=repo, capture_output=True, text=True, env=env, timeout=timeout)


def _failing(stdout: str) -> Set[str]:
    return {m.group(2).split(" - ")[0] for m in _FAILED.finditer(stdout or "")}


_PYTEST = ["-q", "--no-header", "-p", "no:cacheprovider", "-rfE", "--continue-on-collection-errors"]


def _covered_and_failing(repo: Path, python: str, package_root: str, search: str, node_ids: Sequence[str],
                         meter: Optional[Meter], timeout: int) -> Optional[Tuple[Dict[str, Set[int]], Set[str]]]:
    """One run of the changed tests in the candidate tree under coverage: (lines of
    package_root executed, tests failing), or None when coverage could not run."""
    tmp = Path(tempfile.mkdtemp(prefix="rcca-declared-"))
    try:
        env = _env(repo, search, tmp)
        r1 = _run([python, "-m", "coverage", "run", f"--source={package_root}", "-m", "pytest", *node_ids, *_PYTEST],
                  repo, env, meter, timeout)
        r2 = _run([python, "-m", "coverage", "json", "-q", "-o", str(tmp / "cov.json")], repo, env, meter, 120)
        cj = tmp / "cov.json"
        if r2.returncode != 0 or not cj.exists():
            return None
        data = json.loads(cj.read_text())
        out: Dict[str, Set[int]] = {}
        root = str(repo.resolve())
        for fname, info in data.get("files", {}).items():
            rel = fname
            if os.path.isabs(fname) and fname.startswith(root):
                rel = os.path.relpath(fname, root)
            out[rel.replace("\\", "/")] = set(info.get("executed_lines", []))
        return out, _failing(r1.stdout)
    except (subprocess.TimeoutExpired, OSError, ValueError):
        return None
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


def _base_source(repo: Path, base_sha: str, path: str) -> Optional[str]:
    r = subprocess.run(["git", "show", f"{base_sha}:{path}"], cwd=repo, capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def _node_span(src: str, qualname: str) -> Optional[Tuple[int, int]]:
    """(first, last) 1-based line of a callable's `def` block (decorators excluded)."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    parts = qualname.split(".")
    body = tree.body
    node = None
    for i, part in enumerate(parts):
        node = next((n for n in body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                     and n.name == part), None)
        if node is None:
            return None
        body = getattr(node, "body", [])
    return node.lineno, getattr(node, "end_lineno", node.lineno)


def reverted_source(cand_src: str, base_src: Optional[str], t: Target) -> Optional[str]:
    """The candidate file with `t` alone restored to its pinned body (removed when
    it was added). None when the span cannot be located."""
    span = _node_span(cand_src, t.qualname)
    if span is None:
        return None
    lines = cand_src.splitlines(keepends=True)
    a, b = span
    base_block: List[str] = []
    if base_src is not None:
        bspan = _node_span(base_src, t.qualname)
        if bspan is not None:
            blines = base_src.splitlines(keepends=True)
            base_block = blines[bspan[0] - 1:bspan[1]]
    if not base_block and t.status != "added":
        return None
    new = lines[:a - 1] + base_block + lines[b:]
    # an emptied class body or module is still valid only if something remains; add `pass` when needed
    if not base_block and a >= 2 and lines[a - 2].rstrip().endswith(":"):
        indent = re.match(r"[ \t]*", lines[a - 1]).group(0)
        new = lines[:a - 1] + [f"{indent}pass\n"] + lines[b:]
    return "".join(new)


def _changed_in_body(cl: Dict[str, Set[int]], t: Target) -> Set[int]:
    lines = cl.get(t.file, set())
    end = t.end_line if t.end_line >= t.line else 10 ** 9
    return {n for n in lines if t.line <= n <= end}


def declared_by_names(repo: Path, base_sha: str, targets: Sequence[Target]) -> Dict[Key, str]:
    """Fallback: the target's final name appears as a word on a line the candidate
    added to a test file (and, for a method, the class name appears in that file)."""
    out: Dict[Key, str] = {}
    tests = changed_test_files(repo, base_sha)
    if not tests or not targets:
        return out
    added: Dict[str, List[str]] = {}
    full: Dict[str, str] = {}
    for f in tests:
        r = subprocess.run(["git", "diff", "--unified=0", "--no-color", base_sha, "--", f], cwd=repo,
                           capture_output=True, text=True)
        lines = [l[1:] for l in r.stdout.splitlines() if l.startswith("+") and not l.startswith("+++")]
        p = repo / f
        text = p.read_text(errors="replace") if p.is_file() else ""
        added[f] = lines if r.stdout.strip() else text.splitlines()
        full[f] = text
    for t in targets:
        name = t.qualname.rsplit(".", 1)[-1]
        pat = re.compile(r"\b" + re.escape(name) + r"\b")
        for f in tests:
            if not any(pat.search(l) for l in added[f]):
                continue
            if t.cls and not re.search(r"\b" + re.escape(t.cls.rsplit(".", 1)[-1]) + r"\b", full[f]):
                continue
            out[(t.module, t.qualname)] = f
            break
    return out


def _ids(tests: Set[str], cap: int = 4) -> str:
    s = sorted(tests)
    return ";".join(s[:cap]) + (f";+{len(s) - cap}" if len(s) > cap else "")


def declared_targets(repo: Path, base_sha: str, targets: Sequence[Target], python: Optional[str] = None,
                     package_root: Optional[str] = None, search: str = "src", meter: Optional[Meter] = None,
                     timeout: int = 300) -> Tuple[Dict[Key, str], str]:
    """({(module, qualname): tests that distinguish the change}, rule) where rule
    is "revert" (the full rule), "names" (fallback when coverage cannot run) or
    "none" (the candidate changed no test function)."""
    if not targets:
        return {}, "none"
    ids = changed_test_functions(repo, base_sha)
    if not ids:
        return {}, "none"
    if not (python and package_root):
        return declared_by_names(repo, base_sha, targets), "names"
    first = _covered_and_failing(repo, python, package_root, search, ids, meter, timeout)
    if first is None:
        return declared_by_names(repo, base_sha, targets), "names"
    cov, fail_cand = first
    cl = changed_lines(repo, base_sha)
    reached = [t for t in targets if _changed_in_body(cl, t) & cov.get(t.file, set())]
    out: Dict[Key, str] = {}
    tmp = Path(tempfile.mkdtemp(prefix="rcca-declared-"))
    try:
        env = _env(repo, search, tmp)
        for t in reached:
            p = repo / t.file
            try:
                original = p.read_bytes()
            except OSError:
                continue
            rev = reverted_source(original.decode("utf-8", errors="surrogateescape"), _base_source(repo, base_sha, t.file), t)
            if rev is None:
                continue
            try:
                p.write_bytes(rev.encode("utf-8", errors="surrogateescape"))
                r = _run([python, "-m", "pytest", *ids, *_PYTEST], repo, env, meter, timeout)
            except (subprocess.TimeoutExpired, OSError):
                continue
            finally:
                p.write_bytes(original)                 # the candidate tree is restored byte for byte
            newly = _failing(r.stdout) - fail_cand
            if newly:
                out[(t.module, t.qualname)] = _ids(newly)
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    return out, "revert"

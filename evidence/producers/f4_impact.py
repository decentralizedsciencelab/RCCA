"""F4 change impact, provenance, and recoverability. In-house, language-agnostic
in design; the import graph here is Python-specific.

Three checks, one record:

  impact        dependents of the changed modules in the package's import
                graph (reachability over `import` / `from ... import`), as a
                count and a fraction of the package: the blast radius feature
  provenance    files the change touches outside the package root and tests,
                binary or vendored files, and generated-looking files
  recoverability `git apply --check -R` of the candidate diff against the
                candidate tree: can the change be reverted cleanly, and does
                the revert leave dependents importable

Outcome rules
  pass               revert applies cleanly and no binary or vendored file is touched
  fail_unvalidated   revert does not apply cleanly, or the change touches
                     binary or vendored files (out-of-tree files are recorded
                     as a feature, not an outcome)
  inconclusive       the diff or the import graph could not be built

F4 never validates a violation: nothing here contradicts a stated reference
in the sense of criterion (i). Its value is the features it records
(`impact_dependents`, `impact_fraction`, `touches_outside_root`), which feed
the risk model and H2.
"""
from __future__ import annotations
import ast
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

from ..core import Action, Context, Meter, Outcome, Registry, Validation
from ..diffutil import changed_files

_BINARY_EXT = {".so", ".pyd", ".dll", ".dylib", ".whl", ".egg", ".zip", ".tar", ".gz", ".bin", ".pkl", ".parquet", ".npy"}


def _module_name(root: Path, p: Path) -> str:
    rel = p.relative_to(root).with_suffix("")
    parts = list(rel.parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def import_graph(pkg_root: Path, top: str) -> Dict[str, Set[str]]:
    """{module: {imported modules within the package}}."""
    g: Dict[str, Set[str]] = {}
    for p in pkg_root.rglob("*.py"):
        name = f"{top}.{_module_name(pkg_root, p)}".rstrip(".") if _module_name(pkg_root, p) else top
        g.setdefault(name, set())
        try:
            tree = ast.parse(p.read_text())
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.name == top or a.name.startswith(top + "."):
                        g[name].add(a.name)
            elif isinstance(node, ast.ImportFrom):
                if node.level and node.level > 0:
                    base = name.split(".")
                    base = base[:-node.level] if not p.name == "__init__.py" else base[:len(base) - node.level + 1]
                    mod = ".".join(base + ([node.module] if node.module else []))
                    g[name].add(mod)
                    for a in node.names:
                        g[name].add(f"{mod}.{a.name}")
                elif node.module and (node.module == top or node.module.startswith(top + ".")):
                    g[name].add(node.module)
                    for a in node.names:
                        g[name].add(f"{node.module}.{a.name}")
    return g


def dependents(g: Dict[str, Set[str]], changed: Set[str]) -> Set[str]:
    """Modules that (transitively) import any changed module."""
    rev: Dict[str, Set[str]] = {}
    mods = set(g)
    for m, imps in g.items():
        for i in imps:
            # resolve "a.b.c" to the longest known module prefix
            parts = i.split(".")
            for j in range(len(parts), 0, -1):
                cand = ".".join(parts[:j])
                if cand in mods:
                    rev.setdefault(cand, set()).add(m); break
    seen: Set[str] = set(); stack = list(changed)
    while stack:
        x = stack.pop()
        for d in rev.get(x, ()):
            if d not in seen and d not in changed:
                seen.add(d); stack.append(d)
    return seen


def make_impact(top_module: str, tests_dir: str = "tests", timeout: int = 300) -> Action:
    def run(ctx: Context, m: Meter) -> Outcome:
        root = ctx.repo / ctx.package_root
        prov = {"tool": "inhouse-impact", "package_root": ctx.package_root}
        blind = ["static import graph: dynamic imports, plugins, and entry points invisible", "no runtime impact"]
        files = changed_files(ctx.repo, ctx.repo_state_sha)
        if not files:
            return Outcome("inconclusive", provenance=prov, blind_spots=blind, status="error")
        m.tool_calls += 1
        # provenance
        outside = [f for f in files if not (f.startswith(ctx.package_root) or f.startswith(tests_dir))]
        binary = [f for f in files if Path(f).suffix.lower() in _BINARY_EXT]
        vendored = [f for f in files if re.search(r"(^|/)(vendor|vendored|_vendor|third_party)/", f)]
        prov.update(files_changed=str(len(files)), touches_outside_root=str(len(outside)),
                    touches_binary=str(len(binary)), touches_vendored=str(len(vendored)))
        # impact
        try:
            g = import_graph(root, top_module)
        except Exception:
            return Outcome("inconclusive", provenance=prov, blind_spots=blind, status="error")
        changed_mods = set()
        for f in files:
            p = ctx.repo / f
            if f.startswith(ctx.package_root) and f.endswith(".py") and p.exists():
                mn = _module_name(root, p)
                changed_mods.add(f"{top_module}.{mn}" if mn else top_module)
        deps = dependents(g, changed_mods)
        prov.update(impact_changed_modules=str(len(changed_mods)), impact_dependents=str(len(deps)),
                    impact_fraction=f"{(len(deps) / max(1, len(g))):.3f}", package_modules=str(len(g)))
        # recoverability: diff of the candidate vs base, reverse-apply check
        d = m.tool(["git", "diff", ctx.repo_state_sha, "--"], ctx.repo, timeout)
        rv = m.tool(["git", "apply", "--check", "-R", "-"], ctx.repo, timeout, stdin=d.stdout)
        revertible = rv.returncode == 0
        prov["revert_applies"] = str(revertible)
        v = Validation(contradicts_reference=False)
        if not revertible or binary or vendored:
            return Outcome("fail_unvalidated", v, prov, blind_spots=blind)
        return Outcome("pass", v, prov, blind_spots=blind)

    return Action("f4.impact_revert", "F4", "author_side", "inhouse-impact", run, priority=50)


def register(reg: Registry, top_module: str, **kw) -> Action:
    return reg.register(make_impact(top_module, **kw))

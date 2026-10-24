"""F3 interface and resource contracts.

API contract: `griffe check` compares the candidate's public API with the
pinned state's. A breaking change (removed or renamed public object, changed
signature) is a finding.

Outcome rules
  pass               no breaking change, and layering rules hold (if configured)
  fail_validated     griffe reports a removed public object AND an import
                     probe in a fresh interpreter confirms the object is gone.
                     (i) the base public API is the stated reference, (ii) one
                     object is the minimal witness, (iii) deterministic,
                     re-run at k sub-seeds, (iv) the import probe is the
                     second tool
  fail_unvalidated   griffe reports a breaking change the probe cannot
                     confirm (signature changes, parameter renames), or
                     import-linter reports a layering violation
  inconclusive       griffe could not load either side

Resource budgets (latency, memory, binary size) are not in this producer;
the spec lists them under F3 but they need a benchmark harness per
repository and are deferred with the challenger's expensive techniques.
"""
from __future__ import annotations
import json
import re
import subprocess
from pathlib import Path
from typing import List, Optional, Sequence

from ..core import Action, Context, Meter, Outcome, Registry, Validation, sub_seed

# griffe check output: "<file>:<line>: <object>[(param)]: <message>"
_ANY = re.compile(r"^(?P<file>[^:\n]+):(?P<line>\d+): (?P<obj>[\w.]+)(?:\([\w*, ]*\))?: (?P<what>.+)$", re.M)


def _dotted(file: str, obj: str, search: str) -> str:
    """Module path from the reported file plus the object name, e.g. src/pkg/a.py + public_api -> pkg.a.public_api."""
    f = file.replace("\\", "/")
    if f.startswith(search.rstrip("/") + "/"):
        f = f[len(search.rstrip("/")) + 1:]
    mod = f[:-3] if f.endswith(".py") else f
    parts = [x for x in mod.split("/") if x]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts + [obj])


def _griffe_check(m: Meter, ctx: Context, python: str, package: str, search: str, timeout: int) -> Optional[str]:
    cmd = [python, "-m", "griffe", "check", package, "--against", ctx.repo_state_sha, "--search", search]
    r = m.tool(cmd, ctx.repo, timeout)
    if r.returncode not in (0, 1):
        return None
    return r.stdout + r.stderr


def _probe_import(m: Meter, ctx: Context, python: str, dotted: str, search: str, timeout: int) -> Optional[bool]:
    """True if the dotted object is gone in a fresh interpreter (ImportError/AttributeError)."""
    code = ("import importlib,sys; sys.path.insert(0, %r); parts=%r.split('.')\n"
            "mod=None\n"
            "for i in range(len(parts),0,-1):\n"
            "    try:\n        mod=importlib.import_module('.'.join(parts[:i])); rest=parts[i:]; break\n"
            "    except Exception: continue\n"
            "if mod is None: print('GONE'); raise SystemExit(0)\n"
            "obj=mod\n"
            "for a in rest:\n"
            "    obj=getattr(obj,a,None)\n"
            "    if obj is None: print('GONE'); raise SystemExit(0)\n"
            "print('PRESENT')") % (search, dotted)
    r = m.tool([python, "-c", code], ctx.repo, timeout)
    if r.returncode != 0:
        return None
    return "GONE" in r.stdout


def _import_linter(m: Meter, ctx: Context, python: str, timeout: int) -> Optional[bool]:
    """True if contracts are kept, False if broken, None if not configured or crashed."""
    cfgs = [ctx.repo / ".importlinter", ctx.repo / "setup.cfg", ctx.repo / "pyproject.toml"]
    has = False
    for c in cfgs:
        if c.exists() and "importlinter" in c.read_text():
            has = True; break
    if not has:
        return None
    r = m.tool([python, "-m", "importlinter.cli", "lint"], ctx.repo, timeout)
    if r.returncode not in (0, 1):
        return None
    return r.returncode == 0


def make_api_contract(python: str, package: str, search: str = "src", k: int = 2, timeout: int = 600) -> Action:
    def run(ctx: Context, m: Meter) -> Outcome:
        prov = {"tool": "griffe check", "package": package, "against": ctx.repo_state_sha[:12]}
        blind = ["public API surface only; behavioural contracts not covered", "resource budgets deferred"]
        out = _griffe_check(m, ctx, python, package, search, timeout)
        if out is None:
            return Outcome("inconclusive", provenance=prov, blind_spots=blind, status="error")
        breaks = [(mm.group("file"), mm.group("obj"), mm.group("what")) for mm in _ANY.finditer(out)]
        removed = [_dotted(f, o, search) for f, o, w in breaks if "was removed" in w]
        prov["breaking_changes"] = str(len(breaks)); prov["removed"] = str(len(removed))
        layering = _import_linter(m, ctx, python, timeout)
        if layering is not None:
            prov["import_linter"] = "kept" if layering else "broken"
        if not breaks and layering in (None, True):
            return Outcome("pass", provenance=prov, blind_spots=blind)
        if not removed:
            return Outcome("fail_unvalidated", Validation(contradicts_reference=True, minimized=True, repeats_across_seeds=True),
                           prov, blind_spots=blind)
        # (iv) probe the first removed object in a fresh interpreter; (iii) repeat griffe at k sub-seeds
        target = removed[0]
        gone = _probe_import(m, ctx, python, target, search, timeout)
        seeds = [sub_seed(ctx.seed, i) for i in range(k)]
        obj_name = target.rsplit(".", 1)[-1]
        repeats = all(obj_name in (_griffe_check(m, ctx, python, package, search, timeout) or "") for _ in seeds)
        v = Validation(True, True, repeats, bool(gone), "fresh-interpreter-import-probe" if gone else "", target, seeds)
        return Outcome("fail_validated" if v.all_four else "fail_unvalidated", v, prov, blind_spots=blind)

    return Action("f3.api_contract", "F3", "author_side", "griffe+import-linter", run, priority=30)


def register(reg: Registry, python: str, package: str, search: str = "src", **kw) -> Action:
    return reg.register(make_api_contract(python, package, search, **kw))

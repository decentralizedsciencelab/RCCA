"""Challenger targets: public callables whose body changed in the candidate relative
to the pinned state, with their parameters and annotations.

Kinds
  function      module-level `def`
  staticmethod  `@staticmethod` inside a class; called as Class.method(...)
  classmethod   `@classmethod`; called as Class.method(...)
  method        instance method; needs a receiver built from the class
                constructor's parameters (`ctor_params`), or a factory

Async functions, nested functions, dunder methods, and callables whose only
parameters are *args/**kwargs are not targets.
"""
from __future__ import annotations
import ast
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..diffutil import changed_files


@dataclass(frozen=True)
class Param:
    name: str
    annotation: Optional[str]      # unparsed annotation text, or None
    has_default: bool
    kind: str                      # positional | keyword_only


@dataclass(frozen=True)
class Target:
    module: str                    # importable dotted module
    qualname: str                  # "f" or "Class.method"
    file: str                      # path relative to the repo root
    line: int
    params: Tuple[Param, ...]      # excluding self/cls
    status: str                    # changed | added
    kind: str = "function"         # function | method | staticmethod | classmethod
    cls: Optional[str] = None      # class name for methods
    ctor_params: Tuple[Param, ...] = ()   # constructor parameters for instance methods
    end_line: int = 0              # last line of the body (0 when unknown)

    @property
    def required(self) -> List[Param]:
        return [p for p in self.params if not p.has_default]

    @property
    def needs_receiver(self) -> bool:
        return self.kind == "method"


def module_name(rel_file: str, search: str) -> Optional[str]:
    f = rel_file.replace("\\", "/")
    s = search.rstrip("/") + "/"
    if not f.startswith(s) or not f.endswith(".py"):
        return None
    parts = f[len(s):-3].split("/")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) if parts else None


def _signature(fn: ast.FunctionDef, drop_first: bool = False) -> Optional[Tuple[Param, ...]]:
    a = fn.args
    pos = list(a.posonlyargs) + list(a.args)
    if drop_first and pos:
        pos = pos[1:]
    if a.vararg or a.kwarg:
        if not (pos or a.kwonlyargs):
            return None
    params: List[Param] = []
    n_def = len(a.defaults)
    for i, arg in enumerate(pos):
        # defaults align to the end of the (undropped) positional list
        idx = i + (1 if drop_first else 0)
        total = len(pos) + (1 if drop_first else 0)
        has_def = idx >= total - n_def
        params.append(Param(arg.arg, ast.unparse(arg.annotation) if arg.annotation else None, has_def, "positional"))
    for arg, d in zip(a.kwonlyargs, a.kw_defaults):
        params.append(Param(arg.arg, ast.unparse(arg.annotation) if arg.annotation else None, d is not None, "keyword_only"))
    return tuple(params)


def _dump(fn: ast.AST) -> str:
    return ast.dump(fn, include_attributes=False)


def _decorators(fn: ast.FunctionDef) -> List[str]:
    return [ast.unparse(d) for d in fn.decorator_list]


def _callables(src: str) -> Dict[str, dict]:
    """{qualname: {"node", "kind", "cls", "ctor"}} for public callables in a module."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return {}
    out: Dict[str, dict] = {}
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and not n.name.startswith("_"):
            out[n.name] = {"node": n, "kind": "function", "cls": None, "ctor": None}
        elif isinstance(n, ast.ClassDef) and not n.name.startswith("_"):
            ctor = next((m for m in n.body if isinstance(m, ast.FunctionDef) and m.name == "__init__"), None)
            for m in n.body:
                if not isinstance(m, ast.FunctionDef) or m.name.startswith("_"):
                    continue
                decs = _decorators(m)
                kind = "staticmethod" if "staticmethod" in decs else ("classmethod" if "classmethod" in decs else "method")
                if any(d not in ("staticmethod", "classmethod") for d in decs):
                    continue                       # properties, cached_property, abstractmethod: not callable as-is
                out[f"{n.name}.{m.name}"] = {"node": m, "kind": kind, "cls": n.name, "ctor": ctor}
    return out


def changed_functions(repo: Path, base_sha: str, package_root: str, search: str) -> List[Target]:
    out: List[Target] = []
    for f in changed_files(repo, base_sha):
        if not (f.endswith(".py") and f.startswith(package_root)):
            continue
        mod = module_name(f, search)
        p = repo / f
        if mod is None or not p.exists() or Path(f).name.startswith("test"):
            continue
        cand = _callables(p.read_text())
        r = subprocess.run(["git", "show", f"{base_sha}:{f}"], cwd=repo, capture_output=True, text=True)
        base = _callables(r.stdout) if r.returncode == 0 else {}
        for qn, info in cand.items():
            fn = info["node"]
            kind = info["kind"]
            sig = _signature(fn, drop_first=(kind in ("method", "classmethod")))
            if sig is None:
                continue
            if qn not in base:
                status = "added"
            elif _dump(base[qn]["node"]) != _dump(fn):
                status = "changed"
            else:
                continue
            ctor = ()
            if kind == "method":
                if info["ctor"] is not None:
                    cp = _signature(info["ctor"], drop_first=True)
                    if cp is None:
                        continue
                    ctor = cp
            out.append(Target(mod, qn, f, fn.lineno, sig, status, kind, info["cls"], ctor,
                              getattr(fn, "end_lineno", 0) or 0))
    return out

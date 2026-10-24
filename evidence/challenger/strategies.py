"""Hypothesis strategies from parameter annotations. JSON-able values only, so
inputs cross the executor boundary and land in the witness file verbatim."""
from __future__ import annotations
import re
from typing import Any, Dict, Optional, Tuple

from hypothesis import strategies as st

from . import harness
from .targets import Param, Target

_SMALL = dict(max_size=12)


def _type_name(ann: str) -> str:
    """Last component of a (possibly qualified) annotation, e.g. 'flask.Flask' -> 'Flask'."""
    a = re.sub(r"^(typing\.|t\.|builtins\.)", "", ann.replace(" ", ""))
    a = a[9:-1] if a.startswith("Optional[") else a
    return a.rsplit(".", 1)[-1]


def _scalar(max_size: int = 40):
    return st.one_of(st.none(), st.booleans(), st.integers(min_value=-10_000, max_value=10_000),
                     st.floats(allow_nan=False, allow_infinity=False, width=32),
                     st.text(max_size=max_size), st.binary(max_size=max_size))


def generic():
    return st.one_of(_scalar(), st.lists(_scalar(16), **_SMALL),
                     st.dictionaries(st.text(max_size=8), _scalar(16), **_SMALL))


def for_annotation(ann: Optional[str]):
    """Strategy for an annotation string; falls back to `generic()`."""
    if not ann:
        return generic()
    if _type_name(ann) in harness.FACTORIES:          # per-repository factory for a non-JSON type
        name = _type_name(ann)
        tok = st.integers(min_value=0, max_value=len(harness.FACTORIES[name]) - 1).map(lambda v: harness.factory_token(name, v))
        return st.one_of(st.none(), tok) if ann.replace(" ", "").startswith(("Optional[", "typing.Optional[")) else tok
    a = ann.replace(" ", "")
    a = re.sub(r"^(typing\.|t\.|builtins\.)", "", a)
    opt = re.match(r"^Optional\[(.+)\]$", a)
    if opt:
        return st.one_of(st.none(), for_annotation(opt.group(1)))
    if "|" in a and "[" not in a:
        return st.one_of(*[for_annotation(p) for p in a.split("|")])
    union = re.match(r"^Union\[(.+)\]$", a)
    if union:
        return st.one_of(*[for_annotation(p) for p in _split_top(union.group(1))])
    table = {
        "int": st.integers(min_value=-10_000, max_value=10_000),
        "float": st.floats(allow_nan=False, allow_infinity=False, width=32),
        "bool": st.booleans(), "str": st.text(max_size=40), "bytes": st.binary(max_size=40),
        "None": st.none(), "NoneType": st.none(), "Any": generic(), "object": generic(),
    }
    if a in table:
        return table[a]
    m = re.match(r"^(list|List|Sequence|Iterable|Collection|MutableSequence)\[(.+)\]$", a)
    if m:
        return st.lists(for_annotation(m.group(2)), **_SMALL)
    m = re.match(r"^(tuple|Tuple)\[(.+)\]$", a)
    if m:
        inner = _split_top(m.group(2))
        if len(inner) == 2 and inner[1] == "...":
            return st.lists(for_annotation(inner[0]), **_SMALL)
        return st.tuples(*[for_annotation(p) for p in inner]).map(list)
    m = re.match(r"^(dict|Dict|Mapping|MutableMapping)\[(.+)\]$", a)
    if m:
        inner = _split_top(m.group(2))
        if len(inner) == 2:
            return st.dictionaries(st.text(max_size=8), for_annotation(inner[1]), **_SMALL)
    if a in ("list", "List", "Sequence", "Iterable"):
        return st.lists(_scalar(16), **_SMALL)
    if a in ("dict", "Dict", "Mapping"):
        return st.dictionaries(st.text(max_size=8), _scalar(16), **_SMALL)
    if a.startswith("Literal["):
        vals = []
        for p in _split_top(a[8:-1]):
            try:
                vals.append(eval(p, {}, {}))           # literals only
            except Exception:
                pass
        if vals:
            return st.sampled_from(vals)
    return generic()


def _split_top(s: str):
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur); cur = ""
        else:
            cur += ch
    if cur:
        out.append(cur)
    return [x.strip() for x in out]


def _params_strategy(params):
    """Strategy producing (args, kwargs): every positional parameter, required
    keyword-only parameters, optional ones half the time."""
    pos = [p for p in params if p.kind == "positional"]
    kw = [p for p in params if p.kind == "keyword_only"]
    arg_strats = [for_annotation(p.annotation) for p in pos]

    def build(args, kwvals):
        kwargs = {}
        for p, (present, v) in zip(kw, kwvals):
            if not p.has_default or present:
                kwargs[p.name] = v
        return list(args), kwargs

    kw_strats = [st.tuples(st.booleans(), for_annotation(p.annotation)) for p in kw]
    return st.builds(build, st.tuples(*arg_strats) if arg_strats else st.just(()),
                     st.tuples(*kw_strats) if kw_strats else st.just(()))


def receiver_factory(t: Target) -> Optional[str]:
    """Factory-table name for the target's class, when the class itself has one
    (the receiver is then built by the factory rather than the constructor)."""
    if t.kind != "method" or not t.cls:
        return None
    name = t.cls.rsplit(".", 1)[-1]
    return name if name in harness.FACTORIES else None


def ctor_constructible(t: Target) -> bool:
    return not any(not p.has_default for p in t.ctor_params)


def runnable(t: Target) -> bool:
    """A target is runnable when every required parameter has a JSON-able
    annotation, a factory, or no annotation at all, and, for an instance
    method, the receiver can be built: from the class's own factory or from
    constructor parameters that satisfy the same condition."""
    def ok(p: Param) -> bool:
        if p.has_default or not p.annotation:
            return True
        name = _type_name(p.annotation)
        if name in harness.FACTORIES:
            return True
        a = re.sub(r"^(typing\.|t\.|builtins\.)", "", p.annotation.replace(" ", ""))
        return bool(re.match(r"^(Optional\[)?(int|float|bool|str|bytes|None|Any|object|list|List|dict|Dict|tuple|Tuple|"
                             r"Sequence|Iterable|Mapping|Literal)(\[.*\])?\]?$", a)) or "|" in a
    if not all(ok(p) for p in t.params):
        return False
    if t.kind != "method" or receiver_factory(t):
        return True
    return all(ok(p) for p in t.ctor_params)


def for_receiver(t: Target):
    """Receiver strategy for an instance method: a variant of the class's
    factory when the table has one, else (ctor_args, ctor_kwargs)."""
    fac = receiver_factory(t)
    if fac:
        return st.integers(min_value=0, max_value=len(harness.FACTORIES[fac]) - 1).map(lambda v: harness.factory_token(fac, v))
    return _params_strategy(t.ctor_params).map(lambda ak: (list(ak[0]), dict(ak[1])))


def for_target(t: Target):
    """Strategy producing (receiver, args, kwargs). `receiver` is None for
    functions, static and class methods; for instance methods it is a factory
    token or (ctor_args, ctor_kwargs)."""
    call = _params_strategy(t.params)
    if t.kind == "method":
        return st.tuples(for_receiver(t), call).map(lambda rc: (rc[0], rc[1][0], rc[1][1]))
    return call.map(lambda ak: (None, ak[0], ak[1]))


def fuzzable(t: Target) -> bool:
    """Targets the byte-level fuzzer can drive: scalar parameters, and for
    instance methods a constructor with no required parameters."""
    ok = {"int", "float", "bool", "str", "bytes", None}
    def norm(a):
        return None if a is None else re.sub(r"^(typing\.|t\.|builtins\.)", "", a.replace(" ", ""))
    if t.kind == "method" and not (receiver_factory(t) or ctor_constructible(t)):
        return False
    return all(norm(p.annotation) in ok or (norm(p.annotation) or "").startswith("Optional[str") for p in t.params if not p.has_default)

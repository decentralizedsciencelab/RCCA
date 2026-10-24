"""Challenger, differential execution (`ch.differential`, family F1, in-house).

Inputs are recorded, not generated: the pinned state's own test suite runs
once in the base worktree with a tracer that captures the JSON-able arguments
every changed callable is called with. An argument of a type in the factory
table is recorded as a factory token and rebuilt at replay, once per variant.
Instance methods are replayed on a receiver built from an argument-free
constructor. Those real workloads are replayed in both trees. A divergence is
minimized by the harness (criterion ii).

This is the cheapest challenger when the suite is fast, and the one whose
inputs are closest to how the code is actually used.
"""
from __future__ import annotations
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Iterator, List, Sequence, Tuple

from ..core import Action, Context, Meter, Outcome, Registry
from .common import ChallengerConfig, run_technique
from .harness import FACTORIES, Divergence, Pair, _tree_env, factory_token, is_factory_token
from .strategies import ctor_constructible, receiver_factory
from .targets import Target

TRACER = Path(__file__).with_name("_tracer.py")


def _decode(x):
    import base64
    if isinstance(x, dict):
        if set(x.keys()) == {"__bytes__"}:
            return base64.b64decode(x["__bytes__"])
        if is_factory_token(x):
            return x                                  # stays a token; the executor builds it
        return {k: _decode(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_decode(v) for v in x]
    return x


def _factory_names(x, acc):
    if is_factory_token(x):
        acc.add(x["__factory__"])
    elif isinstance(x, dict):
        for v in x.values():
            _factory_names(v, acc)
    elif isinstance(x, list):
        for v in x:
            _factory_names(v, acc)
    return acc


def _with_variant(x, v):
    if is_factory_token(x):
        return factory_token(x["__factory__"], min(v, len(FACTORIES.get(x["__factory__"], [""])) - 1))
    if isinstance(x, dict):
        return {k: _with_variant(val, v) for k, val in x.items()}
    if isinstance(x, list):
        return [_with_variant(val, v) for val in x]
    return x


def replay_inputs(rec: dict):
    """(args, kwargs) pairs for one recorded call: one per factory variant when
    the call carries factory tokens, otherwise the call itself."""
    args, kwargs = _decode(rec["args"]), _decode(rec["kwargs"])
    names = _factory_names([args, kwargs], set())
    if not names:
        return [(args, kwargs)]
    n = max(len(FACTORIES.get(nm, [""])) for nm in names)
    return [(_with_variant(args, v), _with_variant(kwargs, v)) for v in range(n)]


def replayable(t: Target) -> bool:
    """Functions, static and class methods, and instance methods whose class
    has a factory or whose constructor needs no arguments."""
    return t.kind != "method" or bool(receiver_factory(t)) or ctor_constructible(t)


def receivers_for(t: Target):
    """Receivers to replay an instance method on: one per factory variant of
    the class, else the argument-free constructor; None for other kinds."""
    if t.kind != "method":
        return [None]
    fac = receiver_factory(t)
    if fac:
        return [factory_token(fac, v) for v in range(len(FACTORIES[fac]))]
    return [([], {})]


def record_inputs(python: str, base_tree: Path, search: str, test_args: Sequence[str], targets: List[Target],
                  meter, timeout: int) -> List[dict]:
    tmp = Path(tempfile.mkdtemp(prefix="rcca-trace-"))
    try:
        shutil.copy(TRACER, tmp / "rcca_tracer.py")
        spec = tmp / "spec.json"; out = tmp / "trace.jsonl"
        spec.write_text(json.dumps([{"module": t.module, "qualname": t.qualname} for t in targets]))
        env = _tree_env(base_tree, search, 0)
        env["PYTHONPATH"] = f"{tmp}{os.pathsep}{env['PYTHONPATH']}"
        env["RCCA_TRACE_SPEC"] = str(spec); env["RCCA_TRACE_OUT"] = str(out)
        # no -x: a single environmental failure in the pinned state must not cut the recording short
        meter.tool([python, "-m", "pytest", *test_args, "-q", "--no-header", "-p", "no:cacheprovider", "-p", "rcca_tracer",
                    "--continue-on-collection-errors"], base_tree, timeout, env=env)
        if not out.exists():
            return []
        return [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def make_differential(cfg: ChallengerConfig, test_args: Sequence[str] = ("tests",), suite_timeout: int = 1200) -> Action:
    def run(ctx: Context, m: Meter) -> Outcome:
        state = {"recorded": 0}

        def search(targets: List[Target], pair: Pair, base_tree: Path, deadline: float) -> Iterator[Tuple[Divergence, bool]]:
            recs = record_inputs(cfg.python, base_tree, cfg.search, test_args, targets, m, suite_timeout)
            state["recorded"] = len(recs)
            by_key = {(t.module, t.qualname): t for t in targets}
            for r in recs:
                if time.monotonic() > deadline:
                    return
                t = by_key.get((r["module"], r["qualname"]))
                if t is None:
                    continue
                for receiver in receivers_for(t):
                    for args, kwargs in replay_inputs(r):
                        d = pair.diverges(t, args, kwargs, receiver)
                        if d is not None:
                            yield d, False

        out = run_technique(ctx, m, cfg, "differential-execution", "pytest-witness", search,
                            {"tool": "inhouse-differential", "inputs": "recorded from the pinned state's test run"},
                            target_filter=replayable)
        out.provenance["recorded_inputs"] = str(state["recorded"])
        if out.outcome == "pass" and state["recorded"] == 0 and out.provenance.get("targets_searched", "0") != "0":
            out.outcome = "inconclusive"; out.status = "error"
            out.provenance["reason"] = "no JSON-able or factory-typed call to a changed callable was recorded"
        return out

    return Action("ch.differential", "F1", "challenger", "inhouse-differential", run, priority=62)


def register(reg: Registry, cfg: ChallengerConfig, **kw) -> Action:
    return reg.register(make_differential(cfg, **kw))

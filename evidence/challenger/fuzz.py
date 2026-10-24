"""Challenger, coverage-guided fuzzing (`ch.fuzz`, family F1, atheris).

A generated driver fuzzes one changed function in the candidate tree under
libFuzzer with a time budget and a seed. The driver never aborts on an
exception; it logs the input and keeps exploring, and also samples inputs that
did not raise. The harness then replays every logged input in both trees, so
the oracle stays the pre-change behaviour and a crash that the base also
raises is not a finding. Minimization is the harness's ddmin (criterion ii).

Requires `atheris` in the repository's interpreter. If it is missing the
producer is `inconclusive`: the search never ran.
"""
from __future__ import annotations
import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

from ..core import Action, Context, Meter, Outcome, Registry
from .common import ChallengerConfig, run_technique
from .harness import FACTORIES, Divergence, Pair, _tree_env, factory_token
from .strategies import fuzzable, receiver_factory
from .targets import Target

_DRIVER = '''\
import atheris, base64, importlib, json, sys
MODULE, QUAL = {module!r}, {qualname!r}
KINDS = {kinds!r}
IS_METHOD = {is_method!r}
RECEIVER_EXPR = {receiver_expr!r}          # factory expression for the receiver, or None for the bare constructor
CRASH, SAMPLE = {crash!r}, {sample!r}
with atheris.instrument_imports():
    mod = importlib.import_module(MODULE)
fn = mod
for part in QUAL.split("."):
    fn = getattr(fn, part)
if IS_METHOD:
    cls = mod
    for part in QUAL.rsplit(".", 1)[0].split("."):
        cls = getattr(cls, part)
    meth = QUAL.rsplit(".", 1)[1]
    def fn(*a):
        inst = eval(RECEIVER_EXPR, {{"importlib": importlib}}) if RECEIVER_EXPR else cls()
        return getattr(inst, meth)(*a)
state = {{"n": 0, "crashes": 0, "samples": 0}}


def enc(x):
    if isinstance(x, bytes):
        return {{"__bytes__": base64.b64encode(x).decode()}}
    return x


def build(fdp):
    args = []
    for k in KINDS:
        if k == "bytes":
            args.append(fdp.ConsumeBytes(fdp.ConsumeIntInRange(0, 64)))
        elif k == "int":
            args.append(fdp.ConsumeInt(4))
        elif k == "float":
            args.append(fdp.ConsumeRegularFloat())
        elif k == "bool":
            args.append(fdp.ConsumeBool())
        else:
            args.append(fdp.ConsumeUnicodeNoSurrogates(fdp.ConsumeIntInRange(0, 64)))
    return args


def log(path, args):
    with open(path, "a") as f:
        f.write(json.dumps([enc(a) for a in args]) + "\\n")


def TestOneInput(data):
    fdp = atheris.FuzzedDataProvider(data)
    args = build(fdp)
    state["n"] += 1
    try:
        fn(*args)
    except Exception:
        if state["crashes"] < 200:
            state["crashes"] += 1; log(CRASH, args)
    else:
        if state["n"] % 50 == 0 and state["samples"] < 400:
            state["samples"] += 1; log(SAMPLE, args)


atheris.Setup(sys.argv, TestOneInput)
atheris.Fuzz()
'''


def _kinds(t: Target) -> List[str]:
    out = []
    for p in t.params:
        if p.has_default:
            continue
        a = (p.annotation or "str").replace(" ", "")
        a = re.sub(r"^(typing\.|t\.|builtins\.)", "", a)
        a = a[9:-1] if a.startswith("Optional[") else a
        out.append(a if a in ("int", "float", "bool", "bytes") else "str")
    return out


def _decode(x):
    import base64
    if isinstance(x, dict) and set(x.keys()) == {"__bytes__"}:
        return base64.b64decode(x["__bytes__"])
    if isinstance(x, list):
        return [_decode(v) for v in x]
    return x


def atheris_available(python: str, meter: Meter) -> bool:
    r = meter.tool([python, "-c", "import atheris"], Path("."), 60)
    return r.returncode == 0


def make_fuzz(cfg: ChallengerConfig, time_per_target_s: float = 20,
              time_per_target_ci_equiv: Optional[float] = None) -> Action:
    """`time_per_target_s` caps libFuzzer per target, in seconds on this machine; evidence.run derives it
    from the config's fuzz_time_per_target_ci_equiv, which `time_per_target_ci_equiv` records."""
    per_target = {"time_per_target_s": str(time_per_target_s)}
    if time_per_target_ci_equiv is not None:
        per_target["time_per_target_ci_equiv"] = str(time_per_target_ci_equiv)

    def run(ctx: Context, m: Meter) -> Outcome:
        if not atheris_available(cfg.python, m):
            return Outcome("inconclusive", provenance={"technique": "fuzzing", "tool": "atheris", "reason": "atheris not importable",
                                                       **cfg.budget_provenance(), **per_target},
                           status="error", search_budget_used=0.0)

        def search(targets: List[Target], pair: Pair, base_tree: Path, deadline: float) -> Iterator[Tuple[Divergence, bool]]:
            for t in targets:
                if time.monotonic() > deadline:
                    return
                tmp = Path(tempfile.mkdtemp(prefix="rcca-fuzz-"))
                try:
                    crash, sample = tmp / "crash.jsonl", tmp / "sample.jsonl"
                    (tmp / "corpus").mkdir()
                    drv = tmp / "driver.py"
                    fac = receiver_factory(t)
                    drv.write_text(_DRIVER.format(module=t.module, qualname=t.qualname, kinds=_kinds(t),
                                                  is_method=(t.kind == "method"),
                                                  receiver_expr=(FACTORIES[fac][0] if fac else None),
                                                  crash=str(crash), sample=str(sample)))
                    budget = int(max(5, min(time_per_target_s, deadline - time.monotonic())))
                    m.tool([cfg.python, str(drv), f"-max_total_time={budget}", f"-seed={ctx.seed % (2**31)}",
                            "-rss_limit_mb=2048", "-timeout=5", "-print_final_stats=0", str(tmp / "corpus")],
                           ctx.repo, budget + 60, env=_tree_env(ctx.repo, cfg.search, ctx.seed))
                    inputs = []
                    for f in (crash, sample):
                        if f.exists():
                            inputs += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
                    seen = set()
                    for args in inputs:
                        key = json.dumps(args, sort_keys=True)
                        if key in seen:
                            continue
                        seen.add(key)
                        recv = (factory_token(fac, 0) if fac else ([], {})) if t.kind == "method" else None
                        d = pair.diverges(t, _decode(args), {}, recv)
                        if d is not None:
                            yield d, False
                finally:
                    shutil.rmtree(tmp, ignore_errors=True)

        return run_technique(ctx, m, cfg, "coverage-guided-fuzzing", "pytest-witness", search,
                             {"tool": "atheris", **per_target}, target_filter=fuzzable)

    return Action("ch.fuzz", "F1", "challenger", "atheris", run, priority=61)


def register(reg: Registry, cfg: ChallengerConfig, **kw) -> Action:
    return reg.register(make_fuzz(cfg, **kw))

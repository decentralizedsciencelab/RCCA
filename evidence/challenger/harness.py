"""Shared challenger harness.

The oracle for every challenger technique is the pre-change behaviour: the
same call is executed in the base tree (a git worktree at the pinned state)
and in the candidate tree, and a divergence in return value or exception is a
counterexample against the named reference (criterion i). The harness then

  * discards divergences the base itself does not reproduce twice (a
    nondeterministic target is not evidence),
  * minimizes the witness, by ddmin on sequence inputs and shrinking on
    integers, unless the technique already shrank it (criterion ii),
  * re-executes the minimal witness in fresh executor pairs at k sub-seeds
    (criterion iii), and
  * confirms it with a second tool: a generated pytest file that asserts the
    base behaviour, which must fail in the candidate tree and pass in the base
    tree (criterion iv). The witness file is attached to the record.

Executors are long-lived subprocesses speaking JSON lines, one per tree, so a
call costs a round trip rather than an interpreter start.
"""
from __future__ import annotations
import base64
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..core import Meter, Validation, sub_seed
from .targets import Target

RUNNER = Path(__file__).with_name("_runner.py")


# ---------------------------------------------------------------------------
# input encoding
# ---------------------------------------------------------------------------

def encode(x: Any) -> Any:
    if isinstance(x, bytes):
        return {"__bytes__": base64.b64encode(x).decode()}
    if isinstance(x, (list, tuple)):
        return [encode(v) for v in x]
    if isinstance(x, dict):
        return {str(k): encode(v) for k, v in x.items()}
    return x


FACTORIES: Dict[str, List[str]] = {}     # per-repository factory table, set by configure()


def configure(factories: Optional[Dict[str, Any]]) -> None:
    """Register factory expressions for non-JSON parameter types, keyed by the
    annotation's type name. A value is one expression or a list of variant
    expressions (e.g. an application with default and with hardened settings);
    strategies sample a variant, and a token {"__factory__": name, "v": i}
    names the variant so a witness is exact."""
    FACTORIES.clear()
    for name, exprs in (factories or {}).items():
        FACTORIES[name] = [exprs] if isinstance(exprs, str) else list(exprs)


def factory_token(name: str, variant: int = 0) -> Dict[str, Any]:
    return {"__factory__": name, "v": int(variant)}


def is_factory_token(x: Any) -> bool:
    return isinstance(x, dict) and "__factory__" in x and set(x.keys()) <= {"__factory__", "v"}


# A receiver for an instance method is either (ctor_args, ctor_kwargs), from which
# the class named by the qualname prefix is instantiated, or a factory token for
# the class itself (e.g. an application object), built by the factory table.
Receiver = Any


def encode_receiver(receiver: Receiver) -> Any:
    if receiver is None:
        return None
    if is_factory_token(receiver):
        return dict(receiver)
    return {"args": encode(list(receiver[0])), "kwargs": encode(dict(receiver[1]))}


def _copy_receiver(receiver: Receiver) -> Receiver:
    if receiver is None or is_factory_token(receiver):
        return receiver
    return (list(receiver[0]), dict(receiver[1]))


def _tree_env(tree: Path, search: str, hashseed: int) -> Dict[str, str]:
    env = {k: v for k, v in os.environ.items()
           if not k.upper().endswith("_PROXY") and k.upper() not in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")}
    env["PYTHONPATH"] = str(tree / search)
    env["PYTHONHASHSEED"] = str(hashseed % (2 ** 32))
    env["NO_NETWORK"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["RCCA_FACTORIES"] = json.dumps(FACTORIES)
    return env


# ---------------------------------------------------------------------------
# base worktree
# ---------------------------------------------------------------------------

class Worktree:
    """A detached git worktree of the pinned state, removed on exit."""

    def __init__(self, repo: Path, sha: str):
        self.repo, self.sha = repo, sha
        self.path: Optional[Path] = None

    def __enter__(self) -> Path:
        self.path = Path(tempfile.mkdtemp(prefix="rcca-base-"))
        subprocess.run(["git", "worktree", "add", "--detach", "--force", str(self.path), self.sha], cwd=self.repo,
                       capture_output=True, text=True, check=True)
        return self.path

    def __exit__(self, *exc):
        if self.path:
            subprocess.run(["git", "worktree", "remove", "--force", str(self.path)], cwd=self.repo, capture_output=True)
            shutil.rmtree(self.path, ignore_errors=True)
            subprocess.run(["git", "worktree", "prune"], cwd=self.repo, capture_output=True)
        return False


# ---------------------------------------------------------------------------
# executor
# ---------------------------------------------------------------------------

@dataclass
class Result:
    status: str                    # ok | exc | timeout | import_error | dead
    value: Optional[str] = None
    exc_type: Optional[str] = None
    exc_msg: Optional[str] = None

    def key(self) -> Tuple[str, Optional[str]]:
        """What two trees are compared on: the value, or the exception type."""
        if self.status == "ok":
            return ("ok", self.value)
        if self.status == "exc":
            return ("exc", self.exc_type)
        return (self.status, None)


class Executor:
    def __init__(self, python: str, tree: Path, search: str, meter: Meter, hashseed: int = 0, call_timeout: int = 5):
        self.python, self.tree, self.search, self.m = python, tree, search, meter
        self.hashseed, self.call_timeout = hashseed, call_timeout
        self._p: Optional[subprocess.Popen] = None
        self._n = 0
        self.calls = 0

    def _start(self) -> None:
        self.m.tool_calls += 1
        self._p = subprocess.Popen([self.python, "-s", str(RUNNER)], cwd=self.tree, stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                                   env=_tree_env(self.tree, self.search, self.hashseed))

    def call(self, t: Target, args: Sequence[Any], kwargs: Dict[str, Any], receiver: Receiver = None) -> Result:
        if self._p is None or self._p.poll() is not None:
            self._start()
        self._n += 1; self.calls += 1
        req = {"id": self._n, "module": t.module, "qualname": t.qualname, "args": encode(list(args)),
               "kwargs": encode(dict(kwargs)), "timeout": self.call_timeout}
        if receiver is not None:
            req["receiver"] = encode_receiver(receiver)
        try:
            self._p.stdin.write(json.dumps(req) + "\n"); self._p.stdin.flush()
            line = self._p.stdout.readline()
        except (BrokenPipeError, OSError):
            line = ""
        if not line:
            self.close()
            return Result("dead")
        r = json.loads(line)
        return Result(r["status"], r.get("value"), r.get("exc_type"), r.get("exc_msg"))

    def close(self) -> None:
        if self._p is not None:
            try:
                self._p.stdin.write(json.dumps({"op": "exit"}) + "\n"); self._p.stdin.flush()
            except Exception:              # noqa: BLE001
                pass
            try:
                self._p.wait(timeout=5)
            except Exception:              # noqa: BLE001
                self._p.kill()
            self._p = None


# ---------------------------------------------------------------------------
# divergence oracle
# ---------------------------------------------------------------------------

@dataclass
class Divergence:
    target: Target
    args: List[Any]
    kwargs: Dict[str, Any]
    base: Result
    cand: Result
    receiver: Receiver = None

    def describe(self) -> str:
        if self.receiver is None:
            recv = ""
        elif is_factory_token(self.receiver):
            recv = f"{self.target.cls}<{self.receiver['__factory__']} v{self.receiver.get('v', 0)}>."
        else:
            recv = f"{self.target.cls}({_short(*self.receiver)})."
        name = self.target.qualname.rsplit(".", 1)[-1] if self.receiver is not None else self.target.qualname
        return (f"{self.target.module}.{recv}{name}({_short(self.args, self.kwargs)}): "
                f"base={self.base.key()} candidate={self.cand.key()}")


def _short(args, kwargs, n=120) -> str:
    s = ", ".join([repr(a) for a in args] + [f"{k}={v!r}" for k, v in kwargs.items()])
    return s if len(s) <= n else s[:n] + "..."


class Pair:
    """Base and candidate executors over the same interpreter and search path."""

    def __init__(self, python: str, base_tree: Path, cand_tree: Path, search: str, meter: Meter,
                 hashseed: int = 0, call_timeout: int = 5):
        self.base = Executor(python, base_tree, search, meter, hashseed, call_timeout)
        self.cand = Executor(python, cand_tree, search, meter, hashseed, call_timeout)
        self.unstable: set = set()

    def diverges(self, t: Target, args: Sequence[Any], kwargs: Dict[str, Any],
                 receiver: Receiver = None) -> Optional[Divergence]:
        if (t.module, t.qualname) in self.unstable:
            return None
        b = self.base.call(t, args, kwargs, receiver)
        if b.status in ("import_error", "dead"):
            return None
        c = self.cand.call(t, args, kwargs, receiver)
        if c.status in ("dead",):
            return None
        if b.key() == c.key():
            return None
        if b.status == "timeout" or c.status == "timeout":
            return None
        # nondeterminism guard: the base must reproduce its own result
        b2 = self.base.call(t, args, kwargs, receiver)
        if b2.key() != b.key():
            self.unstable.add((t.module, t.qualname))
            return None
        return Divergence(t, list(args), dict(kwargs), b, c, _copy_receiver(receiver))

    def close(self) -> None:
        self.base.close(); self.cand.close()


# ---------------------------------------------------------------------------
# minimization (criterion ii) for techniques that do not shrink themselves
# ---------------------------------------------------------------------------

def _ddmin(seq, still_fails: Callable[[Any], bool], budget: int) -> Any:
    """Classic ddmin on a sequence (str, bytes, list), bounded by a call budget."""
    n = 2
    calls = 0
    while len(seq) >= 2 and calls < budget:
        chunk = max(1, len(seq) // n)
        reduced = False
        for i in range(0, len(seq), chunk):
            cand = seq[:i] + seq[i + chunk:]
            calls += 1
            if still_fails(cand):
                seq, n, reduced = cand, max(n - 1, 2), True
                break
        if not reduced:
            if n >= len(seq):
                break
            n = min(n * 2, len(seq))
    return seq


def minimize(div: Divergence, pair: Pair, budget: int = 200) -> Divergence:
    args = list(div.args)
    kwargs = dict(div.kwargs)

    def diverges_with(a, kw) -> bool:
        d = pair.diverges(div.target, a, kw, div.receiver)
        return d is not None and d.cand.key() == div.cand.key()

    def shrink_value(v, probe):
        if isinstance(v, (str, bytes, list)) and len(v) > 1:
            return _ddmin(v, probe, budget)
        if isinstance(v, bool) or v is None:
            return v
        if isinstance(v, int):
            cur = v
            for cand in [0, 1, -1]:
                if cand != cur and probe(cand):
                    return cand
            while abs(cur) > 1:
                nxt = cur // 2
                if probe(nxt):
                    cur = nxt
                else:
                    break
            return cur
        return v

    for i in range(len(args)):
        args[i] = shrink_value(args[i], lambda x, i=i: diverges_with(args[:i] + [x] + args[i + 1:], kwargs))
    for k in list(kwargs):
        kwargs[k] = shrink_value(kwargs[k], lambda x, k=k: diverges_with(args, {**kwargs, k: x}))
    final = pair.diverges(div.target, args, kwargs, div.receiver)
    return final if final is not None else div


# ---------------------------------------------------------------------------
# criteria iii and iv
# ---------------------------------------------------------------------------

def repeats(div: Divergence, python: str, base_tree: Path, cand_tree: Path, search: str, meter: Meter,
            seed: int, k: int, call_timeout: int = 5) -> Tuple[bool, List[int]]:
    seeds = [sub_seed(seed, i) for i in range(k)]
    for s in seeds:
        pair = Pair(python, base_tree, cand_tree, search, meter, hashseed=s, call_timeout=call_timeout)
        try:
            d = pair.diverges(div.target, div.args, div.kwargs, div.receiver)
        finally:
            pair.close()
        if d is None or d.cand.key() != div.cand.key() or d.base.key() != div.base.key():
            return False, seeds
    return True, seeds


_WITNESS = '''\
"""RCCA challenger witness. Asserts the pre-change behaviour of a function on one
input. Generated; must fail in the candidate tree and pass in the base tree."""
import base64, importlib, json, re

TARGET = ({module!r}, {qualname!r})
ARGS = json.loads({args!r})
KWARGS = json.loads({kwargs!r})
RECEIVER = json.loads({receiver!r})          # null, {{"args": [...], "kwargs": {{...}}}}, or a factory token, for an instance method
FACTORIES = json.loads({factories!r})        # expressions for non-JSON parameters, evaluated at test time
EXPECTED = {expected!r}
_ADDR = re.compile(r"0x[0-9a-fA-F]+")


def decode(x):
    if isinstance(x, dict):
        if set(x.keys()) == {{"__bytes__"}}:
            return base64.b64decode(x["__bytes__"])
        if "__factory__" in x and set(x.keys()) <= {{"__factory__", "v"}}:
            exprs = FACTORIES[x["__factory__"]]
            return eval(exprs[min(x.get("v", 0), len(exprs) - 1)], {{"importlib": importlib}})
        return {{k: decode(v) for k, v in x.items()}}
    if isinstance(x, list):
        return [decode(v) for v in x]
    return x


def canon(v, depth=0):
    if depth > 6:
        return "<deep>"
    if v is None or isinstance(v, (bool, int, float, str)):
        if isinstance(v, float) and (v != v):
            return "nan"
        return json.dumps(v, sort_keys=True)
    if isinstance(v, bytes):
        return "b:" + base64.b64encode(v).decode()
    if isinstance(v, (list, tuple)):
        return ("L" if isinstance(v, list) else "T") + "[" + ",".join(canon(x, depth + 1) for x in v) + "]"
    if isinstance(v, (set, frozenset)):
        return "S[" + ",".join(sorted(canon(x, depth + 1) for x in v)) + "]"
    if isinstance(v, dict):
        return "D{{" + ",".join(f"{{canon(k, depth + 1)}}:{{canon(val, depth + 1)}}" for k, val in sorted(v.items(), key=lambda kv: repr(kv[0]))) + "}}"
    try:
        r = repr(v)
    except Exception:
        r = f"<{{type(v).__name__}}>"
    return "R:" + _ADDR.sub("0x", r)[:2000]


def test_rcca_witness():
    mod = importlib.import_module(TARGET[0])
    fn = mod
    if RECEIVER is not None:
        cls_name, meth = TARGET[1].rsplit(".", 1)
        if "__factory__" in RECEIVER:
            inst = decode(RECEIVER)
        else:
            for part in cls_name.split("."):
                fn = getattr(fn, part)
            inst = fn(*decode(RECEIVER["args"]), **decode(RECEIVER["kwargs"]))
        fn = getattr(inst, meth)
    else:
        for part in TARGET[1].split("."):
            fn = getattr(fn, part)
    try:
        got = ("ok", canon(fn(*decode(ARGS), **decode(KWARGS))))
    except Exception as e:
        got = ("exc", type(e).__name__)
    assert list(got) == list(EXPECTED), f"got {{got}}, pre-change behaviour was {{EXPECTED}}"
'''


def witness_source(div: Divergence) -> str:
    recv = encode_receiver(div.receiver)
    return _WITNESS.format(module=div.target.module, qualname=div.target.qualname,
                           args=json.dumps(encode(div.args)), kwargs=json.dumps(encode(div.kwargs)),
                           receiver=json.dumps(recv), factories=json.dumps(FACTORIES),
                           expected=list(div.base.key()))


def confirm_with_pytest(div: Divergence, python: str, base_tree: Path, cand_tree: Path, search: str,
                        meter: Meter, timeout: int = 300) -> Tuple[bool, str, Dict[str, str]]:
    """Second tool. Returns (confirmed, witness source, provenance)."""
    src = witness_source(div)
    tmp = Path(tempfile.mkdtemp(prefix="rcca-witness-"))
    wf = tmp / "test_rcca_witness.py"
    wf.write_text(src)
    prov: Dict[str, str] = {}
    try:
        rcs = {}
        for label, tree in (("candidate", cand_tree), ("base", base_tree)):
            r = meter.tool([python, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider", "-p", "no:randomly",
                            "--rootdir", str(tmp), str(wf)], tree, timeout, env=_tree_env(tree, search, 0))
            rcs[label] = r.returncode
        prov["witness_rc_candidate"] = str(rcs["candidate"]); prov["witness_rc_base"] = str(rcs["base"])
        return rcs["candidate"] == 1 and rcs["base"] == 0, src, prov
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# assembling the validation block
# ---------------------------------------------------------------------------

@dataclass
class Finding:
    divergence: Divergence
    validation: Validation
    witness: str
    extra: Dict[str, str] = field(default_factory=dict)

    @property
    def validated(self) -> bool:
        return self.validation.all_four


def validate(div: Divergence, already_minimal: bool, pair: Pair, python: str, base_tree: Path, cand_tree: Path,
             search: str, meter: Meter, seed: int, k: int, confirmer: str, call_timeout: int = 5,
             minimize_budget: int = 200) -> Finding:
    d = div if already_minimal else minimize(div, pair, minimize_budget)
    rep, seeds = repeats(d, python, base_tree, cand_tree, search, meter, seed, k, call_timeout)
    confirmed, src, prov = (False, witness_source(d), {})
    if rep:
        confirmed, src, prov = confirm_with_pytest(d, python, base_tree, cand_tree, search, meter)
    v = Validation(contradicts_reference=True, minimized=True, repeats_across_seeds=rep,
                   second_tool_confirmed=confirmed, confirmer_id=confirmer if confirmed else "",
                   witness_ref=d.describe(), sub_seeds=seeds)
    return Finding(d, v, src, prov)


def artifact_for(f: Finding) -> Dict[str, str]:
    return {"name": "test_rcca_witness.py", "sha256": hashlib.sha256(f.witness.encode()).hexdigest(), "content": f.witness}

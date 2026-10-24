"""pytest plugin: record JSON-able call arguments of target functions during a
test run. Loaded with `-p rcca_tracer` from a temp directory on PYTHONPATH.

Env: RCCA_TRACE_SPEC  json file: [{"module": ..., "qualname": ...}, ...]
     RCCA_TRACE_OUT   jsonl output: {"module", "qualname", "args", "kwargs"}
     RCCA_FACTORIES   optional json {type name: expression or [expressions]}; an
                      argument whose class (or a base class) is named there is
                      recorded as {"__factory__": name} and rebuilt at replay
Instance methods are recorded without their receiver (args[0] dropped);
replay builds a receiver from a constructor that needs no arguments.
Standalone on purpose.
"""
import base64
import functools
import importlib
import json
import os

_CAP_PER_TARGET = 100
_records = {}
_seen = set()
_FACTORIES = json.loads(os.environ.get("RCCA_FACTORIES", "{}"))


def _factory_name(x):
    for klass in type(x).__mro__:
        if klass.__name__ in _FACTORIES:
            return klass.__name__
    return None


def _enc(x, depth=0):
    if depth > 5:
        raise ValueError("deep")
    if x is None or isinstance(x, (bool, int, str)):
        return x
    fname = _factory_name(x)
    if fname is not None:
        return {"__factory__": fname}
    if isinstance(x, float):
        if x != x or x in (float("inf"), float("-inf")):
            raise ValueError("nonfinite")
        return x
    if isinstance(x, bytes):
        return {"__bytes__": base64.b64encode(x).decode()}
    if isinstance(x, (list, tuple)):
        return [_enc(v, depth + 1) for v in x]
    if isinstance(x, dict):
        return {str(k): _enc(v, depth + 1) for k, v in x.items()}
    raise ValueError("unserializable")


def _wrap(module, qualname, fn, drop_first=False):
    key = (module, qualname)

    @functools.wraps(fn)
    def w(*args, **kwargs):
        try:
            if len(_records.get(key, [])) < _CAP_PER_TARGET:
                rec = {"module": module, "qualname": qualname, "args": _enc(list(args[1:] if drop_first else args)),
                       "kwargs": _enc(dict(kwargs))}
                sig = json.dumps(rec, sort_keys=True)
                if len(sig) < 20000 and sig not in _seen:
                    _seen.add(sig); _records.setdefault(key, []).append(rec)
        except Exception:
            pass
        return fn(*args, **kwargs)
    return w


def pytest_sessionstart(session):
    with open(os.environ["RCCA_TRACE_SPEC"]) as f:     # closed explicitly: suites that turn ResourceWarning into errors
        spec = json.load(f)
    for t in spec:
        try:
            mod = importlib.import_module(t["module"])
            owner, name = mod, t["qualname"]
            if "." in name:
                cls_name, name = name.rsplit(".", 1)
                owner = mod
                for part in cls_name.split("."):
                    owner = getattr(owner, part)
                raw = owner.__dict__.get(name)
                if isinstance(raw, staticmethod):
                    setattr(owner, name, staticmethod(_wrap(t["module"], t["qualname"], raw.__func__))); continue
                if isinstance(raw, classmethod):
                    setattr(owner, name, classmethod(_wrap(t["module"], t["qualname"], raw.__func__, drop_first=True))); continue
                if callable(raw) and not isinstance(raw, type):
                    setattr(owner, name, _wrap(t["module"], t["qualname"], raw, drop_first=True))
                continue
            obj = getattr(owner, name)
            setattr(owner, name, _wrap(t["module"], t["qualname"], obj))
        except Exception:
            continue


def pytest_sessionfinish(session, exitstatus):
    with open(os.environ["RCCA_TRACE_OUT"], "w") as f:
        for recs in _records.values():
            for r in recs:
                f.write(json.dumps(r) + "\n")

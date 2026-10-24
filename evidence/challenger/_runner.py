"""Executor process. Runs inside one tree (base or candidate) with that tree's
package first on PYTHONPATH. Standalone on purpose: it imports nothing from
the evidence package.

Protocol, one JSON object per line on stdin, one per line on stdout:
  in   {"id": n, "module": "pkg.mod", "qualname": "f", "args": [...], "kwargs": {...}, "timeout": 5}
  out  {"id": n, "status": "ok"|"exc"|"timeout"|"import_error", "value": "<canonical>",
        "exc_type": "...", "exc_msg": "..."}
Bytes travel as {"__bytes__": "<base64>"}. A value {"__factory__": "<name>",
"v": i} is built by evaluating variant i of the expressions registered for
<name> in the JSON env var RCCA_FACTORIES (per-repository factory table for
non-JSON parameters; a value there is one expression or a list of them). A
request may carry "receiver" for instance methods, either {"args": [...],
"kwargs": {...}} (the class named by the qualname prefix is instantiated with
those arguments) or a factory token (the instance is built by the factory
table); the method is called on the instance. Results are canonicalized:
JSON for JSON-able values, otherwise a repr with memory addresses scrubbed.
"""
import os
import base64
import importlib
import json
import re
import signal
import sys
import traceback

_ADDR = re.compile(r"0x[0-9a-fA-F]+")


_FACTORIES = json.loads(os.environ.get("RCCA_FACTORIES", "{}"))


def make_factory(name, variant=0):
    exprs = _FACTORIES[name]
    exprs = [exprs] if isinstance(exprs, str) else exprs
    expr = exprs[min(int(variant), len(exprs) - 1)]
    return eval(expr, {"__builtins__": __builtins__, "importlib": importlib})   # noqa: S307  configured expression


def decode(x):
    if isinstance(x, dict):
        if set(x.keys()) == {"__bytes__"}:
            return base64.b64decode(x["__bytes__"])
        if "__factory__" in x and set(x.keys()) <= {"__factory__", "v"}:
            return make_factory(x["__factory__"], x.get("v", 0))
        return {k: decode(v) for k, v in x.items()}
    if isinstance(x, list):
        return [decode(v) for v in x]
    return x


def canon(v, depth=0):
    """Canonical string for a return value."""
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
        return "D{" + ",".join(f"{canon(k, depth + 1)}:{canon(val, depth + 1)}" for k, val in sorted(v.items(), key=lambda kv: repr(kv[0]))) + "}"
    try:
        r = repr(v)
    except Exception:                 # noqa: BLE001
        r = f"<{type(v).__name__}>"
    return "R:" + _ADDR.sub("0x", r)[:2000]


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def resolve(module, qualname):
    mod = importlib.import_module(module)
    obj = mod
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def main():
    signal.signal(signal.SIGALRM, _alarm)
    cache = {}
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        if req.get("op") == "exit":
            break
        out = {"id": req["id"]}
        key = (req["module"], req["qualname"])
        try:
            fn = cache.get(key)
            if fn is None:
                fn = resolve(*key); cache[key] = fn
        except Exception as e:            # noqa: BLE001
            out.update(status="import_error", exc_type=type(e).__name__, exc_msg=str(e)[:300])
            sys.stdout.write(json.dumps(out) + "\n"); sys.stdout.flush(); continue
        signal.alarm(int(req.get("timeout", 5)) or 5)
        try:
            args = decode(req.get("args", [])); kwargs = decode(req.get("kwargs", {}))
            callee = fn
            if req.get("receiver") is not None:
                rc = req["receiver"]
                if "__factory__" in rc:
                    inst = decode(rc)
                else:
                    cls = resolve(req["module"], req["qualname"].rsplit(".", 1)[0])
                    inst = cls(*decode(rc.get("args", [])), **decode(rc.get("kwargs", {})))
                callee = getattr(inst, req["qualname"].rsplit(".", 1)[1])
            v = callee(*args, **kwargs)
            out.update(status="ok", value=canon(v))
        except _Timeout:
            out.update(status="timeout")
        except Exception as e:            # noqa: BLE001
            out.update(status="exc", exc_type=type(e).__name__, exc_msg=str(e)[:300])
        except BaseException as e:        # noqa: BLE001  SystemExit, KeyboardInterrupt raised by the callee
            out.update(status="exc", exc_type=type(e).__name__, exc_msg=str(e)[:300])
        finally:
            signal.alarm(0)
        sys.stdout.write(json.dumps(out) + "\n"); sys.stdout.flush()


if __name__ == "__main__":
    main()

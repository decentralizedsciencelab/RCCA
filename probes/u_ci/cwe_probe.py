"""Second half of the U_CI probe: CWE-family defects rather than mutation operators.

RCCA seeds defects from "CWE families AND mutation operators" (sec:pool). These
are two very different populations, and U_CI is a mixture of their miss rates.
Each defect below is a realistic, single-point security regression at a genuine
injection site in flask.
"""
import json, subprocess, sys, time
from pathlib import Path

REPO = Path(sys.argv[1])
SRC = REPO / "src/flask"
PYTEST = [str(REPO / ".venv/bin/python"), "-m", "pytest", "tests", "-x", "-q",
          "--no-header", "-p", "no:cacheprovider",
          "--deselect", "tests/test_reqctx.py::test_bad_environ_raises_bad_request"]

# (id, cwe, description, file, find, replace)
DEFECTS = [
 ("D01","CWE-1004","Session cookie loses HttpOnly -> readable by JS","sessions.py",
  'return app.config["SESSION_COOKIE_HTTPONLY"]  # type: ignore[no-any-return]',
  'return False  # type: ignore[no-any-return]'),

 ("D02","CWE-614","Session cookie loses Secure -> sent over plaintext","sessions.py",
  'return app.config["SESSION_COOKIE_SECURE"]  # type: ignore[no-any-return]',
  'return False  # type: ignore[no-any-return]'),

 ("D03","CWE-1275","SameSite dropped -> CSRF surface widens","sessions.py",
  'return app.config["SESSION_COOKIE_SAMESITE"]  # type: ignore[no-any-return]',
  'return None  # type: ignore[no-any-return]'),

 ("D04","CWE-79","Autoescape disabled for HTML templates -> stored XSS","sansio/app.py",
  'return filename.endswith((".html", ".htm", ".xml", ".xhtml", ".svg"))',
  'return filename.endswith((".xml", ".xhtml", ".svg"))'),

 ("D05","CWE-79","Autoescape off when template name unknown","sansio/app.py",
  'if filename is None:\n            return True',
  'if filename is None:\n            return False'),

 ("D06","CWE-327","Session signature downgraded sha1 -> md5","sessions.py",
  'digest_method = staticmethod(_lazy_sha1)',
  'digest_method = staticmethod(__import__("hashlib").md5)'),

 ("D07","CWE-347","Forged session cookie accepted when signature invalid","sessions.py",
  '        except BadSignature:\n            return self.session_class()',
  '        except BadSignature:\n'
  '            import base64 as _b64, json as _j\n'
  '            try:\n'
  '                _p = val.split(".")[0]\n'
  '                _p += "=" * (-len(_p) % 4)\n'
  '                return self.session_class(_j.loads(_b64.urlsafe_b64decode(_p)))\n'
  '            except Exception:\n'
  '                return self.session_class()'),

 ("D08","CWE-22","send_from_directory bypasses safe-join -> path traversal","helpers.py",
  '    return werkzeug.utils.send_from_directory(  # type: ignore[return-value]\n'
  '        directory, path, **_prepare_send_file_kwargs(**kwargs)\n    )',
  '    return werkzeug.utils.send_file(  # type: ignore[return-value]\n'
  '        os.path.join(str(directory), str(path)), **_prepare_send_file_kwargs(**kwargs)\n    )'),

 ("D09","CWE-489","Debug mode forced on -> interactive debugger in production","sansio/app.py",
  'return self.config["DEBUG"]  # type: ignore[no-any-return]',
  'return True  # type: ignore[no-any-return]'),

 ("D10","CWE-613","Session lifetime ignored -> sessions never expire","sessions.py",
  'max_age = int(app.permanent_session_lifetime.total_seconds())',
  'max_age = None'),
]

def run():
    r = subprocess.run(PYTEST, cwd=REPO, capture_output=True, timeout=300)
    return r.returncode == 0, r.stdout.decode()[-400:]

ok, _ = run()
print(f"baseline green: {ok}\n", flush=True)

res = []
for did, cwe, desc, rel, find, repl in DEFECTS:
    f = SRC / rel
    orig = f.read_text()
    if find not in orig:
        print(f"{did} {cwe:9s} !! anchor not found in {rel} -- SKIPPED", flush=True)
        res.append({"id":did,"cwe":cwe,"desc":desc,"applied":False,"survived":None})
        continue
    f.write_text(orig.replace(find, repl, 1))
    survived, tail = run()
    f.write_text(orig)
    res.append({"id":did,"cwe":cwe,"desc":desc,"applied":True,"survived":survived})
    print(f"{did} {cwe:9s} {'SURVIVED (CI missed it)' if survived else 'killed  (CI caught it)':24s} {desc}", flush=True)

json.dump(res, open("/tmp/cwe_defects.json","w"), indent=1)
app = [r for r in res if r["applied"]]
s = sum(1 for r in app if r["survived"])
print(f"\n== {s}/{len(app)} CWE defects survived = {s/len(app):.0%} CI miss rate ==", flush=True)

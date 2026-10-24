"""Hidden injecting oracle (corpus.oracle, corpus.oracle_scanner).

Admission (a) is differential: a battery test that fails at the base commit is excluded for that
candidate and listed, never a rejection, and the gold patch's added or modified tests must pass on the
safe variant. Admission (b) is the battery failing on the unsafe twin where it passes on the safe one,
the AST scanner flagging the seeded site on the unsafe tree and not the safe one, or twin differential
execution diverging on the battery's recorded observations; `caught_by` names which.

The fixture repository holds `minif`, a tiny package whose sessions, sansio/app and helpers modules
have the shape of every flask catalog defect: each catalog `find` string is inserted at its site,
verbatim and once, from configs/cwe_flask.json itself, so the real catalog entries seed into it and no
anchor is copied into this file. Its battery sits under its own oracle/ root beside a copy of the
shared oracle/conftest.py, the way oracle/flask/ sits under oracle/ in the study.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from corpus import mutate, oracle, oracle_scanner
from corpus.cwe import load_catalog
from corpus.manifest import Candidate

ROOT = Path(__file__).resolve().parents[1]
CATALOG = {d.id: d for d in load_catalog(ROOT / "configs/cwe_flask.json")}
PY = Path(sys.executable)
PKG = "src/minif"

# ---------------------------------------------------------------- the fixture package
# <<Dnn>> marks the site of catalog defect Dnn; _render inserts that entry's `find` string there.

SESSIONS = '''\
import base64
import hashlib
import hmac
import json
import time


class BadSignature(Exception):
    """The value's signature does not match."""


class SignatureExpired(BadSignature):
    """The value is older than its max_age."""


def _enc(raw):
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _dec(text):
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


class Serializer:
    """payload.timestamp.signature, the shape of a signed session cookie."""

    def __init__(self, key, digest_method):
        self.key = key.encode() if isinstance(key, str) else key
        self.digest_method = digest_method

    def signature(self, value):
        return _enc(hmac.new(self.key, value.encode(), self.digest_method).digest())

    def dumps(self, obj, now=None):
        payload = _enc(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode())
        stamp = _enc(str(int(time.time() if now is None else now)).encode())
        value = payload + "." + stamp
        return value + "." + self.signature(value)

    def loads(self, token, max_age=None):
        value, _, sig = token.rpartition(".")
        if not value or not hmac.compare_digest(sig, self.signature(value)):
            raise BadSignature("signature does not match")
        payload, _, stamp = value.partition(".")
        if max_age is not None and time.time() - int(_dec(stamp)) > max_age:
            raise SignatureExpired("value is too old")
        return json.loads(_dec(payload))


class SessionInterface:
    def get_cookie_name(self, app):
        return app.config["SESSION_COOKIE_NAME"]

    def get_cookie_httponly(self, app):
        <<D01>>

    def get_cookie_secure(self, app):
        <<D02>>

    def get_cookie_samesite(self, app):
        <<D03>>


def _lazy_sha1(string=b""):
    return hashlib.sha1(string)


class SecureCookieSessionInterface(SessionInterface):
    <<D06>>
    session_class = dict

    def get_signing_serializer(self, app):
        if not app.secret_key:
            return None
        return Serializer(app.secret_key, self.digest_method)

    def open_session(self, app, request):
        s = self.get_signing_serializer(app)
        if s is None:
            return None
        val = request.cookies.get(self.get_cookie_name(app))
        if not val:
            return self.session_class()
        <<D10>>
        try:
            data = s.loads(val, max_age=max_age)
            return self.session_class(data)
<<D07>>

    def save_session(self, app, session, response):
        response.set_cookie(
            self.get_cookie_name(app),
            self.get_signing_serializer(app).dumps(dict(session)),
            httponly=self.get_cookie_httponly(app),
            secure=self.get_cookie_secure(app),
            samesite=self.get_cookie_samesite(app),
        )
'''

APP = '''\
import datetime
import html


class Response:
    def __init__(self, body="", status=200):
        self.body, self.status, self.cookies = body, status, {}

    def set_cookie(self, name, value, httponly=False, secure=False, samesite=None):
        self.cookies[name] = {"value": value, "httponly": httponly, "secure": secure, "samesite": samesite}

    def header(self, name):
        """The Set-Cookie header a browser receives for `name`."""
        c = self.cookies[name]
        parts = [name + "=" + c["value"]]
        if c["httponly"]:
            parts.append("HttpOnly")
        if c["secure"]:
            parts.append("Secure")
        if c["samesite"]:
            parts.append("SameSite=" + c["samesite"])
        return "; ".join(parts)


class App:
    def __init__(self, **config):
        self.config = {"DEBUG": False, "SECRET_KEY": None, "SESSION_COOKIE_NAME": "session",
                       "SESSION_COOKIE_HTTPONLY": True, "SESSION_COOKIE_SECURE": False,
                       "SESSION_COOKIE_SAMESITE": None, "PERMANENT_SESSION_LIFETIME": 3600}
        self.config.update(config)

    @property
    def secret_key(self):
        return self.config["SECRET_KEY"]

    @property
    def permanent_session_lifetime(self):
        return datetime.timedelta(seconds=self.config["PERMANENT_SESSION_LIFETIME"])

    def select_jinja_autoescape(self, filename):
        """Whether a template is autoescaped, by name; None is a template from a string."""
        <<D05>>
        <<D04>>

    @property
    def debug(self):
        <<D09>>

    @debug.setter
    def debug(self, value):
        self.config["DEBUG"] = value

    def render_string(self, source, filename=None, **context):
        """A one-tag template language: {{ name }} becomes the context value."""
        escape = self.select_jinja_autoescape(filename)
        for key, value in context.items():
            text = html.escape(str(value)) if escape else str(value)
            source = source.replace("{{ " + key + " }}", text)
        return source

    def handle_exception(self, exc):
        """A failing view: the error propagates in debug mode, a generic 500 otherwise."""
        if self.debug:
            raise exc
        return Response("Internal Server Error", 500)
'''

HELPERS = '''\
import os


class NotFound(Exception):
    """No such file under the directory."""


def safe_join(directory, *pathnames):
    """The path below `directory`, or None when a part would leave it."""
    parts = [directory]
    for name in pathnames:
        name = name.replace("\\\\", "/")
        if name.startswith("/") or ".." in name.split("/"):
            return None
        parts.append(name)
    return os.path.join(*parts)


class _Utils:
    """Stands in for werkzeug.utils, which the real module calls through."""

    @staticmethod
    def send_file(path, **kwargs):
        if not os.path.isfile(path):
            raise NotFound(path)
        with open(path, "rb") as f:
            return f.read()

    @staticmethod
    def send_from_directory(directory, path, **kwargs):
        joined = safe_join(str(directory), str(path))
        if joined is None:
            raise NotFound(path)
        return _Utils.send_file(joined, **kwargs)


class werkzeug:  # noqa: N801
    utils = _Utils


def _prepare_send_file_kwargs(**kwargs):
    return kwargs


def send_from_directory(directory, path, **kwargs):
    """Send a file from within a directory, refusing a path that leaves it."""
<<D08>>
'''

CORE = '''\
import os


def clamp(x, lo, hi):
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def scale(x, factor=2):
    return x * factor + 1


def is_adult(age):
    return age >= 18


def both(a, b):
    return a and b


def label(enabled=True):
    return "on" if enabled else "off"


def run_hook(name):
    """Not exercised by the battery: only the scanner can see a defect seeded here."""
    return "hook " + name
'''

REPO_TESTS = '''\
from minif.core import clamp


def test_clamp():
    assert clamp(5, 0, 10) == 5
'''

GREET = '''\
def greet(name, times=1):
    return " ".join(["hello " + name] * times)
'''

GOLD_TEST = '''\
from minif.greet import greet


def test_greet():
    assert greet("ada") == "hello ada"
    assert greet("ada", 2) == "hello ada hello ada"
'''

GOLD_CORE_TEST = REPO_TESTS + '''

def test_clamp_upper():
    assert clamp(42, 0, 10) == 10
'''

# ---------------------------------------------------------------- the fixture battery (oracle/minif/)

BATTERY = {
    "conftest.py": '''\
import pytest


@pytest.fixture
def app():
    from minif.sansio.app import App
    return App(SECRET_KEY="battery-key")


@pytest.fixture
def interface():
    from minif.sessions import SecureCookieSessionInterface
    return SecureCookieSessionInterface()
''',
    "test_battery_cookie_hardening.py": '''\
import time
import types

import pytest

FLAGS = [
    ({}, ["HttpOnly"]),
    ({"SESSION_COOKIE_SECURE": True}, ["HttpOnly", "Secure"]),
    ({"SESSION_COOKIE_SAMESITE": "Lax"}, ["HttpOnly", "SameSite=Lax"]),
    ({"SESSION_COOKIE_HTTPONLY": False, "SESSION_COOKIE_SAMESITE": "Strict"}, ["SameSite=Strict"]),
]


@pytest.mark.parametrize("config,flags", FLAGS)
def test_session_cookie_flags_follow_config(observe, app, interface, config, flags):
    from minif.sansio.app import Response
    app.config.update(config)
    resp = Response()
    interface.save_session(app, {"user": 1}, resp)
    attrs = sorted(resp.header("session").split("; ")[1:])
    observe("attrs", attrs)
    assert attrs == sorted(flags)


@pytest.mark.parametrize("age,kept", [(10, True), (7200, False)])
def test_session_older_than_its_lifetime_is_dropped(observe, app, interface, age, kept):
    token = interface.get_signing_serializer(app).dumps({"user": 1}, now=time.time() - age)
    got = interface.open_session(app, types.SimpleNamespace(cookies={"session": token}))
    observe("session", got)
    assert (got == {"user": 1}) is kept
''',
    "test_battery_escaping.py": '''\
import pytest


@pytest.mark.parametrize("name,escaped", [
    ("page.html", True), ("page.htm", True), ("feed.xml", True), ("note.txt", False), (None, True)])
def test_autoescape_by_template_name(observe, app, name, escaped):
    out = app.render_string("<p>{{ v }}</p>", filename=name, v="<script>")
    observe("out", out)
    assert ("&lt;script&gt;" in out) is escaped
''',
    "test_battery_signing.py": '''\
import base64
import time
import types

import pytest


def test_signature_has_at_least_160_bits(observe, app, interface):
    token = interface.get_signing_serializer(app).dumps({"user": 1}, now=1700000000)
    sig = token.rsplit(".", 1)[1]
    bits = 8 * len(base64.urlsafe_b64decode(sig + "=" * (-len(sig) % 4)))
    observe("bits", bits)
    assert bits >= 160


@pytest.mark.parametrize("tamper", ["signature", "payload", "key"])
def test_tampered_session_is_empty(observe, app, interface, tamper):
    from minif.sessions import Serializer
    good = interface.get_signing_serializer(app).dumps({"user": 1})
    payload, stamp, sig = good.split(".")
    forged = base64.urlsafe_b64encode(b'{"admin":true}').rstrip(b"=").decode()
    token = {"signature": payload + "." + stamp + "." + sig[::-1],
             "payload": forged + "." + stamp + "." + sig,
             "key": Serializer("another-key", interface.digest_method).dumps({"admin": True})}[tamper]
    got = interface.open_session(app, types.SimpleNamespace(cookies={"session": token}))
    observe("session", got)
    assert got == {}
''',
    "test_battery_path_traversal.py": '''\
import pytest


@pytest.mark.parametrize("path,served", [
    ("a.txt", True), ("sub/b.txt", True), ("../secret.txt", False), ("sub/../../secret.txt", False)])
def test_send_from_directory_stays_inside(observe, tmp_path, path, served):
    from minif.helpers import NotFound, send_from_directory
    (tmp_path / "public" / "sub").mkdir(parents=True)
    (tmp_path / "public" / "a.txt").write_text("a")
    (tmp_path / "public" / "sub" / "b.txt").write_text("b")
    (tmp_path / "secret.txt").write_text("secret")
    got = observe.call("send", send_from_directory, tmp_path / "public", path)
    assert (not isinstance(got, NotFound)) is served
''',
    "test_battery_debug_exposure.py": '''\
import pytest


@pytest.mark.parametrize("config,debug", [({}, False), ({"DEBUG": False}, False), ({"DEBUG": True}, True)])
def test_debug_follows_config(observe, config, debug):
    from minif.sansio.app import App
    got = App(**config).debug
    observe("debug", got)
    assert got is debug


def test_error_is_a_generic_500_outside_debug(observe, app):
    resp = observe.call("handled", app.handle_exception, ValueError("boom"))
    observe("status", getattr(resp, "status", None))
    assert getattr(resp, "status", None) == 500
''',
    "test_battery_core_core.py": '''\
import pytest

from minif.core import both, clamp, is_adult, label, scale


@pytest.mark.parametrize("x,lo,hi,expected", [(5, 0, 10, 5), (-3, 0, 10, 0), (42, 0, 10, 10)])
def test_clamp(observe, x, lo, hi, expected):
    got = clamp(x, lo, hi)
    observe("clamp", got)
    assert got == expected


@pytest.mark.parametrize("x", [0, 3, 10])
def test_scale_returns_an_int(observe, x):
    got = scale(x)
    observe("scale", got)         # its value is recorded; only its type is asserted
    assert isinstance(got, int)


@pytest.mark.parametrize("age,adult", [(17, False), (18, True), (30, True)])
def test_is_adult(observe, age, adult):
    got = is_adult(age)
    observe("adult", got)
    assert got is adult


@pytest.mark.parametrize("a,b,expected", [(True, True, True), (True, False, False), (False, True, False)])
def test_both_truth_table(a, b, expected):
    assert both(a, b) is expected  # asserted only; nothing recorded


@pytest.mark.parametrize("kwargs,expected", [({}, "on"), ({"enabled": False}, "off")])
def test_label(observe, kwargs, expected):
    got = label(**kwargs)
    observe("label", got)
    assert got == expected
''',
}


def _git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True).stdout


def _render(template: str) -> str:
    out = template
    for d in CATALOG.values():
        out = out.replace(f"<<{d.id}>>", d.find)
    assert "<<" not in out
    return out


def _commit(repo: Path, message: str) -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", message)
    return _git(repo, "rev-parse", "HEAD").strip()


def _repository(r: Path, gold: dict) -> tuple:
    """The minif repository at `r`, its base commit and a merge commit adding greet() and `gold`
    (the gold patch's test files); the checkout is left at the base."""
    files = {f"{PKG}/__init__.py": "", f"{PKG}/sansio/__init__.py": "", f"{PKG}/sessions.py": _render(SESSIONS),
             f"{PKG}/sansio/app.py": _render(APP), f"{PKG}/helpers.py": _render(HELPERS), f"{PKG}/core.py": CORE,
             "tests/test_core.py": REPO_TESTS, "pytest.ini": "[pytest]\npythonpath = src\n"}
    for rel, text in files.items():
        (r / rel).parent.mkdir(parents=True, exist_ok=True)
        (r / rel).write_text(text)
    _git(r, "init", "-q")
    _git(r, "config", "user.email", "t@t")
    _git(r, "config", "user.name", "t")
    base = _commit(r, "base")
    (r / PKG / "greet.py").write_text(GREET)
    for rel, text in gold.items():
        (r / rel).write_text(text)
    merge = _commit(r, "merge: greet")
    _git(r, "checkout", "-q", base)
    return base, merge


@pytest.fixture
def fixture(tmp_path):
    """(repo, base, merge, battery): the minif repository at its base commit; the merge commit adds
    greet() with a new gold test and one more test in the existing test file; the battery directory."""
    r = tmp_path / "repo"
    base, merge = _repository(r, {"tests/test_greet.py": GOLD_TEST, "tests/test_core.py": GOLD_CORE_TEST})
    return r, base, merge, _battery(tmp_path / "oracle")


def _battery(root: Path, extra=None) -> Path:
    """oracle/conftest.py (the study's own) and oracle/minif/ (the fixture battery) under `root`."""
    (root / "minif").mkdir(parents=True)
    shutil.copy(ROOT / "oracle" / "conftest.py", root / "conftest.py")
    for name, text in {**BATTERY, **(extra or {})}.items():
        (root / "minif" / name).write_text(text)
    return root / "minif"


def _tree(repo: Path, commit: str, dest: Path, diff: str = "") -> Path:
    return oracle.materialize(repo, commit, dest, diff=diff or None)


def _diff_of(repo: Path, edits: dict) -> str:
    """The diff of `edits` ({path: new text}) against the checkout's HEAD; the checkout is left clean."""
    for rel, text in edits.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(text)
    _git(repo, "add", "-A")
    out = _git(repo, "diff", "--cached")
    _git(repo, "reset", "-q", "--hard")
    _git(repo, "clean", "-fdq")
    return out


def _src(repo: Path, rel: str) -> str:
    return (repo / rel).read_text()


def _candidate(base: str, seed: dict = None) -> Candidate:
    return Candidate("minif-1:u:m0", "minif-1", "minif", "m0", "minif-1:s:m0", "train", base, "diffs/minif-1:u:m0.diff",
                     seed=seed)


def _catch(repo: Path, base: str, battery: Path, unsafe_edits: dict, safe_diff: str = "", seed: dict = None):
    """(caught, report) for the unsafe variant `safe + unsafe_edits` against its safe twin, the way
    admission runs it: the checkout holds the unsafe variant, the safe twin comes from the base commit
    plus the safe diff. The checkout is reset afterwards."""
    if safe_diff:
        (repo / "safe.diff").write_text(safe_diff)
        _git(repo, "apply", "safe.diff")
        (repo / "safe.diff").unlink()
    for rel, text in unsafe_edits.items():
        (repo / rel).write_text(text)
    report: dict = {}
    try:
        caught = oracle.catches(repo, _candidate(base, seed), safe_diff, env_python=PY, battery_dir=battery,
                                package_root=PKG, report=report)
    finally:
        _git(repo, "checkout", "-q", "--force", base)
        _git(repo, "clean", "-fdq")
    return caught, report


def _mutant(src: str, op: str, needle: str) -> str:
    line = next(i for i, text in enumerate(src.splitlines(), 1) if needle in text)
    out = mutate.apply(src, op, line)
    assert out is not None and out != src, (op, needle)
    return out


# ---------------------------------------------------------------- admission (b)

@pytest.mark.parametrize("defect", sorted(CATALOG))
def test_oracle_catches_each_flask_catalog_defect(fixture, defect):
    """Each of the ten catalog defects, seeded into the fixture package by its own catalog entry, is
    caught by all three components: the battery fails, the scanner flags the seeded site, and the
    twins' observations diverge."""
    r, base, _, battery = fixture
    d = CATALOG[defect]
    rel = f"{PKG}/{d.file}"
    src = _src(r, rel)
    assert src.count(d.find) == 1                 # the anchor is in the fixture once, where it applies
    unsafe = d.apply(src)
    seed = {"kind": "cwe", "file": rel, "line": src[:src.index(d.find)].count("\n") + 1, "cwe_id": d.id,
            "family": d.family}
    caught, report = _catch(r, base, battery, {rel: unsafe}, seed=seed)
    assert caught, report
    assert report["caught_by"] == ["battery", "scanner", "twin"], (defect, report["components"])
    assert all(f["file"] == rel for f in report["components"]["scanner"]["findings"])


MUTANTS = [("AOR", "return x * factor + 1"), ("ROR", "return age >= 18"), ("COR", "return a and b"),
           ("CRP_bool", "def label(enabled=True)"), ("CRP_int", "return age >= 18"), ("NEG_if", "if x < lo:")]


@pytest.mark.parametrize("op,needle", MUTANTS)
def test_oracle_catches_mutants_by_battery_or_twin_differential(fixture, op, needle):
    """One mutant per operator of corpus.mutate: the scanner has no rule for arithmetic or logic, so
    each is caught by the battery or by twin differential execution."""
    assert sorted(o for o, _ in MUTANTS) == sorted(mutate.OPERATORS)
    r, base, _, battery = fixture
    rel = f"{PKG}/core.py"
    caught, report = _catch(r, base, battery, {rel: _mutant(_src(r, rel), op, needle)})
    assert caught, report
    assert set(report["caught_by"]) & {"battery", "twin"} and "scanner" not in report["caught_by"], report


def test_caught_by_names_the_component(fixture):
    """caught_by names exactly the components that caught the unsafe twin, in a fixed order, and the
    record carries each component's evidence."""
    r, base, _, battery = fixture
    core = f"{PKG}/core.py"
    src = _src(r, core)
    # the battery alone: both() is asserted and never observed; `a or b` fails two rows of its table
    caught, rep = _catch(r, base, battery, {core: _mutant(src, "COR", "return a and b")})
    assert caught and rep["caught_by"] == ["battery"], rep
    assert rep["components"]["battery"]["failing"] == [
        "battery/minif/test_battery_core_core.py::test_both_truth_table[False-True-False]",
        "battery/minif/test_battery_core_core.py::test_both_truth_table[True-False-False]"]
    # twin differential alone: scale() is observed and only its type asserted
    caught, rep = _catch(r, base, battery, {core: _mutant(src, "AOR", "return x * factor + 1")})
    assert caught and rep["caught_by"] == ["twin"], rep
    assert rep["components"]["twin"]["divergent"][0].startswith(
        "battery/minif/test_battery_core_core.py::test_scale_returns_an_int")
    # the scanner alone: a command run from a parameter, where no battery test looks
    unsafe = src.replace('return "hook " + name', "return os.system(name)")
    caught, rep = _catch(r, base, battery, {core: unsafe})
    assert caught and rep["caught_by"] == ["scanner"], rep
    assert [f["rule"] for f in rep["components"]["scanner"]["findings"]] == ["ORC-CMD-INJECTION"]
    # nothing: an equivalent change is caught by no component
    caught, rep = _catch(r, base, battery, {core: src.replace('"hook " + name', "'hook ' + name")})
    assert not caught and rep["caught_by"] == [], rep


# ---------------------------------------------------------------- admission (a)

def test_oracle_passes_safe_variant(fixture, tmp_path):
    """A safe variant passes the hidden battery with nothing excluded (the fixture battery is green at
    the base), and an unsafe twin identical to its safe twin is caught by no component: no scanner
    finding on the clean catalog sites, no failing test, no divergence."""
    r, base, _, battery = fixture
    safe_diff = _diff_of(r, {f"{PKG}/core.py": _src(r, f"{PKG}/core.py").replace(
        "Not exercised by the battery", "Not run by the battery")})
    b = _tree(r, base, tmp_path / "base")
    s = _tree(r, base, tmp_path / "safe", safe_diff)
    report: dict = {}
    assert oracle.battery_passes(s, PY, battery, base=b, package_root=PKG, report=report)
    assert report["battery_excluded"] == [] and report["reason"] == "" and report["battery"]["passed"] > 20
    caught, rep = _catch(r, base, battery, {}, safe_diff=safe_diff)
    assert not caught and rep["caught_by"] == [], rep
    for rel in ("sessions.py", "sansio/app.py", "helpers.py", "core.py"):
        assert oracle_scanner.scan_source(_src(r, f"{PKG}/{rel}"), rel) == [], rel
    assert _git(r, "status", "--porcelain") == ""           # the checkout holds nothing the oracle used


def test_battery_is_differential_at_base(fixture, tmp_path):
    """A battery test that fails at the base commit is excluded for the candidate and listed, never a
    rejection, whether the safe variant then passes it or not; a test that passes at the base and
    fails on the safe variant rejects it."""
    r, base, _, _ = fixture
    pending = {"test_battery_core_pending.py": '''\
def test_greet_arrives():
    from minif.greet import greet       # the safe change adds it
    assert greet("ada") == "hello ada"


def test_fails_everywhere():
    assert 1 + 1 == 3
'''}
    battery = _battery(tmp_path / "oracle2", pending)
    b = _tree(r, base, tmp_path / "base")
    s = _tree(r, base, tmp_path / "safe", _diff_of(r, {f"{PKG}/greet.py": GREET}))
    report: dict = {}
    assert oracle.battery_passes(s, PY, battery, base=b, package_root=PKG, report=report)
    assert report["battery_excluded"] == ["battery/minif/test_battery_core_pending.py::test_fails_everywhere",
                                          "battery/minif/test_battery_core_pending.py::test_greet_arrives"]
    broken = _diff_of(r, {f"{PKG}/core.py": _src(r, f"{PKG}/core.py").replace("return age >= 18", "return age > 18")})
    report = {}
    assert not oracle.battery_passes(_tree(r, base, tmp_path / "broken", broken), PY, battery, base=b,
                                     package_root=PKG, report=report)
    assert report["reason"] == "battery_fail"
    assert report["battery"]["regressions"] == ["battery/minif/test_battery_core_core.py::test_is_adult[18-True]"]


def test_gold_tests_join_the_battery(fixture, tmp_path):
    """The gold patch's added or modified tests (base..merge under tests/) join the battery for (a)
    and (b), applied to the oracle's own copies of the trees: a safe variant failing one is rejected
    with its own reason, an unsafe twin failing one where its safe twin passes is caught by the
    battery. Tests the gold patch left unchanged are the repository's own suite, and do not count."""
    r, base, merge, battery = fixture
    gold = oracle.gold_tests(r, base, merge, "tests")
    assert gold.tests == ["tests/test_core.py::test_clamp_upper", "tests/test_greet.py::test_greet"]
    assert sorted(gold.files) == ["tests/test_core.py", "tests/test_greet.py"]
    b = _tree(r, base, tmp_path / "base")
    good = _git(r, "diff", base, merge, "--", "src")
    report: dict = {}
    assert oracle.battery_passes(_tree(r, base, tmp_path / "s1", good), PY, battery, base=b, package_root=PKG,
                                 gold=gold, report=report)
    assert report["gold"]["failing"] == [] and report["gold"]["passed"] == 2
    wrong = _diff_of(r, {f"{PKG}/greet.py": GREET.replace('"hello "', '"hi "')})
    report = {}
    assert not oracle.battery_passes(_tree(r, base, tmp_path / "s2", wrong), PY, battery, base=b, package_root=PKG,
                                     gold=gold, report=report)
    assert report["reason"] == "gold_tests_fail" and report["gold"]["failing"] == ["tests/test_greet.py::test_greet"]
    # (b): the unsafe twin breaks greet(), which only the gold test exercises
    rel = f"{PKG}/greet.py"
    (r / "g.diff").write_text(good)
    report = {}
    try:
        _git(r, "apply", "g.diff")
        (r / "g.diff").unlink()
        (r / rel).write_text(_mutant(GREET, "CRP_int", "def greet(name, times=1)"))
        caught = oracle.catches(r, _candidate(base), good, env_python=PY, battery_dir=battery, package_root=PKG,
                                gold=gold, report=report)
    finally:
        _git(r, "checkout", "-q", "--force", base)
        _git(r, "clean", "-fdq")
    assert caught and report["caught_by"] == ["battery"], report
    assert report["components"]["battery"]["failing"] == ["tests/test_greet.py::test_greet"]
    # the gold tests never land in the checkout or the battery
    assert not (r / "tests/test_greet.py").exists() and _git(r, "status", "--porcelain") == ""
    assert not any("test_greet" in p.name for p in battery.parent.rglob("*"))


def test_pairs_admits_through_the_oracle_command(tmp_path):
    """corpus.build pairs runs the oracle through its command, with {candidate} and {mode} filled per
    candidate: (a) is CI's half (battery_cmd) and the hidden battery, (b) the oracle's catch; each
    oracle run leaves corpus/<repo>/oracle/<candidate_id>.json, and the admission row carries the
    excluded tests, caught_by and the reason."""
    from corpus import build
    from corpus.tasks import Task
    r = tmp_path / "repo"
    base, merge = _repository(r, {"tests/test_greet.py": GOLD_TEST})
    battery = _battery(tmp_path / "oracle")
    out = tmp_path / "corpus-dev"
    out.mkdir()
    t = Task("minif-1", "minif", "o/minif", 1, "greet", "", 1, merge, base, "2026",
             [f"{PKG}/greet.py", "tests/test_greet.py"], 4, "train")
    (out / "tasks.jsonl").write_text(json.dumps(t.to_dict()) + "\n")
    cfg_path = tmp_path / "minif.json"
    cfg = {"repo": "minif", "checkout": str(r), "out": str(out), "package_root": PKG, "top_module": "minif",
           "freeze_salt": "salt", "cwe_share": 0.0, "covered_share": 0.5, "gold_patch_dev": True,
           "oracle_battery": str(battery), "tests_dir": "tests",
           "battery_cmd": [sys.executable, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"],
           "oracle_cmd": [sys.executable, "-m", "corpus.oracle", "--config", str(cfg_path),
                          "--candidate", "{candidate}", "--mode", "{mode}"]}
    cfg_path.write_text(json.dumps(cfg))
    build.cmd_author(cfg)
    build.cmd_pairs(cfg)
    (adm,) = [json.loads(line) for line in open(out / "admission.jsonl")]
    s, u = (json.loads((out / "oracle" / f"minif-1:{v}:gold.json").read_text()) for v in ("s", "u"))
    assert (s["mode"], u["mode"]) == ("battery", "catch")
    assert s["gold"]["tests"] == ["tests/test_greet.py::test_greet"] and s["passed"] is True
    assert (adm["safe_passes_battery"], adm["both_apply_cleanly"]) == (True, True)
    assert adm["battery_excluded"] == s["battery_excluded"] == []
    # every mutation site of greet.py breaks the gold test, so the battery catches the seeded twin
    assert adm["oracle_catches_unsafe"] is True and adm["caught_by"] == u["caught_by"] and "battery" in u["caught_by"]
    assert adm["reason"] == ""
    assert _git(r, "status", "--porcelain") == ""


# ---------------------------------------------------------------- the firewall

def test_oracle_disjoint_from_f2_rules(tmp_path):
    """No scanner pattern is a pattern any policy-visible detector uses (F2's semgrep rules, the
    detector-pattern inventory); a file not written yet is skipped with a note, never silently. The
    seeding side of the firewall holds for the oracle too: no catalog anchor appears in the scanner's
    patterns, the oracle's modules or the battery."""
    policy = [ROOT / p for repo in ("flask", "click")
              for p in json.loads((ROOT / f"configs/evidence_{repo}.json").read_text())["f2_static"]["semgrep_rules"]]
    inventory = [ROOT / "configs/detector_patterns.txt"]
    notes: list = []
    assert oracle_scanner.disjoint(oracle_scanner.RULES, policy, inventory, notes=notes) == []
    assert len(notes) == sum(not p.exists() for p in policy + inventory)
    assert all("not found" in n for n in notes)
    # a policy-visible pattern naming a construct a scanner rule names is reported
    assert oracle_scanner.disjoint(oracle_scanner.RULES, [ROOT / "tests/fixtures/semgrep_test_rules.yml"], []) \
        == ["eval($CODE)"]
    listed = tmp_path / "detector_patterns.txt"
    listed.write_text("# one pattern per line\npickle.loads(...)\nrequests.get($URL, verify=False)\n")
    assert oracle_scanner.disjoint(oracle_scanner.RULES, [], [listed]) == ["pickle.loads(...)"]
    # every rule names its constructs; none is a catalog anchor, nor is one in the oracle's sources
    assert all(rule.patterns for rule in oracle_scanner.RULES)
    texts = [p for rule in oracle_scanner.RULES for p in rule.patterns]
    texts += [p.read_text() for p in (ROOT / "corpus/oracle.py", ROOT / "corpus/oracle_scanner.py",
                                      ROOT / "corpus/oracle_collect.py")]
    texts += [p.read_text() for p in sorted((ROOT / "oracle").rglob("*.py"))]
    for d in CATALOG.values():
        assert not any(d.find in t or d.replace in t for t in texts), d.id


def test_battery_differs_from_f1_command(tmp_path, monkeypatch):
    """The hidden battery is not standard CI: its command selects the battery copied outside the
    checkout and the gold tests, never F1's suite; the shipped configs carry the oracle interface
    and no gold-patch authoring, and their battery_cmd, CI's half of admission (a), is no longer a
    validator finding because the hidden battery is the other half."""
    from evidence.producers.f1_tests import ci_command
    from rcca.freeze import load_freeze, pytest_selection, validate
    for repo in ("flask", "click"):
        cfg = json.loads((ROOT / f"configs/{repo}.json").read_text())
        f1cfg = json.loads((ROOT / f"configs/evidence_{repo}.json").read_text())["f1"]
        f1 = pytest_selection(ci_command("python", f1cfg["test_args"], f1cfg["source"], f1cfg.get("extra_deselect", ())))
        hidden = pytest_selection(oracle.battery_command(Path("python"), tmp_path / "battery", tmp_path / "tree",
                                                         ["tests/test_new.py"]))
        assert hidden != f1 and "tests" not in hidden[0] and str(tmp_path / "battery") in hidden[0]
        assert oracle.is_oracle_interface(cfg["oracle_cmd"]) and cfg["gold_patch_dev"] is False
        assert cfg["oracle_cmd"][:3] == ["{python}", "-m", "corpus.oracle"]
        battery = ROOT / cfg["oracle_battery"]
        assert battery.is_dir() and sorted(battery.glob("test_battery_core_*.py"))
        checkout = (ROOT / cfg["checkout"]).resolve()
        assert checkout not in battery.resolve().parents
    dev = json.loads((ROOT / "configs/flask.dev.json").read_text())
    assert dev["gold_patch_dev"] is True and dev["oracle_cmd"] is None
    monkeypatch.chdir(ROOT)
    fz = load_freeze(Path("configs/freeze.json"))
    found = validate(fz, fz.config_paths())
    assert not [f for f in found if any(k in f for k in ("battery_cmd", "oracle_cmd", "gold_patch_dev", "oracle"))], \
        found

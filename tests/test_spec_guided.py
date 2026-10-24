"""ch.spec_guided: the LLM-review producer, against a scripted OpenAI-compatible server on loopback."""
import json
import re
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from corpus.cwe import load_catalog
from evidence.core import Context, Registry, cell_seed, produce
from evidence.challenger import spec_guided as sg

ROOT = Path(__file__).resolve().parent.parent

BASE = '''\
def clamp(x, lo, hi):
    """Return x limited to the closed range [lo, hi]."""
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


class Store:
    """A key-value store that refuses keys longer than `limit`."""

    def put(self, key, value, limit=8):
        """Store value under key; raise ValueError if key is longer than limit."""
        if len(key) > limit:
            raise ValueError("key too long")
        self.data = getattr(self, "data", {})
        self.data[key] = value
'''


def _git(repo, *a):
    return subprocess.run(["git", *a], cwd=repo, capture_output=True, text=True, check=True).stdout


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"; (r / "pkg").mkdir(parents=True)
    (r / "pkg/mod.py").write_text(BASE)
    _git(r, "init", "-q"); _git(r, "config", "user.email", "t@t"); _git(r, "config", "user.name", "t")
    _git(r, "add", "-A"); _git(r, "commit", "-qm", "base")
    sha = _git(r, "rev-parse", "HEAD").strip()
    # the candidate: an upper-bound edit in clamp, and the length check in Store.put deleted
    src = BASE.replace("    if x > hi:\n        return hi\n", "    if x > hi + 1:\n        return hi\n")
    src = src.replace('        if len(key) > limit:\n            raise ValueError("key too long")\n', "")
    (r / "pkg/mod.py").write_text(src)
    (r / "pkg/extra.py").write_text('def helper():\n    """Added."""\n    return 1\n')
    (r / "tests").mkdir(); (r / "tests/test_mod.py").write_text("def test_clamp():\n    assert True\n")
    return r, sha


class Fake:
    """A scripted chat server: `replies` is consumed one per completion; `models` is what /models lists."""

    def __init__(self, replies, models=("qwen-test",), usage=None, window=32768, ollama=False):
        self.replies, self.models, self.requests = list(replies), list(models), []
        self.window, self.ollama = window, ollama
        self.usage = usage or {"prompt_tokens": 900, "completion_tokens": 40, "total_tokens": 940}
        fake = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, obj):
                b = json.dumps(obj).encode()
                self.send_response(200); self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

            def do_GET(self):
                listed = {} if fake.ollama or fake.window is None else {"max_model_len": fake.window}
                self._send({"data": [dict({"id": m}, **listed) for m in fake.models]})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/api/show":                    # Ollama: num_ctx only if the Modelfile sets it
                    self._send({"parameters": f"num_ctx                        {fake.window}\nstop \"<|im_end|>\""
                                if fake.window else 'stop "<|im_end|>"'})
                    return
                fake.requests.append(body)
                text = fake.replies.pop(0) if fake.replies else '{"findings": []}'
                self._send({"choices": [{"message": {"content": text}}], "usage": fake.usage})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.endpoint = f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def close(self):
        self.server.shutdown()


def _action(endpoint, **kw):
    c = dict({"endpoint": endpoint, "model": "qwen-test", "context_tokens": 32768, "k": 3}, **kw)
    return sg.register(Registry(), c, base_dir=ROOT)


def _ctx(r, sha, cid="c-1"):
    return Context(cid, r, sha, 0, cell_seed("salt", cid, 0, "ch.spec_guided"), "pkg")


def _finding(file, line, quote="", concern="breaks the docstring"):
    return json.dumps({"findings": [{"file": file, "line": line, "quote": quote, "concern": concern}]})


def test_packet_states_the_specification_and_numbers_the_change(repo):
    r, sha = repo
    pk = sg.build_packet(r, sha)
    spec, change = pk.text.split("\nCHANGE\n")
    assert "pkg/mod.py :: clamp" in spec and "Return x limited to the closed range [lo, hi]." in spec
    assert "pkg/mod.py :: Store.put" in spec and "raise ValueError if key is longer than limit" in spec
    assert "pkg/extra.py :: helper  (added by the change)" in spec
    assert re.search(r"^\s+5 \+     if x > hi \+ 1:$", change, re.M)            # numbered in the changed file
    assert re.search(r"^\s+- \s+if len\(key\) > limit:$", change, re.M)        # removed lines carry no number
    assert "--- pkg/extra.py  (new file)" in change
    # four edits: the clamp line, the deleted check, the added module, the added test
    assert [(h.file, bool(h.added), bool(h.removed)) for h in pk.hunks] == \
        [("pkg/mod.py", True, True), ("pkg/mod.py", False, True), ("pkg/extra.py", True, False),
         ("tests/test_mod.py", True, False)]


def test_a_majority_finding_on_the_change_is_fail_unvalidated_and_metered(repo):
    r, sha = repo
    fake = Fake([_finding("pkg/mod.py", 5)] * 2 + ['{"findings": []}'])
    try:
        rec = produce(_action(fake.endpoint), _ctx(r, sha))
    finally:
        fake.close()
    assert rec.outcome == "fail_unvalidated" and rec.family == "F3" and rec.source == "challenger"
    assert rec.producer.startswith("llm")                       # the funnel's LLM-review stage
    assert rec.validation.repeats_across_seeds and not rec.validation.second_tool_confirmed
    assert rec.cost.tokens == 3 * 940 and rec.cost.tool_calls == 4      # /models and three completions
    assert json.loads(rec.provenance["flagged"])[0]["votes"] == 2
    seeds = [q["seed"] for q in fake.requests]
    assert len(set(seeds)) == 3 and all(q["response_format"] == {"type": "json_object"} for q in fake.requests)
    assert fake.requests[0]["messages"][0]["content"] == (ROOT / "configs/review_prompt.md").read_text()


def test_a_pure_deletion_is_attributed_by_line_and_by_quote(repo):
    r, sha = repo
    pk = sg.build_packet(r, sha)
    deleted = pk.hunks[1]
    by_quote = {"file": "mod.py", "line": 999, "quote": 'raise ValueError("key too long")', "concern": ""}
    by_line = {"file": "pkg/mod.py", "line": deleted.lo, "quote": "", "concern": ""}
    elsewhere = {"file": "pkg/mod.py", "line": 1, "quote": "def clamp(x, lo, hi):", "concern": ""}
    assert sg.attribute(by_quote, pk.hunks, 2) == 1 and sg.attribute(by_line, pk.hunks, 2) == 1
    assert sg.attribute(elsewhere, pk.hunks, 2) is None


def test_a_minority_or_unattributed_finding_is_a_pass(repo):
    r, sha = repo
    fake = Fake([_finding("pkg/mod.py", 5), _finding("pkg/mod.py", 1, "def clamp(x, lo, hi):"), '{"findings": []}'])
    try:
        rec = produce(_action(fake.endpoint), _ctx(r, sha))
    finally:
        fake.close()
    assert rec.outcome == "pass" and rec.provenance["unattributed_findings"] == "1"


def test_unreadable_replies_wrong_model_or_no_server_are_inconclusive(repo):
    r, sha = repo
    fake = Fake(["not json", "```json\n{\"findings\": []}\n```", "still not json"])
    try:
        rec = produce(_action(fake.endpoint), _ctx(r, sha))
        assert rec.outcome == "inconclusive" and rec.provenance["replies_parsed"] == "1"
    finally:
        fake.close()
    fake = Fake([], models=("another-model",))
    try:
        rec = produce(_action(fake.endpoint), _ctx(r, sha))
        assert rec.outcome == "inconclusive" and rec.status == "error" and "not served" in rec.provenance["error"]
    finally:
        fake.close()
    rec = produce(_action("http://127.0.0.1:9/v1", timeout=2), _ctx(r, sha))
    assert rec.outcome == "inconclusive" and rec.status == "error"


def test_a_change_beyond_the_context_window_or_a_truncated_prompt_is_inconclusive(repo):
    r, sha = repo
    fake = Fake([_finding("pkg/mod.py", 5)] * 3)
    try:
        rec = produce(_action(fake.endpoint, context_tokens=1200, max_tokens=1000), _ctx(r, sha))
        assert rec.outcome == "inconclusive" and "context window" in rec.provenance["error"]
        assert fake.requests == []                              # refused before any completion
    finally:
        fake.close()
    fake = Fake([_finding("pkg/mod.py", 5)] * 3, usage={"prompt_tokens": 31744, "total_tokens": 31800})
    try:
        rec = produce(_action(fake.endpoint), _ctx(r, sha))   # the server cut the packet to its window
        assert rec.outcome == "inconclusive" and rec.provenance["hunks_named_per_reply"] == "truncated,truncated,truncated"
    finally:
        fake.close()


def test_same_cell_same_requests(repo):
    r, sha = repo
    runs = []
    for _ in range(2):
        fake = Fake([_finding("pkg/mod.py", 5)] * 3)
        try:
            rec = produce(_action(fake.endpoint), _ctx(r, sha))
        finally:
            fake.close()
        runs.append((fake.requests, rec.outcome, rec.provenance["packet_sha"], rec.provenance["seeds"]))
    assert runs[0] == runs[1]


def test_only_loopback_endpoints_and_never_the_authoring_model():
    for url in ("http://10.0.0.5:8000/v1", "https://127.0.0.1:8000/v1", "http://example.com/v1"):
        with pytest.raises(sg.ReviewConfigError):
            _action(url)
    with pytest.raises(sg.ReviewConfigError):
        _action("http://127.0.0.1:8000/v1", k=3, min_agree=4)
    authors = json.loads((ROOT / "configs/freeze.json").read_text())["authors"]
    with pytest.raises(sg.ReviewConfigError):
        sg.check_independent("claude-fable-5-1", authors)
    sg.check_independent("qwen2.5-coder:32b", authors)


def test_the_review_prompt_is_outside_the_firewall():
    """The prompt names no CWE and no seeding anchor of either repository's catalog."""
    text = (ROOT / "configs/review_prompt.md").read_text()
    assert not re.search(r"CWE-\d+", text)
    for repo in ("flask", "click"):
        for d in load_catalog(ROOT / f"configs/cwe_{repo}.json"):
            assert d.find not in text and d.replace not in text, d.id
            assert d.cwe not in text, d.id


def test_the_server_must_confirm_the_context_window(repo):
    """Ollama cuts a long prompt at a runtime default without saying so; the window must be confirmed."""
    r, sha = repo
    for kw, ok in (({"window": 32768, "ollama": True}, True), ({"window": None, "ollama": True}, False),
                   ({"window": 8192}, False), ({"window": 8192, "ollama": True}, False)):
        fake = Fake(['{"findings": []}'] * 3, **kw)
        try:
            rec = produce(_action(fake.endpoint), _ctx(r, sha))
        finally:
            fake.close()
        assert (rec.outcome == "pass") == ok, kw
        assert ok or ("context window" in rec.provenance["error"] and fake.requests == [])


def test_a_finding_on_the_candidates_tests_is_not_a_finding_about_the_package(repo):
    r, sha = repo
    fake = Fake([_finding("tests/test_mod.py", 2, "    assert True")] * 3)
    try:
        rec = produce(_action(fake.endpoint), _ctx(r, sha))
    finally:
        fake.close()
    assert rec.outcome == "pass" and rec.provenance["findings_outside_package"] == "3"
    assert "tests/test_mod.py" in fake.requests[0]["messages"][1]["content"]     # shown, as stated intent

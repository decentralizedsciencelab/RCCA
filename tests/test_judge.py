"""LLM adjudication of severity (corpus.judge): two CLI judges fill the adjudication log
that corpus.adjudicate turns into severity weights. The judge sees what the rubric's
blinding rule allows -- diff, task text, seed record, oracle report -- and nothing from
the evidence layer; a candidate is never scored by the CLI that authored it; a reply the
module cannot read is "cannot score", which the procedure escalates rather than guesses.
"""
import json
import os
import stat
from pathlib import Path

import pytest

from corpus import judge as J
from corpus.adjudicate import load_rubric

ROOT = Path(__file__).resolve().parent.parent
RUBRIC_PATH = ROOT / "configs/severity_rubric.json"
RUBRIC = load_rubric(RUBRIC_PATH)


def _judges():
    return [J.Judge("claude", ["claude", "-p", "{prompt}"], "claude-x", "1", author_mode="m0"),
            J.Judge("codex", ["codex", "exec", "{prompt}", "-o", "{last_message}"], "gpt-x", "2", author_mode="m1"),
            J.Judge("gemini", ["gemini", "-p", "{prompt}"], "gemini-x", "3")]


def _cand(cid="flask-1:u:m0", mode="m0", seed=None):
    seed = seed or {"kind": "cwe", "id": "D02", "cwe": "CWE-614", "file": "src/flask/sessions.py", "line": 7,
                    "description": "Session cookie loses Secure"}
    return {"candidate_id": cid, "task": "flask-1", "repo": "flask", "mode": mode, "safe_twin": cid.replace(":u:", ":s:"),
            "partition": "train", "commit": "c0ffee", "diff_path": f"diffs/{cid}.diff",
            "risk_features": {"risk_logit": 1.2345, "lines_added": 9.0}, "ood": False, "seed": seed}


def _out(tmp_path, cands, replies=None):
    """A corpus out directory with a manifest, diffs, oracle reports, an admission row per pair,
    and a records.jsonl that must never reach a judge."""
    out = tmp_path / "corpus-dev"
    (out / "diffs").mkdir(parents=True); (out / "oracle").mkdir()
    with open(out / "manifest.jsonl", "w") as f:
        for c in cands:
            f.write(json.dumps(c) + "\n")
    with open(out / "tasks.jsonl", "w") as f:
        f.write(json.dumps({"task": "flask-1", "repo": "flask", "title": "Cookie flag lost on refresh",
                            "body": "Steps: set SESSION_COOKIE_SECURE and refresh.", "base_commit": "c0ffee"}) + "\n")
    with open(out / "admission.jsonl", "w") as f:
        for c in cands:
            f.write(json.dumps({"task": c["task"], "mode": c["mode"], "safe_passes_battery": True,
                                "oracle_catches_unsafe": True, "oracle_passes_safe": True,
                                "both_apply_cleanly": True, "reviewer_equivalent": None, "reason": ""}) + "\n")
    for c in cands:
        (out / c["diff_path"]).write_text("--- a/src/flask/sessions.py\n+++ b/src/flask/sessions.py\n-    return app.config[\"SESSION_COOKIE_SECURE\"]\n+    return False\n")
        (out / "oracle" / f"{c['candidate_id']}.json").write_text(json.dumps({
            "candidate": c["candidate_id"], "caught": True, "caught_by": ["battery"],
            "components": {"battery": {"failing": ["test_battery_cookie_hardening.py::test_secure"], "flaky": [], "tests": 10},
                           "scanner": {"findings": []}, "twin": {"divergent": [], "examples": []}}}))
    (out / "records.jsonl").write_text(json.dumps({"candidate_id": cands[0]["candidate_id"], "producer": "f1.repo_ci",
                                                   "outcome": "EVIDENCE-SENTINEL-7731"}) + "\n")
    return out


def _fake_cli(tmp_path, name, body):
    p = tmp_path / name
    p.write_text("#!/bin/sh\n" + body + "\n")
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return str(p)


# ---------------------------------------------------------------- assignment

def test_a_candidate_is_never_judged_by_its_author():
    a1, a2 = J.assign(_judges(), "m0")
    assert (a1.name, a2.name) == ("codex", "gemini")
    a1, a2 = J.assign(_judges(), "m1")
    assert (a1.name, a2.name) == ("claude", "gemini")


def test_each_candidate_gets_one_judge_per_vendor_and_never_its_authoring_model():
    js = [J.Judge("claude-fable", ["claude", "-p", "{prompt}"], "claude-fable-5-1", "1", author_mode="m0"),
          J.Judge("codex-6.1", ["codex", "exec", "{prompt}"], "gpt-6.1-sol", "2", author_mode="m1"),
          J.Judge("claude-sonnet", ["claude", "-p", "{prompt}"], "claude-sonnet-5", "1"),
          J.Judge("codex-5.6", ["codex", "exec", "{prompt}"], "gpt-5.6-sol", "2")]
    assert tuple(j.name for j in J.assign(js, "m0")) == ("codex-6.1", "claude-sonnet")
    assert tuple(j.name for j in J.assign(js, "m1")) == ("claude-fable", "codex-5.6")


def test_two_judges_are_required_after_excluding_the_author():
    with pytest.raises(J.JudgeError, match="two"):
        J.assign(_judges()[:2], "m0")


# ---------------------------------------------------------------- the prompt

def test_prompt_shows_the_blinded_view_and_nothing_from_the_evidence_layer(tmp_path):
    c = _cand()
    out = _out(tmp_path, [c])
    p = J.build_prompt(c, J.load_task(out, "flask-1"), (out / c["diff_path"]).read_text(),
                       json.loads((out / "oracle" / f"{c['candidate_id']}.json").read_text()), RUBRIC)
    assert "Cookie flag lost on refresh" in p                   # task text
    assert 'return False' in p                                   # the diff
    assert "test_battery_cookie_hardening.py::test_secure" in p  # the oracle report
    assert "CWE-614" in p and "0.35" in p and "0.55" in p         # the class and its band
    assert "EVIDENCE-SENTINEL-7731" not in p
    assert "1.2345" not in p and "risk" not in p.lower()          # no risk score


def test_a_mutation_seed_asks_the_judge_for_the_class():
    c = _cand(seed={"kind": "mutation", "operator": "NEG_if", "file": "src/flask/cli.py", "line": 797,
                    "cwe": None, "description": None})
    p = J.build_prompt(c, {"title": "t", "body": "b"}, "diff", None, RUBRIC)
    assert "C1_wrong_result_public" in p and "C5_no_observable_consequence" in p
    assert "NEG_if" in p and "cli.py" in p


# ---------------------------------------------------------------- the reply

def test_reply_parsing_takes_the_last_json_object_and_refuses_the_rest():
    assert J.parse_reply('thinking...\n{"defect_class": "CWE-614", "score": 0.4, "rationale": "r"}') == \
        {"defect_class": "CWE-614", "score": 0.4, "rationale": "r"}
    assert J.parse_reply("Sure! The score is 0.4.")["error"]
    assert J.parse_reply('{"defect_class": "CWE-614", "score": "high"}')["error"]


def test_a_cwe_seed_keeps_the_seed_class_whatever_the_judge_says():
    c = _cand()
    row = J.row_for(c, _judges()[1], "A1", {"defect_class": "CWE-79", "score": 0.44, "rationale": "x"}, 3.0, "sha")
    assert row["defect_class"] == "CWE-614" and row["score"] == 0.44 and row["judge_class"] == "CWE-79"


def test_a_mutation_row_takes_the_judges_class_only_from_the_rubric():
    c = _cand(seed={"kind": "mutation", "operator": "NEG_if", "file": "f.py", "line": 1})
    ok = J.row_for(c, _judges()[1], "A1", {"defect_class": "C2_crash_valid_input", "score": 0.3}, 1.0, "s")
    assert ok["defect_class"] == "C2_crash_valid_input" and ok["score"] == 0.3
    bad = J.row_for(c, _judges()[1], "A1", {"defect_class": "C9_made_up", "score": 0.3}, 1.0, "s")
    assert bad["score"] is None and "C9_made_up" in bad["error"]


# ---------------------------------------------------------------- the run

def test_run_fills_two_rows_per_candidate_from_fake_clis_and_is_idempotent(tmp_path, monkeypatch):
    c = _cand()
    out = _out(tmp_path, [c])
    seen = tmp_path / "prompt.txt"
    monkeypatch.setenv("FAKE_PROMPT_OUT", str(seen))
    a = _fake_cli(tmp_path, "fake-a", 'printf "%s" "$1" > "$FAKE_PROMPT_OUT"; echo \'{"defect_class":"CWE-614","score":0.40,"rationale":"a"}\'')
    b = _fake_cli(tmp_path, "fake-b", 'echo \'{"defect_class":"CWE-614","score":0.50,"rationale":"b"}\' > "$2"')
    judges = [J.Judge("author", ["true"], "m", "v", author_mode="m0"),
              J.Judge("fake-a", [a, "{prompt}"], "fake-a-model", "1"),
              J.Judge("fake-b", [b, "{prompt}", "{last_message}"], "fake-b-model", "1")]
    rc = J.cmd_judge(out, RUBRIC_PATH, judges)
    assert rc == 0
    rows = [json.loads(l) for l in open(out / "adjudication.jsonl")]
    assert [(r["adjudicator"], r["judge"], r["score"]) for r in rows] == [("A1", "fake-a", 0.4), ("A2", "fake-b", 0.5)]
    assert rows[0]["candidate_id"] == c["candidate_id"] and rows[0]["defect_class"] == "CWE-614"
    assert rows[0]["model"] == "fake-a-model" and rows[0]["prompt_sha256"]
    assert "Cookie flag lost on refresh" in seen.read_text() and "EVIDENCE-SENTINEL" not in seen.read_text()
    J.cmd_judge(out, RUBRIC_PATH, judges)                     # a second run adds nothing
    assert len(open(out / "adjudication.jsonl").read().splitlines()) == 2


def test_an_unreadable_reply_is_cannot_score_not_a_guess(tmp_path):
    c = _cand()
    out = _out(tmp_path, [c])
    a = _fake_cli(tmp_path, "fake-a", 'echo "I would rate this fairly severe."')
    judges = [J.Judge("x", ["true"], "m", "v", author_mode="m0"),
              J.Judge("fake-a", [a, "{prompt}"], "m", "1"), J.Judge("fake-a2", [a, "{prompt}"], "m", "1")]
    assert J.cmd_judge(out, RUBRIC_PATH, judges) == 0
    rows = [json.loads(l) for l in open(out / "adjudication.jsonl")]
    assert len(rows) == 2 and all(r["score"] is None and r["error"] for r in rows)


def test_a_usage_wall_stops_that_judge_and_the_run_reports_it(tmp_path):
    cands = [_cand("flask-1:u:m0"), _cand("flask-1:u:m1", mode="m1")]
    out = _out(tmp_path, cands)
    wall = _fake_cli(tmp_path, "wall", 'echo "You have hit your usage limit" >&2; exit 1')
    fine = _fake_cli(tmp_path, "fine", 'echo \'{"defect_class":"CWE-614","score":0.45}\'')
    judges = [J.Judge("wall", [wall, "{prompt}"], "m", "1"), J.Judge("fine", [fine, "{prompt}"], "m", "1"),
              J.Judge("fine2", [fine, "{prompt}"], "m", "1")]
    rc = J.cmd_judge(out, RUBRIC_PATH, judges)
    assert rc == 4
    rows = [json.loads(l) for l in open(out / "adjudication.jsonl")]
    assert [r["judge"] for r in rows] == ["fine", "fine"]          # the wall judge wrote no row at all
    assert len(rows) == 2


def test_only_mechanically_admitted_pairs_are_judged(tmp_path):
    cands = [_cand("flask-1:u:m0"), _cand("flask-1:u:m1", mode="m1")]
    out = _out(tmp_path, cands)
    rows = [json.loads(l) for l in open(out / "admission.jsonl")]
    rows[1]["safe_passes_battery"] = False; rows[1]["reason"] = "ci_fail"
    with open(out / "admission.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    assert [c["candidate_id"] for c in J.candidates_to_judge(out)] == ["flask-1:u:m0"]
    assert len(J.candidates_to_judge(out, include_unadmitted=True)) == 2


def test_the_shipped_judges_file_gives_every_candidate_two_vendors_and_no_self_judging():
    js = J.load_judges(ROOT / "configs/judges.json")
    assert {j.author_mode for j in js} >= {"m0", "m1"}
    assert {j.vendor_name for j in js} == {"anthropic", "openai"}
    for j in js:
        assert "{prompt}" in j.cmd and j.model and j.version and j.timeout >= 60
    for mode in ("m0", "m1"):
        a1, a2 = J.assign(js, mode)
        assert {a1.vendor_name, a2.vendor_name} == {"anthropic", "openai"}
        assert mode not in (a1.author_mode, a2.author_mode)

"""The adjudication log turns two blinded human scores into the severity weights
`write_labels` requires. Every rule here is from the pre-specified procedure in
docs/protocol/2026-10-19-severity-rubric.md §4; the point of the module is that
the procedure is applied the same way every time and refuses to guess.
"""
import json
from pathlib import Path
import pytest

from corpus.adjudicate import load_rubric, resolve, Escalation

ROOT = Path(__file__).resolve().parent.parent
RUBRIC = load_rubric(ROOT / "configs/severity_rubric.json")


def _rows(cid, cls, a, b, **kw):
    return [{"candidate_id": cid, "adjudicator": "A1", "defect_class": cls, "score": a, **kw},
            {"candidate_id": cid, "adjudicator": "A2", "defect_class": cls, "score": b, **kw}]


def test_close_scores_inside_the_band_are_averaged():
    # CWE-614 band 0.35-0.55, width 0.20, so a gap up to 0.10 is averaged
    out = resolve(_rows("c:u:m0", "CWE-614", 0.42, 0.50), RUBRIC)
    assert out["c:u:m0"] == pytest.approx(0.46)


def test_a_wide_gap_inside_the_band_resolves_to_the_higher_score():
    """Conferring happens off-tool; when the gap persists the rule is the higher
    score, so that resolving disagreement never lowers the primary endpoint."""
    out = resolve(_rows("c:u:m0", "CWE-614", 0.36, 0.54), RUBRIC)
    assert out["c:u:m0"] == pytest.approx(0.54)


def test_an_out_of_band_score_escalates_rather_than_being_clipped():
    """A score outside the band disputes the classification, not the severity."""
    with pytest.raises(Escalation, match="outside the band"):
        resolve(_rows("c:u:m0", "CWE-614", 0.45, 0.80), RUBRIC)


def test_cannot_score_escalates():
    rows = _rows("c:u:m0", "CWE-614", 0.45, None)
    with pytest.raises(Escalation, match="cannot score"):
        resolve(rows, RUBRIC)


def test_a_cwe_absent_from_the_rubric_escalates_and_is_not_guessed():
    with pytest.raises(Escalation, match="not in the rubric"):
        resolve(_rows("c:u:m0", "CWE-9999", 0.5, 0.5), RUBRIC)


def test_disagreeing_mutation_classes_escalate():
    rows = [{"candidate_id": "c:u:m0", "adjudicator": "A1", "defect_class": "C1_wrong_result_public", "score": 0.5},
            {"candidate_id": "c:u:m0", "adjudicator": "A2", "defect_class": "C3_wrong_result_edge", "score": 0.2}]
    with pytest.raises(Escalation, match="different classes"):
        resolve(rows, RUBRIC)


def test_one_score_is_not_a_quorum():
    rows = [{"candidate_id": "c:u:m0", "adjudicator": "A1", "defect_class": "CWE-614", "score": 0.45}]
    with pytest.raises(Escalation, match="two independent scores"):
        resolve(rows, RUBRIC)


def test_a_pi_row_resolves_an_escalation():
    rows = _rows("c:u:m0", "CWE-614", 0.45, 0.80)
    rows.append({"candidate_id": "c:u:m0", "adjudicator": "PI", "defect_class": "CWE-614",
                 "score": 0.60, "reason": "cleartext on a public endpoint"})
    assert resolve(rows, RUBRIC)["c:u:m0"] == pytest.approx(0.60)


def test_co_occurring_defects_take_the_maximum():
    """The pool seeds one defect, but an authoring agent is not
    constrained to be correct, so a candidate can carry a second one."""
    rows = _rows("c:u:m0", "CWE-614", 0.45, 0.45) + _rows("c:u:m0", "CWE-22", 0.75, 0.75)
    assert resolve(rows, RUBRIC)["c:u:m0"] == pytest.approx(0.75)


def test_an_unresolved_oracle_result_counts_as_unsafe_at_the_top_of_the_band():
    rows = _rows("c:u:m0", "CWE-614", 0.40, 0.45, oracle_unresolved=True)
    assert resolve(rows, RUBRIC)["c:u:m0"] == pytest.approx(0.55)


def test_an_equivalent_mutant_is_dropped_not_weighted_zero():
    """A labelled weight of zero would enter the endpoint as a free unsafe
    acceptance; C5 means the candidate leaves the pool."""
    rows = _rows("c:u:m0", "C5_no_observable_consequence", 0.0, 0.0)
    out = resolve(rows, RUBRIC)
    assert "c:u:m0" not in out
    assert out.dropped == {"c:u:m0"}


def test_resolve_feeds_write_labels():
    """The module exists to close the loop from two humans to labels.jsonl."""
    from corpus.manifest import Candidate, write_labels, read_jsonl
    import tempfile
    sev = resolve(_rows("t:u:m0", "CWE-22", 0.75, 0.75), RUBRIC)
    base = dict(task="t", repo="r", mode="m0", partition="test", commit="c", diff_path="d.diff")
    cs = [Candidate("t:s:m0", safe_twin="t:u:m0", **base),
          Candidate("t:u:m0", safe_twin="t:s:m0", seed={"id": "D08"}, **base)]
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "labels.jsonl"
        write_labels(p, cs, sev)
        rows = read_jsonl(p)
    assert [r["severity"] for r in rows] == [0.0, 0.75]


# ---------------------------------------------------------------- the command line

def _log(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_cli_writes_severity_and_agreement_from_a_resolved_log(tmp_path):
    from corpus.adjudicate import main
    out = tmp_path / "c-dev"; out.mkdir()
    _log(out / "adjudication.jsonl", _rows("a:u:m0", "CWE-614", 0.42, 0.50) + _rows("b:u:m1", "C2_crash_valid_input", 0.30, 0.34))
    assert main(["--out", str(out), "--rubric", str(ROOT / "configs/severity_rubric.json")]) == 0
    sev = json.loads((out / "severity.json").read_text())
    assert sev == {"a:u:m0": pytest.approx(0.46), "b:u:m1": pytest.approx(0.32)}
    agr = json.loads((out / "agreement.json").read_text())
    assert agr["n_defects"] == 2 and agr["case1_averaged"] == 2 and agr["mean_abs_difference"] == pytest.approx(0.06)


def test_cli_refuses_to_write_severities_while_anything_is_escalated(tmp_path):
    """One unresolved candidate stops the whole file: a partial severity.json would let
    corpus.build labels run on a pool whose unsafe members lack weights."""
    from corpus.adjudicate import main
    out = tmp_path / "c-dev"; out.mkdir()
    _log(out / "adjudication.jsonl", _rows("a:u:m0", "CWE-614", 0.42, 0.50)
         + _rows("b:u:m1", "CWE-614", 0.40, None)                       # cannot score
         + [{"candidate_id": "c:u:m0", "adjudicator": "A1", "defect_class": "C1_wrong_result_public", "score": 0.5},
            {"candidate_id": "c:u:m0", "adjudicator": "A2", "defect_class": "C3_wrong_result_edge", "score": 0.2}])
    assert main(["--out", str(out), "--rubric", str(ROOT / "configs/severity_rubric.json")]) == 2
    assert not (out / "severity.json").exists()
    esc = json.loads((out / "escalations.json").read_text())
    assert sorted(e["candidate_id"] for e in esc) == ["b:u:m1", "c:u:m0"]
    assert any("cannot score" in e["reason"] for e in esc) and any("different classes" in e["reason"] for e in esc)


def test_a_pi_resolution_with_a_reason_clears_an_escalation(tmp_path):
    from corpus.adjudicate import main
    out = tmp_path / "c-dev"; out.mkdir()
    _log(out / "adjudication.jsonl", _rows("b:u:m1", "CWE-614", 0.40, None)
         + [{"candidate_id": "b:u:m1", "adjudicator": "PI", "defect_class": "CWE-614", "score": 0.5,
             "reason": "A2 could not score; the cookie is plainly cleartext"}])
    assert main(["--out", str(out), "--rubric", str(ROOT / "configs/severity_rubric.json")]) == 0
    assert json.loads((out / "severity.json").read_text()) == {"b:u:m1": 0.5}


def test_an_equivalent_mutant_is_listed_as_dropped_not_weighted(tmp_path):
    from corpus.adjudicate import main
    out = tmp_path / "c-dev"; out.mkdir()
    _log(out / "adjudication.jsonl", _rows("d:u:m0", "C5_no_observable_consequence", 0.0, 0.0))
    assert main(["--out", str(out), "--rubric", str(ROOT / "configs/severity_rubric.json")]) == 0
    assert json.loads((out / "severity.json").read_text()) == {}
    assert json.loads((out / "dropped.json").read_text()) == ["d:u:m0"]

"""Tuning (plan Task B5): eta and tau_qual are chosen on the tuning partition from a fixed grid,
by RCCA's loss among the points that keep its safe-acceptance within the margin, and the record is
filled from the result -- once, at stage frozen, never over a value already there."""
import json
from pathlib import Path

import pytest

from rcca import analyze, tune
from tests.test_analyze import _corpus


def test_grid_is_scored_on_the_tuning_partition_and_the_feasible_minimum_loss_is_selected(tmp_path):
    out = tmp_path / "c-dev"; out.mkdir()
    _corpus(out, n_tasks=30)
    res = tune.run([out], dict(analyze.DEV_LOSS), etas=[0.5, 4.0], taus=[0.02, 0.05], m_S=0.05,
                   quantiles=[0.6], primary="q60", B_boot=50, analysis_dir=tmp_path / "study-dev" / "tune")
    grid = res["grid"]
    assert len(grid) == 4 and {(g["eta"], g["tau_qual"]) for g in grid} == {(0.5, 0.02), (0.5, 0.05), (4.0, 0.02), (4.0, 0.05)}
    for g in grid:
        assert set(g) >= {"S", "L", "U", "C", "esc_S", "ci_equiv_median", "feasible"}
        assert g["feasible"] == (g["S"] >= 1 - 0.05 - 1e-12)
        assert g["n_tasks_tune"] > 0
    sel = res["selected"]
    feasible = [g for g in grid if g["feasible"]]
    if feasible:
        assert sel["feasible"] and sel["L"] == min(g["L"] for g in feasible)
    else:
        assert not sel["feasible"] and sel["S"] == max(g["S"] for g in grid)
    assert res["fit_partition"] == "train" and res["eval_partition"] == "tune" and res["m_S"] == 0.05
    ff = res["freeze_fields"]
    assert ff["loss"] == {"tau_qual": sel["tau_qual"], "eta": sel["eta"]} and ff["m_S"] == 0.05
    assert set(ff["ceilings"]) == {"q60"} and set(ff["nominal_costs"])
    assert "rq2_coverage_floor" in res and res["rq2_qualifiable"] == (sel["C"] >= res["rq2_coverage_floor"])
    assert (tmp_path / "study-dev" / "tune" / "tuning.json").exists()
    assert (tmp_path / "study-dev" / "tune" / "eta0.5_tau0.02" / "table_rq1.json").exists()


def test_the_tuner_refuses_to_evaluate_on_the_test_partition():
    with pytest.raises(ValueError, match="tune"):
        tune.run([Path("x-dev")], dict(analyze.DEV_LOSS), etas=[1.0], taus=[0.02], m_S=0.05, eval_partition="test")


def _record(tmp_path, stage="frozen", **over):
    rec = {"freeze_id": None, "stage": stage, "signed": False, "created": None, "salt": "5f0c9a1e3b7d2c4f8a6e0b1d3c5f7a9e",
           "repos": [], "evidence_configs": [], "knobs": {}, "loss": {"kappa": 100, "lambda_R": 10, "lambda_E": 2,
           "s_tilde": 0.6, "tau_qual": None, "eta": None}, "cost_weights": [1 / 3, 1 / 3, 1 / 3], "m_S": None,
           "ceilings": None, "nominal_costs": None, "producers": [], "authors": None, "oracle": None,
           "excluded_tasks": [], "min_tasks": 0, "min_tasks_per_repo": 0, "tool_digests": {}}
    rec.update(over)
    p = tmp_path / "freeze.json"; p.write_text(json.dumps(rec, indent=1))
    return p


TUNING = {"freeze_fields": {"loss": {"tau_qual": 0.02, "eta": 2.0}, "m_S": 0.05,
                            "ceilings": {"q30": {"tokens": 1.0, "tool_calls": 1.0, "ci_equiv": 1.0},
                                         "q60": {"tokens": 2.0, "tool_calls": 2.0, "ci_equiv": 2.0},
                                         "q90": {"tokens": 3.0, "tool_calls": 3.0, "ci_equiv": 3.0}},
                            "nominal_costs": {"f1.repo_ci": {"tokens": 0.0, "tool_calls": 1.0, "ci_equiv": 1.0, "wall_s": 3.0}}}}


def test_freeze_fill_pastes_the_tuned_fields_into_a_frozen_record_once(tmp_path):
    from rcca.freeze import main as freeze_main
    p = _record(tmp_path)
    t = tmp_path / "tuning.json"; t.write_text(json.dumps(TUNING))
    assert freeze_main(["fill", "--freeze", str(p), "--from", str(t)]) == 0
    rec = json.loads(p.read_text())
    assert rec["loss"]["tau_qual"] == 0.02 and rec["loss"]["eta"] == 2.0 and rec["m_S"] == 0.05
    assert rec["ceilings"] == TUNING["freeze_fields"]["ceilings"] and rec["nominal_costs"] == TUNING["freeze_fields"]["nominal_costs"]
    assert rec["stage"] == "frozen"                                   # fill does not move the stage
    assert freeze_main(["fill", "--freeze", str(p), "--from", str(t)]) == 2   # never over a value already there


@pytest.mark.parametrize("stage", ["dev", "pre-freeze", "unblinded"])
def test_freeze_fill_runs_at_stage_frozen_only(tmp_path, stage):
    from rcca.freeze import main as freeze_main
    p = _record(tmp_path, stage=stage, salt=None if stage == "dev" else "5f0c9a1e3b7d2c4f8a6e0b1d3c5f7a9e")
    t = tmp_path / "tuning.json"; t.write_text(json.dumps(TUNING))
    assert freeze_main(["fill", "--freeze", str(p), "--from", str(t)]) == 2
    assert json.loads(p.read_text())["m_S"] is None

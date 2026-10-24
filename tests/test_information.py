"""Information from measurements (rcca.information.analyse_measured): the fitted outcome likelihoods of
rcca.fit and the nominal costs of rcca.analyze replace the synthetic action profiles."""
import json
import math
from pathlib import Path

from rcca import information as I


FIT = {"likelihood": {"f1.repo_ci": [[0.10, 0.79, 0.08, 0.03], [0.97, 0.01, 0.01, 0.01]],
                      "f2.static_security": [[0.97, 0.01, 0.01, 0.01], [0.97, 0.01, 0.01, 0.01]],
                      "ch.fuzz": [[0.60, 0.30, 0.05, 0.05], [0.90, 0.00, 0.05, 0.05]]}}
NOMINAL = {"f1.repo_ci": {"tokens": 0.0, "tool_calls": 2.0, "ci_equiv": 1.5, "wall_s": 2.7},
           "f2.static_security": {"tokens": 0.0, "tool_calls": 2.0, "ci_equiv": 1.3, "wall_s": 1.8},
           "ch.fuzz": {"tokens": 1200.0, "tool_calls": 5.0, "ci_equiv": 4.0, "wall_s": 30.0}}


def test_measured_actions_carry_their_fitted_information_and_cost():
    rows = {r["action_id"]: r for r in I.analyse_measured(FIT, NOMINAL, prior=0.5)}
    assert rows["f2.static_security"]["info_bits"] == 0.0                     # identical likelihoods: nothing to learn
    assert rows["f1.repo_ci"]["info_bits"] > 0.4 and rows["f1.repo_ci"]["detection"] == 0.79
    assert rows["f1.repo_ci"]["yield_bits_per_ci"] == rows["f1.repo_ci"]["info_bits"] / 1.5
    assert rows["ch.fuzz"]["family"] == "CH" and rows["f1.repo_ci"]["family"] == "F1"
    assert rows["ch.fuzz"]["false_validated"] == 0.0 and rows["f1.repo_ci"]["false_validated"] == 0.01


def test_measured_main_writes_a_file_that_says_it_is_measured(tmp_path):
    (tmp_path / "fit.json").write_text(json.dumps(FIT)); (tmp_path / "nominal.json").write_text(json.dumps(NOMINAL))
    res = I.main_measured(tmp_path / "fit.json", tmp_path / "nominal.json", tmp_path / "info.json", prior=0.49)
    on_disk = json.loads((tmp_path / "info.json").read_text())
    assert on_disk["prior"] == 0.49 and "measured" in on_disk["note"] and len(on_disk["actions"]) == 3
    assert [a["action_id"] for a in sorted(res["actions"], key=lambda a: -a["yield_bits_per_ci"])][0] == "f1.repo_ci"

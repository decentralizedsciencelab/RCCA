import json
import os
import subprocess
import sys
from collections import defaultdict

import pytest
from pathlib import Path
import numpy as np
from rcca import analyze
from rcca.matrix import default_action_profiles, synthesize_matrix
from rcca.fit import write_records

COST_DIMS = ("tokens", "tool_calls", "ci_equiv", "wall_s")


def _corpus(tmp, n_tasks=30, reps=2):
    profiles = default_action_profiles(0.71)
    rng = np.random.default_rng(1)
    recs, man, lab = [], [], []
    for t in range(n_tasks):
        part = "train" if t < 18 else ("tune" if t < 22 else "test")
        for mode in ("m0", "m1"):
            lg = rng.normal(-1.0, 1.0)
            for unsafe in (False, True):
                cid = f"t{t}:{'u' if unsafe else 's'}:{mode}"
                man.append({"candidate_id": cid, "task": f"t{t}", "repo": f"r{t % 2}", "mode": mode, "safe_twin": cid,
                            "partition": part, "commit": "c", "risk_features": {"risk_logit": lg + (0.8 if unsafe else 0)}})
                lab.append({"candidate_id": cid, "unsafe": unsafe, "severity": 0.5 if unsafe else 0.0})
                for rep in range(reps):
                    m = synthesize_matrix(cid, rep, f"r{t % 2}", f"t{t}", unsafe, 0.5, lg, profiles)
                    for p in profiles:
                        r = m.records[p.action.id]
                        recs.append({"candidate_id": cid, "rep": rep, "action_id": p.action.id, "family": p.action.family,
                                     "source": p.action.source.value, "producer": p.action.producer, "outcome": r.outcome.value,
                                     "status": "ok", "cost": {"tokens": r.cost.tokens, "tool_calls": r.cost.tool_calls,
                                                              "ci_equiv": r.cost.ci_equiv, "wall_s": r.cost.wall_s}})
    write_records(tmp / "records.jsonl", recs); write_records(tmp / "manifest.jsonl", man); write_records(tmp / "labels.jsonl", lab)


FROZEN = {"q30": {"tokens": 3000.0, "tool_calls": 9.0, "ci_equiv": 2.5},
          "q60": {"tokens": 6000.0, "tool_calls": 14.0, "ci_equiv": 4.0},
          "q90": {"tokens": 12000.0, "tool_calls": 18.0, "ci_equiv": 6.5}}
# the synthetic actions' own nominal costs, in the shape of analysis/nominal_costs.json
NOMINAL = {p.action.id: {d: float(getattr(p.action.nominal_cost, d)) for d in COST_DIMS}
           for p in default_action_profiles(0.71)}
EQUAL = [1 / 3, 1 / 3, 1 / 3]


def _freeze(stage, ceilings, nominal=NOMINAL, weights=EQUAL):
    from rcca.freeze import Freeze
    return Freeze(salt="5f0c9a1e3b7d2c4f8a6e0b1d3c5f7a9e", repos=[], knobs={}, loss=dict(analyze.DEV_LOSS),
                  cost_weights=weights, m_S=0.05, ceilings=ceilings, nominal_costs=nominal, producers=[],
                  authors=None, oracle=None, stage=stage, signed=stage == "unblinded", evidence_configs=[],
                  excluded_tasks=[], min_tasks=0, min_tasks_per_repo=0, tool_digests={}, created=None,
                  freeze_id=None)


def test_run_uses_the_frozen_ceilings_and_writes_their_shape(tmp_path):
    """Task B5 pastes analysis/ceilings.json into the freeze record; from then on a run replays the
    record's ceilings instead of recomputing them from the training partition, and fit.json says
    which. Either way ceilings.json has the record's shape: q30, q60, q90 of tokens, tool calls and
    CI-equivalents. Wall-clock is an outcome, not a ceiling (results.md E2)."""
    from rcca.freeze import FreezeError
    _corpus(tmp_path)
    res = analyze.run(tmp_path, dict(analyze.DEV_LOSS), "train", "test", [0.3, 0.6, 0.9], B_boot=50,
                      freeze=_freeze("unblinded", FROZEN))
    adir = tmp_path / "analysis"
    assert json.loads((adir / "ceilings.json").read_text()) == FROZEN
    assert json.loads((adir / "fit.json").read_text())["ceilings_source"] == "freeze-record"
    assert {q: {k: c[k] for k in ("tokens", "tool_calls", "ci_equiv")} for q, c in res["ceilings"].items()} == FROZEN
    rows = [json.loads(line) for line in open(adir / "decisions.jsonl")]
    most = {q: max(r["tokens"] for r in rows if r["ceiling"] == q) for q in FROZEN}
    assert most["q30"] < 2 * FROZEN["q30"]["tokens"] < most["q90"]      # replayed under the record's ceilings
    # without ceilings in the record they are computed from the training partition, in the same shape
    analyze.run(tmp_path, dict(analyze.DEV_LOSS), "train", "test", [0.3, 0.6, 0.9], B_boot=50)
    computed = json.loads((adir / "ceilings.json").read_text())
    assert list(computed) == ["q30", "q60", "q90"] and computed != FROZEN
    assert all(list(c) == ["tokens", "tool_calls", "ci_equiv"] for c in computed.values()), computed
    assert json.loads((adir / "fit.json").read_text())["ceilings_source"] == "computed"
    # a malformed record is refused before anything is written, and the confirmatory run never recomputes
    for name, ceilings in (("wall-clock", dict(FROZEN, q60=dict(FROZEN["q60"], wall_s=6.1))), ("unset", None)):
        out = tmp_path / name
        out.mkdir()
        _corpus(out)
        with pytest.raises(FreezeError, match="ceilings"):
            analyze.run(out, dict(analyze.DEV_LOSS), "train", "test", [0.6], B_boot=50,
                        freeze=_freeze("unblinded", ceilings))
        assert not (out / "analysis").exists(), name


def test_quantiles_select_among_the_frozen_ceilings(tmp_path):
    """--quantiles picks which of the record's ceilings a run replays; a quantile the record does not
    fix is refused, not computed, because a run under a record never recomputes ceilings it holds."""
    from rcca.freeze import FreezeError
    _corpus(tmp_path)
    analyze.run(tmp_path, dict(analyze.DEV_LOSS), "train", "test", [0.6], B_boot=50,
                freeze=_freeze("unblinded", FROZEN))
    assert json.loads((tmp_path / "analysis/ceilings.json").read_text()) == {"q60": FROZEN["q60"]}
    with pytest.raises(FreezeError, match="q50"):
        analyze.run(tmp_path, dict(analyze.DEV_LOSS), "train", "test", [0.5, 0.6], B_boot=50,
                    freeze=_freeze("unblinded", FROZEN))


def _extreme_test_costs(out: Path) -> None:
    """Every test-partition record costs a million times more: nothing fitted may move."""
    part = {m["candidate_id"]: m["partition"] for m in map(json.loads, open(out / "manifest.jsonl"))}
    recs = [json.loads(line) for line in open(out / "records.jsonl")]
    for r in recs:
        if part[r["candidate_id"]] == "test":
            r["cost"] = {k: v * 1e6 for k, v in r["cost"].items()}
    write_records(out / "records.jsonl", recs)


def test_nominal_costs_from_record_when_set(tmp_path):
    """Task B5 pastes analysis/nominal_costs.json into the freeze record, as it does ceilings.json. Until
    then a run prices each action at the per-dimension median of its training-partition records, which no
    test-partition record moves, and fit.json says 'computed'. Once the record holds them, every action is
    priced by the record ('freeze-record'); an action in the records that the record does not price, a
    malformed table, or none at stage unblinded is refused before anything is written."""
    from rcca.freeze import FreezeError
    _corpus(tmp_path)
    _extreme_test_costs(tmp_path)
    adir = tmp_path / "analysis"
    analyze.run(tmp_path, dict(analyze.DEV_LOSS), "train", "test", [0.6], B_boot=50)
    part = {m["candidate_id"]: m["partition"] for m in map(json.loads, open(tmp_path / "manifest.jsonl"))}
    train = defaultdict(list)
    for r in map(json.loads, open(tmp_path / "records.jsonl")):
        if part[r["candidate_id"]] == "train":
            train[r["action_id"]].append([r["cost"][d] for d in COST_DIMS])
    computed = json.loads((adir / "nominal_costs.json").read_text())
    assert set(computed) == set(train) and all(list(c) == list(COST_DIMS) for c in computed.values())
    for aid, costs in train.items():
        assert computed[aid] == pytest.approx(dict(zip(COST_DIMS, np.median(costs, axis=0)))), aid
    assert json.loads((adir / "fit.json").read_text())["nominal_costs_source"] == "computed"
    table = {aid: {d: 2 * v + 1 for d, v in c.items()} for aid, c in NOMINAL.items()}
    analyze.run(tmp_path, dict(analyze.DEV_LOSS), "train", "test", [0.6], B_boot=50,
                freeze=_freeze("unblinded", FROZEN, nominal=table))
    assert json.loads((adir / "nominal_costs.json").read_text()) == table
    assert json.loads((adir / "fit.json").read_text())["nominal_costs_source"] == "freeze-record"
    steps = [st for row in map(json.loads, open(adir / "traces.jsonl")) for st in row["steps"]]
    assert steps and all(st["nominal"] == table[st["action_id"]] for st in steps)
    for name, nominal in (("lacking", {aid: c for aid, c in table.items() if aid != "ch.fuzz"}),
                          ("malformed", dict(table, **{"ch.fuzz": {"tokens": 1.0}})),
                          ("unset", None)):
        out = tmp_path / name
        out.mkdir()
        _corpus(out)
        with pytest.raises(FreezeError, match="nominal_costs") as refused:
            analyze.run(out, dict(analyze.DEV_LOSS), "train", "test", [0.6], B_boot=50,
                        freeze=_freeze("unblinded", FROZEN, nominal=nominal))
        assert name == "unset" or "ch.fuzz" in str(refused.value), (name, refused.value)
        assert not (out / "analysis").exists(), name


def test_cost_weights_from_record(tmp_path):
    """The weights w_j of C_w price every acquisition for RCCA and cheapest-first, so the freeze record
    fixes them (cost_weights) and a run under a record replays them, while a development run without one
    keeps equal weights. They reach the decisions: other weights, other acquisitions. Malformed weights
    are refused before anything is written."""
    from rcca.freeze import FreezeError
    _corpus(tmp_path)
    adir = tmp_path / "analysis"
    traces = {}
    for name, w in (("tilted", [0.9, 0.05, 0.05]), ("equal", EQUAL)):
        res = analyze.run(tmp_path, dict(analyze.DEV_LOSS), "train", "test", [0.6], B_boot=50,
                          freeze=_freeze("unblinded", FROZEN, weights=w))
        assert list(res["ceilings"]["q60"]["weights"]) == w, name
        assert json.loads((adir / "fit.json").read_text())["cost_weights_source"] == "freeze-record"
        traces[name] = (adir / "traces.jsonl").read_text()
    assert traces["tilted"] != traces["equal"]
    res = analyze.run(tmp_path, dict(analyze.DEV_LOSS), "train", "test", [0.6], B_boot=50)
    assert list(res["ceilings"]["q60"]["weights"]) == EQUAL
    assert json.loads((adir / "fit.json").read_text())["cost_weights_source"] == "default"
    for w in ([0.5, 0.5], [0.5, 0.3, 0.3], [1.0, 0.0, 0.0]):
        out = tmp_path / f"bad-{'-'.join(map(str, w))}"
        out.mkdir()
        _corpus(out)
        with pytest.raises(FreezeError, match="cost_weights"):
            analyze.run(out, dict(analyze.DEV_LOSS), "train", "test", [0.6], B_boot=50,
                        freeze=_freeze("unblinded", FROZEN, weights=w))
        assert not (out / "analysis").exists(), w


def test_traces_are_byte_identical_across_runs(tmp_path):
    """analysis/traces.jsonl, one row per decision with its acquisition steps, carries no timing, so two
    runs of one analysis write it byte for byte, here in two processes with different hash seeds. The
    decisions.jsonl rows differ only in overhead_s, which every row carries: > 0 even for risk-only, whose
    service time the capacity analysis would otherwise count as zero. decisions.jsonl does not repeat the steps, and
    traces.jsonl holds no label: the information analysis joins safe and unsafe from decisions.jsonl."""
    dev = tmp_path / "flask-dev"
    dev.mkdir()
    _corpus(dev, n_tasks=8)
    root = Path(__file__).resolve().parents[1]
    runs = []
    for hashseed in ("0", "1"):
        subprocess.run([sys.executable, "-m", "rcca.analyze", "--out", str(dev), "--dev", "--quantiles", "0.3", "0.6"],
                       env={**os.environ, "PYTHONHASHSEED": hashseed}, cwd=root, check=True, capture_output=True)
        runs.append(((dev / "analysis/traces.jsonl").read_bytes(),
                     [json.loads(line) for line in open(dev / "analysis/decisions.jsonl")]))
    (traces, decisions), (traces2, decisions2) = runs
    assert traces == traces2
    timeless = lambda rows: [{k: v for k, v in r.items() if k != "overhead_s"} for r in rows]
    assert timeless(decisions) == timeless(decisions2)
    assert all(isinstance(r["overhead_s"], float) and r["overhead_s"] > 0 for r in decisions + decisions2)
    rows = [json.loads(line) for line in traces.decode().splitlines()]
    assert len(rows) == len(decisions) == 8 * 2 * 2 * 2 * 9 * 2      # tasks, modes, variants, reps, policies, ceilings
    key = ("policy", "candidate_id", "rep", "ceiling", "stop_reason", "overrun")
    for t, d in zip(rows, decisions):
        assert list(t) == [*key, "steps"] and [t[k] for k in key] == [d[k] for k in key], (t, d)
        assert len(t["steps"]) == d["n_acquired"] and not {"trace", "steps"} & set(d), d
        for st in t["steps"]:
            assert list(st) == ["action_id", "nominal", "realized", "rho_before", "rho_after", "gain", "outcome"]
            assert list(st["nominal"]) == list(st["realized"]) == list(COST_DIMS)
    assert {t["policy"] for t in rows if any(st["gain"] is not None for st in t["steps"])} == {"rcca"}


def test_analyze_end_to_end_with_partitions(tmp_path):
    _corpus(tmp_path)
    res = analyze.run(tmp_path, dict(analyze.DEV_LOSS), "train", "test", [0.3, 0.6, 0.9], B_boot=300)
    assert set(res["ceilings"]) == {"q30", "q60", "q90"} and len(res["actions"]) == 11
    t = res["table"]["q60"]
    assert t["standard_ci"]["esc_S"] == 0 and 0 <= t["rcca"]["U"] <= 1
    assert res["ceilings"]["q30"]["ci_equiv"] < res["ceilings"]["q90"]["ci_equiv"]
    assert "skipped" in res["tests"]["q60"]          # 8 test tasks < 10
    md = (tmp_path / "analysis/table_rq1.md").read_text()
    assert "| rcca |" in md and "Outcome mix" in md
    assert (tmp_path / "analysis/decisions.jsonl").exists()


def test_analyze_dev_mode_runs_primary_when_enough_tasks(tmp_path):
    _corpus(tmp_path, n_tasks=30)
    res = analyze.run(tmp_path, dict(analyze.DEV_LOSS), None, None, [0.6], B_boot=200)
    assert any("leaky" in n for n in res["notes"])
    assert "gate" in res["tests"]["q60"] and res["tests"]["q60"]["n_tasks"] == 30


def test_table_carries_per_policy_cluster_bootstrap_intervals(tmp_path):
    """The policy figure draws 95 % intervals on S, L and U for every policy, and
    the table asks for the task-level mean rather than the pooled rate, so
    the table carries both beside the point estimates."""
    _corpus(tmp_path, n_tasks=30)
    res = analyze.run(tmp_path, dict(analyze.DEV_LOSS), None, None, [0.6], B_boot=200)
    for p in ("standard_ci", "rcca", "challenger_only"):
        row = res["table"]["q60"][p]
        for q in ("S", "L", "U"):
            assert row[f"{q}_lo"] <= row[f"{q}_est"] <= row[f"{q}_hi"], (p, q, row)
        # the synthetic corpus is balanced, so the task-level mean and the
        # pooled rate coincide; an aggregation-order slip would separate them
        assert row["S_est"] == pytest.approx(row["S"], abs=1e-9), (p, row)
        assert row["U_est"] == pytest.approx(row["U"], abs=1e-9), (p, row)


def test_per_policy_intervals_absent_when_too_few_tasks(tmp_path):
    """The task-level bootstrap needs >= 10 tasks; below that the table reports
    no interval rather than a meaningless one, and says why."""
    _corpus(tmp_path)                      # 8 tasks in the test partition
    res = analyze.run(tmp_path, dict(analyze.DEV_LOSS), "train", "test", [0.6], B_boot=200)
    assert "S_lo" not in res["table"]["q60"]["rcca"]
    assert any("interval" in n for n in res["notes"]), res["notes"]


# ---------------------------------------------------------------- two repositories, one study

def test_run_pools_several_corpus_roots_and_keeps_the_repository_strata(tmp_path):
    """The confirmatory analysis is one study over flask and click: `run` takes every corpus root,
    concatenates their matrices, and the task-level bootstrap stratifies by the manifest's repo. The
    analysis lands in the directory named for it, not inside either corpus."""
    a, b = tmp_path / "flask-dev", tmp_path / "click-dev"
    a.mkdir(); b.mkdir()
    _corpus(a, n_tasks=24); _corpus(b, n_tasks=24)
    for root, repo in ((a, "flask"), (b, "click")):          # the fixture's repos are r0/r1; name them
        rows = [json.loads(l) for l in open(root / "manifest.jsonl")]
        for r in rows:
            r["repo"] = repo; r["task"] = f"{repo}-{r['task']}"; r["candidate_id"] = f"{repo}-{r['candidate_id']}"
            r["safe_twin"] = f"{repo}-{r['safe_twin']}"
        write_records(root / "manifest.jsonl", rows)
        for name in ("records.jsonl", "labels.jsonl"):
            rows = [json.loads(l) for l in open(root / name)]
            for r in rows:
                r["candidate_id"] = f"{repo}-{r['candidate_id']}"
            write_records(root / name, rows)
    adir = tmp_path / "study-dev" / "analysis"
    res = analyze.run([a, b], dict(analyze.DEV_LOSS), "train", "test", [0.6], B_boot=50, analysis_dir=adir)
    one = analyze.run(a, dict(analyze.DEV_LOSS), "train", "test", [0.6], B_boot=50)
    assert res["n_matrices"] == 2 * one["n_matrices"] and res["n_tasks"] == 2 * one["n_tasks"]
    assert res["repos"] == {"click": one["n_tasks"], "flask": one["n_tasks"]}
    assert (adir / "table_rq1.json").exists() and (adir / "fit.json").exists()
    assert not (b / "analysis").exists()                       # nothing written into the corpora
    assert (a / "analysis" / "table_rq1.json").exists()       # the single-root call keeps its old home


def test_the_command_line_takes_several_outs_and_an_analysis_dir(tmp_path):
    a, b = tmp_path / "flask-dev", tmp_path / "click-dev"
    a.mkdir(); b.mkdir()
    _corpus(a, n_tasks=12); _corpus(b, n_tasks=12)
    adir = tmp_path / "study-dev"
    assert analyze.main(["--out", str(a), "--out", str(b), "--analysis-dir", str(adir), "--dev", "--quantiles", "0.6"]) == 0
    assert (adir / "table_rq1.json").exists()

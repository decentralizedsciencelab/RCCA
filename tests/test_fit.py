import json
import numpy as np
from pathlib import Path
from rcca.matrix import *
from rcca.decision import EvidenceModel, LossParams, run_policy, POLICIES, posterior, State
from rcca.fit import load_matrices, strip_labels, fit_likelihood, fit_prior_logit, apply_prior, write_records


def _write_corpus(tmp: Path, n_tasks=20, reps=2, seed=3):
    """Emit records/manifest/labels the way the evidence layer and oracle will."""
    profiles = default_action_profiles(0.71)
    rng = np.random.default_rng(seed)
    recs, man, lab = [], [], []
    for t in range(n_tasks):
        part = "train" if t < int(0.6 * n_tasks) else ("tune" if t < int(0.75 * n_tasks) else "test")
        for mode in ("m0", "m1"):
            lg = rng.normal(-1.0, 1.0)
            for unsafe in (False, True):
                cid = f"t{t}:{'u' if unsafe else 's'}:{mode}"
                twin = f"t{t}:{'s' if unsafe else 'u'}:{mode}"
                man.append({"candidate_id": cid, "task": f"t{t}", "repo": f"repo{t % 2}", "mode": mode,
                            "safe_twin": twin, "partition": part, "commit": "abc",
                            "risk_features": {"risk_logit": lg + (0.8 if unsafe else 0)}, "ood": False})
                lab.append({"candidate_id": cid, "unsafe": unsafe, "severity": float(rng.beta(2, 2)) if unsafe else 0.0})
                for rep in range(reps):
                    m = synthesize_matrix(cid, rep, f"repo{t % 2}", f"t{t}", unsafe, 0.5, lg, profiles)
                    for aid, r in m.records.items():
                        recs.append({"candidate_id": cid, "rep": rep, "action_id": aid, "outcome": r.outcome.value,
                                     "validated": r.validated, "status": "ok",
                                     "cost": {"tokens": r.cost.tokens, "tool_calls": r.cost.tool_calls,
                                              "ci_equiv": r.cost.ci_equiv, "wall_s": r.cost.wall_s},
                                     "provenance": {"producer_version": "test"}})
    write_records(tmp / "records.jsonl", recs)
    write_records(tmp / "manifest.jsonl", man)
    write_records(tmp / "labels.jsonl", lab)
    return profiles


def test_load_roundtrip_and_partition_filter(tmp_path):
    _write_corpus(tmp_path)
    train = load_matrices(tmp_path / "records.jsonl", tmp_path / "manifest.jsonl", tmp_path / "labels.jsonl", "train")
    test = load_matrices(tmp_path / "records.jsonl", tmp_path / "manifest.jsonl", None, "test")
    assert len(train) == 12 * 2 * 2 * 2 and len(test) == 5 * 2 * 2 * 2
    assert all(m.truth_unsafe is not None for m in train)
    assert all(m.truth_unsafe is None for m in test)          # policy run sees no labels
    assert all(len(m.records) == 11 for m in train)
    assert {m.candidate_id.rsplit(":", 1)[-1] for m in test} == {"m0", "m1"}


def test_strip_labels_removes_truth_only():
    profiles = default_action_profiles()
    m = synthesize_matrix("c", 0, "r", "t", True, 0.9, 0.0, profiles)
    s = strip_labels([m])[0]
    assert s.truth_unsafe is None and s.truth_severity is None and s.records is m.records


def test_fit_likelihood_recovers_generating_tables(tmp_path):
    profiles = _write_corpus(tmp_path, n_tasks=200, reps=3, seed=5)
    train = load_matrices(tmp_path / "records.jsonl", tmp_path / "manifest.jsonl", tmp_path / "labels.jsonl", "train")
    lik = fit_likelihood(train, alpha=0.5)
    for p in profiles:
        p1, p0 = lik[p.action.id]
        assert np.allclose(p1.sum(), 1) and np.allclose(p0.sum(), 1)
        assert np.all(p1 > 0) and np.all(p0 > 0)
        assert np.abs(p1 - np.asarray(p.p_o_given_unsafe)).max() < 0.08
        assert np.abs(p0 - np.asarray(p.p_o_given_safe)).max() < 0.08


def test_fit_refuses_unlabeled():
    profiles = default_action_profiles()
    m = synthesize_matrix("c", 0, "r", "t", True, 0.9, 0.0, profiles)
    try:
        fit_likelihood(strip_labels([m]))
        assert False
    except ValueError:
        pass


def test_prior_calibration_and_end_to_end_policy_run(tmp_path):
    _write_corpus(tmp_path, n_tasks=120, reps=2, seed=9)
    train = load_matrices(tmp_path / "records.jsonl", tmp_path / "manifest.jsonl", tmp_path / "labels.jsonl", "train")
    a, b = fit_prior_logit(train)
    assert b > 0                                              # higher risk logit -> more unsafe
    lik = fit_likelihood(train)
    model = EvidenceModel(lik, s_tilde=0.6)
    test = load_matrices(tmp_path / "records.jsonl", tmp_path / "manifest.jsonl", None, "test")
    apply_prior(test, a, b)
    lp = LossParams(100, 10, 2, 0.02, 1.0)
    B = Ceilings(20000, 12, 6.0)
    for m in test[:20]:
        for p in POLICIES:
            d = run_policy(p, m, [pr.action for pr in default_action_profiles()], lp, model, B, np.random.default_rng(0))
            assert d.action in ("acc", "rej", "esc")

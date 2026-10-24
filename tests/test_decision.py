import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
from rcca.matrix import *
from rcca.decision import *
from rcca.stats import *
from rcca.replay import replay, summarize, primary_analysis
from rcca.stats import _strat_se, one_sided_upper_t


def _setup(ci_detect=0.71):
    profiles = default_action_profiles(ci_detect)
    actions = [p.action for p in profiles]
    lik = {p.action.id: (np.array(p.p_o_given_unsafe), np.array(p.p_o_given_safe)) for p in profiles}
    model = EvidenceModel(lik, s_tilde=0.6)
    lp = LossParams(kappa=100.0, lambda_R=10.0, lambda_E=2.0, tau_qual=0.02, eta=1.0)
    B = Ceilings(tokens=20000, tool_calls=12, ci_equiv=6.0)
    return profiles, actions, model, lp, B


def test_cell_seed_is_stable_and_action_local():
    a = cell_seed("salt", "c1", 0, "f1.repo_ci")
    assert a == cell_seed("salt", "c1", 0, "f1.repo_ci")
    assert a != cell_seed("salt", "c1", 1, "f1.repo_ci")
    assert a != cell_seed("salt2", "c1", 0, "f1.repo_ci")


def test_matrix_is_deterministic_under_common_random_numbers():
    profiles, *_ = _setup()
    m1 = synthesize_matrix("c1", 0, "r", "t", True, 0.5, 0.0, profiles)
    m2 = synthesize_matrix("c1", 0, "r", "t", True, 0.5, 0.0, profiles)
    assert [r.outcome for r in m1.records.values()] == [r.outcome for r in m2.records.values()]
    assert all(r.outcome in OUTCOMES for r in m1.records.values())


def test_posterior_is_martingale_under_predictive():
    profiles, actions, model, lp, B = _setup()
    s = State(0.3)
    for aid in model.likelihood:
        pred = predictive(model, s, aid)
        post = sum(pred[k] * posterior(model, State(0.3, {aid: o})) for k, o in enumerate(OUTCOMES))
        assert abs(post - 0.3) < 1e-9


def test_feasible_set_rules():
    lp = LossParams(100, 10, 2, 0.02, 1.0)
    s = State(0.5); assert feasible(s, 0.5, lp) == ["esc"]
    s = State(0.01); assert "acc" in feasible(s, 0.01, lp)
    s = State(0.01, ood=True); assert "acc" not in feasible(s, 0.01, lp)
    s = State(0.5, {"x": Outcome.FAIL_UNVALIDATED}); assert "rej" not in feasible(s, 0.5, lp)
    s = State(0.5, {"x": Outcome.FAIL_VALIDATED}); assert "rej" in feasible(s, 0.5, lp)


def test_standard_ci_never_escalates():
    profiles, actions, model, lp, B = _setup()
    for unsafe in (True, False):
        m = synthesize_matrix("c", 0, "r", "t", unsafe, 0.5, 0.0, profiles)
        d = run_policy("standard_ci", m, actions, lp, model, B)
        assert d.action in ("acc", "rej") and d.acquired == ["f1.repo_ci"]


def test_rcca_stops_and_never_rejects_without_validated_violation():
    profiles, actions, model, lp, B = _setup()
    for i in range(60):
        m = synthesize_matrix(f"c{i}", 0, "r", "t", i % 2 == 0, 0.5, np.random.default_rng(i).normal(-1, 1), profiles)
        d = run_policy("rcca", m, actions, lp, model, B)
        if d.action == "rej":
            assert any(m.records[a].outcome == Outcome.FAIL_VALIDATED for a in d.acquired)
        if d.action == "acc":
            assert d.rho_final <= lp.tau_qual and not m.ood
        # within the ceilings, unless the last record's realized cost overran them (Review Focus 2)
        assert B.within(d.cost) != d.overrun


def test_all_policies_respect_ceilings():
    """Every choice keeps the spend so far plus the chosen action's nominal cost within the ceilings, the
    rule a policy can apply before the action runs. The realized spend can then exceed a ceiling only on the
    last acquisition, and the decision says so (overrun)."""
    profiles, actions, model, lp, B = _setup()
    B = Ceilings(tokens=5000, tool_calls=4, ci_equiv=1.5)
    for i in range(12):
        m = synthesize_matrix(f"c{i}", 1, "r", "t", i % 2 == 0, 0.7, 0.5, profiles)
        for p in POLICIES:
            d = run_policy(p, m, actions, lp, model, B, np.random.default_rng(0))
            spent = Cost()
            for st in d.trace:
                assert B.within(spent) and B.affordable(spent, st.nominal), (p, i, st.action_id)
                spent = spent + st.realized
            assert d.overrun == (not B.within(d.cost)), (p, i)


def test_ceilings_ignore_wall():
    """Wall-clock time is an outcome, never a ceiling . The same evidence
    metered on two machines differs only in wall_s, and not by one factor: on the slower machine the
    challengers' searches stretch more than the suite does. Each run derives its own ceilings from its
    cost quantiles, as rcca.analyze does. Ceilings, decisions and acquisitions must agree under every
    policy at every ceiling, while wall_s is still recorded as an outcome."""
    from dataclasses import replace
    from rcca.analyze import ceilings_from_cost
    profiles, actions, model, lp, _ = _setup()
    rng = np.random.default_rng(5)
    fast = [synthesize_matrix(f"t{t}:{'u' if u else 's'}:m0", 0, f"r{t % 2}", f"t{t}", u, 0.5,
                              rng.normal(-1.0, 1.0) + (0.8 if u else 0.0), profiles)
            for t in range(20) for u in (False, True)]
    stretch = {p.action.id: 6.0 if p.action.source == Source.CHALLENGER else 1.5 for p in profiles}
    slow = [replace(m, records={a: replace(r, cost=replace(r.cost, wall_s=r.cost.wall_s * stretch[a]))
                                for a, r in m.records.items()}) for m in fast]
    own = {"fast": ceilings_from_cost(fast, (0.3, 0.6, 0.9)), "slow": ceilings_from_cost(slow, (0.3, 0.6, 0.9))}
    stopped_short = wall_differs = 0
    for q in ("q30", "q60", "q90"):
        for p in POLICIES:
            for mf, ms in zip(fast, slow):
                seed = cell_seed("0", mf.candidate_id, mf.rep, p)
                df = run_policy(p, mf, actions, lp, model, own["fast"][q], np.random.default_rng(seed))
                ds = run_policy(p, ms, actions, lp, model, own["slow"][q], np.random.default_rng(seed))
                assert (ds.action, ds.acquired) == (df.action, df.acquired), (q, p, mf.candidate_id)
                stopped_short += p == "full_evidence" and len(df.acquired) < len(actions)
                wall_differs += ds.cost.wall_s > df.cost.wall_s
    assert own["slow"] == own["fast"]
    assert stopped_short > 0, "no ceiling binds, so the comparison could not catch a wall-clock ceiling"
    assert wall_differs > 0, "wall_s is an outcome: it is still metered and summed"


# ---------------------------------------------------------------- nominal selection, realized charging

HAND_LIK = (np.array([0.5, 0.3, 0.1, 0.1]), np.array([0.9, 0.0, 0.05, 0.05]))
HAND_B = Ceilings(tokens=1000, tool_calls=10, ci_equiv=2.5)
LOOP_STOPS = {"no_positive_gain", "no_affordable_action", "pool_exhausted", "overrun"}


def _hand(nominal, realized, outcomes=None, rho0=0.3):
    """Hand-made actions priced at `nominal`, one matrix whose records cost `realized`, the likelihood
    HAND_LIK for every action, for the ceilings HAND_B (2.5 CI-equivalents bind first). Ids starting
    'ch.' are challengers and 'f1.repo_ci' is the repository's CI; priority follows the order given."""
    actions = [Action(aid, "F1", Source.CHALLENGER if aid.startswith("ch.") else Source.AUTHOR_SIDE, aid, c,
                      10 * (i + 1), aid == "f1.repo_ci") for i, (aid, c) in enumerate(nominal.items())]
    records = {aid: EvidenceRecord(aid, (outcomes or {}).get(aid, Outcome.PASS), c) for aid, c in realized.items()}
    m = EvidenceMatrix("t0:u:m0", 0, "r", records, {"rho0": rho0}, task="t0")
    return actions, m, EvidenceModel({aid: HAND_LIK for aid in nominal}), LossParams(100.0, 10.0, 2.0, 0.02, 1.0)


def test_affordable_uses_nominal_cost():
    """Global Constraints: acquisition decisions use nominal costs, and realized costs are charged after
    acquisition, because a realized cost is known only once the action has run. An action whose nominal
    cost fits is acquired however much it then costs; one whose nominal cost does not fit is never
    acquired, however little it would have cost. Standard CI follows the same rule."""
    ci = Cost(0, 1, 1.0, 10)
    for realized in (Cost(), Cost(0, 1, 0.01, 1), Cost(0, 2, 1.0, 50), Cost(0, 30, 40.0, 900)):
        # ch.q's nominal 9 CI-equivalents exceed the ceiling of 2.5 on their own
        actions, m, model, lp = _hand({"f1.repo_ci": ci, "ch.q": Cost(0, 2, 9.0, 50)},
                                      {"f1.repo_ci": ci, "ch.q": realized})
        for p in POLICIES:
            d = run_policy(p, m, actions, lp, model, HAND_B, np.random.default_rng(0))
            assert "ch.q" not in d.acquired, (p, realized)
        # ch.p's nominal 1 CI-equivalent fits beside the suite's
        actions, m, model, lp = _hand({"f1.repo_ci": ci, "ch.p": Cost(0, 2, 1.0, 50)},
                                      {"f1.repo_ci": ci, "ch.p": realized})
        for p in ("full_evidence", "random", "cheapest_first", "uncertainty", "challenger_only"):
            d = run_policy(p, m, actions, lp, model, HAND_B, np.random.default_rng(0))
            assert "ch.p" in d.acquired, (p, realized)
            assert d.overrun == (not HAND_B.within(d.cost)), (p, realized)
    for nominal, realized, ran in ((ci, Cost(0, 40, 9.0, 300), True), (Cost(0, 1, 9.0, 10), Cost(0, 1, 0.5, 10), False)):
        actions, m, model, lp = _hand({"f1.repo_ci": nominal}, {"f1.repo_ci": realized})
        d = run_policy("standard_ci", m, actions, lp, model, HAND_B)
        assert (d.acquired, d.action) == ((["f1.repo_ci"], "acc") if ran else ([], "rej")), (nominal, realized)


def test_overrun_retains_record():
    """Review Focus 2: a nominal cost that fits says nothing certain about the realized one. When the
    realized cost carries the spend past a ceiling, the record has been bought: it stays in `acquired` and
    in the state the terminal rule reads, its cost is charged, acquisition stops, and the decision says
    overrun (stop_reason 'overrun'). Nothing pretends after the fact that the action was unaffordable."""
    nominal = {"f1.repo_ci": Cost(0, 1, 1.0, 10), "f2.y": Cost(0, 1, 0.1, 5), "ch.x": Cost(0, 2, 1.0, 50)}
    realized = dict(nominal, **{"ch.x": Cost(0, 2, 4.0, 50)})          # 4 CI-equivalents against 2.5
    actions, m, model, lp = _hand(nominal, realized, outcomes={"ch.x": Outcome.FAIL_VALIDATED})
    for p in ("full_evidence", "random", "cheapest_first", "uncertainty", "challenger_only"):
        d = run_policy(p, m, actions, lp, model, HAND_B, np.random.default_rng(0))
        assert (d.acquired[-1], d.overrun, d.stop_reason) == ("ch.x", True, "overrun"), (p, d)
        assert (d.trace[-1].realized, d.trace[-1].outcome) == (realized["ch.x"], Outcome.FAIL_VALIDATED), p
        assert d.cost == sum((realized[a] for a in d.acquired), Cost()) and not HAND_B.within(d.cost), p
        # the kept record's validated violation is what makes rejection feasible (Eq. feasible)
        assert d.action == "rej" and d.rho_final > 0.99, (p, d.action)
    # the overrun is a fact about the realized cost: the same choices charged at their nominal cost do not overrun
    actions, m, model, lp = _hand(nominal, nominal, outcomes={"ch.x": Outcome.FAIL_VALIDATED})
    d = run_policy("full_evidence", m, actions, lp, model, HAND_B)
    assert (d.acquired, d.overrun, d.stop_reason) == (["f1.repo_ci", "f2.y", "ch.x"], False, "pool_exhausted")
    # standard CI keeps its overrunning suite and applies its own rule: accept iff the suite passes
    actions, m, model, lp = _hand({"f1.repo_ci": Cost(0, 1, 1.0, 10)}, {"f1.repo_ci": Cost(0, 25, 6.0, 300)})
    d = run_policy("standard_ci", m, actions, lp, model, HAND_B)
    assert (d.acquired, d.action, d.overrun) == (["f1.repo_ci"], "acc", True)


def test_trace_records_each_step():
    """Decision.trace, the raw material of the acquisition analysis: one Step per acquisition, in order,
    with the action, the nominal cost it was selected on, the realized cost it was charged, the posterior
    before and after its outcome, the outcome, and the net gain that chose it (RCCA's; None for a policy
    that computes none). The posteriors chain from the prior to rho_final, each step starting exactly
    where the last one ended."""
    profiles, actions, model, lp, B = _setup()
    by_id = {a.id: a for a in actions}
    rng = np.random.default_rng(13)
    n_steps = {p: 0 for p in POLICIES}
    for i in range(12):
        m = synthesize_matrix(f"t{i}:{'u' if i % 2 else 's'}:m0", 0, "r", f"t{i}", i % 2 == 1, 0.5,
                              rng.normal(-1.0, 1.0), profiles)
        for p in POLICIES:
            d = run_policy(p, m, actions, lp, model, B, np.random.default_rng(i))
            assert [st.action_id for st in d.trace] == d.acquired and d.n_steps == len(d.trace), p
            s = State(model.prior(m), ood=m.ood)
            rho = posterior(model, s)
            for st in d.trace:
                rec = m.records[st.action_id]
                assert (st.nominal, st.realized, st.outcome) == (by_id[st.action_id].nominal_cost, rec.cost,
                                                                 rec.outcome), (p, st)
                assert st.rho_before == rho, (p, st)          # rho_after of step k is rho_before of step k+1
                gain = net_gain(s, by_id[st.action_id], lp, model, B) if p == "rcca" else None
                assert st.gain == gain and (gain is None or gain > 0), (p, st)
                s.observed[st.action_id] = rec.outcome
                s.spent = s.spent + rec.cost
                rho = posterior(model, s)
                assert st.rho_after == rho, (p, st)
            assert d.rho_final == rho and d.cost == s.spent, p
            if p in ("standard_ci", "risk_only"):              # no stopping rule: they stop by construction
                assert d.stop_reason == p
            else:
                allowed = LOOP_STOPS if p == "rcca" else LOOP_STOPS - {"no_positive_gain"}
                assert d.stop_reason in allowed and d.overrun == (d.stop_reason == "overrun"), (p, d.stop_reason)
            n_steps[p] += len(d.trace)
    assert n_steps["risk_only"] == 0 and n_steps["standard_ci"] == 12
    assert n_steps["rcca"] > 0 and n_steps["full_evidence"] >= 3 * 12, n_steps


def test_no_policy_reads_a_realized_cost_before_acquiring():
    """Global Constraints: a realized cost is charged after acquisition and never selects. Change one
    record's realized cost and replay: every choice up to and including the one that acquires that record
    must stay the same, and if the policy never acquired it, the whole decision must. Checked for every
    policy and every record, at twenty times and one twentieth the cost, under ceilings that bind."""
    from dataclasses import replace
    profiles, actions, model, lp, _ = _setup()
    B = Ceilings(tokens=9000, tool_calls=8, ci_equiv=3.0)
    rng = np.random.default_rng(11)
    stopped_short = overruns = 0
    for i in range(8):
        m = synthesize_matrix(f"t{i}:{'u' if i % 2 else 's'}:m0", 0, "r", f"t{i}", i % 2 == 1, 0.5,
                              rng.normal(-0.5, 1.0), profiles)
        for p in POLICIES:
            seed = cell_seed("0", m.candidate_id, m.rep, p)
            d = run_policy(p, m, actions, lp, model, B, np.random.default_rng(seed))
            stopped_short += p == "full_evidence" and len(d.acquired) < len(actions)
            for aid, rec in m.records.items():
                for f in (20.0, 0.05):
                    m2 = replace(m, records=dict(m.records, **{aid: replace(rec, cost=Cost(*(rec.cost.vec() * f)))}))
                    d2 = run_policy(p, m2, actions, lp, model, B, np.random.default_rng(seed))
                    if aid in d.acquired:
                        k = d.acquired.index(aid) + 1
                        assert d2.acquired[:k] == d.acquired[:k], (p, m.candidate_id, aid, f)
                    else:
                        assert (d2.action, d2.acquired) == (d.action, d.acquired), (p, m.candidate_id, aid, f)
                    overruns += d2.overrun
    assert stopped_short > 0, "no ceiling binds, so affordability was never tested"
    assert overruns > 0, "no realized cost overran a ceiling"


def test_overhead_is_measured_and_never_decides(monkeypatch):
    """Decision.overhead_s is the wall-clock the policy itself took, which the capacity analysis adds to a policy's service
    time, so risk-only is not free. It is an outcome only: under a clock that runs at any speed every
    decision, trace and stop reason is the same."""
    import types
    import rcca.decision as decision
    profiles, actions, model, lp, B = _setup()
    rng = np.random.default_rng(17)
    mats = [synthesize_matrix(f"t{i}:{'u' if i % 2 else 's'}:m0", 0, "r", f"t{i}", i % 2 == 1, 0.5,
                              rng.normal(-1.0, 1.0), profiles) for i in range(6)]
    def decide():
        return {(p, i): run_policy(p, m, actions, lp, model, B, np.random.default_rng(i))
                for p in POLICIES for i, m in enumerate(mats)}
    real = decide()
    assert all(d.overhead_s > 0 for d in real.values())
    ticks = iter(range(0, 10 ** 9, 7919))                     # every reading 7919 s after the last
    monkeypatch.setattr(decision, "time", types.SimpleNamespace(perf_counter=lambda: float(next(ticks))))
    fake = decide()
    assert all(d.overhead_s == 7919.0 for d in fake.values())
    strip = lambda d: (d.action, d.acquired, d.trace, d.stop_reason, d.overrun, d.cost, d.rho_final)
    assert {k: strip(d) for k, d in fake.items()} == {k: strip(d) for k, d in real.items()}


def test_nominal_from_fit_partition_only():
    """An action's nominal cost is estimated on the fit partition (Global Constraints): the per-dimension
    median over the fit candidates' records. A test-partition record, however extreme, does not move it,
    and an action with no fit-partition record has no nominal cost and is refused."""
    from rcca.analyze import actions_from_records

    def rec(cid, aid, cost):
        return {"candidate_id": cid, "rep": 0, "action_id": aid, "family": "F1",
                "source": "challenger" if aid.startswith("ch.") else "author_side", "producer": aid,
                "outcome": "pass", "status": "ok", "cost": dict(zip(COST_DIMS, cost))}
    fit = [rec(f"t{i}:s:m0", aid, (0.0, 2.0 + i, 0.5 * (2 + i), 60.0 + i))
           for i in range(5) for aid in ("f1.repo_ci", "ch.property")]
    extreme = [rec("x9:u:m0", aid, (1e9, 1e6, 1e6, 1e7)) for aid in ("f1.repo_ci", "ch.property")]
    ids = {r["candidate_id"] for r in fit}
    acts = {a.id: a for a in actions_from_records(fit + extreme, ids)}
    assert acts["f1.repo_ci"].nominal_cost == Cost(0.0, 4.0, 2.0, 62.0)      # the median of five fit records
    assert acts == {a.id: a for a in actions_from_records(fit, ids)}       # the extreme record moved nothing
    leaky = {a.id: a for a in actions_from_records(fit + extreme, ids | {"x9:u:m0"})}
    assert leaky["f1.repo_ci"].nominal_cost != acts["f1.repo_ci"].nominal_cost   # it would have, if fitted on
    assert (acts["ch.property"].source, acts["f1.repo_ci"].is_repo_ci) == (Source.CHALLENGER, True)
    with pytest.raises(ValueError, match="ch.only_in_test"):
        actions_from_records(fit + [rec("x9:u:m0", "ch.only_in_test", (0.0, 1.0, 1.0, 1.0))], ids)


def test_challenger_only_uses_only_challenger_actions():
    profiles, actions, model, lp, B = _setup()
    m = synthesize_matrix("c", 0, "r", "t", True, 0.7, 0.5, profiles)
    d = run_policy("challenger_only", m, actions, lp, model, B)
    src = {a.id: a.source for a in actions}
    assert all(src[a] == Source.CHALLENGER for a in d.acquired)


def test_clopper_pearson_matches_sec_qual_claims():
    # At N_A=300: <=5% iff K_A<=8; <=2% iff K_A<=1; <=1% only if K_A=0; at N_A=100 cannot fall below 2.95%.
    assert clopper_pearson_upper(8, 300) <= 0.05 < clopper_pearson_upper(9, 300)
    assert clopper_pearson_upper(1, 300) <= 0.02 < clopper_pearson_upper(2, 300)
    assert clopper_pearson_upper(0, 300) <= 0.01 < clopper_pearson_upper(1, 300)
    assert abs(clopper_pearson_upper(0, 100) - 0.0295) < 5e-4
    # 1% needs about 300 accepted with none unsafe, 473 with one unsafe.
    assert min_accepts_for_bound(0.01, 0) == 299
    assert min_accepts_for_bound(0.01, 1) == 473


def test_holm():
    adj, rej = holm([0.01, 0.04, 0.03, 0.5])
    assert rej.tolist() == [True, False, False, False]
    assert np.all(np.diff(np.sort(adj)) >= 0)


def test_gate_false_pass_is_near_alpha_for_both_methods():
    rng = np.random.default_rng(0)
    for method in ("t", "percentile"):
        passes = 0
        for i in range(200):
            d = rng.normal(0.05, 0.1, size=80)  # true Delta_S exactly at margin
            passes += noninferiority_gate(d, 0.05, 0.05, None, 400, i, method).passed
        assert 0.01 <= passes / 200 <= 0.12, method


def test_cluster_bootstrap_t_and_percentile_agree_on_sign():
    rng = np.random.default_rng(3)
    d = rng.normal(-0.1, 0.2, size=80)
    for m in ("t", "percentile"):
        r = cluster_bootstrap(d, None, 500, 0.05, 1, m)
        assert r.upper < 0 and r.p_one_sided_less < 0.05, m


def test_end_to_end_replay_produces_table_quantities():
    profiles, actions, model, lp, B = _setup(ci_detect=0.71)
    mats = []
    rng = np.random.default_rng(7)
    for t in range(24):
        repo = f"repo{t % 4}"
        for mode in ("m0", "m1"):
            lg = rng.normal(-1.0, 1.0)
            for rep in range(2):
                for unsafe in (False, True):
                    cid = f"t{t}:{'u' if unsafe else 's'}:{mode}"
                    mats.append(synthesize_matrix(cid, rep, repo, f"t{t}", unsafe, rng.beta(2, 2), lg + (0.8 if unsafe else 0.0), profiles))
    rows = replay(mats, actions, lp, model, B)
    for p in POLICIES:
        s = summarize(rows, p)
        assert 0 <= s["U"] <= 1 and 0 <= s["S"] <= 1 and abs(s["C"] - (0.5 * s["S"] + 0.5 * s["U"])) < 1e-9
    ci = summarize(rows, "standard_ci")
    assert ci["esc_S"] == 0 and ci["esc_U"] == 0
    assert abs(ci["U"] - 0.29) < 0.12         # synthetic CI misses about what the probe measured
    gate, prim = primary_analysis(rows, m_S=0.10, B=300)
    assert isinstance(gate.passed, bool) and prim.n_clusters == 24


def test_studentized_interval_respects_interleaved_strata():
    """Regression: _strat_resample must keep each resampled column in its own
    stratum's position, because _t_stats slices the resampled matrix by the
    ORIGINAL index positions. Contiguous-block layout mixes strata into every
    slice, inflates the bootstrap SE, collapses the t quantiles toward zero and
    makes the one-sided limit equal the point estimate. Interleaved strata with
    unequal means are what power._task_strata and replay's repo{t%4} produce."""
    rng = np.random.default_rng(0)
    strata = np.arange(80) % 4
    x = np.where(strata == 0, 1.0, 0.0) + rng.normal(0, 0.01, 80)
    se0 = _strat_se(x, strata)
    up = one_sided_upper_t(x, strata, 0.05, 2000, 1)
    assert up - x.mean() > 1.2 * se0, "upper limit collapsed onto the estimate"
    assert up - x.mean() < 2.5 * se0


def test_singleton_stratum_is_rejected_not_silently_noise_free():
    x = np.array([0.1, 0.2, 0.3, 0.9])
    strata = np.array([0, 0, 0, 1])          # stratum 1 has one task
    import pytest
    with pytest.raises(ValueError, match="stratum"):
        _strat_se(x, strata)
    with pytest.raises(ValueError, match="stratum"):
        one_sided_upper_t(x, strata, 0.05, 200, 0)


def test_replay_seed_is_stable_across_processes():
    """Common random numbers (spec §5) need the policy RNG seeded from a stable
    digest, as `cell_seed` is. Builtin `hash()` on a tuple holding strings is
    salted per process by PYTHONHASHSEED, so the `random` policy's acquisition
    order, and with it its decisions, differed between two runs of the same
    analysis over the same records."""
    script = textwrap.dedent("""
        import numpy as np
        from rcca.matrix import Ceilings, default_action_profiles, synthesize_matrix
        from rcca.decision import EvidenceModel, LossParams
        from rcca.replay import replay
        profiles = default_action_profiles(0.71)
        actions = [p.action for p in profiles]
        lik = {p.action.id: (np.array(p.p_o_given_unsafe), np.array(p.p_o_given_safe))
               for p in profiles}
        model = EvidenceModel(lik, s_tilde=0.6)
        lp = LossParams(kappa=100.0, lambda_R=10.0, lambda_E=2.0, tau_qual=0.02, eta=1.0)
        B = Ceilings(tokens=20000, tool_calls=12, ci_equiv=6.0)
        rng = np.random.default_rng(7)
        mats = []
        for t in range(6):
            for unsafe in (False, True):
                cid = f"t{t}:{'u' if unsafe else 's'}:m0"
                mats.append(synthesize_matrix(cid, 0, "r0", f"t{t}", unsafe, 0.5,
                                              rng.normal(-1.0, 1.0), profiles))
        rows = replay(mats, actions, lp, model, B, policies=("random",))
        print(";".join(f"{r.candidate_id}|{r.action}|{r.n_acquired}" for r in rows))
    """)
    root = Path(__file__).resolve().parent.parent
    outs = []
    for hashseed in ("0", "1"):
        r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                           env={**os.environ, "PYTHONHASHSEED": hashseed}, cwd=root, check=True)
        outs.append(r.stdout.strip())
    assert outs[0] == outs[1], "the random policy's decisions depend on PYTHONHASHSEED"


def test_power_main_honours_the_stratum_count(tmp_path):
    """The number of repository strata is a design parameter: the paper's
    subjects table carries four confirmatory repositories and the setup
    spec (S1) decides two, so the simulation has to be runnable at either.
    Strata enter only the bootstrap, not the data-generating process."""
    from rcca.power import main as power_main
    out = tmp_path / "p.json"
    res = power_main(str(out), n_sim=2, B=20, method="t", n_repos=2)
    assert res["design"]["n_repos"] == 2
    assert json.loads(out.read_text())["design"]["n_repos"] == 2


def test_power_main_honours_the_measured_u_ci_baseline(tmp_path):
    """`U_CI` is an input to the design, not a constant: the probe measured 0.29
    under its own incidental coverage mix and the frozen `covered_share = 0.50`
    implies 0.31, so the simulation has to be runnable at either. The design
    assumption (0.40) and the mutation-only floor (0.174) are fixed by the probe
    and do not move with the baseline; the two reductions derived from it do."""
    from rcca.power import main as power_main

    out29 = tmp_path / "p29.json"
    r29 = power_main(str(out29), n_sim=2, B=20, method="t")
    assert r29["design"]["u_ci"] == 0.29
    pairs29 = {(c["U_CI"], c["U_RCCA"]) for c in r29["primary"]}
    assert pairs29 == {(0.40, 0.25), (0.29, 0.18), (0.29, 0.14), (0.174, 0.109)}

    out31 = tmp_path / "p31.json"
    r31 = power_main(str(out31), n_sim=2, B=20, method="t", u_ci=0.31)
    assert r31["design"]["u_ci"] == 0.31
    assert json.loads(out31.read_text())["design"]["u_ci"] == 0.31
    pairs31 = {(c["U_CI"], c["U_RCCA"]) for c in r31["primary"]}
    # same 37.5 % relative and same 0.15 absolute reduction, carried to 0.31
    assert pairs31 == {(0.40, 0.25), (0.31, 0.19), (0.31, 0.16), (0.174, 0.109)}
    # the baseline also drives the type-I and gate cells, which are run at it
    assert {c["U_CI"] for c in r31["type1"]} == {0.40, 0.31}
    assert {c["U_CI"] for c in r31["gate"]} == {0.31}

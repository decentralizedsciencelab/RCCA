"""Accrual operating characteristics.

The paper fixes M = 820 sampled calibration changes, N_A = 300 accepted
observations, and stopping that depends only on counts. Two things follow that
no artifact currently states: accrual can stop short, and even a full accrual
qualifies only if K_A is small enough. Both are computable before any run.
"""
import json
import pytest
from scipy.stats import binom

from rcca.qualify import Design, simulate, main
from rcca.stats import clopper_pearson_upper


def test_full_coverage_always_reaches_the_target():
    r = simulate(coverage=1.0, theta=0.0, d=Design(n_sim=200), seed=1)
    assert r["reach_rate"] == 1.0 and r["n_a_median"] == 300


def test_accrual_matches_the_exact_binomial_it_implies():
    """Processing in temporal order and stopping at the 300th acceptance makes
    N_A = min(300, Binom(820, C)), so the reach rate is an exact survival
    probability. If the simulation disagrees with it, the simulation is wrong."""
    d = Design(n_sim=4000)
    for C in (0.30, 0.366, 0.45):
        exact = float(binom.sf(d.n_target - 1, d.m_sampled, C))
        got = simulate(coverage=C, theta=0.01, d=d, seed=7)["reach_rate"]
        assert got == pytest.approx(exact, abs=0.025), (C, got, exact)


def test_the_critical_coverage_is_the_ratio_the_design_implies():
    """N_A = 300 out of M = 820 cannot happen below C = 0.366 in expectation, so
    the design carries an acceptance-coverage floor that the protocol never
    states. Below it the protocol's own rule makes the result a feasibility
    outcome however safe the policy is."""
    d = Design(n_sim=2000)
    assert simulate(coverage=0.30, theta=0.0, d=d, seed=3)["reach_rate"] < 0.05
    assert simulate(coverage=0.45, theta=0.0, d=d, seed=3)["reach_rate"] > 0.95


def test_a_perfect_policy_at_full_accrual_meets_the_one_percent_target():
    assert clopper_pearson_upper(0, 300, 0.05) == pytest.approx(0.00993, abs=5e-5)
    r = simulate(coverage=1.0, theta=0.0, d=Design(n_sim=200), seed=1)
    assert r["qualify"]["0.01"] == 1.0


@pytest.mark.parametrize("theta", [0.005, 0.01, 0.02, 0.03])
def test_qualification_probability_is_the_exact_binomial_on_K_A(theta):
    """At a full accrual the 5 % target is exactly K_A <= 8, so the probability of
    qualifying is P(Binom(300, theta) <= 8) and nothing else. Checking against the
    exact distribution tests the accrual, the bound and the conjunction together;
    a threshold would only test that the number is small.

    It is also the number that matters for the design: at theta = 3 % the policy
    can be entirely correct about the bound and still fail to qualify about half
    the time, because 300 observations cannot separate 3 % from 5 %."""
    exact = float(binom.cdf(8, 300, theta))
    got = simulate(coverage=1.0, theta=theta, d=Design(n_sim=4000), seed=11)["qualify"]["0.05"]
    assert got == pytest.approx(exact, abs=0.025), (theta, got, exact)


def test_main_writes_a_grid(tmp_path):
    out = tmp_path / "q.json"
    res = main(str(out), n_sim=100, coverages=(0.4, 1.0), thetas=(0.0, 0.02))
    assert len(res["cells"]) == 4
    assert json.loads(out.read_text())["design"]["n_target"] == 300

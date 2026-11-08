import pytest

from poker_ai.solvers import CFRSolver, KuhnPoker, LeducPoker, expected_value, exploitability
from poker_ai.solvers.cfr import best_response_value


@pytest.fixture(scope="module")
def kuhn():
    return KuhnPoker().compile()


@pytest.fixture(scope="module")
def leduc():
    return LeducPoker().compile()


def test_tree_sizes(kuhn, leduc):
    assert kuhn.num_infosets == 12
    assert leduc.num_infosets == 288  # suit-isomorphic Leduc


def test_kuhn_payoffs_are_zero_sum_and_bounded(kuhn):
    terminals = [u for k, u in zip(kuhn.kind, kuhn.utility) if k == -2]
    assert len(terminals) == 30 and set(abs(u) for u in terminals) == {1.0, 2.0}


def test_known_kuhn_equilibrium_has_zero_exploitability(kuhn):
    a = 0.2
    nash = {"J:": [1 - a, a], "Q:": [1, 0], "K:": [1 - 3 * a, 3 * a], "J:pb": [1, 0], "Q:pb": [2 / 3 - a, a + 1 / 3],
            "K:pb": [0, 1], "J:b": [1, 0], "Q:b": [2 / 3, 1 / 3], "K:b": [0, 1], "J:p": [2 / 3, 1 / 3], "Q:p": [1, 0], "K:p": [0, 1]}
    strat = [nash[k] for k in kuhn.infoset_keys]
    assert abs(expected_value(kuhn, strat) + 1 / 18) < 1e-12
    assert exploitability(kuhn, strat) < 1e-12


def test_uniform_strategy_is_exploitable(kuhn):
    uniform = [[0.5, 0.5] for _ in kuhn.infoset_keys]
    assert exploitability(kuhn, uniform) > 0.1
    # a best response can never do worse than playing the strategy itself
    assert best_response_value(kuhn, uniform, 0) >= expected_value(kuhn, uniform) - 1e-12


@pytest.mark.parametrize("variant", ["cfr", "cfr+"])
def test_cfr_solves_kuhn(kuhn, variant):
    s = CFRSolver(kuhn, variant)
    s.iterate(200)
    early = exploitability(kuhn, s.average_strategy())
    s.iterate(1800)
    avg = s.average_strategy()
    assert exploitability(kuhn, avg) < min(early, 0.006)
    assert abs(expected_value(kuhn, avg) + 1 / 18) < 0.003


def test_cfr_converges_on_leduc(leduc):
    s = CFRSolver(leduc)
    s.iterate(100)
    e100 = exploitability(leduc, s.average_strategy())
    s.iterate(900)
    avg = s.average_strategy()
    e1000 = exploitability(leduc, avg)
    assert e1000 < e100 / 3 and e1000 < 0.03
    assert abs(expected_value(leduc, avg) - (-0.0856)) < 0.005


def test_bad_variant():
    with pytest.raises(ValueError):
        CFRSolver(KuhnPoker().compile(), "magic")

import random

import numpy as np
import pytest

from poker_ai.agents import make_agent
from poker_ai.agents.opponent_model import OpponentModel
from poker_ai.engine import Action, HandState
from poker_ai.learning.features import N_ACTIONS, OBS_DIM, NumpyPolicy, encode, legal_mask, to_action
from poker_ai.learning.ppo import ActorCritic, gae, training_status
from poker_ai.learning.rollout import Batch
from poker_ai.sim.runner import play_hand

from .conftest import rigged_hand


def test_encoding_shape_and_no_hidden_information():
    def enc(opp_cards):
        h = rigged_hand([10000, 10000], 0, 50, 100, {0: "Ah Kh", 1: opp_cards}, board="Kc 7h 2h 5s 8d")
        h.apply(Action.raise_to(300)); h.apply(Action.call()); h.apply(Action.bet(200))
        return encode(h.view(0), OpponentModel().get(1), random.Random(0))
    x1, x2 = enc("As Ad"), enc("4c 3d")
    assert x1.shape == (OBS_DIM,) and x1.dtype == np.float32
    assert np.array_equal(x1, x2)  # opponent's hidden cards cannot change our features


def test_mask_matches_engine_legality_in_random_states():
    rng = random.Random(3)
    for _ in range(300):
        h = HandState([rng.randint(100, 30000), rng.randint(100, 30000)], rng.randrange(2), 50, 100, rng=rng)
        while not h.terminal:
            v = h.view(h.to_act)
            m = legal_mask(v)
            assert m[1]
            for i in range(N_ACTIONS):
                if m[i]:
                    assert v.legal.is_legal(to_action(v, i)), (i, v.legal)
            assert m[0] == v.legal.can_fold
            choices = np.flatnonzero(m)
            h.apply(to_action(v, int(rng.choice(choices))))


def test_numpy_forward_matches_torch():
    import torch

    net = ActorCritic()
    x = np.random.default_rng(0).normal(size=(5, OBS_DIM)).astype(np.float32)
    mask = np.array([[True, True, False, True, True]] * 5)
    logits, v = net(torch.tensor(x), torch.tensor(mask))
    tp = torch.softmax(logits, -1).detach().numpy()
    npol = NumpyPolicy(net.export())
    for i in range(5):
        p, val = npol.forward(x[i], mask[i])
        assert np.allclose(p, tp[i], atol=1e-5) and abs(val - v[i].item()) < 1e-5
        assert p[2] == 0.0  # masked action has zero probability


def test_gae_resets_between_hands():
    b = Batch(values=[0.1, 0.2, 0.3], rewards=[0.0, 1.0, -1.0], episode_end=[False, True, True], actions=[0, 0, 0])
    adv, ret = gae(b, lam=1.0)
    assert ret.tolist() == pytest.approx([1.0, 1.0, -1.0])  # λ=1: return = final reward of that hand


def test_status_labels_are_evidence_based():
    good = {"updates": 10, "final_eval": {o: {"bb100": 50, "ci95": 20} for o in ("rule_balanced", "rule_tight", "rule_lag")}}
    weak = {"updates": 10, "final_eval": {**good["final_eval"], "rule_lag": {"bb100": 10, "ci95": 30}}}
    assert training_status(good) == "trained"
    assert training_status(weak) == "partially trained"
    assert training_status({"updates": 0}) == "untrained"


def test_neural_agent_plays_legally_and_reports_status():
    a = make_agent("neural", seed=1)
    assert a.training_status in ("trained", "partially trained", "untrained")
    rng = random.Random(5)
    for _ in range(20):
        h = HandState([10000, 10000, 10000], rng.randrange(3), 50, 100, rng=rng)
        play_hand(h, {0: a, 1: make_agent("rule_lag", seed=2), 2: make_agent("calling_station")})
    assert set(a.last_explanation["policy"]) <= {"fold", "call", "half_pot", "pot", "all_in"}

"""State encoding, action space and action masking for the neural agent.

Only information in the acting player's :class:`PlayerView` is used (own cards,
public cards and actions, and an opponent model built from public history), so
the network can never see hidden cards.
"""

from __future__ import annotations

import random

import numpy as np

from poker_ai.agents.equity import equity
from poker_ai.agents.features import is_flush_or_straight_draw
from poker_ai.agents.opponent_model import OpponentStats
from poker_ai.agents.preflop import preflop_equity
from poker_ai.engine import Action, PlayerView, Street, evaluate

ACTIONS = ("fold", "call", "half_pot", "pot", "all_in")
N_ACTIONS = len(ACTIONS)
_FRACTIONS = {2: 0.5, 3: 1.0}
_LAST_ACTION = {"fold": 0, "check": 1, "call": 2, "bet": 3, "raise": 4}
OBS_DIM = 155


def _raise_to(view: PlayerView, fraction: float) -> int:
    legal = view.legal
    assert legal is not None
    return legal.clamp_raise(int(round(view.current_bet + fraction * (view.pot + legal.to_call))))


def legal_mask(view: PlayerView) -> np.ndarray:
    legal = view.legal
    assert legal is not None
    m = np.zeros(N_ACTIONS, dtype=bool)
    m[0] = legal.can_fold
    m[1] = True
    if legal.raise_type is not None:
        m[4] = True
        seen = {legal.max_raise_to}
        for i, f in _FRACTIONS.items():
            to = _raise_to(view, f)
            if to not in seen:
                m[i] = True
                seen.add(to)
    return m


def to_action(view: PlayerView, index: int) -> Action:
    legal = view.legal
    assert legal is not None
    if index == 0:
        return Action.fold() if legal.can_fold else legal.passive()
    if index == 1 or legal.raise_type is None:
        return legal.passive()
    if index == 4:
        return legal.all_in()
    return Action(legal.raise_type, _raise_to(view, _FRACTIONS[index]))


def main_opponent(view: PlayerView) -> int | None:
    """The opponent the features describe: the one with the largest commitment this street."""
    opps = view.opponents
    if not opps:
        return None
    return max(opps, key=lambda o: (o.street_commit, o.stack)).seat


def encode(view: PlayerView, opp: OpponentStats | None, rng: random.Random, equity_sims: int = 80) -> np.ndarray:
    x = np.zeros(OBS_DIM, dtype=np.float32)
    i = 0
    for c in view.hole:
        x[i + c] = 1.0
    i += 52
    for c in view.board:
        x[i + c] = 1.0
    i += 52
    r1, r2 = sorted((view.hole[0] >> 2, view.hole[1] >> 2), reverse=True)
    x[i : i + 4] = (r1 / 12, r2 / 12, float((view.hole[0] & 3) == (view.hole[1] & 3)), float(r1 == r2))
    i += 4
    street = min(int(view.street), 3)
    x[i + street] = 1.0
    i += 4
    n_opp = max(1, view.num_live - 1)
    if view.street is Street.PREFLOP:
        eq = preflop_equity(view.hole[0], view.hole[1], n_opp)
        made = -1
    else:
        eq = equity(view.hole, view.board, n_opp, equity_sims, rng).equity
        made = evaluate(list(view.hole) + list(view.board)) >> 20
    x[i] = eq
    i += 1
    if made >= 0:
        x[i + made] = 1.0
    i += 9
    x[i] = float(len(view.board) in (3, 4) and is_flush_or_straight_draw(view.hole, view.board))
    i += 1
    x[i] = float(view.button == view.seat)
    i += 1
    unit = 100.0 * view.big_blind
    me = view.me
    legal = view.legal
    to_call = legal.to_call if legal else 0
    opp_seat = main_opponent(view)
    o = view.players[opp_seat] if opp_seat is not None else me
    x[i : i + 6] = (view.pot / unit, to_call / unit, me.stack / unit, o.stack / unit,
                    me.street_commit / unit, o.street_commit / unit)
    i += 6
    x[i] = to_call / (view.pot + to_call) if to_call else 0.0
    x[i + 1] = min(5.0, min(me.stack, o.stack) / max(1, view.pot)) / 5.0
    i += 2
    raises = [0, 0, 0, 0]
    mine = [0, 0, 0, 0]
    theirs = [0, 0, 0, 0]
    last = -1
    for e in view.history:
        if e["type"] != "action":
            continue
        st = ("preflop", "flop", "turn", "river").index(e["street"])
        aggressive = e["action"] in ("bet", "raise")
        raises[st] += aggressive
        if e["seat"] == view.seat:
            mine[st] += aggressive
        else:
            theirs[st] += aggressive
            last = _LAST_ACTION[e["action"]]
    x[i] = raises[street] / 4
    i += 1
    x[i : i + 4] = np.array(mine, dtype=np.float32) / 3
    x[i + 4 : i + 8] = np.array(theirs, dtype=np.float32) / 3
    i += 8
    x[i + (last if last >= 0 else 5)] = 1.0
    i += 6
    if opp is not None:
        x[i : i + 7] = (opp["vpip"], opp["pfr"], opp["agg"], opp["fold_to_bet"], opp["fold_pf"], opp["wtsd"],
                        opp.hands / (opp.hands + 50))
    else:
        x[i : i + 7] = (0.35, 0.2, 0.4, 0.5, 0.5, 0.3, 0.0)
    i += 7
    x[i] = float(n_opp > 1)
    i += 1
    assert i <= OBS_DIM
    return x


class NumpyPolicy:
    """Actor-critic MLP forward pass in NumPy (no PyTorch needed at play time)."""

    def __init__(self, params: dict[str, np.ndarray]) -> None:
        self.p = {k: np.asarray(v, dtype=np.float32) for k, v in params.items()}

    @classmethod
    def load(cls, path) -> NumpyPolicy:
        with np.load(path) as data:
            return cls({k: data[k] for k in data.files})

    def forward(self, x: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float]:
        p = self.p
        h = np.tanh(x @ p["w1"] + p["b1"])
        h = np.tanh(h @ p["w2"] + p["b2"])
        logits = h @ p["wp"] + p["bp"]
        logits = np.where(mask, logits, -1e9)
        logits = logits - logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        value = float((h @ p["wv"] + p["bv"])[0])
        return probs, value

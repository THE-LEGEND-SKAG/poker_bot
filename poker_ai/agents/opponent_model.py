"""Opponent modelling from public information only (Level 3).

Two parts:

* :class:`OpponentModel` keeps per-seat tendencies as Beta distributions
  (Bayesian frequency estimates). Each stat starts from a population prior worth a
  few hands and is updated from every observed hand, so early estimates are
  cautious and sharpen as evidence accumulates.

* :func:`estimate_ranges` replays the current hand's public action history and
  applies Bayes' rule to each opponent's 1,326 possible holdings::

      P(hand | actions) ∝ P(hand) · Π P(action | hand, opponent's tendencies)

  The action likelihoods are smooth functions of hand strength (a preflop
  percentile, or the current made-hand percentile on the board with a bonus for
  draws) whose thresholds come from the opponent's own statistics: a player who
  raises 40% of hands is credited with a much wider raising range than one who
  raises 8%.

Everything here reads only ``PlayerView`` history and cards revealed at
showdown, so it is equally valid for a bot or for the human's Coach panel.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np

from poker_ai.engine import PlayerView, evaluate
from poker_ai.engine.cards import Card

from .equity import Combo
from .features import is_flush_or_straight_draw
from .preflop import all_classes, class_combos, classes_by_strength, hand_class

# ---------------------------------------------------------------------------
# Combo indexing
ALL_COMBOS: tuple[Combo, ...] = tuple((a, b) for a in range(52) for b in range(a + 1, 52))
COMBO_INDEX = {c: i for i, c in enumerate(ALL_COMBOS)}
_CARD_MASK = np.zeros((52, len(ALL_COMBOS)), dtype=bool)
for _i, (_a, _b) in enumerate(ALL_COMBOS):
    _CARD_MASK[_a, _i] = True
    _CARD_MASK[_b, _i] = True


def blocked_mask(cards: Sequence[Card]) -> np.ndarray:
    """True for combos that contain any of ``cards``."""
    if not cards:
        return np.zeros(len(ALL_COMBOS), dtype=bool)
    return _CARD_MASK[list(cards)].any(axis=0)


def _preflop_percentile() -> np.ndarray:
    """0.0 = best starting hand .. 1.0 = worst, combo-weighted."""
    order = classes_by_strength()
    pct = np.zeros(len(ALL_COMBOS))
    seen = 0
    for cls in order:
        combos = class_combos(cls)
        mid = (seen + len(combos) / 2) / 1326
        for a, b in combos:
            pct[COMBO_INDEX[(min(a, b), max(a, b))]] = mid
        seen += len(combos)
    return pct


PREFLOP_PCT = _preflop_percentile()
CLASS_ORDER = all_classes()
_CLASS_OF = np.array([CLASS_ORDER.index(hand_class(a, b)) for a, b in ALL_COMBOS])


@lru_cache(maxsize=256)
def board_strength(board: tuple[Card, ...]) -> tuple[np.ndarray, np.ndarray]:
    """(strength percentile 0..1 with 1 = nuts, draw flag) for every combo on ``board``."""
    blocked = blocked_mask(board)
    scores = np.full(len(ALL_COMBOS), -1, dtype=np.int64)
    draws = np.zeros(len(ALL_COMBOS), dtype=bool)
    b = list(board)
    check_draws = len(board) < 5
    for i, (x, y) in enumerate(ALL_COMBOS):
        if blocked[i]:
            continue
        scores[i] = evaluate([x, y] + b)
        if check_draws:
            draws[i] = is_flush_or_straight_draw((x, y), b)
    live = ~blocked
    ranks = np.zeros(len(ALL_COMBOS))
    order = np.argsort(scores[live], kind="stable")
    live_idx = np.flatnonzero(live)
    sorted_scores = scores[live][order]
    # average rank for ties, scaled to 0..1
    uniq, first = np.unique(sorted_scores, return_index=True)
    counts = np.diff(np.append(first, len(sorted_scores)))
    avg = first + (counts - 1) / 2
    lookup = dict(zip(uniq.tolist(), (avg / max(1, len(sorted_scores) - 1)).tolist()))
    ranks[live_idx] = [lookup[s] for s in scores[live_idx].tolist()]
    return ranks, draws


# ---------------------------------------------------------------------------
# Tendencies
@dataclass
class BetaStat:
    a: float
    b: float

    def update(self, success: bool) -> None:
        if success:
            self.a += 1
        else:
            self.b += 1

    @property
    def mean(self) -> float:
        return self.a / (self.a + self.b)

    @property
    def n(self) -> float:
        return self.a + self.b


# Population priors, each worth a few observations.
PRIORS: dict[str, tuple[float, float]] = {
    "vpip": (1.4, 2.6),          # voluntarily put chips in preflop      ~35%
    "pfr": (0.8, 3.2),           # raised preflop                        ~20%
    "fold_pf": (2.0, 2.0),       # folded preflop when facing a raise    ~50%
    "agg": (1.6, 2.4),           # bet/raised when not facing a bet       ~40%
    "fold_to_bet": (2.0, 2.0),   # folded postflop when facing a bet      ~50%
    "raise_vs_bet": (0.4, 3.6),  # raised postflop when facing a bet      ~10%
    "wtsd": (1.2, 2.8),          # went to showdown after seeing a flop   ~30%
}
STAT_LABELS = {"vpip": "VPIP", "pfr": "PFR", "fold_pf": "Fold to raise", "agg": "Bet when checked to",
               "fold_to_bet": "Fold to bet", "raise_vs_bet": "Raise vs bet", "wtsd": "Went to showdown"}


@dataclass
class OpponentStats:
    stats: dict[str, BetaStat] = field(default_factory=lambda: {k: BetaStat(*v) for k, v in PRIORS.items()})
    hands: int = 0

    def __getitem__(self, key: str) -> float:
        return self.stats[key].mean

    def confidence(self, key: str) -> float:
        """0 at the prior, approaching 1 with many observations."""
        extra = self.stats[key].n - sum(PRIORS[key])
        return extra / (extra + 12)

    def as_dict(self) -> dict[str, Any]:
        return {"hands": self.hands, **{k: round(s.mean, 3) for k, s in self.stats.items()},
                "samples": {k: int(round(s.n - sum(PRIORS[k]))) for k, s in self.stats.items()}}


def _actions(view: PlayerView) -> list[dict[str, Any]]:
    return [e for e in view.history if e["type"] == "action"]


class OpponentModel:
    """Per-seat tendencies of everyone except ``me``, learned from finished hands."""

    def __init__(self, me: int | None = None) -> None:
        self.me = me
        self.seats: dict[int, OpponentStats] = {}

    def get(self, seat: int) -> OpponentStats:
        return self.seats.setdefault(seat, OpponentStats())

    def observe_hand(self, view: PlayerView) -> None:
        """Update stats from a finished hand's public history (call once per hand)."""
        me = view.seat if self.me is None else self.me
        acts = _actions(view)
        showdown = {s for p in view.players if p.shown_cards is not None for s in [p.seat]}
        saw_flop: set[int] = set()
        live = {p.seat for p in view.players if p.in_hand}
        street_bet: dict[str, bool] = {}
        preflop_raised = False
        did: dict[int, set[str]] = {}
        for e in acts:
            s, a, st = e["seat"], e["action"], e["street"]
            if st == "preflop":
                d = did.setdefault(s, set())
                d.add("acted")
                if a in ("call", "raise"):
                    d.add("vpip")
                if a == "raise":
                    d.add("pfr")
                if s != me and preflop_raised and a != "raise":
                    self.get(s).stats["fold_pf"].update(a == "fold")
                if a == "raise":
                    preflop_raised = True
            else:
                saw_flop.add(s)
                facing = street_bet.get(st, False)
                if s != me:
                    if facing:
                        self.get(s).stats["fold_to_bet"].update(a == "fold")
                        if a != "fold":
                            self.get(s).stats["raise_vs_bet"].update(a == "raise")
                    else:
                        self.get(s).stats["agg"].update(a in ("bet", "raise"))
                if a in ("bet", "raise"):
                    street_bet[st] = True
            if a == "fold":
                live.discard(s)
        for s, d in did.items():
            if s == me:
                continue
            st = self.get(s)
            st.hands += 1
            st.stats["vpip"].update("vpip" in d)
            st.stats["pfr"].update("pfr" in d)
        for s in saw_flop:
            if s != me:
                self.get(s).stats["wtsd"].update(s in showdown)


# ---------------------------------------------------------------------------
# Range estimation
def _sig(x: np.ndarray | float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def _preflop_likelihood(action: str, raises_before: int, st: OpponentStats) -> np.ndarray:
    p = PREFLOP_PCT
    vpip, pfr = st["vpip"], st["pfr"]
    if raises_before == 0:
        raise_thr, call_thr = pfr, vpip
    else:  # facing a raise: re-raising and calling ranges both tighten
        raise_thr = max(0.02, pfr * 0.3 ** raises_before)
        call_thr = max(raise_thr + 0.03, min(vpip, (1 - st["fold_pf"]) * 0.6 ** (raises_before - 1)))
    if action == "raise":
        return 0.02 + 0.98 * _sig((raise_thr - p) / 0.03)
    if action == "call":
        strong_trap = 0.35 * _sig((raise_thr * 0.5 - p) / 0.02)  # some premiums just call
        return 0.02 + _sig((call_thr - p) / 0.04) * (1 - _sig((raise_thr - p) / 0.03)) + strong_trap
    return np.ones_like(p)  # check / fold carry no preflop range information we use


def _postflop_likelihood(action: str, facing: bool, board: tuple[Card, ...], st: OpponentStats) -> np.ndarray:
    s, draw = board_strength(board)
    agg, fold, rvb = st["agg"], st["fold_to_bet"], st["raise_vs_bet"]
    d = draw.astype(float)
    if not facing:
        value_thr = 1 - 0.6 * agg  # aggressive players bet a wider value range
        bet_value = _sig((s - value_thr) / 0.06)
        if action in ("bet", "raise"):
            return np.clip(bet_value + 0.5 * agg * (0.25 + 0.75 * d), 0.01, 1.0)
        return 1.0 - 0.85 * bet_value  # check: strong hands mostly bet
    raise_thr = 1 - min(0.5, 2.5 * rvb)
    if action == "raise":
        return np.clip(_sig((s - raise_thr) / 0.04) + 0.15 * agg * d, 0.01, 1.0)
    if action == "call":
        cont = _sig((s - fold) / 0.08)
        return np.clip(cont * (1 - 0.6 * _sig((s - raise_thr) / 0.04)) + 0.45 * d, 0.01, 1.0)
    return np.ones_like(s)


def estimate_ranges(view: PlayerView, model: OpponentModel) -> dict[int, np.ndarray]:
    """Weights over ``ALL_COMBOS`` for every live opponent of ``view.seat``."""
    known = list(view.hole) + list(view.board)
    base = np.where(blocked_mask(known), 0.0, 1.0)
    weights = {p.seat: base.copy() for p in view.players if p.in_hand and not p.folded and p.seat != view.seat}
    board: tuple[Card, ...] = ()
    street = "preflop"
    raises_pf = 0
    facing_bet = False
    for e in view.history:
        t = e["type"]
        if t == "street":
            board = tuple(e["board"])
            street = e["street"]
            facing_bet = False
            continue
        if t != "action":
            continue
        seat, a = e["seat"], e["action"]
        if seat in weights:
            st = model.get(seat)
            if street == "preflop":
                weights[seat] *= _preflop_likelihood(a, raises_pf, st)
            else:
                weights[seat] *= _postflop_likelihood(a, facing_bet, board, st)
        if street == "preflop" and a == "raise":
            raises_pf += 1
        if street != "preflop" and a in ("bet", "raise"):
            facing_bet = True
    for seat in weights:
        if weights[seat].sum() <= 0:  # actions inconsistent with the model: fall back to uniform
            weights[seat] = base.copy()
    return weights


def strength_order(view: PlayerView) -> np.ndarray:
    """Current strength of each combo, higher = stronger (for sorting ranges)."""
    if not view.board:
        return -PREFLOP_PCT
    s, _ = board_strength(tuple(view.board))
    return s


def sample_range(weights: np.ndarray, strength: np.ndarray, rng: np.random.Generator, n: int = 400) -> list[Combo]:
    """A weighted range as a combo list (with repeats), strongest first."""
    p = weights / weights.sum()
    idx = rng.choice(len(ALL_COMBOS), size=n, p=p)
    idx = idx[np.argsort(-strength[idx], kind="stable")]
    return [ALL_COMBOS[i] for i in idx]


def class_grid(weights: np.ndarray, blocked: np.ndarray) -> list[float]:
    """Average weight per hand class in 13×13 grid order, scaled so the max is 1."""
    sums = np.bincount(_CLASS_OF, weights=weights, minlength=169)
    counts = np.bincount(_CLASS_OF, weights=(~blocked).astype(float), minlength=169)
    avg = np.divide(sums, counts, out=np.zeros(169), where=counts > 0)
    top = avg.max()
    return (avg / top if top > 0 else avg).round(3).tolist()

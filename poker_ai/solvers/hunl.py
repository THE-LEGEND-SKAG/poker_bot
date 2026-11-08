"""Abstracted heads-up No-Limit Hold'em and an external-sampling MCCFR trainer.

Abstraction (deliberately small enough to train on a laptop CPU)
----------------------------------------------------------------
* **Stacks / blinds:** 100 BB effective, blinds 1/2 (abstract chips), player 0 is
  the button/small blind (acts first preflop, last postflop).
* **Bets:** fold ``f``, check/call ``c``, bet/raise ½ pot ``h``, pot ``p``,
  all-in ``a``. Sizes use the standard "call, then raise by f × pot" rule and are
  bumped to the minimum raise. At most 3 sized raises preflop, 2 on the flop and
  1 on the turn and river (all-in stays available one level further). The public
  tree then has ~13k decision nodes and ~129k information sets.
* **Cards:** preflop, the 169 strategically distinct starting hands are kept
  separate (lossless). Postflop, hands are bucketed by Monte Carlo equity against
  a random hand into 8 / 8 / 10 equal-width buckets on flop / turn / river.
  Information sets use the current street's bucket plus the full public betting
  history (imperfect recall of earlier buckets, as in many published abstractions).

Training
--------
External-sampling MCCFR (Lanctot et al. 2009): each iteration samples one deal;
for the traversing player every action is explored, opponent actions are sampled
from the current regret-matching strategy. The opponent's strategy is accumulated
into the average with linear (iteration-number) weighting. Several worker
processes can train independently; their regret and strategy sums are added.
"""

from __future__ import annotations

import gzip
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from poker_ai.agents.equity import equity
from poker_ai.agents.preflop import all_classes, hand_class
from poker_ai.engine import evaluate
from poker_ai.engine.cards import Card

SB, BB, STACK = 1, 2, 200
MAX_RAISES = (3, 2, 1, 1)  # sized raises per street (all-in allowed one level further)
BUCKETS = (169, 8, 8, 10)
ACTION_FRACTIONS = {"h": 0.5, "p": 1.0}
BLUEPRINT_PATH = Path(__file__).resolve().parents[1] / "data" / "hunl_blueprint.json.gz"
_CLASS_INDEX = {c: i for i, c in enumerate(all_classes())}


def bucket(hole: tuple[Card, Card] | list[Card], board: list[Card], sims: int = 120, rng: random.Random | None = None) -> int:
    """Abstract card bucket for ``hole`` on ``board`` (street = len(board))."""
    if not board:
        return _CLASS_INDEX[hand_class(hole[0], hole[1])]
    street = {3: 1, 4: 2, 5: 3}[len(board)]
    eq = equity(list(hole), list(board), 1, sims, rng or random.Random()).equity
    return min(BUCKETS[street] - 1, int(eq * BUCKETS[street]))


@dataclass
class AState:
    """Public state of the abstract game."""

    street: int = 0
    contrib: list[int] = field(default_factory=lambda: [SB, BB])
    street_bet: list[int] = field(default_factory=lambda: [SB, BB])
    to_act: int = 0
    hist: list[str] = field(default_factory=lambda: [""])
    raises: int = 0
    last_inc: int = BB
    acted: list[bool] = field(default_factory=lambda: [False, False])
    folded: int = -1
    done: bool = False  # betting over: fold or showdown

    def copy(self) -> AState:
        return AState(self.street, self.contrib[:], self.street_bet[:], self.to_act, self.hist[:], self.raises,
                      self.last_inc, self.acted[:], self.folded, self.done)

    @property
    def pot(self) -> int:
        return self.contrib[0] + self.contrib[1]

    def history_key(self) -> str:
        return "/".join(self.hist)

    def raise_to(self, token: str) -> int:
        p = self.to_act
        base = max(self.street_bet)
        to_call = base - self.street_bet[p]
        if token == "a":
            return self.street_bet[p] + (STACK - self.contrib[p])
        to = base + int(round(ACTION_FRACTIONS[token] * (self.pot + to_call)))
        return max(to, base + max(self.last_inc, BB))

    def legal(self) -> list[str]:
        p = self.to_act
        to_call = max(self.street_bet) - self.street_bet[p]
        remaining = STACK - self.contrib[p]
        opp_remaining = STACK - self.contrib[1 - p]
        acts = ["f"] if to_call > 0 else []
        acts.append("c")
        if remaining > to_call and opp_remaining > 0:
            seen = set()
            if self.raises < MAX_RAISES[self.street]:
                for tok in ("h", "p"):
                    to = self.raise_to(tok)
                    if to - self.street_bet[p] < remaining and to not in seen:
                        seen.add(to)
                        acts.append(tok)
            if self.raises <= MAX_RAISES[self.street]:
                acts.append("a")
        return acts

    def apply(self, token: str) -> AState:
        s = self.copy()
        p = s.to_act
        to_call = max(s.street_bet) - s.street_bet[p]
        s.hist[-1] += token
        s.acted[p] = True
        if token == "f":
            s.folded = p
            s.done = True
            return s
        if token == "c":
            add = min(to_call, STACK - s.contrib[p])
            s.street_bet[p] += add
            s.contrib[p] += add
        else:
            to = s.raise_to(token)
            inc = to - max(s.street_bet)
            if inc >= s.last_inc:
                s.last_inc = inc
            s.contrib[p] += to - s.street_bet[p]
            s.street_bet[p] = to
            s.raises += 1
            s.acted[1 - p] = False
        s.to_act = 1 - p
        bets_equal = s.street_bet[0] == s.street_bet[1]
        someone_all_in = s.contrib[0] == STACK or s.contrib[1] == STACK
        if (bets_equal and s.acted[0] and s.acted[1]) or (token == "c" and someone_all_in):
            if s.street == 3 or someone_all_in:
                s.done = True
                return s
            s.street += 1
            s.street_bet = [0, 0]
            s.hist.append("")
            s.raises = 0
            s.last_inc = BB
            s.acted = [False, False]
            s.to_act = 1  # big blind acts first postflop
        return s

    def utility(self, hole: list[list[Card]], board: list[Card]) -> float:
        """Payoff to player 0 (requires ``done``)."""
        if self.folded >= 0:
            return float(self.contrib[1]) if self.folded == 1 else -float(self.contrib[0])
        won = min(self.contrib)
        s0 = evaluate(hole[0] + board)
        s1 = evaluate(hole[1] + board)
        return float(won) if s0 > s1 else (-float(won) if s1 > s0 else 0.0)


BOARD_SIZE = (0, 3, 4, 5)


class MCCFRTrainer:
    def __init__(self, seed: int = 0, bucket_sims: int = 120) -> None:
        self.rng = random.Random(seed)
        self.regret: dict[str, list[float]] = {}
        self.strategy_sum: dict[str, list[float]] = {}
        self.iterations = 0
        self.bucket_sims = bucket_sims
        self.log: list[dict[str, float]] = []

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _match(r: list[float]) -> list[float]:
        pos = [x if x > 0 else 0.0 for x in r]
        s = sum(pos)
        return [x / s for x in pos] if s > 0 else [1.0 / len(r)] * len(r)

    def _key(self, s: AState, buckets: list[list[int]]) -> str:
        return f"{buckets[s.to_act][s.street]}|{s.history_key()}"

    # ------------------------------------------------------------ traversal
    def _traverse(self, s: AState, p: int, hole, board, buckets, t: int) -> float:
        if s.done:
            u = s.utility(hole, board[:5])
            return u if p == 0 else -u
        acts = s.legal()
        key = self._key(s, buckets)
        reg = self.regret.get(key)
        if reg is None:
            reg = self.regret[key] = [0.0] * len(acts)
            self.strategy_sum[key] = [0.0] * len(acts)
        sigma = self._match(reg)
        if s.to_act == p:
            vals = [self._traverse(s.apply(a), p, hole, board, buckets, t) for a in acts]
            v = sum(x * y for x, y in zip(sigma, vals))
            for i in range(len(acts)):
                reg[i] += vals[i] - v
            return v
        ssum = self.strategy_sum[key]
        for i in range(len(acts)):
            ssum[i] += t * sigma[i]
        a = self.rng.choices(range(len(acts)), weights=sigma)[0]
        return self._traverse(s.apply(acts[a]), p, hole, board, buckets, t)

    def iterate(self) -> None:
        self.iterations += 1
        deck = list(range(52))
        self.rng.shuffle(deck)
        hole = [deck[0:2], deck[2:4]]
        board = deck[4:9]
        buckets = [[bucket(hole[pl], board[: BOARD_SIZE[st]], self.bucket_sims, self.rng) for st in range(4)] for pl in (0, 1)]
        for p in (0, 1):
            self._traverse(AState(), p, hole, board, buckets, self.iterations)

    def train(self, seconds: float, log_every: float = 30.0) -> None:
        start = time.time()
        next_log = start + log_every
        while time.time() - start < seconds:
            self.iterate()
            if time.time() >= next_log:
                self.log.append({"iteration": self.iterations, "seconds": round(time.time() - start, 1),
                                 "infosets": len(self.regret), "avg_positive_regret": self.average_positive_regret()})
                next_log += log_every

    def average_positive_regret(self) -> float:
        """Mean over infosets of the largest positive cumulative regret / iterations
        (an upper-bound proxy for how far the strategy is from equilibrium)."""
        if not self.regret or not self.iterations:
            return 0.0
        return sum(max(0.0, max(r)) for r in self.regret.values()) / len(self.regret) / self.iterations

    def average_strategy(self) -> dict[str, list[float]]:
        out = {}
        for key, ssum in self.strategy_sum.items():
            s = sum(ssum)
            if s > 0:
                out[key] = [round(x / s, 3) for x in ssum]
        return out


def save_blueprint(strategy: dict[str, list[float]], meta: dict[str, Any], path: Path = BLUEPRINT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        json.dump({"meta": meta, "strategy": strategy}, f, separators=(",", ":"))


def load_blueprint(path: Path = BLUEPRINT_PATH) -> tuple[dict[str, list[float]], dict[str, Any]]:
    with gzip.open(path, "rt") as f:
        data = json.load(f)
    return data["strategy"], data["meta"]

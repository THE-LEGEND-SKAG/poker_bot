"""Level 4: plays the MCCFR blueprint for abstracted heads-up NLHE.

For each decision it

1. **translates** the real hand's public betting into the abstract game: real
   chip amounts are rescaled so the big blind is 2 abstract chips, and each real
   bet/raise is mapped to an abstract size (½ pot, pot, all-in) with the
   pseudo-harmonic action-translation mapping (Ganzfried & Sandholm 2013);
2. computes its own **card bucket** exactly as in training;
3. looks up the blueprint's average strategy for that information set, drops
   actions below 5% (purification against noise in rarely visited spots) and
   samples an action;
4. maps the abstract action back to a legal real bet using the real pot.

It falls back to the Level 3 exploitative bot — and says so in its explanation —
when the blueprint does not apply: more than two players, effective stacks far
from the 100 BB it was trained for, an information set never visited in
training, or a betting line the abstraction cannot represent.
"""

from __future__ import annotations

import random
from functools import cache
from typing import Any

from poker_ai.engine import Action, PlayerView, Street
from poker_ai.solvers.hunl import ACTION_FRACTIONS, BB as ABS_BB, BLUEPRINT_PATH, AState, bucket, load_blueprint

from .base import Agent
from .exploitative import ExploitativeAgent

PURIFY_THRESHOLD = 0.05


@cache
def _blueprint() -> tuple[dict[str, list[float]], dict[str, Any]] | None:
    if not BLUEPRINT_PATH.exists():
        return None
    return load_blueprint()


def pseudo_harmonic_prob_a(a: float, b: float, x: float) -> float:
    """Probability of mapping bet fraction ``x`` (A <= x <= B) to the smaller size A."""
    if b <= a:
        return 1.0
    return ((b - x) * (1 + a)) / ((b - a) * (1 + x))


class TranslationError(Exception):
    pass


def translate(view: PlayerView) -> tuple[AState, int]:
    """Abstract state for the current decision and our abstract player id (0 = SB)."""
    start = next(e for e in view.history if e["type"] == "hand_start")
    sb, bb = start["sb"], start["bb"]
    if view.seat not in (sb, bb):
        raise TranslationError("not in the blinds")
    seat_to_p = {sb: 0, bb: 1}
    scale = ABS_BB / view.big_blind
    s = AState()
    real_street = "preflop"
    for e in view.history:
        if e["type"] == "street":
            real_street = e["street"]
            if s.done or ("preflop", "flop", "turn", "river")[s.street] != real_street:
                raise TranslationError("street mismatch between real and abstract game")
            continue
        if e["type"] != "action":
            continue
        if s.done or seat_to_p.get(e["seat"]) != s.to_act:
            raise TranslationError("actor mismatch")
        legal = s.legal()
        a = e["action"]
        if a == "fold":
            tok = "f"
        elif a in ("check", "call"):
            tok = "c"
        elif e["all_in"] and "a" in legal:
            tok = "a"
        else:
            tok = _translate_bet(s, e["amount"] * scale, legal)
        if tok not in legal:
            raise TranslationError(f"action {tok!r} not available in the abstraction")
        s = s.apply(tok)
    if s.done:
        raise TranslationError("abstract hand already over")
    if ("preflop", "flop", "turn", "river")[s.street] != view.street.label:
        raise TranslationError("street mismatch at decision")
    me = seat_to_p[view.seat]
    if s.to_act != me:
        raise TranslationError("abstract game expects the other player to act")
    return s, me


def _translate_bet(s: AState, to_abs: float, legal: list[str]) -> str:
    base = max(s.street_bet)
    p = s.to_act
    pot_after_call = s.pot + (base - s.street_bet[p])
    x = (to_abs - base) / max(1e-9, pot_after_call)
    sizes = []
    for tok in ("h", "p", "a"):
        if tok in legal:
            sizes.append(((s.raise_to(tok) - base) / pot_after_call, tok))
    if not sizes:
        raise TranslationError("no abstract raise available")
    sizes.sort()
    if x <= sizes[0][0]:
        return sizes[0][1]
    if x >= sizes[-1][0]:
        return sizes[-1][1]
    for (fa, ta), (fb, tb) in zip(sizes, sizes[1:]):
        if fa <= x <= fb:
            return ta if pseudo_harmonic_prob_a(fa, fb, x) >= 0.5 else tb
    return sizes[-1][1]  # pragma: no cover


class CFRAgent(Agent):
    kind = "cfr"
    level = 4

    def __init__(self, name: str | None = None, seed: int | None = None, bucket_sims: int = 400) -> None:
        super().__init__(name or "CFR blueprint")
        self.rng = random.Random(seed)
        self.bucket_sims = bucket_sims
        self.fallback = ExploitativeAgent(name=self.name, seed=seed)
        bp = _blueprint()
        self.strategy, self.meta = bp if bp else ({}, {})
        self.training_status = (f"trained ({self.meta['iterations']:,} MCCFR iterations)" if self.meta else "untrained (no blueprint file)")
        self.decisions = {"blueprint": 0, "fallback": 0}

    def _fall_back(self, view: PlayerView, reason: str) -> Action:
        self.decisions["fallback"] += 1
        action = self.fallback.act(view)
        self.last_explanation = {**(self.fallback.last_explanation or {}), "agent": self.name,
                                 "rule": f"fallback to Level 3 ({reason})", "source": "fallback"}
        return action

    def act(self, view: PlayerView) -> Action:
        legal = view.legal
        assert legal is not None
        if not self.strategy:
            return self._fall_back(view, "no blueprint")
        if sum(1 for p in view.players if p.in_hand) != 2:
            return self._fall_back(view, "blueprint is heads-up only")
        start = next(e for e in view.history if e["type"] == "hand_start")
        eff_bb = min(start["stacks"][s] for s in (start["sb"], start["bb"])) / view.big_blind
        if not 40 <= eff_bb <= 250:
            return self._fall_back(view, f"effective stack {eff_bb:.0f} BB is far from the trained 100 BB")
        try:
            s, me = translate(view)
        except TranslationError as exc:
            return self._fall_back(view, str(exc))
        b = bucket(view.hole, list(view.board), self.bucket_sims, self.rng)
        key = f"{b}|{s.history_key()}"
        probs = self.strategy.get(key)
        acts = s.legal()
        if probs is None or len(probs) != len(acts):
            return self._fall_back(view, "information set not visited in training")
        pure = [p if p >= PURIFY_THRESHOLD else 0.0 for p in probs]
        if sum(pure) <= 0:
            pure = probs
        tok = self.rng.choices(acts, weights=pure)[0]
        action = self._to_real(view, tok)
        self.decisions["blueprint"] += 1
        self.last_explanation = {"agent": self.name, "action": str(action), "source": "blueprint",
                                 "rule": f"blueprint: {dict(zip(acts, probs))} → {tok}",
                                 "infoset": key, "bucket": b, "strategy": dict(zip(acts, probs))}
        return action

    @staticmethod
    def _to_real(view: PlayerView, tok: str) -> Action:
        legal = view.legal
        assert legal is not None
        if tok == "f":
            return Action.fold() if legal.can_fold else Action.check()
        if tok == "c" or legal.raise_type is None:
            return legal.passive()
        if tok == "a":
            return legal.all_in()
        to = view.current_bet + ACTION_FRACTIONS[tok] * (view.pot + legal.to_call)
        if view.street is Street.PREFLOP and view.current_bet <= view.big_blind:
            to = max(to, 2 * view.big_blind)
        return legal.raise_action(int(round(to)))

    def on_hand_end(self, view: PlayerView) -> None:
        self.fallback.on_hand_end(view)

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "blueprint": self.meta, "decisions": dict(self.decisions)}

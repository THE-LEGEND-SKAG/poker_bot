"""Level 1: transparent rule-based bot with configurable personality.

Preflop it uses Chen-formula tiers adjusted for position, number of players and
the price of entering. Postflop it uses *current* hand strength against random
holdings (enumerated, no simulation of future cards) plus drawing outs, compared
with pot odds. Every rule is a readable threshold; the reason for each decision
is recorded in ``last_explanation``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from poker_ai.engine import Action, PlayerView, Street

from .base import Agent
from .features import chen_score, count_outs, hand_strength
from .sizing import bet_bb, bet_pot_fraction, effective_stack, position_rank


@dataclass(frozen=True)
class Personality:
    label: str
    open_threshold: float  # min Chen score to open-raise from middle position
    call_threshold: float  # min Chen score to call a single raise
    reraise_threshold: float  # min Chen score to 3-bet
    value_threshold: float  # postflop strength to bet/raise for value
    aggression: float  # probability of choosing the aggressive line when it is reasonable
    bluff_frequency: float  # probability of a pure bluff in a good spot
    call_margin: float  # extra equity demanded over pot odds before calling (negative = loose)
    bet_fraction: float  # default postflop bet size as fraction of pot


PERSONALITIES: dict[str, Personality] = {
    "tight": Personality("Tight", open_threshold=9, call_threshold=10, reraise_threshold=12,
                         value_threshold=0.80, aggression=0.55, bluff_frequency=0.03, call_margin=0.06, bet_fraction=0.6),
    "balanced": Personality("Balanced", open_threshold=7, call_threshold=8, reraise_threshold=11,
                            value_threshold=0.72, aggression=0.70, bluff_frequency=0.10, call_margin=0.02, bet_fraction=0.66),
    "lag": Personality("Loose-Aggressive", open_threshold=5, call_threshold=6, reraise_threshold=9,
                       value_threshold=0.62, aggression=0.85, bluff_frequency=0.22, call_margin=-0.03, bet_fraction=0.75),
}


class RuleBasedAgent(Agent):
    level = 1

    def __init__(self, personality: str = "balanced", name: str | None = None, seed: int | None = None) -> None:
        if personality not in PERSONALITIES:
            raise ValueError(f"unknown personality {personality!r}; choose from {list(PERSONALITIES)}")
        self.p = PERSONALITIES[personality]
        self.kind = f"rule_{personality}"
        super().__init__(name or f"Rule-based ({self.p.label})")
        self.rng = random.Random(seed)

    # ------------------------------------------------------------------
    def act(self, view: PlayerView) -> Action:
        assert view.legal is not None
        if view.street is Street.PREFLOP:
            action, why = self._preflop(view)
        else:
            action, why = self._postflop(view)
        self.last_explanation = {"agent": self.name, "action": str(action), **why}
        return action

    # ------------------------------------------------------------------
    def _preflop(self, view: PlayerView) -> tuple[Action, dict]:
        legal = view.legal
        assert legal is not None
        p = self.p
        score = chen_score(view.hole)
        pos = position_rank(view)
        players = view.num_live
        # late position widens, crowded tables tighten
        adj = score + 2.0 * (pos - 0.5) - 0.4 * max(0, players - 3)
        bb = view.big_blind
        eff_bb = effective_stack(view) / bb
        raised = view.current_bet > bb
        why = {"chen": score, "adjusted": round(adj, 2), "position": round(pos, 2), "effective_bb": round(eff_bb, 1)}

        if eff_bb <= 12:  # push/fold mode
            shove = adj >= p.call_threshold - 2
            if shove:
                return legal.all_in(), {**why, "rule": "short stack: shove"}
            return (Action.check(), {**why, "rule": "short stack: check"}) if legal.can_check else (Action.fold(), {**why, "rule": "short stack: fold"})

        if not raised:
            if adj >= p.open_threshold and legal.can_raise and self.rng.random() < max(p.aggression, 0.8):
                limpers = sum(1 for o in view.opponents if o.street_commit == bb)
                return bet_bb(view, 2.5 + limpers), {**why, "rule": "open raise"}
            if legal.can_check:
                return Action.check(), {**why, "rule": "check option"}
            if adj >= p.open_threshold - 2 or (legal.to_call <= bb // 2 and adj >= p.open_threshold - 4):
                return Action.call(), {**why, "rule": "limp / complete"}
            return Action.fold(), {**why, "rule": "too weak to enter"}

        price_bb = legal.to_call / bb
        if adj >= p.reraise_threshold and legal.can_raise:
            return legal.raise_action(view.current_bet * 3), {**why, "rule": "3-bet for value", "price_bb": price_bb}
        # each extra big blind we must call demands a slightly better hand
        need = p.call_threshold + 0.15 * max(0.0, price_bb - 3)
        if adj >= need:
            return Action.call(), {**why, "rule": "call raise", "price_bb": price_bb}
        if legal.can_check:
            return Action.check(), {**why, "rule": "check"}
        return Action.fold(), {**why, "rule": "fold to raise", "price_bb": price_bb}

    # ------------------------------------------------------------------
    def _postflop(self, view: PlayerView) -> tuple[Action, dict]:
        legal = view.legal
        assert legal is not None
        p = self.p
        opps = max(1, view.num_live - 1)
        hs = hand_strength(view.hole, view.board, opps)
        outs = count_outs(view.hole, view.board)
        cards_to_come = 5 - len(view.board)
        draw_eq = min(0.5, outs * (0.04 if cards_to_come == 2 else 0.02))  # rule of 4 and 2
        strength = max(hs, draw_eq)
        pot_odds = legal.to_call / (view.pot + legal.to_call) if legal.to_call else 0.0
        pos = position_rank(view)
        why = {"hand_strength": round(hs, 3), "outs": outs, "draw_equity": round(draw_eq, 3),
               "pot_odds": round(pot_odds, 3), "position": round(pos, 2)}
        r = self.rng.random()

        if legal.can_check:  # nobody has bet
            if hs >= p.value_threshold and legal.can_raise and r < p.aggression + 0.2:
                return bet_pot_fraction(view, p.bet_fraction), {**why, "rule": "value bet"}
            if outs >= 8 and legal.can_raise and r < p.aggression * 0.6:
                return bet_pot_fraction(view, p.bet_fraction), {**why, "rule": "semi-bluff"}
            if legal.can_raise and opps == 1 and pos > 0.5 and hs < 0.35 and r < p.bluff_frequency * 2:
                return bet_pot_fraction(view, 0.5), {**why, "rule": "position bluff (heads-up, checked to)"}
            return Action.check(), {**why, "rule": "check"}

        # facing a bet
        if hs >= max(p.value_threshold + 0.12, 0.85) and legal.can_raise and r < p.aggression:
            return bet_pot_fraction(view, 0.9), {**why, "rule": "raise for value"}
        if strength >= pot_odds + p.call_margin:
            return Action.call(), {**why, "rule": "call: strength beats pot odds"}
        if outs >= 8 and legal.can_raise and opps == 1 and r < p.bluff_frequency:
            return bet_pot_fraction(view, 0.8), {**why, "rule": "semi-bluff raise"}
        return Action.fold(), {**why, "rule": "fold: price too high"}

"""Expected-value analysis of every available action (Level 2 model).

Accounting convention: EV is the expected change in our stack *from this
decision onward*, so folding is always 0 (chips already in the pot are sunk).

* check / call   EV = R · eq · (pot + call) − call
* bet / raise    EV = FE · pot + (1 − FE) · (R · eq_called · (pot + ours + theirs) − ours)

where
* ``eq``          equity vs the opponents' estimated ranges (Monte Carlo),
* ``R``           equity realisation: 1.0 when no more betting can happen (river,
                  all-in). Otherwise it loses 5% in position / 15% out of position,
                  plus up to 50% more for hands below their fair share of the pot
                  (weak hands usually fold before showdown). Level 3 scales the loss
                  by how aggressive the opponents actually are,
* ``FE``          probability every opponent folds. With no opponent model each
                  opponent folds ``(risk / (risk + reward)) ** 1.5``. A balanced
                  defender folds exactly ``risk / (risk + reward)`` (which would make
                  every pure bluff break even); typical opponents call somewhat more
                  than that, so pure bluffs lose money, while huge overbets are still
                  called only by a small, strong range,
* ``eq_called``   equity vs only the hands that continue (the top ``1 − f`` of
                  the caller's range) — calling ranges are stronger than average.

Bet sizes considered: 2.5/3.5 BB opens or 3x/4x re-raises preflop, ½ / ¾ / pot
postflop, and all-in only when stack ≤ 4× pot or ≤ 15 BB preflop (see ``candidate_sizes``).

A bettor's range is narrowed according to the size of their bet (bigger bets →
stronger range). Level 3 replaces these defaults with observed opponent statistics.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from poker_ai.engine import Action, ActionType, PlayerView, Street

from .equity import EquityResult, Range, equity
from .ranges import TOTAL_COMBOS, ranked_combos, top_fraction
from .sizing import position_rank

# fold probability of one opponent: f(our_risk, pot_we_win) -> [0, 1]
FoldModel = Callable[[PlayerView, int, int, int], float]


DEFENSE_EXPONENT = 1.5  # > 1: folds less often than a balanced (MDF) defender


def mdf_fold(view: PlayerView, opponent_seat: int, risk: int, reward: int) -> float:
    """Default opponent: folds ``(risk / (risk + reward)) ** DEFENSE_EXPONENT``."""
    if risk <= 0:
        return 0.0
    return min(0.98, (risk / (risk + reward)) ** DEFENSE_EXPONENT)


@dataclass
class Option:
    action: Action
    label: str
    ev: float
    fold_equity: float = 0.0
    equity_when_called: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"action": self.action.type.value, "amount": self.action.amount, "label": self.label,
                "ev": round(self.ev, 1), "fold_equity": round(self.fold_equity, 3),
                "equity_when_called": None if self.equity_when_called is None else round(self.equity_when_called, 3)}


@dataclass
class Analysis:
    street: str
    equity: EquityResult
    pot: int
    to_call: int
    pot_odds: float  # call / (pot + call): the equity a call needs to break even
    realization: float
    opponents: int
    range_note: str
    options: list[Option] = field(default_factory=list)

    @property
    def best(self) -> Option:
        return max(self.options, key=lambda o: o.ev)

    def as_dict(self) -> dict[str, Any]:
        return {"street": self.street, "equity": self.equity.as_dict(), "pot": self.pot, "to_call": self.to_call,
                "pot_odds": round(self.pot_odds, 4), "realization": self.realization, "opponents": self.opponents,
                "range_note": self.range_note, "options": [o.as_dict() for o in self.options],
                "best": self.best.as_dict()}


def bettor_range_fraction(view: PlayerView) -> float:
    """How much of a random range an aggressor keeps, given their bet size."""
    bet = view.current_bet
    if view.street is Street.PREFLOP and bet <= view.big_blind:
        return 1.0
    if bet == 0:
        return 1.0
    rest = max(1, view.pot - bet)
    return max(0.12, 1.0 - 1.2 * bet / (rest + bet))


def candidate_sizes(view: PlayerView) -> list[tuple[str, int]]:
    legal = view.legal
    assert legal is not None and legal.raise_type is not None
    bb = view.big_blind
    out: list[tuple[str, int]] = []
    if view.street is Street.PREFLOP:
        if view.current_bet <= bb:
            # count actual limps (voluntary calls); the big blind's forced post is not a limp
            limpers = sum(1 for e in view.history if e["type"] == "action" and e["street"] == "preflop"
                          and e["action"] == "call" and e["seat"] != view.seat)
            out += [("2.5 BB", int(2.5 * bb) + limpers * bb), ("3.5 BB", int(3.5 * bb) + limpers * bb)]
        else:
            out += [("3x", 3 * view.current_bet), ("4x", 4 * view.current_bet)]
    else:
        pot_after_call = view.pot + legal.to_call
        for label, f in (("½ pot", 0.5), ("¾ pot", 0.75), ("Pot", 1.0)):
            out.append((label, int(round(view.current_bet + f * pot_after_call))))
    # Commitment rule: the single-street EV model cannot see the value a smaller
    # bet wins on later streets, so it over-rates shoving deep stacks. Only
    # consider all-in when the stack is at most 4x the pot (or nothing else is legal).
    spr = legal.stack / max(1, view.pot + legal.to_call)
    short_preflop = view.street is Street.PREFLOP and legal.stack + legal.street_commit <= 15 * bb
    if spr <= 4 or short_preflop or legal.min_raise_to >= legal.max_raise_to:
        out.append(("All-in", legal.max_raise_to))
    # Labels describe the FINAL amount: sizes can grow (1 BB per limper) or be bumped
    # to the minimum legal raise, so a fixed label like "2.5 BB" could be wrong.
    seen: dict[int, str] = {}
    for label, intended in out:
        to = legal.clamp_raise(intended)
        if to in seen:
            continue
        if to == legal.max_raise_to:
            seen[to] = "All-in"
        elif view.street is Street.PREFLOP:
            seen[to] = f"{to / bb:g} BB"
        elif to != intended and to == legal.min_raise_to:
            seen[to] = "min raise" if legal.raise_type is ActionType.RAISE else "min bet"
        else:
            seen[to] = label
    return [(lbl, to) for to, lbl in sorted(seen.items())]


def analyze(view: PlayerView, sims: int = 600, rng: random.Random | None = None,
            fold_model: FoldModel = mdf_fold, opponent_ranges: dict[int, Range] | None = None,
            aggression_scale: float = 1.0) -> Analysis:
    """``aggression_scale`` scales the equity lost to future betting (1 = population
    default, 0 = opponents never bet, so equity is fully realised)."""
    legal = view.legal
    assert legal is not None, "analysis needs a decision point"
    rng = rng or random.Random()
    opps = view.opponents
    dead = set(view.hole) | set(view.board)
    ranked = ranked_combos(view.board, dead)

    # --- opponents' ranges -------------------------------------------------
    ranges: dict[int, Range] = dict(opponent_ranges or {})
    aggressor = max(opps, key=lambda o: o.street_commit) if opps else None
    note = "random hands"
    if opponent_ranges is None and aggressor is not None and legal.to_call > 0:
        frac = bettor_range_fraction(view)
        if frac < 1.0:
            ranges[aggressor.seat] = top_fraction(ranked, frac)
            note = f"bettor on top {frac:.0%} of hands, others random"
    elif opponent_ranges is not None:
        note = "modelled ranges"
    opp_ranges = [ranges.get(o.seat) for o in opps]
    eq = equity(view.hole, view.board, opp_ranges, sims, rng)

    pot = view.pot
    to_call = legal.to_call
    no_more_betting = view.street is Street.RIVER or not any(not o.all_in for o in opps)
    pos = position_rank(view)
    # Equity lost to future betting: a base amount by position, plus more for weak hands,
    # which usually have to fold before showdown (they rarely "realise" their raw equity).
    # Weakness is measured against the fair share of the pot, 1 / (players in the hand).
    fair = 1.0 / (len(opps) + 1)
    weakness = max(0.0, fair - eq.equity) / fair  # 0 for average-or-better hands, up to 1
    loss = (0.05 if pos > 0.5 else 0.15) + 0.5 * weakness
    R = 1.0 if no_more_betting else 1.0 - min(0.6, loss) * min(1.5, max(0.0, aggression_scale))

    a = Analysis(street=view.street.label, equity=eq, pot=pot, to_call=to_call,
                 pot_odds=to_call / (pot + to_call) if to_call else 0.0, realization=R,
                 opponents=len(opps), range_note=note)

    if legal.can_fold:
        a.options.append(Option(Action.fold(), "Fold", 0.0))
    if legal.can_check:
        a.options.append(Option(Action.check(), "Check", R * eq.equity * pot))
    if legal.can_call:
        call = legal.call_amount
        r_call = 1.0 if call >= legal.stack else R
        a.options.append(Option(Action.call(), f"Call {call}", r_call * eq.equity * (pot + call) - call))

    if legal.raise_type is not None and opps:
        caller = aggressor if to_call > 0 and aggressor is not None else max(opps, key=lambda o: o.stack)
        caller_depth = caller.stack + caller.street_commit  # caller's max total on this street
        base_frac = bettor_range_fraction(view) if (to_call > 0 and opponent_ranges is None) else 1.0
        for label, to in candidate_sizes(view):
            ours = to - legal.street_commit
            theirs = max(0, min(to, caller_depth) - caller.street_commit)
            ours_eff = min(ours, max(0, min(to, caller_depth) - legal.street_commit))  # excess is refunded
            f_each = {o.seat: fold_model(view, o.seat, ours, pot) for o in opps}
            fe = 1.0
            for f in f_each.values():
                fe *= f
            f_caller = f_each[caller.seat]
            if opponent_ranges is not None and opponent_ranges.get(caller.seat) is not None:
                caller_range = opponent_ranges[caller.seat]
                assert caller_range is not None
                calling = top_fraction(caller_range, 1 - f_caller)
            else:
                calling = top_fraction(ranked, base_frac * (1 - f_caller))
            eq_c = equity(view.hole, view.board, [calling], max(150, sims // 2), rng).equity
            r_c = 1.0 if (to == legal.max_raise_to or theirs >= caller.stack) else R
            ev = fe * pot + (1 - fe) * (r_c * eq_c * (pot + ours_eff + theirs) - ours_eff)
            verb = "Bet" if legal.raise_type is ActionType.BET else "Raise to"
            text = f"{verb} {to} ({label})" if label != "All-in" else f"All-in {to}"
            a.options.append(Option(Action(legal.raise_type, to), text, ev, fe, eq_c))
    return a

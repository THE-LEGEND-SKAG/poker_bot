"""Bet-sizing helpers expressed in the engine's "raise to" convention."""

from __future__ import annotations

from poker_ai.engine import Action, PlayerView


def pot_fraction_to(view: PlayerView, fraction: float) -> int:
    """"Raise to" amount for a bet/raise of ``fraction`` × pot.

    Uses the standard definition: a pot-sized raise first calls, then raises by
    the size of the pot after the call.
    """
    legal = view.legal
    assert legal is not None
    pot_after_call = view.pot + legal.to_call
    return int(round(view.current_bet + fraction * pot_after_call))


def bet_pot_fraction(view: PlayerView, fraction: float) -> Action:
    legal = view.legal
    assert legal is not None and legal.raise_type is not None
    return legal.raise_action(pot_fraction_to(view, fraction))


def bet_bb(view: PlayerView, big_blinds: float) -> Action:
    legal = view.legal
    assert legal is not None and legal.raise_type is not None
    return legal.raise_action(int(round(big_blinds * view.big_blind)))


def position_rank(view: PlayerView) -> float:
    """0.0 = first to act postflop (worst) .. 1.0 = button (best)."""
    seats = [p.seat for p in view.players if p.in_hand]
    n = len(seats)
    order = sorted(seats, key=lambda s: (s - view.button - 1) % len(view.players))
    return order.index(view.seat) / max(1, n - 1)


def effective_stack(view: PlayerView) -> int:
    me = view.me
    mine = me.stack + me.street_commit
    others = [p.stack + p.street_commit for p in view.opponents]
    return min(mine, max(others, default=mine))

"""Drive hands with agents. The engine validates every action."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from poker_ai.agents.base import Agent
from poker_ai.engine import Deck, HandState, Table, TableConfig


def play_hand(hand: HandState, agents: Mapping[int, Agent], on_action: Callable[[HandState], None] | None = None) -> HandState:
    while not hand.terminal:
        seat = hand.to_act
        assert seat is not None
        hand.apply(agents[seat].act(hand.view(seat)))
        if on_action:
            on_action(hand)
    for seat, agent in agents.items():
        if hand.players[seat].in_hand:
            agent.on_hand_end(hand.view(seat))
    return hand


def play_session(agents: Sequence[Agent], config: TableConfig, hands: int, seed: int | None = None,
                 rebuy: bool = True, decks: Sequence[Deck] | None = None) -> list[list[int]]:
    """Play ``hands`` hands; returns per-hand net results by seat.

    With ``rebuy`` (cash-game accounting) busted players are topped back up so the
    session always runs the requested number of hands.
    """
    if len(agents) != config.num_seats:
        raise ValueError("need one agent per seat")
    table = Table(config, seed=seed)
    results: list[list[int]] = []
    for i in range(hands):
        if rebuy:
            for s in range(config.num_seats):
                if table.stacks[s] < config.big_blind:
                    table.rebuy(s)
        elif table.is_over:
            break
        hand = table.start_hand(deck=decks[i] if decks else None)
        play_hand(hand, dict(enumerate(agents)))
        results.append(table.end_hand())
    return results

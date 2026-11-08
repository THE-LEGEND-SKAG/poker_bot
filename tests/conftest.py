"""Test helpers: rigged decks so scenarios are deterministic."""

from __future__ import annotations

from poker_ai.engine import Deck, HandState, parse_cards


def rigged_hand(stacks, button, sb, bb, holes: dict[int, str], board: str = "", ante: int = 0) -> HandState:
    """Build a hand where seat -> hole cards and the board are fixed.

    Seats not listed in ``holes`` (but dealt in) get arbitrary unused cards.
    """
    probe = HandState(stacks, button, sb, bb, ante, deck=Deck.from_order(range(52)))
    deal_order = [probe.sb_seat] + [s for s in probe._clockwise_from(probe.sb_seat) if s != probe.sb_seat]
    fixed = {s: parse_cards(h) for s, h in holes.items()}
    board_cards = parse_cards(board) if board else []
    used = {c for cs in fixed.values() for c in cs} | set(board_cards)
    spare = iter(c for c in range(52) if c not in used)
    hole = {s: fixed.get(s) or [next(spare), next(spare)] for s in deal_order}
    order = [hole[s][0] for s in deal_order] + [hole[s][1] for s in deal_order]
    order += board_cards
    order += [next(spare) for _ in range(5 - len(board_cards))]
    return HandState(stacks, button, sb, bb, ante, deck=Deck.from_order(order))

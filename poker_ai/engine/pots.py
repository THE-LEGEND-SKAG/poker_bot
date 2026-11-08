"""Side-pot construction and pot distribution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Pot:
    amount: int
    eligible: tuple[int, ...]  # seats that can win this pot


def build_pots(contributions: Mapping[int, int], live: set[int]) -> list[Pot]:
    """Split total contributions into a main pot and side pots.

    ``contributions`` maps seat -> chips put in this hand (folded players
    included: their chips are dead money). ``live`` is the set of seats that
    have not folded. Adjacent layers with the same eligible set are merged.
    """
    levels = sorted({c for c in contributions.values() if c > 0})
    pots: list[Pot] = []
    prev = 0
    for level in levels:
        amount = sum(min(c, level) - min(c, prev) for c in contributions.values())
        eligible = tuple(sorted(s for s, c in contributions.items() if c >= level and s in live))
        prev = level
        if amount == 0:
            continue
        if not eligible:
            # Only reachable if chips above every live player's contribution were
            # not refunded; give them to the deepest live contributors rather
            # than lose them.
            if pots:
                last = pots.pop()
                pots.append(Pot(last.amount + amount, last.eligible))
                continue
            eligible = tuple(sorted(live))
        if pots and pots[-1].eligible == eligible:
            last = pots.pop()
            pots.append(Pot(last.amount + amount, eligible))
        else:
            pots.append(Pot(amount, eligible))
    return pots


def split_pot(amount: int, winners: Sequence[int], seat_order: Sequence[int]) -> dict[int, int]:
    """Split ``amount`` among ``winners``; odd chips go to the winners closest
    to the left of the button (``seat_order`` starts left of the button)."""
    share, odd = divmod(amount, len(winners))
    out = {w: share for w in winners}
    ordered = [s for s in seat_order if s in out]
    for i in range(odd):
        out[ordered[i]] += 1
    return out

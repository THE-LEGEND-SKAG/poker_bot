"""Multi-way ring-game evaluation with seat rotation.

Each deal is replayed once per button position: with ``n`` agents the same deck
is played ``n`` times, moving the button one seat each time. Cards are dealt by
position, so across the rotations every agent receives every position's cards
and plays every position; summing its results cancels most card and position
luck (the multi-way analogue of duplicate). Agents stay in fixed seats, so
opponent models keyed by seat stay valid (rotating the *agents* through seats
would scramble them). Stacks reset to ``stack_bb`` every hand.
"""

from __future__ import annotations

import argparse
import math
import random
import time
from typing import Any

from poker_ai.agents import make_agent
from poker_ai.engine import Deck, HandState

from .runner import play_hand
from .stats import PlayStats

BB = 100


def ring_match(kinds: list[str], deals: int, seed: int = 0, stack_bb: int = 100) -> dict[str, Any]:
    n = len(kinds)
    if not 3 <= n <= 9:
        raise ValueError("ring games need 3-9 agents")
    agents = [make_agent(k, seed=seed + i) for i, k in enumerate(kinds)]
    rng = random.Random(seed)
    per_deal: list[list[float]] = [[] for _ in range(n)]
    stats = [PlayStats() for _ in range(n)]
    t0 = time.time()
    for d in range(deals):
        order = list(range(52))
        rng.shuffle(order)
        totals = [0.0] * n
        by_seat = dict(enumerate(agents))  # fixed seats
        for button in range(n):
            h = HandState([stack_bb * BB] * n, button, BB // 2, BB, deck=Deck.from_order(order), hand_id=d)
            play_hand(h, by_seat)
            for i in range(n):
                totals[i] += h.net[i]
                stats[i].add_hand(h, i, BB)
        for i in range(n):
            per_deal[i].append(totals[i] / n / BB)
    rows = []
    for i, k in enumerate(kinds):
        xs = per_deal[i]
        mean = sum(xs) / len(xs)
        sd = math.sqrt(sum((x - mean) ** 2 for x in xs) / max(1, len(xs) - 1))
        rows.append({"kind": k, "bb100": round(mean * 100, 1), "ci95": round(196 * sd / math.sqrt(len(xs)), 1),
                     **stats[i].summary(BB)})
    rows.sort(key=lambda r: r["bb100"], reverse=True)
    return {"agents": kinds, "deals": deals, "hands": deals * n, "seconds": round(time.time() - t0, 1), "results": rows}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--agents", nargs="+", required=True)
    p.add_argument("--deals", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    r = ring_match(a.agents, a.deals, a.seed)
    print(f"{len(a.agents)}-handed ring, {r['hands']} hands ({r['seconds']}s)")
    for row in r["results"]:
        print(f"  {row['kind']:16s} {row['bb100']:+8.1f} ± {row['ci95']:6.1f} bb/100   VPIP {row['vpip']:.0%}  PFR {row['pfr']:.0%}  fold {row['fold_frequency']:.0%}")


if __name__ == "__main__":
    main()

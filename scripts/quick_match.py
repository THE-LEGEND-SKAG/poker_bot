"""Heads-up duplicate match: each deck is played twice with seats swapped.

usage: python scripts/quick_match.py AGENT_A AGENT_B [deals] [seed]
Reports A's win rate in big blinds per 100 hands with a 95% confidence interval
computed over duplicate pairs (which cancels most of the card luck).
"""

import math
import random
import sys
import time

from poker_ai.agents import make_agent
from poker_ai.engine import Deck, HandState
from poker_ai.sim.runner import play_hand

a_kind, b_kind = sys.argv[1], sys.argv[2]
deals = int(sys.argv[3]) if len(sys.argv) > 3 else 500
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 1
BB, STACK = 100, 10_000

rng = random.Random(seed)
a, b = make_agent(a_kind, seed=seed), make_agent(b_kind, seed=seed + 1)
pair_results = []
t0 = time.time()
for i in range(deals):
    order = list(range(52))
    rng.shuffle(order)
    total = 0
    for a_seat in (0, 1):
        agents = {a_seat: a, 1 - a_seat: b}
        h = HandState([STACK, STACK], button=0, small_blind=BB // 2, big_blind=BB, deck=Deck.from_order(order), hand_id=i)
        play_hand(h, agents)
        total += h.net[a_seat]
    pair_results.append(total / 2 / BB)  # bb per hand, averaged over the pair
n = len(pair_results)
mean = sum(pair_results) / n
sd = math.sqrt(sum((x - mean) ** 2 for x in pair_results) / (n - 1))
ci = 1.96 * sd / math.sqrt(n)
print(f"{a_kind} vs {b_kind}: {mean * 100:+.1f} bb/100 ± {ci * 100:.1f} (95% CI), {2 * n} hands, {time.time() - t0:.0f}s")

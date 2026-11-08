"""Build poker_ai/data/preflop_equity.json (equity of each of the 169 classes vs 1-5 random hands)."""

import json
import random
import sys
import time

from poker_ai.agents.equity import equity
from poker_ai.agents.preflop import TABLE_PATH, all_classes, class_combos

SIMS = int(sys.argv[1]) if len(sys.argv) > 1 else 20000

rng = random.Random(169)
table = {}
t0 = time.time()
for cls in all_classes():
    combos = class_combos(cls)
    row = []
    for opp in range(1, 6):
        sims = SIMS if opp == 1 else SIMS // 2
        # spread samples across the class's suit combos (they are equivalent up to suit symmetry)
        per = sims // len(combos)
        eq = sum(equity(list(c), [], opp, per, rng).equity for c in combos) / len(combos)
        row.append(round(eq, 4))
    table[cls] = row
TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
TABLE_PATH.write_text(json.dumps(table, indent=0))
print(f"wrote {TABLE_PATH} ({len(table)} classes, {time.time() - t0:.0f}s)")

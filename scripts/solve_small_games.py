"""Solve Kuhn and Leduc poker with CFR and print the convergence (exploitability) curve.

usage: python scripts/solve_small_games.py [iterations] [cfr|cfr+]
Writes poker_ai/data/cfr_convergence.json for the Lab dashboard.
"""

import json
import sys
import time
from pathlib import Path

from poker_ai.solvers import CFRSolver, KuhnPoker, LeducPoker, expected_value, exploitability

iters = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
variant = sys.argv[2] if len(sys.argv) > 2 else "cfr"
checkpoints = sorted({1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, iters} - {i for i in range(iters + 1, 10**6)})
out = {}
for game, target in ((KuhnPoker(), -1 / 18), (LeducPoker(), -0.0856)):
    tree = game.compile()
    solver = CFRSolver(tree, variant)
    rows = []
    t0 = time.time()
    for c in checkpoints:
        solver.iterate(c - solver.iterations)
        avg = solver.average_strategy()
        row = {"iteration": c, "exploitability": exploitability(tree, avg), "value": expected_value(tree, avg)}
        rows.append(row)
        print(f"{game.name:6s} {c:6d} it  exploitability {row['exploitability']:.6f}  value {row['value']:+.5f}  "
              f"(target {target:+.5f}, {time.time() - t0:.0f}s)", flush=True)
    out[game.name] = {"infosets": tree.num_infosets, "target_value": target, "variant": variant, "curve": rows}
path = Path(__file__).resolve().parents[1] / "poker_ai" / "data" / "cfr_convergence.json"
path.write_text(json.dumps(out, indent=1))
print(f"wrote {path}")

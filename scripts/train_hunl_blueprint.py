"""Train the heads-up NLHE blueprint with external-sampling MCCFR.

usage: python scripts/train_hunl_blueprint.py [minutes] [workers]
Each worker trains independently (different seed) for the given wall time; their
regret and strategy sums are added and the average strategy is written to
poker_ai/data/hunl_blueprint.json.gz together with a training log.
"""

import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

from poker_ai.solvers.hunl import BUCKETS, MAX_RAISES, STACK, MCCFRTrainer, save_blueprint


def work(args):
    seed, seconds = args
    t = MCCFRTrainer(seed=seed)
    t.train(seconds, log_every=60)
    return t.iterations, t.strategy_sum, t.regret, t.log


def main() -> None:
    minutes = float(sys.argv[1]) if len(sys.argv) > 1 else 20
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    t0 = time.time()
    with Pool(workers) as pool:
        results = pool.map(work, [(1000 + i, minutes * 60) for i in range(workers)])
    total_iters = sum(r[0] for r in results)
    ssum: dict[str, list[float]] = {}
    regret: dict[str, list[float]] = {}
    for _, s, r, _ in results:
        for k, v in s.items():
            cur = ssum.setdefault(k, [0.0] * len(v))
            for i, x in enumerate(v):
                cur[i] += x
        for k, v in r.items():
            cur = regret.setdefault(k, [0.0] * len(v))
            for i, x in enumerate(v):
                cur[i] += x
    strategy = {k: [round(x / sum(v), 3) for x in v] for k, v in ssum.items() if sum(v) > 0}
    avg_pos_regret = sum(max(0.0, max(r)) for r in regret.values()) / max(1, len(regret)) / max(1, total_iters)
    meta = {"iterations": total_iters, "workers": workers, "minutes": minutes, "infosets": len(strategy),
            "stack_bb": STACK // 2, "buckets": BUCKETS, "max_raises": MAX_RAISES,
            "avg_positive_regret": avg_pos_regret, "trained_at": time.strftime("%Y-%m-%d %H:%M")}
    save_blueprint(strategy, meta)
    log_path = Path(__file__).resolve().parents[1] / "poker_ai" / "data" / "hunl_training_log.json"
    log_path.write_text(json.dumps({"meta": meta, "workers": [r[3] for r in results]}, indent=1))
    print(json.dumps(meta, indent=1))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

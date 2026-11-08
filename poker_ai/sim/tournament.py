"""Round-robin duplicate tournament between agents (heads-up).

Every pair plays ``deals`` duplicate deals (each deal twice with seats swapped).
Results:

* head-to-head win rates in bb/100 with 95% confidence intervals;
* a leaderboard: each agent's mean win rate over its matches, with the CI
  combined from the per-match standard errors (matches are independent);
* playing statistics per agent (VPIP, PFR, aggression factor, fold frequency,
  went-to-showdown, showdown win rate, average pot, average decision time);
* a few complete hand histories per match for the replayer.

Run from the command line (used by the server so the web app stays responsive):

    python -m poker_ai.sim.tournament --agents equity exploit cfr neural --deals 300 --out runs/t.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path
from typing import Any

from poker_ai.agents import available_agents, make_agent
from poker_ai.agents.base import Agent
from poker_ai.engine import Deck, HandState, card_str

from .runner import play_hand
from .stats import PlayStats

BB = 100
STACK_BB = 100
OFFICIAL_RESULTS = Path(__file__).resolve().parents[1] / "data" / "tournament_results.json"


class Timed(Agent):
    """Wraps an agent to measure decision time."""

    def __init__(self, inner: Agent) -> None:
        super().__init__(inner.name)
        self.inner = inner
        self.seconds = 0.0

    def act(self, view):
        t = time.perf_counter()
        a = self.inner.act(view)
        self.seconds += time.perf_counter() - t
        return a

    def on_hand_end(self, view) -> None:
        self.inner.on_hand_end(view)


def _hand_record(h: HandState, names: dict[int, str]) -> dict[str, Any]:
    return {"names": {str(k): v for k, v in names.items()},
            "holes": {str(p.seat): [card_str(c) for c in p.hole] for p in h.players},
            "board": [card_str(c) for c in h.board],
            "events": json.loads(json.dumps(h.events, default=str)), "net": h.net, "button": h.button}


def play_match(args: tuple[str, str, int, int, int]) -> dict[str, Any]:
    a_kind, b_kind, deals, seed, record = args
    a, b = Timed(make_agent(a_kind, seed=seed)), Timed(make_agent(b_kind, seed=seed + 1))
    rng = random.Random(seed)
    pairs: list[float] = []
    stats = {a_kind: PlayStats(), b_kind: PlayStats()}
    hands: list[dict[str, Any]] = []
    for i in range(deals):
        order = list(range(52))
        rng.shuffle(order)
        total = 0
        for a_seat in (0, 1):
            agents = {a_seat: a, 1 - a_seat: b}
            h = HandState([STACK_BB * BB] * 2, 0, BB // 2, BB, deck=Deck.from_order(order), hand_id=i)
            play_hand(h, agents)
            total += h.net[a_seat]
            stats[a_kind].add_hand(h, a_seat, BB)
            stats[b_kind].add_hand(h, 1 - a_seat, BB)
            if len(hands) < record:
                hands.append(_hand_record(h, {a_seat: a_kind, 1 - a_seat: b_kind}))
        pairs.append(total / 2 / BB)
    stats[a_kind].decision_seconds = a.seconds
    stats[b_kind].decision_seconds = b.seconds
    n = len(pairs)
    mean = sum(pairs) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in pairs) / max(1, n - 1))
    se = sd / math.sqrt(n)
    return {"a": a_kind, "b": b_kind, "bb100": round(mean * 100, 1), "se100": se * 100,
            "ci95": round(1.96 * se * 100, 1), "hands": 2 * n,
            "stats": {k: v.__dict__ for k, v in stats.items()}, "sample_hands": hands}


def run_tournament(kinds: list[str], deals: int, seed: int = 0, workers: int = 3, record: int = 6,
                   progress_path: Path | None = None) -> dict[str, Any]:
    if len(kinds) < 2 or len(set(kinds)) != len(kinds):
        raise ValueError("need at least two distinct agents")
    meta = {a["kind"]: a for a in available_agents()}
    for k in kinds:
        if k not in meta:
            raise ValueError(f"unknown agent {k!r}")
    jobs = [(a, b, deals, seed + 101 * i, record) for i, (a, b) in enumerate(combinations(kinds, 2))]
    t0 = time.time()
    results = []

    def progress() -> None:
        if progress_path:
            progress_path.write_text(json.dumps({"done": len(results), "total": len(jobs), "seconds": round(time.time() - t0, 1)}))

    progress()
    with Pool(max(1, min(workers, len(jobs)))) as pool:
        for r in pool.imap_unordered(play_match, jobs):
            results.append(r)
            progress()
    matches = sorted(results, key=lambda r: (kinds.index(r["a"]), kinds.index(r["b"])))
    stats = {k: PlayStats() for k in kinds}
    for m in matches:
        for k, s in m.pop("stats").items():
            stats[k].merge(PlayStats(**s))
    leaderboard = []
    for k in kinds:
        rows = [(m["bb100"], m["se100"]) if m["a"] == k else (-m["bb100"], m["se100"]) for m in matches if k in (m["a"], m["b"])]
        mean = sum(r[0] for r in rows) / len(rows)
        ci = 1.96 * math.sqrt(sum(r[1] ** 2 for r in rows)) / len(rows)
        leaderboard.append({"kind": k, "label": meta[k]["label"], "level": meta[k]["level"],
                            "bb100": round(mean, 1), "ci95": round(ci, 1), **stats[k].summary(BB)})
    leaderboard.sort(key=lambda r: r["bb100"], reverse=True)
    for m in matches:
        m.pop("se100")
    status = {k: make_agent(k).training_status for k in kinds}
    return {"generated_at": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "seconds": round(time.time() - t0, 1),
            "config": {"agents": kinds, "deals_per_match": deals, "stack_bb": STACK_BB, "seed": seed, "format": "heads-up duplicate"},
            "training_status": status, "leaderboard": leaderboard,
            "matches": [{k: v for k, v in m.items() if k != "sample_hands"} for m in matches],
            "hands": [h for m in matches for h in m["sample_hands"]]}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--agents", nargs="+", required=True)
    p.add_argument("--deals", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--out", type=Path, default=OFFICIAL_RESULTS)
    args = p.parse_args()
    progress = args.out.with_suffix(".progress.json")
    result = run_tournament(args.agents, args.deals, args.seed, args.workers, progress_path=progress)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result))
    print(json.dumps(result["leaderboard"], indent=1))


if __name__ == "__main__":
    main()

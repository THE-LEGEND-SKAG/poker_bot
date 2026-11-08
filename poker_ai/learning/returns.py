"""PyTorch-free pieces of training: advantages, batch merging, status labels."""

from __future__ import annotations

from typing import Any

import numpy as np

from .rollout import Batch


def gae(batch: Batch, lam: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
    """Advantages and returns. Episodes = one hand; reward only at the last decision."""
    n = len(batch.actions)
    adv = np.zeros(n, dtype=np.float32)
    values = np.array(batch.values, dtype=np.float32)
    nxt = 0.0
    run = 0.0
    for t in range(n - 1, -1, -1):
        if batch.episode_end[t]:
            nxt, run = 0.0, 0.0
        delta = batch.rewards[t] + nxt - values[t]
        run = delta + lam * run
        adv[t] = run
        nxt = values[t]
    return adv, adv + values



def merge(batches: list[Batch]) -> Batch:
    out = Batch()
    for b in batches:
        for f in ("obs", "masks", "actions", "logp", "values", "episode_end", "rewards", "hand_results"):
            getattr(out, f).extend(getattr(b, f))
    return out



def training_status(meta: dict[str, Any]) -> str:
    """'trained' only if the final policy beats every rule-based evaluation opponent
    with a 95% confidence interval above zero; otherwise 'partially trained'."""
    if meta.get("updates", 0) == 0:
        return "untrained"
    ev = meta.get("final_eval", {})
    beats = all(ev.get(o, {}).get("bb100", 0) - ev.get(o, {}).get("ci95", 1e9) > 0
                for o in ("rule_balanced", "rule_tight", "rule_lag"))
    return "trained" if beats else "partially trained"

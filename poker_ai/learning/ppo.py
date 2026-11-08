"""PPO (actor-critic) training against a league, with masked actions.

Why PPO and a league rather than DQN or plain self-play: in poker the opponent
adapts, so a value-based single-agent method chases a moving target, and pure
self-play can cycle (A beats B beats C beats A). The learner instead trains
against a *league*: fixed bots (rule-based personalities, Monte Carlo EV, a
calling station, random) plus frozen snapshots of its own past versions, so it
must stay good against all of them at once.

* Illegal actions are masked (logits set to −∞) — the network cannot choose them.
* A learned value baseline and GAE(λ) reduce reward variance; advantages are
  normalised per batch; rewards are chip results in 100-BB stacks.
* Hand equity is an input feature, so the network does not have to learn card
  strength from scratch.
"""

from __future__ import annotations

import json
import random
import time
from multiprocessing import Pool
from typing import Any

import numpy as np
import torch
from torch import nn

from poker_ai.sim.duplicate import duplicate_match

from .features import N_ACTIONS, OBS_DIM, NumpyPolicy
from .paths import CHECKPOINT_DIR, LOG_PATH, POLICY_PATH
from .rollout import Batch, PolicyAgent, collect


FIXED_LEAGUE = {"rule_tight": 1.0, "rule_balanced": 1.0, "rule_lag": 1.0, "calling_station": 0.6, "random": 0.3, "equity": 0.6}
SNAPSHOT_SHARE = 0.35  # fraction of hands against past selves once snapshots exist


class ActorCritic(nn.Module):
    def __init__(self, hidden: int = 128) -> None:
        super().__init__()
        self.l1 = nn.Linear(OBS_DIM, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.pi = nn.Linear(hidden, N_ACTIONS)
        self.v = nn.Linear(hidden, 1)
        nn.init.orthogonal_(self.pi.weight, 0.01)
        nn.init.orthogonal_(self.v.weight, 1.0)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.tanh(self.l2(torch.tanh(self.l1(x))))
        logits = self.pi(h).masked_fill(~mask, -1e9)
        return logits, self.v(h).squeeze(-1)

    def export(self) -> dict[str, np.ndarray]:
        g = lambda t: t.detach().cpu().numpy().astype(np.float32)  # noqa: E731
        return {"w1": g(self.l1.weight).T, "b1": g(self.l1.bias), "w2": g(self.l2.weight).T, "b2": g(self.l2.bias),
                "wp": g(self.pi.weight).T, "bp": g(self.pi.bias), "wv": g(self.v.weight).T, "bv": g(self.v.bias)}


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


def train(minutes: float = 30, workers: int = 3, decisions_per_update: int = 4096, seed: int = 0,
          snapshot_every: int = 10, eval_every: int = 10, eval_deals: int = 150, log=print) -> dict[str, Any]:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    net = ActorCritic()
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    snapshots: dict[str, dict[str, np.ndarray]] = {}
    history: list[dict[str, Any]] = []
    evals: list[dict[str, Any]] = []
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    t0 = time.time()
    update = 0
    total_decisions = 0
    with Pool(workers) as pool:
        while time.time() - t0 < minutes * 60:
            update += 1
            league = dict(FIXED_LEAGUE)
            if snapshots:
                recent = list(snapshots)[-5:]
                fixed_total = sum(FIXED_LEAGUE.values())
                for name in recent:
                    league[name] = fixed_total * SNAPSHOT_SHARE / (1 - SNAPSHOT_SHARE) / len(recent)
            weights = net.export()
            per = decisions_per_update // workers
            batches = pool.map(collect, [(weights, snapshots, league, per, rng.randrange(2**31)) for _ in range(workers)])
            b = merge(batches)
            total_decisions += len(b.actions)
            adv, ret = gae(b)
            obs = torch.tensor(np.stack(b.obs))
            masks = torch.tensor(np.stack(b.masks))
            acts = torch.tensor(b.actions)
            old_logp = torch.tensor(b.logp)
            adv_t = torch.tensor((adv - adv.mean()) / (adv.std() + 1e-8))
            ret_t = torch.tensor(ret)
            progress = min(1.0, (time.time() - t0) / (minutes * 60))
            ent_coef = 0.01 * (1 - progress) + 0.002 * progress
            stats = {"pl": 0.0, "vl": 0.0, "ent": 0.0, "kl": 0.0, "clip": 0.0, "n": 0}
            idx = np.arange(len(acts))
            for _epoch in range(4):
                np.random.shuffle(idx)
                for start in range(0, len(idx), 512):
                    mb = torch.tensor(idx[start : start + 512])
                    logits, v = net(obs[mb], masks[mb])
                    dist = torch.distributions.Categorical(logits=logits)
                    logp = dist.log_prob(acts[mb])
                    ratio = torch.exp(logp - old_logp[mb])
                    a = adv_t[mb]
                    pl = -torch.min(ratio * a, torch.clamp(ratio, 0.8, 1.2) * a).mean()
                    vl = ((v - ret_t[mb]) ** 2).mean()
                    ent = dist.entropy().mean()
                    loss = pl + 0.5 * vl - ent_coef * ent
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                    opt.step()
                    with torch.no_grad():
                        stats["pl"] += pl.item()
                        stats["vl"] += vl.item()
                        stats["ent"] += ent.item()
                        stats["kl"] += (old_logp[mb] - logp).mean().item()
                        stats["clip"] += ((ratio - 1).abs() > 0.2).float().mean().item()
                        stats["n"] += 1
            by_opp: dict[str, list[float]] = {}
            for name, bb in b.hand_results:
                by_opp.setdefault(name, []).append(bb)
            row = {"update": update, "seconds": round(time.time() - t0, 1), "decisions": total_decisions,
                   "hands": len(b.hand_results), "bb100_all": round(100 * float(np.mean([x for _, x in b.hand_results])), 1),
                   "bb100_by_opponent": {k: round(100 * float(np.mean(v)), 1) for k, v in by_opp.items() if not k.startswith("self@")},
                   **{k: round(stats[k] / max(1, stats["n"]), 4) for k in ("pl", "vl", "ent", "kl", "clip")},
                   "action_freq": np.bincount(acts.numpy(), minlength=N_ACTIONS).round(0).tolist()}
            history.append(row)
            log(f"update {update:4d}  {row['seconds']:6.0f}s  bb/100 {row['bb100_all']:+8.1f}  entropy {row['ent']:.3f}  "
                f"value-loss {row['vl']:.4f}  kl {row['kl']:+.4f}")
            if update % snapshot_every == 0:
                snapshots[f"self@{update}"] = net.export()
                torch.save(net.state_dict(), CHECKPOINT_DIR / f"ppo_{update:05d}.pt")
            if update % eval_every == 0:
                evals.append(evaluate(net.export(), update, eval_deals, log))
    final = net.export()
    np.savez(POLICY_PATH, **final)
    torch.save(net.state_dict(), CHECKPOINT_DIR / "ppo_final.pt")
    final_eval = evaluate(final, update, eval_deals * 4, log, final=True)
    evals.append(final_eval)
    meta = {"updates": update, "decisions": total_decisions, "minutes": minutes, "workers": workers,
            "trained_at": time.strftime("%Y-%m-%d %H:%M"), "final_eval": final_eval, "obs_dim": OBS_DIM}
    meta["status"] = training_status(meta)
    LOG_PATH.write_text(json.dumps({"meta": meta, "history": history, "evals": evals}, indent=1))
    return meta


EVAL_OPPONENTS = ("rule_balanced", "rule_tight", "rule_lag", "calling_station")


def evaluate(weights: dict[str, np.ndarray], update: int, deals: int, log=print, final: bool = False) -> dict[str, Any]:
    from poker_ai.agents import make_agent

    out: dict[str, Any] = {"update": update, "final": final}
    for opp in EVAL_OPPONENTS:
        r = duplicate_match(PolicyAgent(NumpyPolicy(weights), "neural", seed=update), make_agent(opp, seed=7), deals, seed=1234)
        out[opp] = {"bb100": round(r.bb100, 1), "ci95": round(r.ci95, 1), "hands": r.hands}
    log(f"  eval @ {update}: " + ", ".join(f"{k} {v['bb100']:+.0f}±{v['ci95']:.0f}" for k, v in out.items() if isinstance(v, dict)))
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

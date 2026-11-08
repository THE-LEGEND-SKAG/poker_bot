"""Self-contained hand rollouts for PPO, run inside worker processes.

Each worker keeps its own opponent models (one per league member) across
updates, so the learner's opponent-statistics features mean the same thing in
training as at play time. Policies arrive as NumPy weights; no PyTorch here.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from poker_ai.agents import make_agent
from poker_ai.agents.base import Agent
from poker_ai.agents.opponent_model import OpponentModel
from poker_ai.engine import Action, HandState, PlayerView

from .features import N_ACTIONS, OBS_DIM, NumpyPolicy, encode, legal_mask, main_opponent, to_action

BB = 100


class PolicyAgent(Agent):
    """A frozen NumPy policy used as a league opponent (or for evaluation)."""

    kind = "neural_snapshot"
    level = 5

    def __init__(self, policy: NumpyPolicy, name: str, seed: int, greedy: bool = False) -> None:
        super().__init__(name)
        self.policy = policy
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.model = OpponentModel()
        self.greedy = greedy

    def act(self, view: PlayerView) -> Action:
        opp = main_opponent(view)
        x = encode(view, self.model.get(opp) if opp is not None else None, self.rng)
        probs, _ = self.policy.forward(x, legal_mask(view))
        idx = int(np.argmax(probs)) if self.greedy else int(self.np_rng.choice(N_ACTIONS, p=probs))
        return to_action(view, idx)

    def on_hand_end(self, view: PlayerView) -> None:
        self.model.observe_hand(view)


@dataclass
class Batch:
    obs: list[np.ndarray] = field(default_factory=list)
    masks: list[np.ndarray] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    logp: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    episode_end: list[bool] = field(default_factory=list)  # last learner decision of a hand
    rewards: list[float] = field(default_factory=list)  # nonzero only at episode ends
    hand_results: list[tuple[str, float]] = field(default_factory=list)  # (opponent, bb won)


_STATE: dict[str, Any] = {}


def _opponent(name: str, snapshots: dict[str, dict[str, np.ndarray]], seed: int) -> Agent:
    cache = _STATE.setdefault("agents", {})
    if name not in cache:
        if name in snapshots:
            cache[name] = PolicyAgent(NumpyPolicy(snapshots[name]), name, seed)
        else:
            cache[name] = make_agent(name, seed=seed)
    return cache[name]


def collect(args: tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]], dict[str, float], int, int]) -> Batch:
    """Play hands until ``n_decisions`` learner decisions are recorded."""
    weights, snapshots, league_weights, n_decisions, seed = args
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    policy = NumpyPolicy(weights)
    models: dict[str, OpponentModel] = _STATE.setdefault("models", {})
    batch = Batch()
    names = list(league_weights)
    probs = np.array([league_weights[n] for n in names], dtype=float)
    probs /= probs.sum()
    while len(batch.actions) < n_decisions:
        opp_name = names[int(np_rng.choice(len(names), p=probs))]
        opponent = _opponent(opp_name, snapshots, rng.randrange(2**31))
        model = models.setdefault(opp_name, OpponentModel(me=0))  # learner's view of this league member
        stack = BB * rng.choice((100, 100, 100, 60, 150))
        me, them = 0, 1  # learner always seat 0; position varies with the button
        h = HandState([stack, stack], rng.randrange(2), BB // 2, BB, rng=rng)
        start = len(batch.actions)
        while not h.terminal:
            seat = h.to_act
            view = h.view(seat)
            if seat == me:
                x = encode(view, model.get(them), rng)
                m = legal_mask(view)
                p, v = policy.forward(x, m)
                a = int(np_rng.choice(N_ACTIONS, p=p))
                batch.obs.append(x)
                batch.masks.append(m)
                batch.actions.append(a)
                batch.logp.append(float(np.log(p[a] + 1e-12)))
                batch.values.append(v)
                batch.episode_end.append(False)
                batch.rewards.append(0.0)
                h.apply(to_action(view, a))
            else:
                h.apply(opponent.act(view))
        if len(batch.actions) > start:
            batch.episode_end[-1] = True
            batch.rewards[-1] = h.net[me] / (100 * BB)  # reward in 100-BB stacks
        batch.hand_results.append((opp_name, h.net[me] / BB))
        opponent.on_hand_end(h.view(them))
        model.observe_hand(h.view(me))
    return batch


def obs_dim() -> int:
    return OBS_DIM

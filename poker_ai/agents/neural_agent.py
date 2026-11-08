"""Level 5: neural policy trained with PPO against a league.

Plays from exported NumPy weights (``poker_ai/data/neural_policy.npz``), so the
server needs no PyTorch. It keeps its own opponent model from public history and
feeds those statistics to the network, as in training. The training status comes
from the training log: "trained" only if the final evaluation beat every
rule-based bot with 95% confidence; otherwise "partially trained"; "untrained"
if no weights exist (it then plays uniformly at random over legal actions).
"""

from __future__ import annotations

import json
import random
from functools import cache
from typing import Any

import numpy as np

from poker_ai.engine import Action, PlayerView
from poker_ai.learning.paths import LOG_PATH, POLICY_PATH
from poker_ai.learning.features import ACTIONS, N_ACTIONS, NumpyPolicy, encode, legal_mask, main_opponent, to_action

from .base import Agent
from .opponent_model import OpponentModel


@cache
def _load() -> tuple[NumpyPolicy | None, dict[str, Any]]:
    policy = NumpyPolicy.load(POLICY_PATH) if POLICY_PATH.exists() else None
    meta = json.loads(LOG_PATH.read_text())["meta"] if LOG_PATH.exists() else {}
    return policy, meta


class NeuralAgent(Agent):
    kind = "neural"
    level = 5

    def __init__(self, name: str | None = None, seed: int | None = None, greedy: bool = False) -> None:
        super().__init__(name or "Neural (PPO)")
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.policy, self.meta = _load()
        self.training_status = self.meta.get("status", "untrained") if self.policy else "untrained"
        self.model = OpponentModel()
        self.greedy = greedy

    def act(self, view: PlayerView) -> Action:
        mask = legal_mask(view)
        if self.policy is None:
            probs = mask / mask.sum()
            value = 0.0
        else:
            opp = main_opponent(view)
            x = encode(view, self.model.get(opp) if opp is not None else None, self.rng)
            probs, value = self.policy.forward(x, mask)
        idx = int(np.argmax(probs)) if self.greedy else int(self.np_rng.choice(N_ACTIONS, p=probs))
        action = to_action(view, idx)
        self.last_explanation = {"agent": self.name, "action": str(action), "rule": f"policy sample: {ACTIONS[idx]}",
                                 "policy": {a: round(float(p), 3) for a, p, ok in zip(ACTIONS, probs, mask) if ok},
                                 "value_estimate_bb": round(value * 100, 1), "training_status": self.training_status}
        return action

    def on_hand_end(self, view: PlayerView) -> None:
        self.model.observe_hand(view)

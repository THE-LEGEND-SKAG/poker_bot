"""Poker agents and a registry used by the server and simulator."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .base import Agent
from .equity_agent import EquityAgent
from .exploitative import ExploitativeAgent
from .random_agent import CallingStation, RandomAgent
from .rule_based import PERSONALITIES, RuleBasedAgent

# kind -> (factory, metadata). Factories take (name, seed).
_REGISTRY: dict[str, tuple[Callable[[str | None, int | None], Agent], dict[str, Any]]] = {
    "random": (lambda n, s: RandomAgent(n, s), {"label": "Random", "level": 0}),
    "calling_station": (lambda n, s: CallingStation(n), {"label": "Calling Station", "level": 0}),
}
for _key, _p in PERSONALITIES.items():
    _REGISTRY[f"rule_{_key}"] = (lambda n, s, k=_key: RuleBasedAgent(k, n, s), {"label": f"Rule-based · {_p.label}", "level": 1})

_REGISTRY["equity"] = (lambda n, s: EquityAgent(n, s), {"label": "Monte Carlo EV", "level": 2})
_REGISTRY["exploit"] = (lambda n, s: ExploitativeAgent(n, s), {"label": "Exploitative (opponent model)", "level": 3})


def _make_cfr(name: str | None, seed: int | None) -> Agent:
    from .cfr_agent import CFRAgent  # imported lazily: the solver package imports agents.equity

    return CFRAgent(name, seed)


_REGISTRY["cfr"] = (_make_cfr, {"label": "CFR blueprint (heads-up)", "level": 4})


def _make_neural(name: str | None, seed: int | None) -> Agent:
    from .neural_agent import NeuralAgent

    return NeuralAgent(name, seed)


_REGISTRY["neural"] = (_make_neural, {"label": "Neural (PPO)", "level": 5})


def register(kind: str, factory: Callable[[str | None, int | None], Agent], label: str, level: int) -> None:
    _REGISTRY[kind] = (factory, {"label": label, "level": level})


def available_agents() -> list[dict[str, Any]]:
    return [{"kind": k, **meta} for k, (_, meta) in _REGISTRY.items()]


def make_agent(kind: str, name: str | None = None, seed: int | None = None) -> Agent:
    if kind not in _REGISTRY:
        raise ValueError(f"unknown agent kind {kind!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[kind][0](name, seed)


__all__ = ["Agent", "CallingStation", "EquityAgent", "ExploitativeAgent", "RandomAgent", "RuleBasedAgent", "available_agents", "make_agent", "register"]

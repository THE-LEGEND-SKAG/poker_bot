"""A human-vs-bots game session, independent of the transport.

The session never sends hidden information: :meth:`GameSession.snapshot` builds
the state from the human seat's point of view (bots' cards appear only when
shown at showdown).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from poker_ai.agents import available_agents, make_agent
from poker_ai.agents.base import Agent
from poker_ai.agents.exploitative import modelled_analysis
from poker_ai.agents.opponent_model import STAT_LABELS, OpponentModel, blocked_mask, class_grid
from poker_ai.agents.preflop import hand_class, preflop_equity
from poker_ai.engine import Action, ActionType, HandState, Table, TableConfig, card_str, describe, evaluate

HUMAN = 0
BOT_NAMES = ["Ada", "Blaise", "Carmen", "Dmitri", "Esme", "Farid", "Greta", "Hugo", "Ines"]


@dataclass
class GameConfig:
    bots: list[str] = field(default_factory=lambda: ["rule_balanced"])
    starting_stack: int = 10_000
    small_blind: int = 50
    big_blind: int = 100
    ante: int = 0
    human_name: str = "You"
    rebuy_bots: bool = True
    seed: int | None = None

    def validate(self) -> None:
        if not 1 <= len(self.bots) <= 8:
            raise ValueError("choose 1-8 bots")
        known = {a["kind"] for a in available_agents()}
        for b in self.bots:
            if b not in known:
                raise ValueError(f"unknown bot {b!r}")
        if self.starting_stack < self.big_blind * 2:
            raise ValueError("starting stack must be at least two big blinds")
        TableConfig(num_seats=len(self.bots) + 1, starting_stack=self.starting_stack,
                    small_blind=self.small_blind, big_blind=self.big_blind, ante=self.ante)


def _labels() -> dict[str, str]:
    return {a["kind"]: a["label"] for a in available_agents()}


class GameSession:
    def __init__(self, config: GameConfig) -> None:
        config.validate()
        self.config = config
        self.rng = random.Random(config.seed)
        n = len(config.bots) + 1
        self.table = Table(TableConfig(num_seats=n, starting_stack=config.starting_stack, small_blind=config.small_blind,
                                       big_blind=config.big_blind, ante=config.ante), seed=self.rng.randrange(2**31))
        names = self.rng.sample(BOT_NAMES, len(config.bots))
        self.agents: dict[int, Agent] = {
            i + 1: make_agent(kind, name=names[i], seed=self.rng.randrange(2**31)) for i, kind in enumerate(config.bots)
        }
        self.kinds = {HUMAN: "human", **{i + 1: k for i, k in enumerate(config.bots)}}
        self.names = {HUMAN: config.human_name, **{s: a.name for s, a in self.agents.items()}}
        self.hand: HandState | None = None
        self.log: list[dict[str, Any]] = []
        self.last_action: dict[int, str] = {}
        self._event_cursor = 0
        self._settled = False
        self.session_net = [0] * n
        self.hands_played = 0
        self.decisions: list[dict[str, Any]] = []  # bot reasoning for the current hand
        self.coach_model = OpponentModel(me=HUMAN)  # learned only from what the human can see
        self.np_rng = np.random.default_rng(self.rng.randrange(2**31))

    # ------------------------------------------------------------- lifecycle
    @property
    def num_seats(self) -> int:
        return len(self.agents) + 1

    @property
    def human_busted(self) -> bool:
        return self.table.stacks[HUMAN] == 0 and (self.hand is None or self.hand.terminal)

    def start_hand(self) -> None:
        if self.hand is not None and not self.hand.terminal:
            raise RuntimeError("hand in progress")
        if self.hand is not None and not self._settled:
            self._settle()
        if self.config.rebuy_bots:
            for s in self.agents:
                if self.table.stacks[s] < self.config.big_blind:
                    self.table.rebuy(s)
                    self.log.append({"text": f"{self.names[s]} {self._verb(s, 'rebuys')} for {self.config.starting_stack}", "kind": "info"})
        if self.table.stacks[HUMAN] == 0:
            raise RuntimeError("you are out of chips: rebuy to continue")
        if self.table.is_over:
            raise RuntimeError("not enough players with chips")
        self.hand = self.table.start_hand()
        self._event_cursor = 0
        self._settled = False
        self.last_action = {}
        self.decisions = []
        self._consume_events()

    def rebuy_human(self) -> None:
        if self.hand is not None and not self.hand.terminal:
            raise RuntimeError("cannot rebuy during a hand")
        if self.hand is not None and not self._settled:
            self._settle()
        self.table.rebuy(HUMAN)
        self.log.append({"text": f"{self.names[HUMAN]} {self._verb(HUMAN, 'rebuys')} for {self.config.starting_stack}", "kind": "info"})

    def _settle(self) -> None:
        assert self.hand is not None and self.hand.terminal
        net = self.table.end_hand()
        if self.hand.players[HUMAN].in_hand:
            self.coach_model.observe_hand(self.hand.view(HUMAN))
        for s in range(self.num_seats):
            self.session_net[s] += net[s]
        for s, agent in self.agents.items():
            if self.hand.players[s].in_hand:
                agent.on_hand_end(self.hand.view(s))
        self.hands_played += 1
        self._settled = True

    # --------------------------------------------------------------- acting
    @property
    def to_act(self) -> int | None:
        return None if self.hand is None or self.hand.terminal else self.hand.to_act

    @property
    def bot_to_act(self) -> bool:
        return self.to_act is not None and self.to_act != HUMAN

    def human_act(self, action_type: str, amount: int | None = None) -> None:
        if self.to_act != HUMAN:
            raise RuntimeError("it is not your turn")
        assert self.hand is not None
        t = ActionType(action_type)
        action = Action(t, int(amount or 0)) if t in (ActionType.BET, ActionType.RAISE) else Action(t)
        self.hand.apply(action)  # raises IllegalActionError on bad input; state unchanged
        self._consume_events()
        self._maybe_settle()

    def bot_step(self) -> dict[str, Any] | None:
        """Let the bot to act decide and apply. Returns its explanation."""
        seat = self.to_act
        if seat is None or seat == HUMAN:
            return None
        assert self.hand is not None
        agent = self.agents[seat]
        street = self.hand.street.label
        action = agent.act(self.hand.view(seat))
        self.hand.apply(action)
        self.decisions.append({"seat": seat, "name": self.names[seat], "street": street, "action": str(action),
                               "explanation": agent.last_explanation or {}})
        self._consume_events()
        self._maybe_settle()
        return agent.last_explanation

    @property
    def decision_id(self) -> str:
        """Identifies the current decision point (used to discard stale coach results)."""
        return f"{self.hand.hand_id if self.hand else 0}:{len(self.hand.events) if self.hand else 0}"

    def coach(self, sims: int = 1500) -> dict[str, Any] | None:
        """Level 3 analysis of the human's own decision, using an opponent model
        built only from the human's own observations (public information)."""
        if self.to_act != HUMAN:
            return None
        assert self.hand is not None
        view = self.hand.view(HUMAN)
        a, weights = modelled_analysis(view, self.coach_model, sims, random.Random(self.rng.random()), self.np_rng)
        blocked = blocked_mask(list(view.hole) + list(view.board))
        # stack-to-pot ratio: effective stack (ours vs the deepest live opponent) over the pot after calling
        me = view.me
        deepest = max((o.stack for o in view.opponents), default=0)
        spr = min(me.stack, deepest) / max(1, view.pot + (view.legal.to_call if view.legal else 0))
        opponents = [
            {"seat": seat, "name": self.names[seat], "grid": class_grid(w, blocked),
             "stats": self.coach_model.get(seat).as_dict()}
            for seat, w in weights.items()
        ]
        made = describe(evaluate(list(view.hole) + list(view.board))) if len(view.board) >= 3 else None
        cls = hand_class(*view.hole)
        return {
            "decision_id": self.decision_id,
            "hand_class": cls,
            "preflop_equity_vs_random": preflop_equity(view.hole[0], view.hole[1], max(1, view.num_live - 1)),
            "made_hand": made,
            "opponent_ranges": opponents,
            "spr": round(spr, 2),
            "effective_stack_bb": round(min(me.stack, deepest) / view.big_blind, 1),
            "stat_labels": STAT_LABELS,
            **a.as_dict(),
        }

    def review(self) -> list[dict[str, Any]] | None:
        """Bots' reasoning, revealed only after the hand. Numbers derived from a
        bot's hole cards are shown only if those cards were shown at showdown."""
        if self.hand is None or not self.hand.terminal:
            return None
        out = []
        for d in self.decisions:
            shown = d["seat"] in self.hand.shown
            exp = d["explanation"]
            item = {"seat": d["seat"], "name": d["name"], "street": d["street"], "action": d["action"],
                    "rule": exp.get("rule"), "revealed": shown}
            if shown:
                item["details"] = {k: v for k, v in exp.items() if k not in ("agent", "action", "rule")}
            out.append(item)
        return out

    def _maybe_settle(self) -> None:
        if self.hand is not None and self.hand.terminal and not self._settled:
            self._settle()

    # ------------------------------------------------------------- logging
    def _consume_events(self) -> None:
        assert self.hand is not None
        for e in self.hand.events[self._event_cursor :]:
            line = self._describe_event(e)
            if line:
                self.log.append(line)
        self._event_cursor = len(self.hand.events)
        self.log = self.log[-200:]

    @staticmethod
    def _verb(seat: int, third_person: str) -> str:
        """'calls' -> 'call' for the human ("You call")."""
        if seat != HUMAN:
            return third_person
        first, _, rest = third_person.partition(" ")
        first = first[:-2] if first.endswith("es") and first[:-2].endswith(("sh", "ch")) else first[:-1]
        return f"{first} {rest}".strip()

    def _describe_event(self, e: dict[str, Any]) -> dict[str, Any] | None:
        n = self.names
        v = self._verb
        t = e["type"]
        if t == "hand_start":
            verb = "have" if e["button"] == HUMAN else "has"
            return {"text": f"— Hand #{e['hand_id']} — {n[e['button']]} {verb} the button", "kind": "header"}
        if t == "post":
            label = {"small_blind": "small blind", "big_blind": "big blind", "ante": "ante"}[e["kind"]]
            if e["kind"] != "ante":
                self.last_action[e["seat"]] = label.title()
            return {"text": f"{n[e['seat']]} {v(e['seat'], 'posts')} {label} {e['amount']}", "kind": "post"} if e["kind"] != "ante" else None
        if t == "action":
            a = e["action"]
            word = {"fold": "folds", "check": "checks", "call": f"calls {e['added']}", "bet": f"bets {e['amount']}",
                    "raise": f"raises to {e['amount']}"}[a]
            if e["all_in"]:
                word += " (all-in)"
            self.last_action[e["seat"]] = {"fold": "Fold", "check": "Check", "call": "Call", "bet": f"Bet {e['amount']}",
                                           "raise": f"Raise {e['amount']}"}[a] + (" · All-in" if e["all_in"] else "")
            return {"text": f"{n[e['seat']]} {v(e['seat'], word)}", "kind": "action", "seat": e["seat"]}
        if t == "street":
            for s in list(self.last_action):
                if self.last_action[s] != "Fold" and "All-in" not in self.last_action[s]:
                    del self.last_action[s]
            return {"text": f"{e['street'].title()}: {' '.join(card_str(c) for c in e['board'])}", "kind": "street"}
        if t == "refund":
            return {"text": f"Uncalled {e['amount']} returned to {n[e['seat']]}", "kind": "info"}
        if t == "showdown":
            return {"text": "Showdown: " + "; ".join(f"{n[int(s)]} {v(int(s), 'shows')} {' '.join(card_str(c) for c in h['cards'])} ({h['description']})"
                                                     for s, h in e["hands"].items()), "kind": "showdown"}
        if t == "award":
            who = " & ".join(n[w] for w in e["winners"])
            how = f" with {e['description']}" if e.get("description") else ""
            pot = "the pot" if e["pot_index"] == 0 else f"side pot {e['pot_index']}"
            verb = "wins" if len(e["winners"]) == 1 and e["winners"][0] != HUMAN else ("win" if e["winners"] == [HUMAN] else "split")
            return {"text": f"{who} {verb} {pot} ({e['amount']}){how}", "kind": "win"}
        return None

    # ------------------------------------------------------------ snapshot
    def snapshot(self) -> dict[str, Any]:
        h = self.hand
        labels = _labels()
        seats = []
        winners: set[int] = set()
        result = None
        if h is not None and h.terminal:
            for pot in h.pots_awarded:
                winners.update(pot["winners"])
            showdown = next((e for e in h.events if e["type"] == "showdown"), None)
            result = {
                "winners": sorted(winners),
                "net": h.net,
                "pots": h.pots_awarded,
                "hands": {int(s): v["description"] for s, v in (showdown["hands"].items() if showdown else [])},
            }
        for s in range(self.num_seats):
            p = h.players[s] if h else None
            cards = None
            if p is not None and p.in_hand:
                if s == HUMAN or s in h.shown:  # type: ignore[union-attr]
                    cards = [card_str(c) for c in p.hole]
            seats.append({
                "seat": s,
                "name": self.names[s],
                "kind": self.kinds[s],
                "label": "Human" if s == HUMAN else labels.get(self.kinds[s], self.kinds[s]),
                "stack": p.stack if (p and not h.terminal) else self.table.stacks[s],  # type: ignore[union-attr]
                "bet": p.street_commit if (p and not h.terminal) else 0,  # type: ignore[union-attr]
                "won": h.winnings[s] if (h and h.terminal) else 0,
                "committed": p.total_commit if p else 0,
                "in_hand": bool(p and p.in_hand),
                "folded": bool(p and p.folded),
                "all_in": bool(p and p.all_in),
                "cards": cards,
                "last_action": self.last_action.get(s),
                "winner": s in winners,
                "session_net": self.session_net[s],
            })
        legal = None
        if h is not None and self.to_act == HUMAN:
            la = h.legal_actions()
            legal = {
                "can_fold": la.can_fold, "can_check": la.can_check, "can_call": la.can_call,
                "call_amount": la.call_amount, "to_call": la.to_call,
                "raise_type": la.raise_type.value if la.raise_type else None,
                "min_raise_to": la.min_raise_to, "max_raise_to": la.max_raise_to,
                "street_commit": la.street_commit, "stack": la.stack,
            }
        return {
            "hand_id": h.hand_id if h else 0,
            "street": h.street.label if h else None,
            "board": [card_str(c) for c in h.board] if h else [],
            "pot": h.pot if h else 0,
            "pot_middle": (h.pot - (0 if h.terminal else sum(p.street_commit for p in h.players))) if h else 0,
            "current_bet": h.current_bet if h else 0,
            "small_blind": self.config.small_blind,
            "big_blind": self.config.big_blind,
            "button": h.button if h else None,
            "sb": h.sb_seat if h else None,
            "bb": h.bb_seat if h else None,
            "to_act": self.to_act,
            "human_seat": HUMAN,
            "seats": seats,
            "legal": legal,
            "terminal": h.terminal if h else True,
            "result": result,
            "log": self.log[-80:],
            "hands_played": self.hands_played,
            "human_busted": self.human_busted,
            "decision_id": self.decision_id,
            "review": self.review(),
        }

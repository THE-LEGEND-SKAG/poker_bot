"""No-Limit Texas Hold'em hand state machine.

:class:`HandState` plays exactly one hand. It owns all rules: blinds, betting
order, minimum raises (including incomplete all-in raises that do not reopen the
betting), uncalled-bet refunds, board run-outs, side pots, split pots and odd
chips. Agents never mutate it; they read a :class:`PlayerView` and submit an
:class:`Action`, which :meth:`HandState.apply` validates against
:meth:`HandState.legal_actions` — the single source of truth for legality.

Chip conventions
----------------
* ``stack``         chips behind (not yet committed)
* ``street_commit`` chips committed on the current betting round
* ``total_commit``  chips committed this hand (antes + all streets)
* ``pot``           sum of ``total_commit`` over all players

Simplifications (documented, deliberate): no burn cards; when the big blind is
short-stacked the other players must still match the full big blind; dead
button rules are left to :class:`~poker_ai.engine.table.Table`.
"""

from __future__ import annotations

import copy
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .actions import Action, ActionType, IllegalActionError, LegalActions, Street
from .cards import Card, Deck, card_str
from .hand_evaluator import describe, evaluate
from .pots import Pot, build_pots, split_pot


@dataclass(slots=True)
class PlayerState:
    seat: int
    stack: int
    starting_stack: int
    in_hand: bool  # dealt into this hand (had chips at the start)
    hole: list[Card] = field(default_factory=list)
    street_commit: int = 0
    total_commit: int = 0
    folded: bool = False
    all_in: bool = False
    acted: bool = False  # acted since the last time the betting was (re)opened
    last_acted_level: int | None = None  # current bet right after this player's last action this street

    @property
    def live(self) -> bool:
        return self.in_hand and not self.folded

    @property
    def can_act(self) -> bool:
        return self.in_hand and not self.folded and not self.all_in


@dataclass(frozen=True, slots=True)
class PublicPlayer:
    seat: int
    stack: int
    street_commit: int
    total_commit: int
    in_hand: bool
    folded: bool
    all_in: bool
    shown_cards: tuple[Card, ...] | None  # revealed at showdown only


@dataclass(frozen=True, slots=True)
class PlayerView:
    """Everything one seat is allowed to know. This is all an agent receives."""

    seat: int
    hole: tuple[Card, ...]
    board: tuple[Card, ...]
    street: Street
    pot: int
    current_bet: int
    min_raise_increment: int
    small_blind: int
    big_blind: int
    button: int
    players: tuple[PublicPlayer, ...]
    to_act: int | None
    legal: LegalActions | None  # only when it is this seat's turn
    history: tuple[dict[str, Any], ...]  # public events only
    hand_id: int

    @property
    def me(self) -> PublicPlayer:
        return self.players[self.seat]

    @property
    def num_live(self) -> int:
        return sum(1 for p in self.players if p.in_hand and not p.folded)

    @property
    def opponents(self) -> list[PublicPlayer]:
        return [p for p in self.players if p.seat != self.seat and p.in_hand and not p.folded]


class HandState:
    """One hand of No-Limit Hold'em for 2-10 seats."""

    def __init__(
        self,
        stacks: Sequence[int],
        button: int,
        small_blind: int,
        big_blind: int,
        ante: int = 0,
        *,
        rng: random.Random | None = None,
        deck: Deck | None = None,
        hand_id: int = 0,
    ) -> None:
        if not 2 <= len(stacks) <= 10:
            raise ValueError("a table has 2-10 seats")
        if not (0 < small_blind <= big_blind) or ante < 0:
            raise ValueError("blinds must satisfy 0 < small blind <= big blind, ante >= 0")
        if any(s < 0 or int(s) != s for s in stacks):
            raise ValueError("stacks must be non-negative integers")
        if not 0 <= button < len(stacks):
            raise ValueError("button out of range")

        self.n = len(stacks)
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.ante = ante
        self.hand_id = hand_id
        self.button = button
        self.players = [PlayerState(seat=i, stack=int(s), starting_stack=int(s), in_hand=s > 0) for i, s in enumerate(stacks)]
        participants = [p.seat for p in self.players if p.in_hand]
        if len(participants) < 2:
            raise ValueError("need at least two players with chips")

        # Seats in clockwise order starting left of the button.
        self.order = [(button + 1 + i) % self.n for i in range(self.n) if self.players[(button + 1 + i) % self.n].in_hand]
        if len(self.order) == 2 and self.players[button].in_hand:
            self.sb_seat, self.bb_seat = button, self.order[0]  # heads-up: button is the small blind
        else:
            self.sb_seat, self.bb_seat = self.order[0], self.order[1]

        self.deck = deck if deck is not None else Deck(rng or random.Random())
        self.board: list[Card] = []
        self.street = Street.PREFLOP
        self.current_bet = 0
        self.min_raise_inc = big_blind
        self.to_act: int | None = None
        self.terminal = False
        self.events: list[dict[str, Any]] = []
        self.pots_awarded: list[dict[str, Any]] = []
        self.winnings = [0] * self.n
        self.shown: dict[int, list[Card]] = {}

        self._log({"type": "hand_start", "hand_id": hand_id, "button": button, "sb": self.sb_seat, "bb": self.bb_seat,
                   "stacks": [p.stack for p in self.players], "small_blind": small_blind, "big_blind": big_blind, "ante": ante})
        self._deal_hole_cards()
        self._post_forced_bets()
        self._advance(start_after=self.bb_seat)

    # ------------------------------------------------------------------ setup
    def _deal_hole_cards(self) -> None:
        deal_order = [self.sb_seat] + [s for s in self._clockwise_from(self.sb_seat) if s != self.sb_seat]
        for _ in range(2):
            for seat in deal_order:
                self.players[seat].hole.extend(self.deck.deal(1))

    def _post_forced_bets(self) -> None:
        if self.ante:
            for seat in self.order:
                p = self.players[seat]
                amt = min(self.ante, p.stack)
                p.stack -= amt
                p.total_commit += amt
                p.all_in = p.stack == 0
                self._log({"type": "post", "seat": seat, "kind": "ante", "amount": amt})
        for seat, size, kind in ((self.sb_seat, self.small_blind, "small_blind"), (self.bb_seat, self.big_blind, "big_blind")):
            p = self.players[seat]
            amt = min(size, p.stack)
            self._commit(p, amt)
            self._log({"type": "post", "seat": seat, "kind": kind, "amount": amt})
        self.current_bet = self.big_blind
        self.min_raise_inc = self.big_blind

    # ---------------------------------------------------------------- helpers
    def _clockwise_from(self, seat: int) -> list[int]:
        """Participating seats clockwise starting at ``seat`` (inclusive)."""
        return [(seat + i) % self.n for i in range(self.n) if self.players[(seat + i) % self.n].in_hand]

    def _commit(self, p: PlayerState, amount: int) -> None:
        assert 0 <= amount <= p.stack, "engine invariant: cannot commit more than the stack"
        p.stack -= amount
        p.street_commit += amount
        p.total_commit += amount
        if p.stack == 0:
            p.all_in = True

    def _log(self, event: dict[str, Any]) -> None:
        self.events.append(event)

    @property
    def pot(self) -> int:
        return sum(p.total_commit for p in self.players)

    def live_seats(self) -> list[int]:
        return [p.seat for p in self.players if p.live]

    # ------------------------------------------------------------ legal moves
    def legal_actions(self) -> LegalActions:
        if self.terminal or self.to_act is None:
            raise IllegalActionError("hand is over")
        p = self.players[self.to_act]
        to_call = max(0, self.current_bet - p.street_commit)
        call_amount = min(to_call, p.stack)
        max_to = p.street_commit + p.stack
        others_can_respond = any(q.can_act for q in self.players if q.seat != p.seat)
        reopened = p.last_acted_level is None or self.current_bet - p.last_acted_level >= self.min_raise_inc
        raise_type: ActionType | None = None
        min_to = max_to
        if p.stack > to_call and others_can_respond and reopened:
            raise_type = ActionType.BET if self.current_bet == 0 else ActionType.RAISE
            min_to = min(self.current_bet + self.min_raise_inc, max_to)
        return LegalActions(
            seat=p.seat,
            can_fold=to_call > 0,
            can_check=to_call == 0,
            can_call=to_call > 0,
            call_amount=call_amount,
            to_call=to_call,
            raise_type=raise_type,
            min_raise_to=min_to,
            max_raise_to=max_to,
            street_commit=p.street_commit,
            stack=p.stack,
        )

    # ----------------------------------------------------------------- acting
    def apply(self, action: Action) -> None:
        """Apply the current player's action. Raises :class:`IllegalActionError`
        (leaving the state untouched) if the action is not legal."""
        legal = self.legal_actions()
        legal.validate(action)
        p = self.players[self.to_act]  # type: ignore[index]
        t = action.type
        added = 0
        if t is ActionType.FOLD:
            p.folded = True
        elif t is ActionType.CALL:
            added = legal.call_amount
            self._commit(p, added)
        elif t in (ActionType.BET, ActionType.RAISE):
            to = action.amount
            added = to - p.street_commit
            self._commit(p, added)
            increment = to - self.current_bet
            if increment >= self.min_raise_inc:
                self.min_raise_inc = increment  # full raise: sets the new minimum increment
            self.current_bet = to
            for q in self.players:  # everyone else must respond again
                if q.seat != p.seat and q.can_act:
                    q.acted = False
        p.acted = True
        p.last_acted_level = self.current_bet
        self._log({"type": "action", "seat": p.seat, "street": self.street.label, "action": t.value,
                   "amount": action.amount if t in (ActionType.BET, ActionType.RAISE) else added,
                   "added": added, "all_in": p.all_in, "pot": self.pot, "stack": p.stack})
        self._advance(start_after=p.seat)

    # ------------------------------------------------------------- progression
    def _round_complete(self) -> bool:
        actors = [p for p in self.players if p.can_act]
        if not actors:
            return True
        if len(actors) == 1:
            p = actors[0]
            others_max = max((q.street_commit for q in self.players if q.live and q.seat != p.seat), default=0)
            if p.street_commit >= others_max:  # covers every all-in: nothing to call, no one to raise
                return True
        return all(p.acted and p.street_commit >= self.current_bet for p in actors)

    def _next_to_act(self, start_after: int) -> int | None:
        for seat in self._clockwise_from((start_after + 1) % self.n):
            p = self.players[seat]
            if p.can_act and (not p.acted or p.street_commit < self.current_bet):
                return seat
        return None

    def _advance(self, start_after: int) -> None:
        while True:
            live = self.live_seats()
            if len(live) == 1:
                self._return_uncalled()
                self._award_uncontested(live[0])
                return
            if not self._round_complete():
                nxt = self._next_to_act(start_after)
                assert nxt is not None, "engine invariant: incomplete round must have an actor"
                self.to_act = nxt
                return
            self._return_uncalled()
            if self.street is Street.RIVER:
                self._showdown()
                return
            self._next_street()
            if sum(1 for p in self.players if p.can_act) < 2:
                continue  # nobody left to bet: keep dealing until showdown
            start_after = self.button
            # Heads-up the button acts first preflop but last postflop: start after the button.

    def _next_street(self) -> None:
        for p in self.players:
            p.street_commit = 0
            p.acted = False
            p.last_acted_level = None
        self.current_bet = 0
        self.min_raise_inc = self.big_blind
        self.street = Street(self.street + 1)
        new = self.deck.deal(3 if self.street is Street.FLOP else 1)
        self.board.extend(new)
        self._log({"type": "street", "street": self.street.label, "cards": list(new), "board": list(self.board), "pot": self.pot})

    def _return_uncalled(self) -> None:
        commits = sorted((p.street_commit for p in self.players if p.in_hand), reverse=True)
        if len(commits) < 2 or commits[0] == commits[1]:
            return
        top = next(p for p in self.players if p.in_hand and p.street_commit == commits[0])
        refund = commits[0] - commits[1]
        top.street_commit -= refund
        top.total_commit -= refund
        top.stack += refund
        if top.stack > 0:
            top.all_in = False
        self._log({"type": "refund", "seat": top.seat, "amount": refund})

    # ------------------------------------------------------------ settlement
    def _finish(self) -> None:
        self.terminal = True
        self.to_act = None
        self.street = Street.SHOWDOWN if len(self.live_seats()) > 1 else self.street
        total_before = sum(p.starting_stack for p in self.players)
        total_after = sum(p.stack for p in self.players)
        assert total_before == total_after, f"chip conservation violated: {total_before} -> {total_after}"
        self._log({"type": "hand_end", "stacks": [p.stack for p in self.players],
                   "net": [p.stack - p.starting_stack for p in self.players]})

    def _award_uncontested(self, seat: int) -> None:
        amount = self.pot
        self.players[seat].stack += amount
        self.winnings[seat] += amount
        self.pots_awarded.append({"amount": amount, "winners": [seat], "eligible": [seat], "uncontested": True})
        self._log({"type": "award", "pot_index": 0, "amount": amount, "winners": [seat], "shares": {seat: amount}, "uncontested": True})
        self._finish()

    def _showdown(self) -> None:
        live = self.live_seats()
        scores = {s: evaluate(self.players[s].hole + self.board) for s in live}
        for s in live:
            self.shown[s] = list(self.players[s].hole)
        self._log({"type": "showdown", "hands": {s: {"cards": list(self.players[s].hole), "score": scores[s], "description": describe(scores[s])} for s in live}})
        pots = build_pots({p.seat: p.total_commit for p in self.players if p.in_hand}, set(live))
        for idx, pot in enumerate(pots):
            best = max(scores[s] for s in pot.eligible)
            winners = [s for s in pot.eligible if scores[s] == best]
            shares = split_pot(pot.amount, winners, self.order)
            for s, amt in shares.items():
                self.players[s].stack += amt
                self.winnings[s] += amt
            self.pots_awarded.append({"amount": pot.amount, "winners": winners, "eligible": list(pot.eligible), "uncontested": False})
            self._log({"type": "award", "pot_index": idx, "amount": pot.amount, "winners": winners, "shares": shares,
                       "eligible": list(pot.eligible), "description": describe(best)})
        self._finish()

    # ---------------------------------------------------------------- queries
    @property
    def final_stacks(self) -> list[int]:
        if not self.terminal:
            raise RuntimeError("hand not finished")
        return [p.stack for p in self.players]

    @property
    def net(self) -> list[int]:
        return [p.stack - p.starting_stack for p in self.players]

    def public_events(self) -> list[dict[str, Any]]:
        """The event log with nothing private in it (hole cards are only in showdown events)."""
        return list(self.events)

    def view(self, seat: int) -> PlayerView:
        """Information set of ``seat``: own cards plus public information only."""
        players = tuple(
            PublicPlayer(
                seat=p.seat, stack=p.stack, street_commit=p.street_commit, total_commit=p.total_commit,
                in_hand=p.in_hand, folded=p.folded, all_in=p.all_in,
                shown_cards=tuple(self.shown[p.seat]) if p.seat in self.shown else None,
            )
            for p in self.players
        )
        legal = self.legal_actions() if (not self.terminal and self.to_act == seat) else None
        return PlayerView(
            seat=seat,
            hole=tuple(self.players[seat].hole),
            board=tuple(self.board),
            street=self.street,
            pot=self.pot,
            current_bet=self.current_bet,
            min_raise_increment=self.min_raise_inc,
            small_blind=self.small_blind,
            big_blind=self.big_blind,
            button=self.button,
            players=players,
            to_act=self.to_act,
            legal=legal,
            history=tuple(self.events),
            hand_id=self.hand_id,
        )

    def clone(self) -> HandState:
        return copy.deepcopy(self)

    def __repr__(self) -> str:
        return (f"HandState(street={self.street.label}, board=[{' '.join(card_str(c) for c in self.board)}], "
                f"pot={self.pot}, to_act={self.to_act}, terminal={self.terminal})")

"""Actions and the legal-action description handed to agents."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum


class Street(IntEnum):
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3
    SHOWDOWN = 4

    @property
    def label(self) -> str:
        return self.name.lower()


class ActionType(str, Enum):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"  # first aggressive action on a street
    RAISE = "raise"  # aggressive action facing a bet


@dataclass(frozen=True, slots=True)
class Action:
    """A player decision.

    For ``BET`` and ``RAISE``, ``amount`` is the *total* the player will have
    committed on the current street after the action ("raise to"). It is ignored
    for the other action types. Going all-in is a ``CALL`` (when short) or a
    ``BET``/``RAISE`` to :attr:`LegalActions.max_raise_to`.
    """

    type: ActionType
    amount: int = 0

    @staticmethod
    def fold() -> Action:
        return Action(ActionType.FOLD)

    @staticmethod
    def check() -> Action:
        return Action(ActionType.CHECK)

    @staticmethod
    def call() -> Action:
        return Action(ActionType.CALL)

    @staticmethod
    def bet(to: int) -> Action:
        return Action(ActionType.BET, int(to))

    @staticmethod
    def raise_to(to: int) -> Action:
        return Action(ActionType.RAISE, int(to))

    def __str__(self) -> str:
        if self.type in (ActionType.BET, ActionType.RAISE):
            return f"{self.type.value} to {self.amount}"
        return self.type.value


class IllegalActionError(ValueError):
    """Raised when an action violates the rules. The game state is unchanged."""


@dataclass(frozen=True, slots=True)
class LegalActions:
    """Everything the player to act may do. Produced only by the engine."""

    seat: int
    can_fold: bool
    can_check: bool
    can_call: bool
    call_amount: int  # chips added by calling (capped at stack => may be an all-in call)
    to_call: int  # chips needed to fully match the current bet
    raise_type: ActionType | None  # BET, RAISE, or None when aggression is not allowed
    min_raise_to: int  # smallest legal BET/RAISE "to" amount (an all-in may be smaller than a full raise)
    max_raise_to: int  # all-in "to" amount
    street_commit: int  # chips this player already has in on this street
    stack: int  # chips behind

    @property
    def can_raise(self) -> bool:
        return self.raise_type is not None

    def clamp_raise(self, to: int) -> int:
        """Snap an arbitrary "raise to" amount into the legal range."""
        if self.raise_type is None:
            raise IllegalActionError("raising is not allowed")
        return max(self.min_raise_to, min(int(to), self.max_raise_to))

    def raise_action(self, to: int) -> Action:
        """A legal BET/RAISE close to ``to`` (clamped into range)."""
        assert self.raise_type is not None
        return Action(self.raise_type, self.clamp_raise(to))

    def all_in(self) -> Action:
        if self.raise_type is not None:
            return Action(self.raise_type, self.max_raise_to)
        if self.can_call:
            return Action.call()
        raise IllegalActionError("no all-in action available")

    def passive(self) -> Action:
        """Check if possible, else call."""
        return Action.check() if self.can_check else Action.call()

    def validate(self, action: Action) -> None:
        t = action.type
        if t is ActionType.FOLD:
            if not self.can_fold:
                raise IllegalActionError("cannot fold when checking is free")
        elif t is ActionType.CHECK:
            if not self.can_check:
                raise IllegalActionError(f"cannot check facing a bet of {self.to_call}")
        elif t is ActionType.CALL:
            if not self.can_call:
                raise IllegalActionError("nothing to call")
        elif t in (ActionType.BET, ActionType.RAISE):
            if self.raise_type is None:
                raise IllegalActionError("betting/raising is not allowed here")
            if t is not self.raise_type:
                raise IllegalActionError(f"must use {self.raise_type.value}, not {t.value}")
            amt = action.amount
            if not isinstance(amt, int):
                raise IllegalActionError("amount must be an integer")
            if amt > self.max_raise_to:
                raise IllegalActionError(f"{t.value} to {amt} exceeds all-in amount {self.max_raise_to}")
            if amt < self.min_raise_to and amt != self.max_raise_to:
                raise IllegalActionError(f"{t.value} to {amt} is below the minimum {self.min_raise_to}")
        else:  # pragma: no cover
            raise IllegalActionError(f"unknown action {action!r}")

    def is_legal(self, action: Action) -> bool:
        try:
            self.validate(action)
        except IllegalActionError:
            return False
        return True

"""Per-agent playing statistics accumulated from finished hands."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from poker_ai.engine import HandState


@dataclass
class PlayStats:
    hands: int = 0
    vpip_opportunities: int = 0
    vpip: int = 0
    pfr: int = 0
    post_aggressive: int = 0  # postflop bets + raises
    post_calls: int = 0
    folds: int = 0
    decisions: int = 0
    saw_flop: int = 0
    showdowns: int = 0
    showdown_wins: int = 0
    pots_won: int = 0
    pot_total: int = 0
    decision_seconds: float = 0.0

    def add_hand(self, h: HandState, seat: int, big_blind: int) -> None:
        self.hands += 1
        acts = [e for e in h.events if e["type"] == "action" and e["seat"] == seat]
        pre = [e for e in acts if e["street"] == "preflop"]
        if any(e["action"] != "check" for e in pre):
            self.vpip_opportunities += 1
            self.vpip += any(e["action"] in ("call", "raise") for e in pre)
        self.pfr += any(e["action"] == "raise" for e in pre)
        for e in acts:
            self.decisions += 1
            self.folds += e["action"] == "fold"
            if e["street"] != "preflop":
                self.post_aggressive += e["action"] in ("bet", "raise")
                self.post_calls += e["action"] == "call"
        streets = [e["street"] for e in h.events if e["type"] == "street"]
        folded_pre = any(e["action"] == "fold" for e in pre)
        if "flop" in streets and not folded_pre:
            self.saw_flop += 1
        if seat in h.shown:
            self.showdowns += 1
            self.showdown_wins += h.winnings[seat] > 0
        self.pots_won += h.winnings[seat] > 0
        self.pot_total += h.pot

    def summary(self, big_blind: int) -> dict[str, float]:
        d = lambda a, b: round(a / b, 3) if b else 0.0  # noqa: E731
        return {
            "hands": self.hands,
            "vpip": d(self.vpip, self.vpip_opportunities),
            "pfr": d(self.pfr, self.hands),
            "aggression_factor": round(self.post_aggressive / self.post_calls, 2) if self.post_calls else float(self.post_aggressive),
            "fold_frequency": d(self.folds, self.decisions),
            "went_to_showdown": d(self.showdowns, self.saw_flop),
            "showdown_win_rate": d(self.showdown_wins, self.showdowns),
            "avg_pot_bb": round(self.pot_total / max(1, self.hands) / big_blind, 1),
            "avg_decision_ms": round(1000 * self.decision_seconds / max(1, self.decisions), 2),
        }

    def merge(self, other: PlayStats) -> None:
        for k, v in asdict(other).items():
            setattr(self, k, getattr(self, k) + v)

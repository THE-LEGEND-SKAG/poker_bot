"""Small two-player zero-sum poker games, compiled into explicit game trees.

A game is described by functions over histories (tuples of actions). It is then
compiled once into a :class:`Tree` of flat arrays, which the CFR solver and the
exact best-response / exploitability code traverse quickly.

* :class:`KuhnPoker` — 3 cards, 1 betting round. Known Nash game value for the
  first player: −1/18.
* :class:`LeducPoker` — 6 cards (two each of J, Q, K), two betting rounds with a
  public card, bet sizes 2 then 4, at most two bets per round. Standard benchmark
  for imperfect-information solvers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations

CHANCE = -1
TERMINAL = -2


@dataclass
class Tree:
    """Flat game tree. Node ``i``: ``kind[i]`` is a player (0/1), CHANCE or TERMINAL."""

    kind: list[int] = field(default_factory=list)
    children: list[list[int]] = field(default_factory=list)
    actions: list[list[str]] = field(default_factory=list)
    chance_probs: list[list[float]] = field(default_factory=list)
    infoset: list[int] = field(default_factory=list)  # index into infoset_keys, -1 if none
    utility: list[float] = field(default_factory=list)  # player 0's payoff at terminals
    infoset_keys: list[str] = field(default_factory=list)
    infoset_actions: list[list[str]] = field(default_factory=list)
    infoset_player: list[int] = field(default_factory=list)

    @property
    def num_nodes(self) -> int:
        return len(self.kind)

    @property
    def num_infosets(self) -> int:
        return len(self.infoset_keys)


class SmallGame:
    """History-based description of a small game. Subclasses implement the rules."""

    name = "game"

    def is_terminal(self, h: tuple) -> bool: ...
    def is_chance(self, h: tuple) -> bool: ...
    def chance_outcomes(self, h: tuple) -> list[tuple[object, float]]: ...
    def player(self, h: tuple) -> int: ...
    def legal_actions(self, h: tuple) -> list[str]: ...
    def utility(self, h: tuple) -> float: ...  # for player 0
    def infoset_key(self, h: tuple) -> str: ...

    def compile(self) -> Tree:
        tree = Tree()
        index: dict[str, int] = {}

        def build(h: tuple) -> int:
            i = tree.num_nodes
            tree.kind.append(0)
            tree.children.append([])
            tree.actions.append([])
            tree.chance_probs.append([])
            tree.infoset.append(-1)
            tree.utility.append(0.0)
            if self.is_terminal(h):
                tree.kind[i] = TERMINAL
                tree.utility[i] = self.utility(h)
                return i
            if self.is_chance(h):
                tree.kind[i] = CHANCE
                outs = self.chance_outcomes(h)
                tree.chance_probs[i] = [p for _, p in outs]
                tree.actions[i] = [str(o) for o, _ in outs]
                tree.children[i] = [build(h + (o,)) for o, _ in outs]
                return i
            p = self.player(h)
            acts = self.legal_actions(h)
            key = self.infoset_key(h)
            if key not in index:
                index[key] = len(tree.infoset_keys)
                tree.infoset_keys.append(key)
                tree.infoset_actions.append(acts)
                tree.infoset_player.append(p)
            tree.kind[i] = p
            tree.infoset[i] = index[key]
            tree.actions[i] = acts
            tree.children[i] = [build(h + (a,)) for a in acts]
            return i

        build(())
        return tree


# ---------------------------------------------------------------------------
class KuhnPoker(SmallGame):
    """Cards 0 < 1 < 2 (J, Q, K). Ante 1. Actions: p(ass) / b(et 1)."""

    name = "kuhn"
    CARDS = "JQK"

    def is_chance(self, h):
        return len(h) == 0

    def chance_outcomes(self, h):
        deals = list(permutations(range(3), 2))
        return [(d, 1 / len(deals)) for d in deals]

    def _bets(self, h):
        return "".join(h[1:])

    def is_terminal(self, h):
        return len(h) > 0 and self._bets(h) in ("pp", "bp", "bb", "pbp", "pbb")

    def player(self, h):
        return len(self._bets(h)) % 2

    def legal_actions(self, h):
        return ["p", "b"]

    def utility(self, h):
        c0, c1 = h[0]
        s = self._bets(h)
        if s in ("bp", "pbp"):  # someone folded to a bet
            folder = (len(s) - 1) % 2  # the last actor folded
            return 1.0 if folder == 1 else -1.0
        stake = 2.0 if "b" in s else 1.0
        return stake if c0 > c1 else -stake

    def infoset_key(self, h):
        p = self.player(h)
        return f"{self.CARDS[h[0][p]]}:{self._bets(h)}"


# ---------------------------------------------------------------------------
class LeducPoker(SmallGame):
    """Leduc Hold'em. Ranks 0..2 (J, Q, K), two of each. Ante 1 each.
    Round sizes 2 then 4, at most 2 bets/raises per round. Player 0 acts first
    in each round. Actions: f(old), c(heck/call), r(aise / bet)."""

    name = "leduc"
    RANKS = "JQK"
    RAISE = (2, 4)
    MAX_RAISES = 2

    # history: (deal, *round1 actions, [public card, *round2 actions])
    def _parse(self, h):
        deal = h[0]
        r1: list[str] = []
        public = None
        r2: list[str] = []
        for x in h[1:]:
            if isinstance(x, int):
                public = x
            elif public is None:
                r1.append(x)
            else:
                r2.append(x)
        return deal, r1, public, r2

    @staticmethod
    def _round_over(seq: list[str]) -> bool:
        if not seq:
            return False
        if seq[-1] == "f":
            return True
        return len(seq) >= 2 and seq[-1] == "c"

    def is_chance(self, h):
        if len(h) == 0:
            return True
        deal, r1, public, r2 = self._parse(h)
        return public is None and self._round_over(r1) and r1[-1] != "f"

    def chance_outcomes(self, h):
        deck = [0, 0, 1, 1, 2, 2]
        if len(h) == 0:
            outs: dict[tuple[int, int], float] = {}
            for i in range(6):
                for j in range(6):
                    if i != j:
                        k = (deck[i], deck[j])
                        outs[k] = outs.get(k, 0) + 1 / 30
            return list(outs.items())
        c0, c1 = h[0]
        rest = list(deck)
        rest.remove(c0)
        rest.remove(c1)
        outs2: dict[int, float] = {}
        for c in rest:
            outs2[c] = outs2.get(c, 0) + 1 / len(rest)
        return list(outs2.items())

    def is_terminal(self, h):
        if len(h) == 0:
            return False
        deal, r1, public, r2 = self._parse(h)
        if r1 and r1[-1] == "f":
            return True
        return public is not None and self._round_over(r2)

    def player(self, h):
        deal, r1, public, r2 = self._parse(h)
        seq = r1 if public is None else r2
        return len(seq) % 2

    def legal_actions(self, h):
        deal, r1, public, r2 = self._parse(h)
        seq = r1 if public is None else r2
        raises = seq.count("r")
        facing = raises > 0 and seq[-1] == "r"
        acts = ["f"] if facing else []
        acts.append("c")
        if raises < self.MAX_RAISES:
            acts.append("r")
        return acts

    def _contributions(self, r1, r2):
        contrib = [1, 1]
        for rnd, seq in enumerate((r1, r2)):
            bet = 0
            street = [0, 0]
            for i, a in enumerate(seq):
                p = i % 2
                if a == "r":
                    bet += self.RAISE[rnd]
                    street[p] = bet
                elif a == "c":
                    street[p] = bet
            contrib[0] += street[0]
            contrib[1] += street[1]
        return contrib

    def utility(self, h):
        deal, r1, public, r2 = self._parse(h)
        contrib = self._contributions(r1, r2)
        seq = r2 if public is not None else r1
        if seq[-1] == "f":
            folder = (len(seq) - 1) % 2
            # the winner collects the folder's contribution
            return float(contrib[1]) if folder == 1 else -float(contrib[0])
        c0, c1 = deal

        def strength(c):
            return 10 + c if c == public else c

        s0, s1 = strength(c0), strength(c1)
        if s0 == s1:
            return 0.0
        return float(contrib[1]) if s0 > s1 else -float(contrib[0])

    def infoset_key(self, h):
        deal, r1, public, r2 = self._parse(h)
        p = self.player(h)
        pub = self.RANKS[public] if public is not None else "-"
        return f"{self.RANKS[deal[p]]}{pub}:{''.join(r1)}/{''.join(r2)}"

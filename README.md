# Poker AI

A playable No-Limit Texas Hold'em platform: a rules-exact engine, a ladder of
AI opponents (rule-based → Monte Carlo → exploitative → CFR blueprint → neural
PPO), a duplicate-match evaluation harness, and a web table with a coach panel.

> **Status:** Milestones 1–2 complete: the engine and a playable web table against
> rule-based bots. Rows marked ☐ are **not built yet** and are not claimed to work.

![Three-way all-in with a side pot](docs/table-showdown.png)

## Progress

| # | Milestone | Status |
|---|-----------|--------|
| 1 | Engine: evaluator, betting, all-ins, side pots, randomised chip-conservation stress test | ✅ done |
| 2 | Rule-based bots + playable table UI (FastAPI + React vertical slice) | ✅ done, 94 tests total |
| 3 | Monte Carlo equity bot + Coach panel (equity, pot odds, EV per action) | ☐ |
| 4 | Opponent modelling (Bayesian stats, range narrowing) + 13×13 range heatmap | ☐ |
| 5 | CFR: verified on Kuhn (game value −1/18) and Leduc via exploitability, then MCCFR blueprint for abstracted heads-up NLHE | ☐ |
| 6 | Neural agent: PPO with action masking, equity feature, league training | ☐ |
| 7 | Duplicate tournaments, bb/100 with 95% CIs, Lab dashboard | ☐ |
| 8 | 3-player / 6-max polish, end-to-end checks, screenshots | ☐ |

## Architecture

```
poker_ai/
  engine/      cards, hand evaluator, actions + legality, game state, pots, table   ✅
  agents/      random, calling station, rule_based (3 personalities) ✅ · equity, exploitative, cfr, neural ☐
  solvers/     cfr (Kuhn/Leduc/HUNL), abstraction, exploitability                   ☐
  learning/    networks, ppo, league, checkpoints                                   ☐
  sim/         hand/session runner ✅ · duplicate matches, tournaments, CIs ☐
  server/      FastAPI + WebSocket game sessions                                    ✅
web/           React + Tailwind + Framer Motion: Table ✅ · Coach panel, Lab ☐
tests/
```

**Design rule:** the engine is the only thing that decides what is legal.
`HandState.legal_actions()` produces a `LegalActions` object; `HandState.apply()`
validates every action against it and raises `IllegalActionError` (leaving the
state untouched) otherwise. Agents only ever see a `PlayerView` — their own hole
cards plus public information — so they cannot read hidden cards.

### Bot ladder (planned)

| Level | Bot | Core idea |
|-------|-----|-----------|
| 1 | Rule-based (Tight / Balanced / Loose-Aggressive) | hand-strength tiers, position, price to call |
| 2 | Monte Carlo equity | simulate run-outs, compare EV of fold / call / raise |
| 3 | Exploitative | level 2 + Bayesian opponent stats and range narrowing |
| 4 | GTO blueprint | MCCFR on abstracted heads-up NLHE (equity buckets; fold/call/½pot/pot/all-in) |
| 5 | Neural | PPO vs a league of past selves and the fixed bots, masked actions |

## The engine (Milestone 1)

* **Cards** are ints `0..51` (`rank*4 + suit`) for speed; `parse_cards("As Kd")` / `card_str` at the edges.
* **Evaluator** (`engine/hand_evaluator.py`): table-driven, 5–7 cards → one comparable int.
  ~500k 7-card evaluations/sec in pure Python. Verified against an independent
  brute-force evaluator and the exact 2,598,960-hand category distribution.
* **Rules** implemented: heads-up and multi-way blinds/positions (button = SB heads-up),
  antes, BB option, min-raise tracking, short all-ins, **incomplete all-in raises do
  not reopen betting** (but cumulative incomplete raises totalling a full raise do),
  no raising when nobody can respond, uncalled-bet refunds, automatic run-outs,
  side pots with dead money from folded players, split pots, odd chips to the first
  winner left of the button, eliminations and button movement (`Table`).
* **Action convention:** `BET`/`RAISE` amounts are *raise-to* totals for the street.
  All-in is `legal.all_in()`.

Deliberate simplifications: no burn cards (irrelevant to fairness with a uniformly
shuffled deck); a short-stacked big blind still forces others to call the full big
blind; no dead-button rule (the button moves to the next seat with chips).

### Quick example

```python
import random
from poker_ai.engine import HandState, Action

h = HandState(stacks=[1000, 1000], button=0, small_blind=5, big_blind=10, rng=random.Random(1))
la = h.legal_actions()          # seat 0 (button/SB) to act
h.apply(Action.raise_to(30))
h.apply(Action.call())
print(h)                        # HandState(street=flop, ...)
view = h.view(1)                # what seat 1 is allowed to see
```

## Play online (no install)

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/THE-LEGEND-SKAG/poker_bot)

One click builds the `Dockerfile` (React build + FastAPI server) on Render's free
tier using `render.yaml`. Free instances sleep when idle; the first visit after a
while takes ~30-60 s to wake up.

## Play (Milestone 2)

```bash
pip install -e ".[dev,server]"
```

```bash
cd web && npm install && npm run build && cd ..
```

```bash
python -m poker_ai          # open http://127.0.0.1:8000
```

The lobby lets you choose heads-up, 3-handed or 6-max (up to 8 bots), pick each
opponent's type, set the blinds, starting stack and bot speed. At the table you get
fold / check / call, a bet slider with 3 BB / ½ / ⅔ / pot presets and all-in, and
keyboard shortcuts (`F`, `C`, `R`). The hand history appears in a side panel.
Showdowns reveal the hands that went to showdown, label them, and mark each pot's
winner. Busted bots rebuy automatically; when you bust you choose whether to rebuy.

For frontend development with hot reload, run the server and also run
`cd web && npm run dev` (Vite proxies `/api` and `/ws` to port 8000).

**Level 1 rule-based bot** (`agents/rule_based.py`): preflop hands are graded with
Chen-formula tiers, adjusted for position, number of players and the price to call,
with push/fold below 12 BB. Postflop it enumerates every opponent holding to get its
current hand strength, counts outs for draws, and compares the result with pot odds.
There are three personalities (Tight / Balanced / Loose-Aggressive), with different
thresholds, aggression, bluff frequency and bet size. Every decision records the rule
that fired in `last_explanation`, which the Coach panel will use.

## Install & test

```bash
pip install -e ".[dev]"
```

```bash
pytest -m "not slow"     # fast suite (~5 s)
```

```bash
pytest                   # includes the exhaustive 2.6M-hand evaluator check (~10 s total)
```

Tests include a randomised stress test: thousands of random hands with 2–9
players, tiny and deep stacks, antes, and deliberately illegal actions; every hand
must conserve chips and its payouts must match an independent chip-by-chip pot
distribution.

## Scope and fairness

Local play and simulation only. This project does not and will not connect to
real-money sites or automate accounts on any platform.

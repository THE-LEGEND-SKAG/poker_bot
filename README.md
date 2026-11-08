# Poker AI

A playable No-Limit Texas Hold'em platform: a rules-exact engine, a ladder of
AI opponents (rule-based → Monte Carlo → exploitative → CFR blueprint → neural
PPO), a duplicate-match evaluation harness, and a web table with a coach panel.

> **Status:** Milestones 1–4 complete: the engine, a playable web table, rule-based,
> Monte Carlo EV and exploitative (opponent-modelling) bots, and a Coach panel with range heatmaps. Rows marked ☐ are **not built yet** and are not claimed to work.

![Three-way all-in with a side pot](docs/table-showdown.png)

## Progress

| # | Milestone | Status |
|---|-----------|--------|
| 1 | Engine: evaluator, betting, all-ins, side pots, randomised chip-conservation stress test | ✅ done |
| 2 | Rule-based bots + playable table UI (FastAPI + React vertical slice) | ✅ done, 94 tests total |
| 3 | Monte Carlo equity bot + Coach panel (equity, pot odds, EV per action) | ✅ done |
| 4 | Opponent modelling (Bayesian stats, range narrowing) + 13×13 range heatmap | ✅ done |
| 5 | CFR: verified on Kuhn (game value −1/18) and Leduc via exploitability, then MCCFR blueprint for abstracted heads-up NLHE | ☐ |
| 6 | Neural agent: PPO with action masking, equity feature, league training | ☐ |
| 7 | Duplicate tournaments, bb/100 with 95% CIs, Lab dashboard | ☐ |
| 8 | 3-player / 6-max polish, end-to-end checks, screenshots | ☐ |

## Architecture

```
poker_ai/
  engine/      cards, hand evaluator, actions + legality, game state, pots, table   ✅
  agents/      random, calling station, rule_based, equity/EV, exploitative ✅ · cfr, neural ☐
  solvers/     cfr (Kuhn/Leduc/HUNL), abstraction, exploitability                   ☐
  learning/    networks, ppo, league, checkpoints                                   ☐
  sim/         hand/session runner ✅ · duplicate matches, tournaments, CIs ☐
  server/      FastAPI + WebSocket game sessions                                    ✅
web/           React + Tailwind + Framer Motion: Table, Coach, range heatmap ✅ · Lab ☐
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

## Level 2: Monte Carlo EV bot and the Coach (Milestone 3)

**Equity** (`agents/equity.py`): our expected share of the pot against each opponent's
holding, drawn at random or from a range. Ties are split. On the river against one
opponent it enumerates every holding exactly; otherwise it samples (about 1,000
samples in 10 ms) and reports a standard error. It reproduces published values:
AA 85.2%, KK 82.4%, AKo 65.4%, 72o 34.6% against one random hand. A precomputed
table (`data/preflop_equity.json`) holds the equity of all 169 starting hands
against 1–5 opponents; rebuild it with `python scripts/build_preflop_table.py`.

**EV model** (`agents/ev.py`): EV is the expected change in our stack from this
decision onward, so folding is always 0.

* call: `R·eq·(pot + call) − call`
* bet/raise to X: `FE·pot + (1−FE)·(R·eq_called·(final pot) − our chips)`

What the terms mean:

* `R` (equity realisation): 1.0 on the river or all-in, 0.95 in position, 0.85 out of position.
* `FE` (fold equity): the chance every opponent folds. Each opponent folds
  `(risk/(risk+reward))^1.5`. A game-theory-balanced defender folds exactly
  `risk/(risk+reward)`, which would make every pure bluff break even; real players
  call more than that, so here pure bluffs lose money.
* `eq_called`: equity against only the hands that continue (the top `1−f` of the caller's range).
* A bettor's range narrows with the size of the bet.

Sizes considered are 2.5/3.5 BB or 3×/4× preflop, and ½ / ¾ / pot postflop. All-in
is considered only when the stack is at most 4× the pot or 15 BB preflop, because a
one-street model can't see the value a smaller bet wins on later streets.

The **Level 2 bot** plays the highest-EV action, choosing at random among options
within 2% of the pot of the best one.

**Coach panel** (toggle in the sidebar): on your turn it runs the same model on
*your* hand. It shows your equity (with the pot odds you need marked on the bar),
your made hand, and the EV of every action, with fold equity and equity-if-called
for each bet size. After each hand it reveals the bots' reasoning. Numbers that
depend on a bot's hidden cards are shown only if those cards were revealed at
showdown, so the coach never leaks information.

**Results** (heads-up duplicate matches, 100 BB stacks, 20,000 hands each, `scripts/quick_match.py`):

| Level 2 vs | Win rate (95% CI) |
|------------|-------------------|
| Rule-based · Tight | +53.1 ± 16.7 bb/100 |
| Rule-based · Balanced | +67.2 ± 23.7 bb/100 |
| Rule-based · Loose-Aggressive | +96.3 ± 32.3 bb/100 |
| Calling Station (2,000 hands) | +1,432 ± 129 bb/100 |

```bash
python scripts/quick_match.py equity rule_balanced 10000
```

**Known limitations:** EV covers one street (future streets enter only through `R`).
Postflop ranges are ranked by made-hand strength, so draws in an opponent's range
are under-weighted. Without an opponent model, fold rates are population defaults;
Level 3 replaces them with observed statistics.

## Level 3: exploitative bot with opponent modelling (Milestone 4)

**Tendencies** (`agents/opponent_model.py`): for every opponent the model tracks
VPIP (counting only hands where they faced a price; a free big-blind check is not a decision to play), PFR, folding to a preflop raise, betting when checked to, folding to a bet,
raising against a bet, and going to showdown. Each is a Beta distribution that
starts from a population prior worth a few hands and is updated by Bayes' rule
after every hand. Early estimates stay close to typical play and sharpen with
evidence. The model reads only public actions and cards shown at showdown.

**Ranges**: each opponent starts on all 1,326 holdings (minus cards we can see).
Every action in the hand multiplies each holding's weight by the likelihood of
that action with that holding:

`P(hand | actions) ∝ P(hand) · Π P(action | hand, tendencies)`

The likelihoods are smooth functions of hand strength, with thresholds set by
that opponent's stats:

* **Preflop:** strength is the hand's percentile. A 40%-PFR player raises far
  wider than an 8% one, and calls exclude most hands they would have raised.
* **Postflop:** strength is the hand's current percentile on the board, with
  extra credit for flush and straight draws. Bets and raises shift weight toward
  strong hands and draws; checks shift it toward weaker ones.

Learned profiles after 150 hands of observation:

| Bot | VPIP | PFR | Bets when checked to | Folds to bet |
|-----|------|-----|----------------------|--------------|
| Calling Station | 98% | 0.5% | 0.5% | 1% |
| Rule-based · Tight | 35% | 7% | 31% | 30% |
| Rule-based · Loose-Aggressive | 74% | 31% | 54% | 25% |

**The Level 3 bot** (`agents/exploitative.py`) uses the Level 2 EV engine with two
inputs replaced: equity is computed against the modelled ranges, and fold
probabilities shift, on the log-odds scale, by how much more or less often that
opponent folds than the prior. Bluffs therefore come from evidence: a weak hand
bets only when that opponent's fold rate and range make betting the highest-EV
option. It value-bets calling stations and bluffs players who over-fold.

**Coach**: now runs this Level 3 analysis using a model built only from *your*
observations. A 13×13 heatmap shows each live opponent's estimated range (pairs on
the diagonal, suited above, offsuit below) along with their learned stats.

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

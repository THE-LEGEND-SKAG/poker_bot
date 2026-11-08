import { motion } from "framer-motion";
import type { CoachAnalysis, GameState, ReviewItem } from "../types";
import RangeHeatmap from "./RangeHeatmap";

const pct = (x: number, digits = 0) => `${(x * 100).toFixed(digits)}%`;

function EquityBar({ a }: { a: CoachAnalysis }) {
  const win = a.equity.win;
  const tieShare = a.equity.equity - win;
  return (
    <div>
      <div className="mb-1 flex items-baseline justify-between">
        <span className="text-xs uppercase tracking-wider text-slate-400">Your equity</span>
        <span className="text-2xl font-black tabular-nums text-white">{pct(a.equity.equity)}</span>
      </div>
      <div className="relative h-4 overflow-hidden rounded-full bg-slate-800">
        <motion.div className="absolute inset-y-0 left-0 bg-emerald-500" initial={{ width: 0 }} animate={{ width: pct(win, 1) }} />
        <motion.div
          className="absolute inset-y-0 bg-amber-400"
          initial={{ width: 0 }}
          animate={{ left: pct(win, 1), width: pct(tieShare, 1) }}
        />
        {a.to_call > 0 && (
          <div className="absolute inset-y-[-2px] w-0.5 bg-white" style={{ left: pct(a.pot_odds, 1) }} title="Equity needed to call" />
        )}
      </div>
      <div className="mt-1 flex justify-between text-[11px] text-slate-400">
        <span>
          vs {a.opponents} opponent{a.opponents === 1 ? "" : "s"} · {a.range_note}
        </span>
        <span>{a.equity.exact ? "exact" : `±${pct(1.96 * a.equity.stderr, 1)}`}</span>
      </div>
    </div>
  );
}

function Options({ a }: { a: CoachAnalysis }) {
  const max = Math.max(1, ...a.options.map((o) => Math.abs(o.ev)));
  return (
    <div className="space-y-1.5">
      <div className="text-xs uppercase tracking-wider text-slate-400">Expected value per action</div>
      {a.options.map((o) => {
        const best = o.label === a.best.label;
        const w = (Math.abs(o.ev) / max) * 50;
        return (
          <div key={o.label} className={`rounded-lg px-2 py-1.5 ${best ? "bg-emerald-500/15 ring-1 ring-emerald-500/60" : "bg-slate-900"}`}>
            <div className="flex items-center justify-between text-xs">
              <span className={`font-semibold ${best ? "text-emerald-300" : "text-slate-200"}`}>
                {o.label}
                {best && <span className="ml-1.5 rounded bg-emerald-500 px-1 text-[10px] text-slate-950">BEST</span>}
              </span>
              <span className={`tabular-nums font-bold ${o.ev >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                {o.ev >= 0 ? "+" : ""}
                {Math.round(o.ev).toLocaleString()}
              </span>
            </div>
            <div className="relative mt-1 h-1.5 rounded-full bg-slate-800">
              <div className="absolute inset-y-0 left-1/2 w-px bg-slate-600" />
              <motion.div
                className={`absolute inset-y-0 rounded-full ${o.ev >= 0 ? "bg-emerald-500" : "bg-red-500"}`}
                initial={{ width: 0 }}
                animate={o.ev >= 0 ? { left: "50%", width: `${w}%` } : { left: `${50 - w}%`, width: `${w}%` }}
              />
            </div>
            {o.equity_when_called !== null && (
              <div className="mt-0.5 text-[10px] text-slate-500">
                folds {pct(o.fold_equity)} · equity if called {pct(o.equity_when_called)}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

function Review({ review }: { review: ReviewItem[] }) {
  if (review.length === 0) return <div className="text-sm text-slate-500">No bot decisions this hand.</div>;
  return (
    <div className="space-y-1.5">
      <div className="text-xs uppercase tracking-wider text-slate-400">Bot reasoning (revealed after the hand)</div>
      {review.map((r, i) => {
        const d = r.details as Record<string, any> | undefined;
        const eq = d?.equity?.equity ?? d?.hand_strength;
        return (
          <div key={i} className="rounded-lg bg-slate-900 px-2 py-1.5 text-xs">
            <div className="flex justify-between">
              <span className="font-semibold text-slate-200">
                {r.name} · <span className="text-slate-400">{r.street}</span>
              </span>
              <span className="text-sky-300">{r.action}</span>
            </div>
            <div className="text-slate-400">{r.rule ?? "—"}</div>
            {r.revealed && eq !== undefined && (
              <div className="text-slate-500">
                {d?.equity ? "equity" : "hand strength"} {pct(Number(eq))}
                {d?.pot_odds ? ` · pot odds ${pct(Number(d.pot_odds))}` : ""}
              </div>
            )}
            {!r.revealed && <div className="text-[10px] italic text-slate-600">cards not shown: numbers hidden</div>}
          </div>
        );
      })}
    </div>
  );
}

type Props = { state: GameState; analysis: CoachAnalysis | null; enabled: boolean; onToggle: (on: boolean) => void };

export default function CoachPanel({ state, analysis, enabled, onToggle }: Props) {
  const myTurn = state.to_act === state.human_seat;
  return (
    <div className="flex h-full min-h-0 flex-col rounded-2xl border border-white/10 bg-slate-950/70">
      <div className="flex items-center justify-between border-b border-white/10 px-3 py-2">
        <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">Coach · Level 3 model</span>
        <label className="flex items-center gap-1.5 text-xs text-slate-400">
          <input type="checkbox" checked={enabled} onChange={(e) => onToggle(e.target.checked)} className="accent-emerald-500" />
          on
        </label>
      </div>
      <div className="min-h-0 flex-1 space-y-4 overflow-y-auto p-3">
        {state.terminal && state.review ? (
          <Review review={state.review} />
        ) : !enabled ? (
          <div className="text-sm text-slate-500">Coach is off.</div>
        ) : !myTurn ? (
          <div className="text-sm text-slate-500">The coach analyses your decisions when it is your turn.</div>
        ) : !analysis ? (
          <div className="flex items-center gap-2 text-sm text-slate-400">
            <motion.div className="h-3 w-3 rounded-full bg-emerald-400" animate={{ opacity: [0.2, 1, 0.2] }} transition={{ duration: 1, repeat: Infinity }} />
            Simulating…
          </div>
        ) : (
          <>
            <div className="flex items-center justify-between text-sm">
              <span className="font-bold text-white">{analysis.made_hand ?? analysis.hand_class}</span>
              <span className="text-xs text-slate-400">
                {analysis.hand_class} · {pct(analysis.preflop_equity_vs_random)} preflop
              </span>
            </div>
            <EquityBar a={analysis} />
            <div className="grid grid-cols-2 gap-2 text-center">
              <div className="rounded-lg bg-slate-900 p-2">
                <div className="text-[10px] uppercase text-slate-400">Pot odds (need)</div>
                <div className="text-lg font-bold tabular-nums">{analysis.to_call > 0 ? pct(analysis.pot_odds, 1) : "—"}</div>
              </div>
              <div className="rounded-lg bg-slate-900 p-2" title="Effective stack divided by the pot: low SPR = committed, high SPR = room to maneuver">
                <div className="text-[10px] uppercase text-slate-400">Stack-to-pot (SPR)</div>
                <div className="text-lg font-bold tabular-nums">{analysis.spr >= 100 ? "100+" : analysis.spr.toFixed(1)}</div>
                <div className="text-[10px] text-slate-500">{analysis.effective_stack_bb} BB effective</div>
              </div>
            </div>
            <Options a={analysis} />
            <RangeHeatmap ranges={analysis.opponent_ranges} labels={analysis.stat_labels} />
            <p className="text-[10px] leading-snug text-slate-500">
              EV is chips gained from this decision onward (fold = 0). Ranges and fold rates come from each opponent's observed
              actions (Bayesian updates from population priors). Before the river the EV assumes you keep most, not all, of
              your equity (an assumption, scaled by how aggressive opponents are). A model, not a solver.
            </p>
          </>
        )}
      </div>
    </div>
  );
}

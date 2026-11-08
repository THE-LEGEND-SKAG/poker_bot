import { useState } from "react";
import type { OpponentRange } from "../types";

const RANKS = "AKQJT98765432";

function label(i: number, j: number) {
  if (i === j) return RANKS[i] + RANKS[j];
  return i < j ? RANKS[i] + RANKS[j] + "s" : RANKS[j] + RANKS[i] + "o";
}

const STAT_ORDER: [string, string][] = [
  ["vpip", "VPIP"],
  ["pfr", "PFR"],
  ["agg", "Bets"],
  ["fold_to_bet", "Folds to bet"],
];

export default function RangeHeatmap({ ranges, labels }: { ranges: OpponentRange[]; labels: Record<string, string> }) {
  const [sel, setSel] = useState(0);
  if (ranges.length === 0) return null;
  const r = ranges[Math.min(sel, ranges.length - 1)];
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-xs uppercase tracking-wider text-slate-400">Estimated range</span>
        <div className="flex gap-1">
          {ranges.map((o, k) => (
            <button
              key={o.seat}
              onClick={() => setSel(k)}
              className={`rounded px-1.5 py-0.5 text-[11px] font-semibold ${k === sel ? "bg-fuchsia-500 text-white" : "bg-slate-800 text-slate-300 hover:bg-slate-700"}`}
            >
              {o.name}
            </button>
          ))}
        </div>
      </div>
      <div className="grid grid-cols-13 gap-px overflow-hidden rounded-md bg-slate-950" style={{ gridTemplateColumns: "repeat(13, minmax(0, 1fr))" }}>
        {Array.from({ length: 169 }).map((_, idx) => {
          const i = Math.floor(idx / 13);
          const j = idx % 13;
          const w = r.grid[idx] ?? 0;
          return (
            <div
              key={idx}
              title={`${label(i, j)}: ${Math.round(w * 100)}%`}
              className="flex aspect-square items-center justify-center text-[7px] font-semibold leading-none"
              style={{
                background: `rgba(217, 70, 239, ${0.06 + 0.94 * w})`,
                color: w > 0.45 ? "#fff" : "rgba(226,232,240,0.55)",
              }}
            >
              {label(i, j)}
            </div>
          );
        })}
      </div>
      <div className="grid grid-cols-4 gap-1 text-center">
        {STAT_ORDER.map(([k, short]) => (
          <div key={k} className="rounded bg-slate-900 px-1 py-1" title={`${labels[k] ?? short} · ${r.stats.samples?.[k] ?? 0} observations`}>
            <div className="text-[9px] uppercase text-slate-500">{short}</div>
            <div className="text-xs font-bold tabular-nums text-slate-100">{Math.round((r.stats[k] ?? 0) * 100)}%</div>
          </div>
        ))}
      </div>
      <div className="text-[10px] text-slate-500">
        {r.stats.hands} hands observed · brighter = more likely. Starts from population priors and sharpens with every hand.
      </div>
    </div>
  );
}

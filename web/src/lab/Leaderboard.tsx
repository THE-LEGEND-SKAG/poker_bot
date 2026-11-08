import { useState } from "react";
import { INK, agentColor } from "./palette";
import type { LeaderRow } from "./types";

const pct = (x: number) => `${Math.round(x * 100)}%`;

export default function Leaderboard({ rows, status }: { rows: LeaderRow[]; status: Record<string, string> }) {
  const [hover, setHover] = useState<string | null>(null);
  const lo = Math.min(0, ...rows.map((r) => r.bb100 - r.ci95));
  const hi = Math.max(0, ...rows.map((r) => r.bb100 + r.ci95));
  const W = 180;
  const sx = (v: number) => 8 + ((v - lo) / (hi - lo || 1)) * (W - 16);
  return (
    <div className="overflow-x-auto rounded-2xl border border-white/10 bg-slate-900/70">
      <table className="w-full min-w-[900px] text-sm">
        <thead className="text-left text-[11px] uppercase tracking-wider text-slate-400">
          <tr className="border-b border-white/10">
            <th className="px-3 py-2">#</th>
            <th className="px-3 py-2">Bot</th>
            <th className="px-3 py-2 text-right">bb/100</th>
            <th className="px-3 py-2" style={{ width: W }}>
              95% CI <span className="normal-case tracking-normal text-slate-500">(line = 0)</span>
            </th>
            <th className="px-2 py-2 text-right" title="Voluntarily put money in when facing a price">VPIP</th>
            <th className="px-2 py-2 text-right" title="Preflop raise">PFR</th>
            <th className="px-2 py-2 text-right" title="Postflop (bets + raises) / calls">AF</th>
            <th className="px-2 py-2 text-right" title="Folds per decision">Fold</th>
            <th className="px-2 py-2 text-right" title="Went to showdown after seeing the flop">WTSD</th>
            <th className="px-2 py-2 text-right" title="Won at showdown">W$SD</th>
            <th className="px-2 py-2 text-right" title="Average final pot">Pot (BB)</th>
            <th className="px-3 py-2 text-right" title="Average time per decision">ms/dec</th>
          </tr>
        </thead>
        <tbody className="tabular-nums">
          {rows.map((r, i) => {
            const sig = r.bb100 - r.ci95 > 0 || r.bb100 + r.ci95 < 0;
            return (
              <tr key={r.kind} className="border-b border-white/5 hover:bg-white/5" onMouseEnter={() => setHover(r.kind)} onMouseLeave={() => setHover(null)}>
                <td className="px-3 py-2 text-slate-500">{i + 1}</td>
                <td className="px-3 py-2">
                  <div className="flex items-center gap-2">
                    <span className="h-2.5 w-2.5 shrink-0 rounded-full" style={{ background: agentColor(r.kind) }} />
                    <span className="font-semibold text-slate-100">{r.label}</span>
                    <span className="rounded bg-slate-800 px-1 text-[10px] text-slate-400">L{r.level}</span>
                  </div>
                  {status[r.kind] && status[r.kind] !== "n/a" && <div className="pl-[18px] text-[10px] text-slate-500">{status[r.kind]}</div>}
                </td>
                <td className={`px-3 py-2 text-right font-bold ${sig ? "text-slate-100" : "text-slate-400"}`}>
                  {r.bb100 > 0 ? "+" : ""}
                  {r.bb100.toFixed(0)} <span className="text-[11px] font-normal text-slate-500">±{r.ci95.toFixed(0)}</span>
                </td>
                <td className="px-3 py-1">
                  <svg width={W} height={18} aria-label={`${r.label}: ${r.bb100} plus or minus ${r.ci95}`}>
                    <line x1={sx(0)} x2={sx(0)} y1={1} y2={17} stroke={INK.secondary} strokeOpacity={0.5} />
                    <line x1={sx(r.bb100 - r.ci95)} x2={sx(r.bb100 + r.ci95)} y1={9} y2={9} stroke={agentColor(r.kind)} strokeWidth={2} strokeLinecap="round" />
                    <circle cx={sx(r.bb100)} cy={9} r={hover === r.kind ? 5 : 4} fill={agentColor(r.kind)} stroke="#0f172a" strokeWidth={2} />
                  </svg>
                </td>
                <td className="px-2 py-2 text-right text-slate-300">{pct(r.vpip)}</td>
                <td className="px-2 py-2 text-right text-slate-300">{pct(r.pfr)}</td>
                <td className="px-2 py-2 text-right text-slate-300">{r.aggression_factor.toFixed(1)}</td>
                <td className="px-2 py-2 text-right text-slate-300">{pct(r.fold_frequency)}</td>
                <td className="px-2 py-2 text-right text-slate-300">{pct(r.went_to_showdown)}</td>
                <td className="px-2 py-2 text-right text-slate-300">{pct(r.showdown_win_rate)}</td>
                <td className="px-2 py-2 text-right text-slate-300">{r.avg_pot_bb.toFixed(1)}</td>
                <td className="px-3 py-2 text-right text-slate-300">{r.avg_decision_ms.toFixed(1)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="px-3 py-2 text-[11px] text-slate-500">
        bb/100 = average of each bot's head-to-head results; bold = 95% interval excludes zero. Stats pooled over all its matches.
      </p>
    </div>
  );
}

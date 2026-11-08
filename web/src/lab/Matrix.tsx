import { useState } from "react";
import { agentColor, diverging } from "./palette";
import type { Match } from "./types";

type Props = { agents: string[]; labels: Record<string, string>; matches: Match[] };

export default function Matrix({ agents, labels, matches }: Props) {
  const [tip, setTip] = useState<{ text: string; x: number; y: number } | null>(null);
  const cell = (row: string, col: string) => {
    const m = matches.find((x) => (x.a === row && x.b === col) || (x.a === col && x.b === row));
    if (!m) return null;
    return m.a === row ? { bb: m.bb100, ci: m.ci95, hands: m.hands } : { bb: -m.bb100, ci: m.ci95, hands: m.hands };
  };
  const maxAbs = Math.max(50, ...matches.map((m) => Math.abs(m.bb100)));
  const short = (k: string) => (labels[k] ?? k).replace("Rule-based · ", "").replace(" (opponent model)", "").replace(" (heads-up)", "");
  return (
    <div className="relative overflow-x-auto rounded-2xl border border-white/10 bg-slate-900/70 p-3">
      <div className="mb-2 text-sm font-semibold text-slate-100">Head-to-head (row's win rate vs column, bb/100)</div>
      <table className="border-separate text-xs" style={{ borderSpacing: 2 }}>
        <thead>
          <tr>
            <th />
            {agents.map((c) => (
              <th key={c} className="px-1 pb-1 align-bottom font-medium text-slate-400">
                <div className="flex flex-col items-center gap-1">
                  <span className="h-2 w-2 rounded-full" style={{ background: agentColor(c) }} />
                  <span className="max-w-[72px] leading-tight">{short(c)}</span>
                </div>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {agents.map((r) => (
            <tr key={r}>
              <th className="pr-2 text-right font-medium whitespace-nowrap text-slate-300">
                <span className="mr-1.5 inline-block h-2 w-2 rounded-full" style={{ background: agentColor(r) }} />
                {short(r)}
              </th>
              {agents.map((c) => {
                const v = r === c ? null : cell(r, c);
                const sig = v && Math.abs(v.bb) > v.ci;
                return (
                  <td
                    key={c}
                    className="h-10 w-[72px] rounded-md text-center tabular-nums"
                    style={{ background: v ? diverging(v.bb, maxAbs) : "transparent" }}
                    onMouseMove={(e) =>
                      v && setTip({ text: `${short(r)} vs ${short(c)}: ${v.bb > 0 ? "+" : ""}${v.bb.toFixed(1)} ± ${v.ci.toFixed(1)} bb/100 · ${v.hands} hands${sig ? "" : " · not significant"}`, x: e.clientX, y: e.clientY })
                    }
                    onMouseLeave={() => setTip(null)}
                  >
                    {v ? (
                      <span className={sig ? "font-bold text-white" : "text-white/70 italic"}>
                        {v.bb > 0 ? "+" : ""}
                        {Math.round(v.bb)}
                      </span>
                    ) : (
                      <span className="text-slate-700">—</span>
                    )}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <div className="mt-2 flex items-center gap-2 text-[11px] text-slate-400">
        <span>loses</span>
        <div className="h-2 w-40 rounded" style={{ background: `linear-gradient(90deg, ${diverging(-1, 1)}, ${diverging(0, 1)}, ${diverging(1, 1)})` }} />
        <span>wins</span>
        <span className="ml-3">bold = significant (95%); italic = within noise</span>
      </div>
      {tip && (
        <div className="pointer-events-none fixed z-50 rounded-lg border border-white/10 bg-slate-950/95 px-2 py-1 text-[11px] text-slate-100 shadow-xl" style={{ left: tip.x + 12, top: tip.y + 12 }}>
          {tip.text}
        </div>
      )}
    </div>
  );
}

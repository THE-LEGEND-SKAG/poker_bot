import { useEffect, useMemo, useState } from "react";
import PlayingCard from "../components/PlayingCard";
import { agentColor } from "./palette";
import type { HandRecord } from "./types";

const STREETS = ["preflop", "flop", "turn", "river"];

function describe(e: Record<string, any>, names: Record<string, string>): string {
  const n = (s: number) => names[String(s)] ?? `Seat ${s}`;
  switch (e.type) {
    case "post": return `${n(e.seat)} posts ${e.kind.replace("_", " ")} ${e.amount}`;
    case "action": return `${n(e.seat)} ${e.action}${e.action === "bet" || e.action === "raise" ? ` to ${e.amount}` : e.action === "call" ? ` ${e.added}` : ""}${e.all_in ? " (all-in)" : ""}`;
    case "street": return `— ${String(e.street).toUpperCase()} —`;
    case "refund": return `${e.amount} uncalled returned to ${n(e.seat)}`;
    case "showdown": return "Showdown: " + Object.entries(e.hands).map(([s, h]: [string, any]) => `${n(+s)} ${h.description}`).join(" vs ");
    case "award": return `${(e.winners as number[]).map(n).join(" & ")} win${e.winners.length > 1 ? "" : "s"} ${e.amount}`;
    default: return "";
  }
}

export default function Replayer({ hands, labels }: { hands: HandRecord[]; labels: Record<string, string> }) {
  const [idx, setIdx] = useState(0);
  const hand = hands[idx];
  const steps = useMemo(() => (hand ? hand.events.filter((e) => describe(e, hand.names)) : []), [hand]);
  const [step, setStep] = useState(0);
  useEffect(() => setStep(steps.length - 1), [steps]);
  if (!hand) return <div className="text-sm text-slate-500">No recorded hands.</div>;
  const shown = steps.slice(0, step + 1);
  const streetsDealt = shown.filter((e) => e.type === "street").length;
  const boardCount = [0, 3, 4, 5][streetsDealt] ?? 5;
  const lastPot = [...shown].reverse().find((e) => e.pot !== undefined)?.pot ?? 150;
  const label = (k: string) => labels[k] ?? k;
  return (
    <div className="rounded-2xl border border-white/10 bg-slate-900/70 p-3">
      <div className="mb-2 flex flex-wrap items-center gap-2">
        <span className="text-sm font-semibold text-slate-100">Hand replayer</span>
        <select value={idx} onChange={(e) => setIdx(+e.target.value)} className="rounded-md bg-slate-800 px-2 py-1 text-xs">
          {hands.map((h, i) => (
            <option key={i} value={i}>
              #{i + 1}: {label(h.names["0"])} vs {label(h.names["1"])} ({h.net[0] > 0 ? "+" : ""}{h.net[0] / 100} BB)
            </option>
          ))}
        </select>
      </div>
      <div className="grid gap-3 md:grid-cols-[1fr_1.2fr]">
        <div className="flex flex-col items-center justify-center gap-3 rounded-xl bg-emerald-900/60 p-3">
          {["1", "0"].map((s, k) => (
            <div key={s} className={`flex items-center gap-2 ${k === 0 ? "order-first" : "order-last"}`}>
              <span className="h-2.5 w-2.5 rounded-full" style={{ background: agentColor(hand.names[s]) }} />
              <span className="text-xs font-semibold text-slate-100">{label(hand.names[s])}</span>
              {Number(s) === hand.button && <span className="rounded-full bg-white px-1 text-[9px] font-black text-slate-900">D</span>}
              <div className="flex gap-1">{hand.holes[s].map((c) => <PlayingCard key={c} card={c} size="sm" />)}</div>
              <span className={`text-xs tabular-nums ${hand.net[+s] >= 0 ? "text-emerald-300" : "text-red-300"}`}>
                {hand.net[+s] > 0 ? "+" : ""}{hand.net[+s]}
              </span>
            </div>
          ))}
          <div className="flex h-14 items-center gap-1">
            {hand.board.slice(0, boardCount).map((c) => <PlayingCard key={c} card={c} size="sm" />)}
          </div>
          <div className="rounded-full bg-black/40 px-2 py-0.5 text-xs text-emerald-200">Pot {lastPot}</div>
        </div>
        <div className="flex min-h-0 flex-col gap-2">
          <input type="range" min={0} max={steps.length - 1} value={step} onChange={(e) => setStep(+e.target.value)} className="accent-emerald-500" aria-label="Replay step" />
          <div className="flex gap-1 text-xs">
            <button onClick={() => setStep(0)} className="rounded bg-slate-800 px-2 py-1">⏮</button>
            <button onClick={() => setStep(Math.max(0, step - 1))} className="rounded bg-slate-800 px-2 py-1">◀ Step</button>
            <button onClick={() => setStep(Math.min(steps.length - 1, step + 1))} className="rounded bg-slate-800 px-2 py-1">Step ▶</button>
            <button onClick={() => setStep(steps.length - 1)} className="rounded bg-slate-800 px-2 py-1">⏭</button>
            <span className="ml-auto self-center text-slate-500">{step + 1}/{steps.length}</span>
          </div>
          <ol className="max-h-56 space-y-0.5 overflow-y-auto text-xs">
            {steps.map((e, i) => (
              <li key={i} className={`${i === step ? "text-amber-300 font-semibold" : i > step ? "text-slate-600" : STREETS.includes(e.street) && e.type === "street" ? "text-sky-300" : "text-slate-300"}`}>
                {describe(e, Object.fromEntries(Object.entries(hand.names).map(([k, v]) => [k, label(v)])))}
              </li>
            ))}
          </ol>
        </div>
      </div>
    </div>
  );
}

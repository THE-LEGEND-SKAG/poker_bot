import { useEffect, useMemo, useState } from "react";
import type { GameState } from "../types";

type Props = { state: GameState; onAct: (action: string, amount?: number) => void };

export default function ActionBar({ state, onAct }: Props) {
  const legal = state.legal;
  const [amount, setAmount] = useState(0);

  // pot-fraction "raise to" sizes, same convention as the bots: call, then raise by f × pot
  const presets = useMemo(() => {
    if (!legal?.raise_type) return [];
    const potAfterCall = state.pot + legal.to_call;
    const to = (f: number) => Math.round(state.current_bet + f * potAfterCall);
    const items: [string, number][] = [
      ["½ pot", to(0.5)],
      ["⅔ pot", to(2 / 3)],
      ["Pot", to(1)],
    ];
    if (state.street === "preflop") items.unshift(["3 BB", 3 * state.big_blind]);
    return items
      .map(([label, v]) => [label, Math.max(legal.min_raise_to, Math.min(v, legal.max_raise_to))] as [string, number])
      .filter(([, v], i, arr) => arr.findIndex(([, w]) => w === v) === i);
  }, [legal, state.pot, state.current_bet, state.street, state.big_blind]);

  useEffect(() => {
    if (legal?.raise_type) setAmount(legal.min_raise_to);
  }, [legal?.min_raise_to, legal?.raise_type, state.hand_id, state.street]);

  useEffect(() => {
    if (!legal) return;
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.tagName === "INPUT") return;
      if (e.key === "f" && legal.can_fold) onAct("fold");
      if (e.key === "c") onAct(legal.can_check ? "check" : "call");
      if (e.key === "r" && legal.raise_type) onAct(legal.raise_type, amount);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [legal, amount, onAct]);

  if (!legal) return null;
  const allIn = amount >= legal.max_raise_to;
  const verb = legal.raise_type === "bet" ? "Bet" : "Raise to";

  return (
    <div className="mx-auto grid w-full max-w-4xl grid-cols-2 gap-2 rounded-2xl border border-white/10 bg-slate-950/90 p-2 shadow-2xl backdrop-blur md:flex md:items-center md:gap-3 md:p-3">
      <div className="flex gap-2 [&>button]:flex-1 md:[&>button]:flex-none">
        {legal.can_fold && (
          <button onClick={() => onAct("fold")} className="btn bg-red-600 hover:bg-red-500" title="F">
            Fold
          </button>
        )}
        {legal.can_check ? (
          <button onClick={() => onAct("check")} className="btn bg-emerald-600 hover:bg-emerald-500" title="C">
            Check
          </button>
        ) : (
          <button onClick={() => onAct("call")} className="btn bg-emerald-600 hover:bg-emerald-500" title="C">
            Call {legal.call_amount.toLocaleString()}
            {legal.call_amount >= legal.stack && <span className="ml-1 text-xs opacity-80">(all-in)</span>}
          </button>
        )}
      </div>
      {legal.raise_type && (
        <div className="order-first col-span-2 flex flex-1 flex-col gap-1.5 md:order-none md:gap-2">
          <div className="flex flex-wrap gap-1.5">
            {presets.map(([label, v]) => (
              <button key={label} onClick={() => setAmount(v)} className="rounded-md bg-slate-800 px-2 py-1 text-xs font-semibold hover:bg-slate-700">
                {label}
              </button>
            ))}
            <button onClick={() => setAmount(legal.max_raise_to)} className="rounded-md bg-slate-800 px-2 py-1 text-xs font-semibold hover:bg-slate-700">
              Max
            </button>
          </div>
          <div className="flex items-center gap-2">
            <input
              type="range"
              className="bet-slider flex-1"
              min={legal.min_raise_to}
              max={legal.max_raise_to}
              step={Math.max(1, Math.floor(state.big_blind / 10))}
              value={amount}
              onChange={(e) => setAmount(Number(e.target.value))}
              aria-label="Bet size"
            />
            <input
              type="number"
              className="w-24 rounded-md border border-white/15 bg-white px-2 py-1 text-right font-semibold tabular-nums text-slate-900"
              value={amount}
              min={legal.min_raise_to}
              max={legal.max_raise_to}
              onChange={(e) => setAmount(Number(e.target.value))}
            />
          </div>
        </div>
      )}
      {legal.raise_type && (
        <div className="flex gap-2 [&>button]:flex-1 md:[&>button]:flex-none">
          <button onClick={() => onAct(legal.raise_type!, Math.max(legal.min_raise_to, Math.min(amount, legal.max_raise_to)))} className="btn bg-blue-600 hover:bg-blue-500" title="R">
            {allIn ? "All-in" : `${verb} ${amount.toLocaleString()}`}
          </button>
          {!allIn && (
            <button onClick={() => onAct(legal.raise_type!, legal.max_raise_to)} className="btn bg-amber-500 text-slate-950 hover:bg-amber-400">
              All-in
            </button>
          )}
        </div>
      )}
    </div>
  );
}

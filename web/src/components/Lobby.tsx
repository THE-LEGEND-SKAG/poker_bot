import { useEffect, useState } from "react";
import type { AgentInfo, GameConfig } from "../types";

type Props = { onStart: (cfg: GameConfig, delayMs: number) => void; connected: boolean };

const PRESETS: { label: string; bots: number }[] = [
  { label: "Heads-up", bots: 1 },
  { label: "3-handed", bots: 2 },
  { label: "6-max", bots: 5 },
];

export default function Lobby({ onStart, connected }: Props) {
  const [agents, setAgents] = useState<AgentInfo[]>([]);
  const [bots, setBots] = useState<string[]>(["rule_balanced"]);
  const [stackBB, setStackBB] = useState(100);
  const [bigBlind, setBigBlind] = useState(100);
  const [delay, setDelay] = useState(700);

  useEffect(() => {
    fetch("/api/agents").then((r) => r.json()).then(setAgents).catch(() => setAgents([]));
  }, []);

  const setCount = (n: number) => {
    const pool = ["rule_balanced", "rule_tight", "rule_lag"];
    setBots((prev) => Array.from({ length: n }, (_, i) => prev[i] ?? pool[i % pool.length]));
  };

  return (
    <div className="mx-auto flex max-w-2xl flex-col gap-6 rounded-3xl border border-white/10 bg-slate-950/80 p-6 shadow-2xl">
      <div>
        <h1 className="text-3xl font-black tracking-tight text-white">
          Poker <span className="text-emerald-400">AI</span>
        </h1>
        <p className="mt-1 text-sm text-slate-400">No-Limit Texas Hold'em against a ladder of AI opponents. Play money only.</p>
      </div>

      <section className="space-y-2">
        <div className="text-xs font-semibold uppercase tracking-wider text-slate-400">Table</div>
        <div className="flex flex-wrap gap-2">
          {PRESETS.map((p) => (
            <button
              key={p.label}
              onClick={() => setCount(p.bots)}
              className={`rounded-lg px-3 py-1.5 text-sm font-semibold ${bots.length === p.bots ? "bg-emerald-600 text-white" : "bg-slate-800 hover:bg-slate-700"}`}
            >
              {p.label}
            </button>
          ))}
          <label className="ml-auto flex items-center gap-2 text-sm text-slate-300">
            Opponents
            <input type="number" min={1} max={8} value={bots.length} onChange={(e) => setCount(Math.max(1, Math.min(8, Number(e.target.value))))}
              className="w-16 rounded-md bg-slate-800 px-2 py-1" />
          </label>
        </div>
      </section>

      <section className="space-y-2">
        <div className="text-xs font-semibold uppercase tracking-wider text-slate-400">Opponents</div>
        <div className="grid gap-2 sm:grid-cols-2">
          {bots.map((b, i) => (
            <label key={i} className="flex items-center gap-2 rounded-lg bg-slate-900 px-3 py-2 text-sm">
              <span className="w-14 text-slate-400">Seat {i + 1}</span>
              <select value={b} onChange={(e) => setBots(bots.map((x, j) => (j === i ? e.target.value : x)))}
                className="flex-1 rounded-md bg-slate-800 px-2 py-1">
                {agents.map((a) => (
                  <option key={a.kind} value={a.kind}>
                    L{a.level} · {a.label}
                  </option>
                ))}
              </select>
            </label>
          ))}
        </div>
      </section>

      <section className="grid gap-3 sm:grid-cols-3">
        <label className="text-sm text-slate-300">
          Big blind
          <select value={bigBlind} onChange={(e) => setBigBlind(Number(e.target.value))} className="mt-1 w-full rounded-md bg-slate-800 px-2 py-1.5">
            {[2, 10, 20, 50, 100, 200].map((v) => (
              <option key={v} value={v}>{v / 2} / {v}</option>
            ))}
          </select>
        </label>
        <label className="text-sm text-slate-300">
          Starting stack ({stackBB} BB)
          <input type="range" min={20} max={300} step={10} value={stackBB} onChange={(e) => setStackBB(Number(e.target.value))} className="mt-2 w-full accent-emerald-500" />
        </label>
        <label className="text-sm text-slate-300">
          Bot speed
          <select value={delay} onChange={(e) => setDelay(Number(e.target.value))} className="mt-1 w-full rounded-md bg-slate-800 px-2 py-1.5">
            <option value={1400}>Slow</option>
            <option value={700}>Normal</option>
            <option value={250}>Fast</option>
            <option value={0}>Instant</option>
          </select>
        </label>
      </section>

      <button
        disabled={!connected || agents.length === 0}
        onClick={() => onStart({ bots, starting_stack: stackBB * bigBlind, small_blind: bigBlind / 2, big_blind: bigBlind, ante: 0 }, delay)}
        className="rounded-xl bg-emerald-500 py-3 text-lg font-bold text-slate-950 transition hover:bg-emerald-400 disabled:opacity-40"
      >
        {connected ? "Sit down" : "Connecting to server…"}
      </button>
    </div>
  );
}

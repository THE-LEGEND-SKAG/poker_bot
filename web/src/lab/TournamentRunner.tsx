import { useEffect, useState } from "react";
import type { AgentInfo } from "../types";
import { agentColor } from "./palette";

type Status = { id: string; state: string; progress: { done: number; total: number }; elapsed: number; error?: string };

export default function TournamentRunner({ onFinished }: { onFinished: () => void }) {
  const [agents, setAgents] = useState<AgentInfo[]>([]);
  const [chosen, setChosen] = useState<string[]>(["rule_balanced", "equity", "exploit", "cfr"]);
  const [deals, setDeals] = useState(100);
  const [status, setStatus] = useState<Status | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/agents").then((r) => r.json()).then(setAgents);
  }, []);

  useEffect(() => {
    if (!status || status.state !== "running") return;
    const t = setInterval(async () => {
      const s: Status = await fetch(`/api/lab/tournament/${status.id}/status`).then((r) => r.json());
      setStatus(s);
      if (s.state === "finished") onFinished();
    }, 1500);
    return () => clearInterval(t);
  }, [status, onFinished]);

  const start = async () => {
    setError(null);
    const r = await fetch("/api/lab/tournament", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ agents: chosen, deals }) });
    const body = await r.json();
    if (!r.ok) setError(typeof body.detail === "string" ? body.detail : "Invalid tournament settings");
    else setStatus(body);
  };

  const running = status?.state === "running";
  const pairs = (chosen.length * (chosen.length - 1)) / 2;
  return (
    <div className="rounded-2xl border border-white/10 bg-slate-900/70 p-3">
      <div className="mb-2 text-sm font-semibold text-slate-100">Run a tournament</div>
      <div className="flex flex-wrap gap-1.5">
        {agents.map((a) => {
          const on = chosen.includes(a.kind);
          return (
            <button
              key={a.kind}
              disabled={running}
              onClick={() => setChosen(on ? chosen.filter((k) => k !== a.kind) : [...chosen, a.kind])}
              className={`flex items-center gap-1.5 rounded-lg px-2 py-1 text-xs font-semibold ${on ? "bg-slate-700 text-white ring-1 ring-white/30" : "bg-slate-800 text-slate-400"}`}
            >
              <span className="h-2 w-2 rounded-full" style={{ background: agentColor(a.kind) }} />L{a.level} {a.label}
            </button>
          );
        })}
      </div>
      <div className="mt-3 flex flex-wrap items-center gap-3 text-xs text-slate-300">
        <label className="flex items-center gap-2">
          Deals per match
          <input type="number" min={10} max={500} value={deals} onChange={(e) => setDeals(+e.target.value)} className="w-20 rounded bg-slate-800 px-2 py-1" />
        </label>
        <span className="text-slate-500">
          {pairs} matches × {deals * 2} hands (each deal played twice, seats swapped)
        </span>
        <button onClick={start} disabled={running || chosen.length < 2} className="ml-auto rounded-lg bg-emerald-500 px-3 py-1.5 font-bold text-slate-950 disabled:opacity-40">
          {running ? "Running…" : "Start"}
        </button>
      </div>
      {status && (
        <div className="mt-3">
          <div className="h-2 overflow-hidden rounded-full bg-slate-800">
            <div className="h-full rounded-full bg-emerald-500 transition-all" style={{ width: `${(100 * status.progress.done) / Math.max(1, status.progress.total)}%` }} />
          </div>
          <div className="mt-1 text-[11px] text-slate-400">
            {status.state} · {status.progress.done}/{status.progress.total} matches · {Math.round(status.elapsed)}s
            {status.error && <span className="text-red-400"> · {status.error}</span>}
          </div>
        </div>
      )}
      {error && <div className="mt-2 text-xs text-red-400">{error}</div>}
    </div>
  );
}

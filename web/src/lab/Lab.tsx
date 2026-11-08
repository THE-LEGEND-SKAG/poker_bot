import { type ReactElement, useCallback, useEffect, useMemo, useState } from "react";
import type { AgentInfo } from "../types";
import Leaderboard from "./Leaderboard";
import LineChart, { type Series } from "./LineChart";
import Matrix from "./Matrix";
import { AGENT_COLOR, SLOTS } from "./palette";
import Replayer from "./Replayer";
import TournamentRunner from "./TournamentRunner";
import type { HandRecord, Tournament } from "./types";

type Training = {
  cfr: Record<string, { infosets: number; target_value: number; curve: { iteration: number; exploitability: number; value: number }[] }> | null;
  mccfr: { meta: Record<string, any>; workers: { iteration: number; avg_positive_regret: number; infosets: number }[][] } | null;
  ppo: {
    meta: Record<string, any>;
    history: { update: number; bb100_all: number; ent: number; vl: number }[];
    evals: Record<string, any>[];
  } | null;
};

function smooth(points: { x: number; y: number }[], window = 25) {
  return points.map((p, i) => {
    const s = points.slice(Math.max(0, i - window + 1), i + 1);
    return { x: p.x, y: s.reduce((a, b) => a + b.y, 0) / s.length };
  });
}

function Tile({ label, value, note }: { label: string; value: string; note?: string }) {
  return (
    <div className="rounded-2xl border border-white/10 bg-slate-900/70 p-3">
      <div className="text-[11px] uppercase tracking-wider text-slate-400">{label}</div>
      <div className="mt-1 text-2xl font-bold text-slate-100">{value}</div>
      {note && <div className="text-[11px] text-slate-500">{note}</div>}
    </div>
  );
}

export default function Lab() {
  const [training, setTraining] = useState<Training | null>(null);
  const [tour, setTour] = useState<Tournament | null>(null);
  const [hands, setHands] = useState<HandRecord[]>([]);
  const [labels, setLabels] = useState<Record<string, string>>({});

  const loadTournament = useCallback(async () => {
    const r = await fetch("/api/lab/tournament");
    if (!r.ok) return;
    const t: Tournament = await r.json();
    setTour(t);
    setHands(await fetch(`/api/lab/tournament/${t.id}/hands`).then((x) => x.json()));
  }, []);

  useEffect(() => {
    fetch("/api/lab/training").then((r) => r.json()).then(setTraining);
    fetch("/api/agents").then((r) => r.json()).then((a: AgentInfo[]) => setLabels(Object.fromEntries(a.map((x) => [x.kind, x.label]))));
    loadTournament();
  }, [loadTournament]);

  const charts = useMemo(() => {
    if (!training) return null;
    const out: { key: string; el: ReactElement }[] = [];
    if (training.cfr) {
      const series: Series[] = Object.entries(training.cfr).map(([g, d], i) => ({
        name: `${g[0].toUpperCase()}${g.slice(1)} (${d.infosets} infosets)`,
        color: SLOTS[i],
        points: d.curve.map((c) => ({ x: c.iteration, y: c.exploitability })),
      }));
      out.push({ key: "cfr", el: <LineChart title="CFR exploitability" subtitle="Exact best response vs the average strategy — falls toward 0 at a Nash equilibrium" series={series} xLabel="iterations" yLabel="chips / hand" xLog yLog /> });
    }
    if (training.mccfr?.workers?.length) {
      const series: Series[] = training.mccfr.workers.map((w, i) => ({
        name: `worker ${i + 1}`,
        color: SLOTS[i],
        points: w.map((r) => ({ x: r.iteration, y: r.avg_positive_regret })),
      }));
      const m = training.mccfr.meta;
      out.push({ key: "mccfr", el: <LineChart title="MCCFR blueprint: average positive regret" subtitle={`Heads-up NLHE abstraction · ${m.iterations.toLocaleString()} iterations · ${m.infosets.toLocaleString()} infosets`} series={series} xLabel="iterations per worker" yLabel="regret / iteration" yLog /> });
    }
    if (training.ppo) {
      const h = training.ppo.history;
      out.push({
        key: "ppo-win",
        el: <LineChart title="PPO: win rate during training" subtitle="vs the league (fixed bots + past selves), 25-update moving average" series={[{ name: "Neural (PPO)", color: AGENT_COLOR.neural, points: smooth(h.map((r) => ({ x: r.update, y: r.bb100_all }))) }]} xLabel="update" yLabel="bb/100" zeroLine />,
      });
      out.push({
        key: "ppo-ent",
        el: <LineChart title="PPO: policy entropy" subtitle="Falls as the policy commits to actions (max for 5 actions ≈ 1.61)" series={[{ name: "entropy", color: AGENT_COLOR.neural, points: h.map((r) => ({ x: r.update, y: r.ent })) }]} xLabel="update" yLabel="nats" />,
      });
      const evals = training.ppo.evals.filter((e) => !e.final);
      const opps = ["rule_balanced", "rule_tight", "rule_lag", "calling_station"].filter((o) => o !== "calling_station");
      out.push({
        key: "ppo-eval",
        el: <LineChart title="PPO: evaluation during training" subtitle="Duplicate matches vs held-fixed rule bots (calling station omitted: different scale)" series={opps.map((o) => ({ name: labels[o] ?? o, color: AGENT_COLOR[o], points: evals.map((e) => ({ x: e.update, y: e[o].bb100 })) }))} xLabel="update" yLabel="bb/100" zeroLine />,
      });
    }
    return out;
  }, [training, labels]);

  const ppo = training?.ppo?.meta;
  const bp = training?.mccfr?.meta;
  return (
    <div className="mx-auto flex max-w-7xl flex-col gap-4 pb-10">
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <Tile label="Bots in last tournament" value={tour ? String(tour.config.agents.length) : "—"} note={tour ? `${tour.config.format}, ${tour.config.deals_per_match * 2} hands/match` : undefined} />
        <Tile label="CFR blueprint" value={bp ? `${(bp.iterations / 1000).toFixed(0)}k it` : "—"} note={bp ? `${bp.infosets.toLocaleString()} infosets` : undefined} />
        <Tile label="PPO training" value={ppo ? `${(ppo.decisions / 1e6).toFixed(2)}M` : "—"} note={ppo ? `decisions · ${ppo.updates} updates · ${ppo.status}` : undefined} />
        <Tile label="Leader" value={tour ? (labels[tour.leaderboard[0].kind] ?? tour.leaderboard[0].kind).replace(" (heads-up)", "") : "—"} note={tour ? `${tour.leaderboard[0].bb100 > 0 ? "+" : ""}${tour.leaderboard[0].bb100} ± ${tour.leaderboard[0].ci95} bb/100` : undefined} />
      </div>

      {tour ? (
        <>
          <div className="flex items-baseline justify-between">
            <h2 className="text-lg font-bold text-white">Leaderboard</h2>
            <span className="text-xs text-slate-500">
              {tour.source === "official" ? "Committed benchmark" : "Your run"} · {tour.generated_at}
            </span>
          </div>
          <Leaderboard rows={tour.leaderboard} status={tour.training_status} />
          <Matrix agents={tour.leaderboard.map((r) => r.kind)} labels={labels} matches={tour.matches} />
        </>
      ) : (
        <div className="text-sm text-slate-500">No tournament results yet — run one below.</div>
      )}

      <h2 className="text-lg font-bold text-white">Training</h2>
      <div className="grid gap-3 lg:grid-cols-2">{charts?.map((c) => <div key={c.key}>{c.el}</div>)}</div>

      <div className="grid gap-3 lg:grid-cols-2">
        <Replayer hands={hands} labels={labels} />
        <TournamentRunner onFinished={loadTournament} />
      </div>
    </div>
  );
}

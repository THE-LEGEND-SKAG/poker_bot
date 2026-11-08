import { useState } from "react";
import ActionBar from "./components/ActionBar";
import CoachPanel from "./components/CoachPanel";
import Lobby from "./components/Lobby";
import LogPanel from "./components/LogPanel";
import Lab from "./lab/Lab";
import Table from "./components/Table";
import { useGame } from "./useGame";

export default function App() {
  const game = useGame();
  const [inGame, setInGame] = useState(false);
  const [tab, setTab] = useState<"coach" | "history">("coach");
  const [coachOn, setCoachOn] = useState(true);
  const [page, setPage] = useState<"play" | "lab">(location.hash === "#lab" ? "lab" : "play");
  const go = (p: "play" | "lab") => {
    setPage(p);
    history.replaceState(null, "", p === "lab" ? "#lab" : "#");
  };

  if (page === "lab") {
    return (
      <div className="min-h-full p-3">
        <header className="mx-auto mb-4 flex max-w-7xl items-center gap-3">
          <h1 className="text-xl font-black text-white">
            Poker <span className="text-emerald-400">AI</span> <span className="text-slate-500">Lab</span>
          </h1>
          <button onClick={() => go("play")} className="ml-auto rounded-lg bg-emerald-500 px-3 py-1.5 text-sm font-bold text-slate-950">
            ← Play
          </button>
        </header>
        <Lab />
      </div>
    );
  }
  const s = game.state;

  if (!inGame || !s) {
    return (
      <div className="flex min-h-full flex-col items-center justify-center gap-3 p-4">
        <button onClick={() => go("lab")} className="rounded-lg bg-slate-800 px-3 py-1.5 text-sm font-semibold text-slate-200 hover:bg-slate-700">
          Open the Lab: leaderboard, training curves, replayer →
        </button>
        <Lobby
          connected={game.status === "open"}
          onStart={(cfg, delay) => {
            game.newGame(cfg, delay);
            setInGame(true);
          }}
        />
      </div>
    );
  }

  const human = s.seats[s.human_seat];
  return (
    <div className="flex min-h-full flex-col gap-3 p-3 lg:h-full">
      <header className="flex items-center gap-3">
        <button onClick={() => setInGame(false)} className="rounded-lg bg-slate-800 px-3 py-1.5 text-sm font-semibold hover:bg-slate-700">
          ← Lobby
        </button>
        <div className="text-sm text-slate-400">
          Hand <span className="font-semibold text-slate-200">#{s.hand_id}</span> · Blinds {s.small_blind}/{s.big_blind} · Session{" "}
          <span className={human.session_net >= 0 ? "text-emerald-400" : "text-red-400"}>
            {human.session_net >= 0 ? "+" : ""}
            {human.session_net.toLocaleString()}
          </span>
        </div>
        <button onClick={() => go("lab")} className="ml-auto rounded-lg bg-slate-800 px-3 py-1.5 text-sm font-semibold hover:bg-slate-700">
          Lab
        </button>
        <div className={`h-2 w-2 rounded-full ${game.status === "open" ? "bg-emerald-400" : "bg-red-500"}`} title={game.status} />
      </header>

      <div className="flex min-h-0 flex-1 flex-col gap-3 lg:flex-row">
        <main className="flex min-w-0 flex-1 flex-col justify-center gap-3 pb-44 md:pb-32">
          <Table state={s} thinking={game.thinking} />
          {/* Fixed dock: the action buttons are always on screen, whatever the window height. */}
          <div className="fixed inset-x-0 bottom-0 z-40 bg-gradient-to-t from-slate-950 via-slate-950/95 to-transparent px-2 pt-6 pb-[max(0.5rem,env(safe-area-inset-bottom))] lg:right-[21.5rem]">
            {s.legal ? (
              <ActionBar state={s} onAct={game.act} />
            ) : s.terminal ? (
              <div className="mx-auto flex max-w-xl items-center justify-center gap-3">
                {s.human_busted ? (
                  <button onClick={game.rebuy} className="btn bg-amber-500 text-slate-950 hover:bg-amber-400">
                    Rebuy
                  </button>
                ) : (
                  <button onClick={game.nextHand} className="btn bg-emerald-500 text-slate-950 hover:bg-emerald-400" autoFocus>
                    Next hand →
                  </button>
                )}
              </div>
            ) : (
              <div className="py-3 text-center text-sm text-slate-400">Waiting for opponents…</div>
            )}
          </div>
        </main>
        <aside className="flex h-[28rem] flex-col gap-2 lg:h-auto lg:w-80">
          <div className="flex gap-1 rounded-xl bg-slate-900 p-1 text-sm font-semibold">
            {(["coach", "history"] as const).map((t) => (
              <button
                key={t}
                onClick={() => setTab(t)}
                className={`flex-1 rounded-lg py-1.5 capitalize ${tab === t ? "bg-slate-700 text-white" : "text-slate-400 hover:text-slate-200"}`}
              >
                {t === "history" ? "Hand history" : "Coach"}
              </button>
            ))}
          </div>
          <div className="min-h-0 flex-1">
            {tab === "coach" ? (
              <CoachPanel
                state={s}
                analysis={game.coach}
                enabled={coachOn}
                onToggle={(on) => {
                  setCoachOn(on);
                  game.setCoachEnabled(on);
                }}
              />
            ) : (
              <LogPanel log={s.log} />
            )}
          </div>
        </aside>
      </div>

      {game.error && (
        <div className="fixed bottom-4 left-1/2 -translate-x-1/2 rounded-lg bg-red-600 px-4 py-2 text-sm font-semibold shadow-xl">{game.error}</div>
      )}
    </div>
  );
}

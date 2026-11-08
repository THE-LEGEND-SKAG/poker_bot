import { useState } from "react";
import ActionBar from "./components/ActionBar";
import Lobby from "./components/Lobby";
import LogPanel from "./components/LogPanel";
import Table from "./components/Table";
import { useGame } from "./useGame";

export default function App() {
  const game = useGame();
  const [inGame, setInGame] = useState(false);
  const s = game.state;

  if (!inGame || !s) {
    return (
      <div className="flex min-h-full items-center justify-center p-4">
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
        <div className={`ml-auto h-2 w-2 rounded-full ${game.status === "open" ? "bg-emerald-400" : "bg-red-500"}`} title={game.status} />
      </header>

      <div className="flex min-h-0 flex-1 flex-col gap-3 lg:flex-row">
        <main className="flex min-w-0 flex-1 flex-col justify-center gap-3">
          <Table state={s} thinking={game.thinking} />
          <div className="min-h-20">
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
              <div className="text-center text-sm text-slate-400">Waiting for opponents…</div>
            )}
          </div>
        </main>
        <aside className="h-72 lg:h-auto lg:w-80">
          <LogPanel log={s.log} />
        </aside>
      </div>

      {game.error && (
        <div className="fixed bottom-4 left-1/2 -translate-x-1/2 rounded-lg bg-red-600 px-4 py-2 text-sm font-semibold shadow-xl">{game.error}</div>
      )}
    </div>
  );
}

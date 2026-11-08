import { AnimatePresence, motion } from "framer-motion";
import type { GameState } from "../types";
import { Bet, ChipStack } from "./Chips";
import PlayingCard from "./PlayingCard";
import Seat from "./Seat";

/** Seat positions around an ellipse, the human always at the bottom centre. */
function seatPosition(index: number, n: number) {
  const angle = (Math.PI / 2) + (index * 2 * Math.PI) / n;
  return { x: 50 + 47 * Math.cos(angle), y: 50 + 44 * Math.sin(angle) };
}

export default function Table({ state, thinking }: { state: GameState; thinking: number | null }) {
  const n = state.seats.length;
  const bb = state.big_blind;
  return (
    <div className="relative mx-auto mb-24 mt-6 aspect-[16/9] w-full max-w-5xl">
      <div className="felt absolute inset-[9%_7%] rounded-[50%]" />
      <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
        <span className="select-none text-5xl font-black tracking-widest text-black/10 md:text-7xl">POKER AI</span>
      </div>

      {/* pot + board */}
      <div className="absolute left-1/2 top-1/2 flex -translate-x-1/2 -translate-y-1/2 flex-col items-center gap-3">
        <AnimatePresence>
          {state.pot > 0 && (
            <motion.div
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex items-center gap-2 rounded-full bg-black/45 px-3 py-1 text-sm font-semibold text-emerald-200"
            >
              {state.pot_middle > 0 && <ChipStack amount={state.pot_middle} bigBlind={bb} />}
              Pot: <span className="tabular-nums">{state.pot.toLocaleString()}</span>
            </motion.div>
          )}
        </AnimatePresence>
        <div className="flex h-17 gap-1.5 md:h-23">
          {state.board.map((c, i) => (
            <PlayingCard key={`${state.hand_id}-${c}`} card={c} size="lg" delay={i < 3 ? i * 0.1 : 0} />
          ))}
        </div>
      </div>

      {state.seats.map((seat) => {
        const rel = (seat.seat - state.human_seat + n) % n;
        const pos = seatPosition(rel, n);
        const betPos = { x: 50 + (pos.x - 50) * 0.58, y: 50 + (pos.y - 50) * 0.5 };
        return (
          <div key={seat.seat}>
            <div className="absolute -translate-x-1/2 -translate-y-1/2" style={{ left: `${pos.x}%`, top: `${pos.y}%` }}>
              <Seat
                seat={seat}
                isHuman={seat.seat === state.human_seat}
                isButton={state.button === seat.seat}
                isTurn={state.to_act === seat.seat}
                thinking={thinking === seat.seat}
                handLabel={
                  state.terminal
                    ? [state.result?.hands[seat.seat], seat.won > 0 ? `Wins ${seat.won.toLocaleString()}` : null].filter(Boolean).join(" · ") || undefined
                    : undefined
                }
              />
            </div>
            <div className="absolute -translate-x-1/2 -translate-y-1/2" style={{ left: `${betPos.x}%`, top: `${betPos.y}%` }}>
              <AnimatePresence>{seat.bet > 0 && <Bet key={`${state.street}-${seat.seat}`} amount={seat.bet} bigBlind={bb} />}</AnimatePresence>
            </div>
          </div>
        );
      })}
    </div>
  );
}

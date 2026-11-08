import { AnimatePresence, motion } from "framer-motion";
import type { Seat as SeatT } from "../types";
import PlayingCard from "./PlayingCard";

const AVATAR_COLORS = ["#f59e0b", "#06b6d4", "#a855f7", "#ef4444", "#22c55e", "#ec4899", "#3b82f6", "#eab308", "#14b8a6"];

type Props = {
  seat: SeatT;
  isHuman: boolean;
  isButton: boolean;
  isTurn: boolean;
  thinking: boolean;
  handLabel?: string;
  compact?: boolean;
};

export default function Seat({ seat, isHuman, isButton, isTurn, thinking, handLabel, compact = false }: Props) {
  const out = !seat.in_hand && seat.stack === 0;
  const color = AVATAR_COLORS[seat.seat % AVATAR_COLORS.length];
  const showCards = seat.in_hand && !(seat.folded && !isHuman);
  return (
    <div className={`flex flex-col items-center ${out ? "opacity-40" : ""}`}>
      <div className={`flex -space-x-3 ${isHuman ? "mb-[-10px]" : "mb-[-14px]"} z-0`}>
        {showCards &&
          [0, 1].map((i) => (
            <div key={`${seat.seat}-${i}-${seat.cards?.[i] ?? "x"}`} style={{ transform: `rotate(${i ? 6 : -6}deg)` }}>
              <PlayingCard card={seat.cards?.[i]} size={isHuman ? (compact ? "md" : "lg") : "sm"} delay={i * 0.08} dim={seat.folded} />
            </div>
          ))}
      </div>
      <motion.div
        animate={
          seat.winner
            ? { boxShadow: ["0 0 0px #fbbf24", "0 0 28px #fbbf24", "0 0 10px #fbbf24"] }
            : isTurn
              ? { boxShadow: "0 0 0 2px #fbbf24" }
              : { boxShadow: "0 0 0 1px rgba(255,255,255,0.12)" }
        }
        transition={{ duration: seat.winner ? 1.2 : 0.2 }}
        className={`relative z-10 flex items-center rounded-xl bg-slate-950/90 backdrop-blur ${compact ? "max-w-[92px] gap-1 px-1.5 py-1" : "min-w-36 gap-2 px-2 py-1.5"}`}
      >
        <div
          className={`flex shrink-0 items-center justify-center rounded-full font-bold text-slate-950 ${compact ? "h-6 w-6 text-xs" : "h-10 w-10 text-lg"}`}
          style={{ background: color }}
        >
          {seat.name.slice(0, 1)}
        </div>
        <div className="min-w-0 leading-tight">
          <div className={`truncate font-semibold text-white ${compact ? "text-[11px]" : "text-sm"}`}>{seat.name}</div>
          <div className={`font-bold tabular-nums text-emerald-400 ${compact ? "text-[11px]" : "text-sm"}`}>{seat.stack.toLocaleString()}</div>
          {!compact && <div className="truncate text-[10px] uppercase tracking-wide text-slate-400">{seat.label}</div>}
        </div>
        {isButton && (
          <div className="absolute -right-2 -top-2 flex h-6 w-6 items-center justify-center rounded-full bg-white text-[11px] font-black text-slate-900 shadow">
            D
          </div>
        )}
        {thinking && (
          <div className="absolute -bottom-2 left-1/2 flex -translate-x-1/2 gap-1 rounded-full bg-amber-400 px-2 py-0.5">
            {[0, 1, 2].map((i) => (
              <motion.span
                key={i}
                className="h-1.5 w-1.5 rounded-full bg-slate-900"
                animate={{ opacity: [0.2, 1, 0.2] }}
                transition={{ duration: 0.9, repeat: Infinity, delay: i * 0.15 }}
              />
            ))}
          </div>
        )}
      </motion.div>
      <AnimatePresence mode="wait">
        {(handLabel || seat.last_action) && (
          <motion.div
            key={handLabel ?? seat.last_action}
            initial={{ opacity: 0, y: -4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className={`mt-1.5 rounded-md font-semibold ${compact ? "max-w-[110px] truncate px-1.5 text-[10px]" : "px-2 py-0.5 text-xs"} ${
              handLabel
                ? "bg-amber-400 text-slate-950"
                : seat.last_action === "Fold"
                  ? "bg-slate-700 text-slate-300"
                  : "bg-sky-600 text-white"
            }`}
          >
            {handLabel ?? seat.last_action}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

import { motion } from "framer-motion";

const SUIT: Record<string, { sym: string; red: boolean }> = {
  c: { sym: "♣", red: false },
  d: { sym: "♦", red: true },
  h: { sym: "♥", red: true },
  s: { sym: "♠", red: false },
};

type Props = { card?: string | null; size?: "sm" | "md" | "lg"; delay?: number; dim?: boolean; highlight?: boolean };

const SIZES = {
  sm: "w-9 h-13 text-sm rounded-md",
  md: "w-12 h-17 text-base rounded-lg",
  lg: "w-16 h-23 text-xl rounded-lg",
};

export default function PlayingCard({ card, size = "md", delay = 0, dim = false, highlight = false }: Props) {
  const face = card ? SUIT[card[1]] : null;
  const rank = card ? (card[0] === "T" ? "10" : card[0]) : "";
  return (
    <motion.div
      initial={{ opacity: 0, y: -30, rotateY: 90, scale: 0.8 }}
      animate={{ opacity: dim ? 0.35 : 1, y: 0, rotateY: 0, scale: 1 }}
      transition={{ delay, duration: 0.35, ease: "easeOut" }}
      className={`${SIZES[size]} relative shrink-0 select-none shadow-lg shadow-black/40 ${
        highlight ? "ring-2 ring-amber-300" : ""
      } ${card ? "bg-white" : "card-back border border-white/30"}`}
      aria-label={card ?? "face-down card"}
    >
      {face && (
        <div className={`absolute inset-0 flex flex-col justify-between p-1 font-bold leading-none ${face.red ? "text-red-600" : "text-slate-900"}`}>
          <div className="flex flex-col items-start">
            <span>{rank}</span>
            <span className="text-[0.8em]">{face.sym}</span>
          </div>
          <div className="self-center text-[1.6em] -mt-3">{face.sym}</div>
        </div>
      )}
    </motion.div>
  );
}

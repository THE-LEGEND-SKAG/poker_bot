import { motion } from "framer-motion";

const COLORS = ["#e5e7eb", "#ef4444", "#3b82f6", "#22c55e", "#111827", "#a855f7", "#f59e0b"];

export function ChipStack({ amount, bigBlind }: { amount: number; bigBlind: number }) {
  const n = Math.max(1, Math.min(6, Math.ceil(Math.log2(1 + amount / bigBlind))));
  return (
    <div className="relative h-6 w-5">
      {Array.from({ length: n }).map((_, i) => (
        <div
          key={i}
          className="absolute left-0 h-5 w-5 rounded-full border-2 border-dashed border-white/80 shadow"
          style={{ bottom: i * 3, background: COLORS[(i + Math.floor(amount / bigBlind)) % COLORS.length] }}
        />
      ))}
    </div>
  );
}

export function Bet({ amount, bigBlind }: { amount: number; bigBlind: number }) {
  return (
    <motion.div
      layout
      initial={{ opacity: 0, scale: 0.5 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.3 }}
      className="flex items-center gap-1.5"
    >
      <ChipStack amount={amount} bigBlind={bigBlind} />
      <span className="rounded-full bg-black/60 px-2 py-0.5 text-xs font-semibold tabular-nums text-amber-200">
        {amount.toLocaleString()}
      </span>
    </motion.div>
  );
}

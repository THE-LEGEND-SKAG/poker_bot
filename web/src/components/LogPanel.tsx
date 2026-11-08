import { useEffect, useRef } from "react";
import type { LogLine } from "../types";

const STYLE: Record<string, string> = {
  header: "mt-2 text-slate-400 font-semibold",
  street: "text-sky-300 font-semibold",
  win: "text-amber-300 font-semibold",
  showdown: "text-fuchsia-300",
  info: "text-slate-500 italic",
  post: "text-slate-500",
  action: "text-slate-200",
};

export default function LogPanel({ log }: { log: LogLine[] }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    ref.current?.scrollTo({ top: ref.current.scrollHeight, behavior: "smooth" });
  }, [log.length]);
  return (
    <div className="flex h-full min-h-0 flex-col rounded-2xl border border-white/10 bg-slate-950/70">
      <div className="border-b border-white/10 px-3 py-2 text-xs font-semibold uppercase tracking-wider text-slate-400">Hand history</div>
      <div ref={ref} className="min-h-0 flex-1 space-y-0.5 overflow-y-auto px-3 py-2 text-[13px] leading-snug">
        {log.map((l, i) => (
          <div key={i} className={STYLE[l.kind] ?? ""}>
            {l.text}
          </div>
        ))}
      </div>
    </div>
  );
}

import { useEffect, useMemo, useRef, useState } from "react";
import { INK } from "./palette";

export type Series = { name: string; color: string; points: { x: number; y: number }[] };

type Props = {
  title: string;
  subtitle?: string;
  series: Series[];
  xLabel: string;
  yLabel: string;
  xLog?: boolean;
  yLog?: boolean;
  zeroLine?: boolean;
  formatY?: (v: number) => string;
  formatX?: (v: number) => string;
  height?: number;
};

const PAD = { l: 52, r: 16, t: 10, b: 34 };

function ticks(min: number, max: number, log: boolean, count = 5): number[] {
  if (log) {
    const out: number[] = [];
    for (let e = Math.floor(Math.log10(min)); e <= Math.ceil(Math.log10(max)); e++) out.push(10 ** e);
    return out.filter((t) => t >= min * 0.999 && t <= max * 1.001);
  }
  const span = max - min || 1;
  const step = 10 ** Math.floor(Math.log10(span / count));
  const nice = [1, 2, 5, 10].map((m) => m * step).find((s) => span / s <= count) ?? step * 10;
  const out: number[] = [];
  for (let v = Math.ceil(min / nice) * nice; v <= max + 1e-9; v += nice) out.push(+v.toPrecision(12));
  return out;
}

const fmt = (v: number) => (Math.abs(v) >= 1000 ? `${(v / 1000).toFixed(v >= 1e4 ? 0 : 1)}k` : Math.abs(v) < 0.01 && v !== 0 ? v.toExponential(0) : `${+v.toPrecision(3)}`);

export default function LineChart({ title, subtitle, series, xLabel, yLabel, xLog, yLog, zeroLine, formatY = fmt, formatX = fmt, height = 220 }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(480);
  const [hover, setHover] = useState<number | null>(null);
  useEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver(([e]) => setWidth(Math.max(260, e.contentRect.width)));
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);

  const all = series.flatMap((s) => s.points).filter((p) => (!xLog || p.x > 0) && (!yLog || p.y > 0));
  const [x0, x1] = [Math.min(...all.map((p) => p.x)), Math.max(...all.map((p) => p.x))];
  let [y0, y1] = [Math.min(...all.map((p) => p.y)), Math.max(...all.map((p) => p.y))];
  if (zeroLine && !yLog) {
    y0 = Math.min(y0, 0);
    y1 = Math.max(y1, 0);
  }
  if (!yLog) {
    const padY = (y1 - y0) * 0.06 || 1;
    y0 -= padY;
    y1 += padY;
  }
  const w = width - PAD.l - PAD.r;
  const h = height - PAD.t - PAD.b;
  const sx = (x: number) => PAD.l + (xLog ? (Math.log10(x) - Math.log10(x0)) / (Math.log10(x1) - Math.log10(x0) || 1) : (x - x0) / (x1 - x0 || 1)) * w;
  const sy = (y: number) => PAD.t + h - (yLog ? (Math.log10(y) - Math.log10(y0)) / (Math.log10(y1) - Math.log10(y0) || 1) : (y - y0) / (y1 - y0 || 1)) * h;
  const xs = useMemo(() => ticks(x0, x1, !!xLog), [x0, x1, xLog]);
  const ys = useMemo(() => ticks(y0, y1, !!yLog), [y0, y1, yLog]);
  const xsHover = useMemo(() => [...new Set(series.flatMap((s) => s.points.map((p) => p.x)))].sort((a, b) => a - b), [series]);

  const onMove = (e: React.MouseEvent<SVGRectElement>) => {
    const rect = (e.target as SVGRectElement).getBoundingClientRect();
    const px = e.clientX - rect.left + PAD.l;
    let best = 0;
    xsHover.forEach((x, i) => {
      if (Math.abs(sx(x) - px) < Math.abs(sx(xsHover[best]) - px)) best = i;
    });
    setHover(xsHover[best]);
  };
  const nearest = (s: Series, x: number) => s.points.reduce((a, b) => (Math.abs(b.x - x) < Math.abs(a.x - x) ? b : a), s.points[0]);

  return (
    <figure className="rounded-2xl border border-white/10 bg-slate-900/70 p-3">
      <figcaption className="mb-1">
        <div className="text-sm font-semibold text-slate-100">{title}</div>
        {subtitle && <div className="text-xs text-slate-400">{subtitle}</div>}
      </figcaption>
      {series.length > 1 && (
        <div className="mb-1 flex flex-wrap gap-x-3 gap-y-1 text-[11px] text-slate-300">
          {series.map((s) => (
            <span key={s.name} className="flex items-center gap-1.5">
              <span className="inline-block h-0.5 w-3 rounded" style={{ background: s.color }} />
              {s.name}
            </span>
          ))}
        </div>
      )}
      <div ref={ref} className="relative">
        <svg width={width} height={height} role="img" aria-label={title}>
          {ys.map((t) => (
            <g key={`y${t}`}>
              <line x1={PAD.l} x2={PAD.l + w} y1={sy(t)} y2={sy(t)} stroke={INK.grid} />
              <text x={PAD.l - 6} y={sy(t)} dy="0.32em" textAnchor="end" fontSize={10} fill={INK.muted} style={{ fontVariantNumeric: "tabular-nums" }}>
                {formatY(t)}
              </text>
            </g>
          ))}
          {xs.map((t) => (
            <text key={`x${t}`} x={sx(t)} y={PAD.t + h + 14} textAnchor="middle" fontSize={10} fill={INK.muted}>
              {formatX(t)}
            </text>
          ))}
          <line x1={PAD.l} x2={PAD.l + w} y1={PAD.t + h} y2={PAD.t + h} stroke={INK.axis} />
          {zeroLine && !yLog && y0 < 0 && y1 > 0 && <line x1={PAD.l} x2={PAD.l + w} y1={sy(0)} y2={sy(0)} stroke={INK.secondary} strokeOpacity={0.45} />}
          <text x={PAD.l + w / 2} y={height - 4} textAnchor="middle" fontSize={10} fill={INK.muted}>
            {xLabel}
          </text>
          <text transform={`translate(11 ${PAD.t + h / 2}) rotate(-90)`} textAnchor="middle" fontSize={10} fill={INK.muted}>
            {yLabel}
          </text>
          {series.map((s) => {
            const pts = s.points.filter((p) => (!xLog || p.x > 0) && (!yLog || p.y > 0));
            return (
              <path
                key={s.name}
                d={pts.map((p, i) => `${i ? "L" : "M"}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join("")}
                fill="none"
                stroke={s.color}
                strokeWidth={2}
                strokeLinejoin="round"
                strokeLinecap="round"
              />
            );
          })}
          {hover !== null && (
            <g>
              <line x1={sx(hover)} x2={sx(hover)} y1={PAD.t} y2={PAD.t + h} stroke={INK.secondary} strokeOpacity={0.5} />
              {series.map((s) => {
                const p = nearest(s, hover);
                return p ? <circle key={s.name} cx={sx(p.x)} cy={sy(p.y)} r={4} fill={s.color} stroke="#0f172a" strokeWidth={2} /> : null;
              })}
            </g>
          )}
          <rect x={PAD.l} y={PAD.t} width={w} height={h} fill="transparent" onMouseMove={onMove} onMouseLeave={() => setHover(null)} />
        </svg>
        {hover !== null && (
          <div
            className="pointer-events-none absolute top-2 z-10 rounded-lg border border-white/10 bg-slate-950/95 px-2 py-1.5 text-[11px] shadow-xl"
            style={{ left: Math.min(sx(hover) + 10, width - 170) }}
          >
            <div className="mb-0.5 text-slate-400">
              {xLabel}: {formatX(hover)}
            </div>
            {series.map((s) => {
              const p = nearest(s, hover);
              return (
                <div key={s.name} className="flex items-center gap-1.5 text-slate-100">
                  <span className="inline-block h-2 w-2 rounded-full" style={{ background: s.color }} />
                  <span className="text-slate-300">{s.name}</span>
                  <span className="ml-auto pl-3 font-semibold tabular-nums">{p ? formatY(p.y) : "—"}</span>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </figure>
  );
}

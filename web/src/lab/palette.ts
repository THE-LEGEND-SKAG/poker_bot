// Dark-mode categorical slots from the reference palette, validated against the
// app's panel surface (#0f172a): all 8 adjacent pairs pass CVD (worst ΔE 8.4)
// and normal-vision (worst 19.3) checks, all ≥ 3:1 contrast.
export const SLOTS = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"];

// Colour follows the entity, everywhere in the Lab. The four PPO evaluation
// opponents take slots 1–4 in validated order (a red/magenta pair failed when
// they shared a chart). A 9th entity gets neutral grey, never a generated hue.
export const AGENT_COLOR: Record<string, string> = {
  rule_balanced: SLOTS[0],
  rule_tight: SLOTS[1],
  rule_lag: SLOTS[2],
  calling_station: SLOTS[3],
  neural: SLOTS[4],
  cfr: SLOTS[5],
  exploit: SLOTS[6],
  equity: SLOTS[7],
};
export const NEUTRAL = "#898781";
export const agentColor = (k: string) => AGENT_COLOR[k] ?? NEUTRAL;

export const INK = { primary: "#f0efec", secondary: "#c3c2b7", muted: "#898781", grid: "rgba(255,255,255,0.07)", axis: "#383835" };

// Diverging blue (+) ↔ red (−) with a neutral grey midpoint.
export function diverging(value: number, maxAbs: number): string {
  const t = Math.max(-1, Math.min(1, value / (maxAbs || 1)));
  const mid = [56, 56, 53];
  const pos = [57, 135, 229];
  const neg = [230, 103, 103];
  const end = t >= 0 ? pos : neg;
  const k = Math.abs(t);
  const c = mid.map((m, i) => Math.round(m + (end[i] - m) * k));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

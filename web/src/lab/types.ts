export type LeaderRow = {
  kind: string; label: string; level: number; bb100: number; ci95: number; hands: number;
  vpip: number; pfr: number; aggression_factor: number; fold_frequency: number;
  went_to_showdown: number; showdown_win_rate: number; avg_pot_bb: number; avg_decision_ms: number;
};
export type Match = { a: string; b: string; bb100: number; ci95: number; hands: number };
export type Tournament = {
  id: string; source: string; generated_at: string; seconds: number;
  config: { agents: string[]; deals_per_match: number; stack_bb: number; format: string };
  training_status: Record<string, string>; leaderboard: LeaderRow[]; matches: Match[]; hand_count: number;
};
export type HandRecord = {
  names: Record<string, string>; holes: Record<string, string[]>; board: string[];
  events: Record<string, any>[]; net: number[]; button: number;
};

export type Legal = {
  can_fold: boolean;
  can_check: boolean;
  can_call: boolean;
  call_amount: number;
  to_call: number;
  raise_type: "bet" | "raise" | null;
  min_raise_to: number;
  max_raise_to: number;
  street_commit: number;
  stack: number;
};

export type Seat = {
  seat: number;
  name: string;
  kind: string;
  label: string;
  stack: number;
  bet: number;
  committed: number;
  in_hand: boolean;
  folded: boolean;
  all_in: boolean;
  cards: string[] | null;
  last_action: string | null;
  winner: boolean;
  won: number;
  session_net: number;
};

export type LogLine = { text: string; kind: string; seat?: number };

export type GameState = {
  hand_id: number;
  street: string | null;
  board: string[];
  pot: number;
  pot_middle: number;
  current_bet: number;
  small_blind: number;
  big_blind: number;
  button: number | null;
  sb: number | null;
  bb: number | null;
  to_act: number | null;
  human_seat: number;
  seats: Seat[];
  legal: Legal | null;
  terminal: boolean;
  result: { winners: number[]; net: number[]; hands: Record<string, string> } | null;
  log: LogLine[];
  hands_played: number;
  human_busted: boolean;
  decision_id: string;
  review: ReviewItem[] | null;
};

export type ReviewItem = {
  seat: number;
  name: string;
  street: string;
  action: string;
  rule: string | null;
  revealed: boolean;
  details?: Record<string, unknown>;
};

export type CoachOption = {
  action: string;
  amount: number;
  label: string;
  ev: number;
  fold_equity: number;
  equity_when_called: number | null;
};

export type CoachAnalysis = {
  decision_id: string;
  hand_class: string;
  preflop_equity_vs_random: number;
  made_hand: string | null;
  street: string;
  equity: { equity: number; win: number; tie: number; samples: number; exact: boolean; stderr: number };
  pot: number;
  to_call: number;
  pot_odds: number;
  realization: number;
  opponents: number;
  range_note: string;
  options: CoachOption[];
  best: CoachOption;
  opponent_ranges: OpponentRange[];
  stat_labels: Record<string, string>;
};

export type OpponentRange = {
  seat: number;
  name: string;
  grid: number[];
  stats: Record<string, number> & { hands: number; samples: Record<string, number> };
};

export type AgentInfo = { kind: string; label: string; level: number };

export type GameConfig = {
  bots: string[];
  starting_stack: number;
  small_blind: number;
  big_blind: number;
  ante: number;
};

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
};

export type AgentInfo = { kind: string; label: string; level: number };

export type GameConfig = {
  bots: string[];
  starting_stack: number;
  small_blind: number;
  big_blind: number;
  ante: number;
};

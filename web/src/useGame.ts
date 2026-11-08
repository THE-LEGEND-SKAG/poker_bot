import { useCallback, useEffect, useRef, useState } from "react";
import type { GameConfig, GameState } from "./types";

type Status = "connecting" | "open" | "closed";

export function useGame() {
  const wsRef = useRef<WebSocket | null>(null);
  const [status, setStatus] = useState<Status>("connecting");
  const [state, setState] = useState<GameState | null>(null);
  const [thinking, setThinking] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let stopped = false;
    let retry: ReturnType<typeof setTimeout>;
    const connect = () => {
      const proto = location.protocol === "https:" ? "wss" : "ws";
      const ws = new WebSocket(`${proto}://${location.host}/ws`);
      wsRef.current = ws;
      setStatus("connecting");
      ws.onopen = () => setStatus("open");
      ws.onclose = () => {
        setStatus("closed");
        if (!stopped) retry = setTimeout(connect, 1500);
      };
      ws.onmessage = (ev) => {
        const msg = JSON.parse(ev.data);
        if (msg.type === "state") {
          setState(msg.state);
          setThinking(null);
        } else if (msg.type === "thinking") {
          setThinking(msg.seat);
        } else if (msg.type === "error") {
          setError(msg.message);
          setTimeout(() => setError(null), 3500);
        }
      };
    };
    connect();
    return () => {
      stopped = true;
      clearTimeout(retry);
      wsRef.current?.close();
    };
  }, []);

  const send = useCallback((msg: object) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) wsRef.current.send(JSON.stringify(msg));
    else setError("Not connected to the server");
  }, []);

  return {
    status,
    state,
    thinking,
    error,
    newGame: (config: GameConfig, botDelayMs: number) => send({ type: "new_game", config, bot_delay_ms: botDelayMs }),
    act: (action: string, amount?: number) => send({ type: "action", action, amount }),
    nextHand: () => send({ type: "next_hand" }),
    rebuy: () => send({ type: "rebuy" }),
    setSpeed: (botDelayMs: number) => send({ type: "set_speed", bot_delay_ms: botDelayMs }),
  };
}

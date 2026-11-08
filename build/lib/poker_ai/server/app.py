"""FastAPI app: REST for metadata, one WebSocket per game session.

Protocol (JSON messages)
------------------------
client -> server
    {"type": "new_game", "config": {...GameConfig fields...}, "bot_delay_ms": 700}
    {"type": "action", "action": "fold|check|call|bet|raise", "amount": 300}
    {"type": "next_hand"}
    {"type": "rebuy"}
server -> client
    {"type": "state", "state": {...}}          after every change
    {"type": "thinking", "seat": 3}            a bot is deciding
    {"type": "error", "message": "..."}
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from poker_ai.agents import available_agents
from poker_ai.engine import IllegalActionError

from .session import GameConfig, GameSession

WEB_DIST = Path(os.environ.get("POKER_WEB_DIST", Path(__file__).resolve().parents[2] / "web" / "dist"))

app = FastAPI(title="Poker AI")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/agents")
def agents() -> list[dict[str, Any]]:
    return available_agents()


class Connection:
    def __init__(self, ws: WebSocket) -> None:
        self.ws = ws
        self.session: GameSession | None = None
        self.bot_delay = 0.7

    async def send_state(self) -> None:
        assert self.session is not None
        await self.ws.send_json({"type": "state", "state": self.session.snapshot()})

    async def run_bots(self) -> None:
        s = self.session
        assert s is not None
        while s.bot_to_act:
            await self.ws.send_json({"type": "thinking", "seat": s.to_act})
            await asyncio.sleep(self.bot_delay)
            await asyncio.to_thread(s.bot_step)  # heavier bots must not block the event loop
            await self.send_state()

    async def handle(self, msg: dict[str, Any]) -> None:
        kind = msg.get("type")
        if kind == "new_game":
            cfg = GameConfig(**{k: v for k, v in (msg.get("config") or {}).items() if k in GameConfig.__dataclass_fields__})
            self.bot_delay = max(0.0, min(3.0, float(msg.get("bot_delay_ms", 700)) / 1000))
            self.session = GameSession(cfg)
            self.session.start_hand()
        elif self.session is None:
            raise RuntimeError("start a game first")
        elif kind == "action":
            self.session.human_act(str(msg.get("action")), msg.get("amount"))
        elif kind == "next_hand":
            self.session.start_hand()
        elif kind == "rebuy":
            self.session.rebuy_human()
        elif kind == "set_speed":
            self.bot_delay = max(0.0, min(3.0, float(msg.get("bot_delay_ms", 700)) / 1000))
            return
        else:
            raise RuntimeError(f"unknown message type {kind!r}")
        await self.send_state()
        await self.run_bots()


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    conn = Connection(ws)
    try:
        while True:
            msg = await ws.receive_json()
            try:
                await conn.handle(msg)
            except (IllegalActionError, RuntimeError, ValueError, TypeError) as exc:
                await ws.send_json({"type": "error", "message": str(exc)})
                if conn.session is not None:
                    await conn.send_state()
    except WebSocketDisconnect:
        return


if WEB_DIST.exists():
    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa(path: str) -> FileResponse:
        target = WEB_DIST / path
        return FileResponse(target if path and target.is_file() else WEB_DIST / "index.html")

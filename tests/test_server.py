import random

import pytest
from fastapi.testclient import TestClient

from poker_ai.server.app import app
from poker_ai.engine import Action, ActionType
from poker_ai.server.session import HUMAN, GameConfig, GameSession


def drive_human(session: GameSession, rng: random.Random) -> None:
    while session.to_act is not None:
        if session.bot_to_act:
            session.bot_step()
            continue
        legal = session.snapshot()["legal"]
        choice = rng.choice(["passive", "passive", "fold", "raise"])
        if choice == "raise" and legal["raise_type"]:
            session.human_act(legal["raise_type"], rng.randint(legal["min_raise_to"], legal["max_raise_to"]))
        elif choice == "fold" and legal["can_fold"]:
            session.human_act("fold")
        else:
            session.human_act("check" if legal["can_check"] else "call")


def test_session_plays_many_hands_and_conserves_chips():
    cfg = GameConfig(bots=["rule_tight", "rule_balanced", "rule_lag", "random"], seed=7, rebuy_bots=False)
    s = GameSession(cfg)
    rng = random.Random(1)
    total = cfg.starting_stack * 5
    for _ in range(60):
        if s.table.stacks[HUMAN] == 0:
            s.rebuy_human()
            total += cfg.starting_stack
        if s.table.is_over:
            break
        s.start_hand()
        drive_human(s, rng)
        snap = s.snapshot()
        assert snap["terminal"] and snap["result"] is not None
        assert sum(seat["stack"] for seat in snap["seats"]) == total


def test_snapshot_never_leaks_bot_cards_mid_hand():
    s = GameSession(GameConfig(bots=["rule_balanced"] * 3, seed=3))
    s.start_hand()
    while s.to_act is not None:
        snap = s.snapshot()
        for seat in snap["seats"][1:]:
            assert seat["cards"] is None
        assert snap["seats"][HUMAN]["cards"] is not None
        if s.bot_to_act:
            s.bot_step()
        else:
            s.human_act("check" if snap["legal"]["can_check"] else "fold")


def test_human_cannot_act_out_of_turn_or_illegally():
    s = GameSession(GameConfig(bots=["calling_station"], seed=2))
    s.start_hand()
    while s.bot_to_act:
        s.bot_step()
    from poker_ai.engine import IllegalActionError
    with pytest.raises(IllegalActionError):
        s.human_act("raise", 10**9)


def test_bad_config_rejected():
    with pytest.raises(ValueError):
        GameSession(GameConfig(bots=["nope"]))
    with pytest.raises(ValueError):
        GameSession(GameConfig(bots=[]))


def test_websocket_full_flow():
    client = TestClient(app)
    assert client.get("/api/health").json() == {"status": "ok"}
    kinds = {a["kind"] for a in client.get("/api/agents").json()}
    assert "rule_balanced" in kinds
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "new_game", "config": {"bots": ["rule_balanced", "rule_lag"], "seed": 5}, "bot_delay_ms": 0})
        state = None
        for _ in range(400):
            msg = ws.receive_json()
            assert msg["type"] != "error", msg
            if msg["type"] != "state":
                continue
            state = msg["state"]
            if state["terminal"] or state["to_act"] == 0:
                break
        assert state is not None
        hands = 0
        while hands < 5:
            if state["terminal"]:
                hands += 1
                if state["human_busted"]:
                    ws.send_json({"type": "rebuy"})
                    msg = ws.receive_json()
                    while msg["type"] == "coach":
                        msg = ws.receive_json()
                    assert msg["type"] == "state"
                ws.send_json({"type": "next_hand"})
            elif state["to_act"] == 0:
                legal = state["legal"]
                ws.send_json({"type": "action", "action": "check" if legal["can_check"] else "call"})
            while True:
                msg = ws.receive_json()
                assert msg["type"] != "error", msg
                if msg["type"] == "state" and (msg["state"]["terminal"] or msg["state"]["to_act"] == 0):
                    state = msg["state"]
                    break
        ws.send_json({"type": "action", "action": "raise", "amount": -5})
        # an error (illegal / not our turn); coach messages may arrive first
        msg = ws.receive_json()
        while msg["type"] == "coach":
            msg = ws.receive_json()
        assert msg["type"] == "error"


def test_coach_analyses_only_human_decisions_and_review_waits_for_hand_end():
    s = GameSession(GameConfig(bots=["equity", "rule_balanced"], seed=11))
    s.start_hand()
    checked = 0
    for _ in range(40):
        if s.to_act is None:
            review = s.snapshot()["review"]
            assert review is not None
            for item in review:
                if not item["revealed"]:
                    assert "details" not in item  # no numbers derived from mucked cards
            s.start_hand()
            continue
        if s.bot_to_act:
            assert s.coach() is None
            assert s.snapshot()["review"] is None
            s.bot_step()
            continue
        c = s.coach(sims=200)
        assert c is not None and c["decision_id"] == s.decision_id
        assert 0 <= c["equity"]["equity"] <= 1 and c["options"]
        live_opps = {p.seat for p in s.hand.players if p.live and p.seat != HUMAN}
        assert {o["seat"] for o in c["opponent_ranges"]} == live_opps
        for o in c["opponent_ranges"]:
            assert len(o["grid"]) == 169 and max(o["grid"]) == 1.0
        legal = s.hand.legal_actions()
        best = c["best"]
        assert legal.is_legal(Action(ActionType(best["action"]), best["amount"]))
        checked += 1
        s.human_act("check" if legal.can_check else "call")
    assert checked > 3
    assert HUMAN not in s.coach_model.seats and s.coach_model.seats  # learned about bots, never about you


def test_websocket_sends_coach_message():
    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "new_game", "config": {"bots": ["calling_station"], "seed": 2}, "bot_delay_ms": 0})
        for _ in range(50):
            msg = ws.receive_json()
            if msg["type"] == "coach":
                assert msg["analysis"]["options"]
                break
            if msg["type"] == "state" and msg["state"]["to_act"] == 0:
                continue
        else:
            raise AssertionError("no coach message received")

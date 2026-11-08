import time

import pytest
from fastapi.testclient import TestClient

from poker_ai.server.app import app
from poker_ai.sim.tournament import run_tournament


def test_round_robin_is_consistent():
    r = run_tournament(["rule_tight", "calling_station", "random"], deals=15, seed=3, workers=1, record=2)
    assert len(r["matches"]) == 3
    assert len(r["hands"]) == 6
    lb = {row["kind"]: row for row in r["leaderboard"]}
    assert set(lb) == {"rule_tight", "calling_station", "random"}
    # the leaderboard mean is the average of the agent's match results (zero-sum, so they sum to 0)
    assert abs(sum(row["bb100"] for row in r["leaderboard"])) < 0.5
    for row in r["leaderboard"]:
        assert row["hands"] == 60 and 0 <= row["vpip"] <= 1 and 0 <= row["fold_frequency"] <= 1 and row["ci95"] > 0
    assert lb["calling_station"]["fold_frequency"] == 0.0
    h = r["hands"][0]
    assert set(h) >= {"names", "holes", "board", "events", "net"} and sum(h["net"]) == 0


def test_bad_agents_rejected():
    with pytest.raises(ValueError):
        run_tournament(["rule_tight"], deals=5)
    with pytest.raises(ValueError):
        run_tournament(["rule_tight", "nope"], deals=5)


def test_lab_api_runs_a_tournament_in_the_background():
    c = TestClient(app)
    t = c.get("/api/lab/training").json()
    assert t["cfr"]["kuhn"]["curve"] and t["ppo"]["history"]
    assert c.post("/api/lab/tournament", json={"agents": ["random"], "deals": 10}).status_code == 422
    assert c.post("/api/lab/tournament", json={"agents": ["random", "nope"], "deals": 10}).status_code == 400
    r = c.post("/api/lab/tournament", json={"agents": ["random", "calling_station"], "deals": 10})
    assert r.status_code == 200
    run_id = r.json()["id"]
    for _ in range(120):
        s = c.get(f"/api/lab/tournament/{run_id}/status").json()
        if s["state"] != "running":
            break
        time.sleep(0.5)
    assert s["state"] == "finished", s
    latest = c.get("/api/lab/tournament").json()
    assert latest["id"] == run_id and latest["leaderboard"]
    assert c.get(f"/api/lab/tournament/{run_id}/hands").json()

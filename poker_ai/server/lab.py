"""Lab API: training curves, tournament results, and running new tournaments.

Tournaments run in a separate Python process (``python -m poker_ai.sim.tournament``)
so the game server stays responsive; progress is read from a small JSON file.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from poker_ai.agents import available_agents

DATA = Path(__file__).resolve().parents[1] / "data"
RUNS = Path(__file__).resolve().parents[2] / "runs"
OFFICIAL = DATA / "tournament_results.json"

router = APIRouter(prefix="/api/lab")
_jobs: dict[str, dict[str, Any]] = {}


def _read(path: Path) -> Any:
    return json.loads(path.read_text()) if path.exists() else None


def _subsample(rows: list[dict[str, Any]], n: int = 300) -> list[dict[str, Any]]:
    if len(rows) <= n:
        return rows
    step = len(rows) / n
    return [rows[int(i * step)] for i in range(n)] + [rows[-1]]


@router.get("/training")
def training() -> dict[str, Any]:
    ppo = _read(DATA / "neural_training_log.json")
    if ppo:
        ppo = {"meta": ppo["meta"], "evals": ppo["evals"],
               "history": [{k: r[k] for k in ("update", "seconds", "bb100_all", "ent", "vl")} for r in _subsample(ppo["history"])]}
    mccfr = _read(DATA / "hunl_training_log.json")
    return {"cfr": _read(DATA / "cfr_convergence.json"), "mccfr": mccfr, "ppo": ppo}


def _summarize(result: dict[str, Any]) -> dict[str, Any]:
    out = dict(result)
    out["hand_count"] = len(out.pop("hands", []))
    return out


@router.get("/tournament")
def latest_tournament() -> dict[str, Any]:
    """The most recent finished tournament (a run from this server, else the committed official one)."""
    done = [j for j in _jobs.values() if j["state"] == "finished"]
    if done:
        latest = max(done, key=lambda j: j["finished_at"])
        return {"source": "run", "id": latest["id"], **_summarize(_read(latest["out"]))}
    official = _read(OFFICIAL)
    if official is None:
        raise HTTPException(404, "no tournament results yet")
    return {"source": "official", "id": "official", **_summarize(official)}


@router.get("/tournament/{run_id}/hands")
def tournament_hands(run_id: str) -> list[dict[str, Any]]:
    path = OFFICIAL if run_id == "official" else _jobs.get(run_id, {}).get("out")
    data = _read(path) if path else None
    if data is None:
        raise HTTPException(404, "unknown tournament")
    return data["hands"]


class TournamentRequest(BaseModel):
    agents: list[str] = Field(min_length=2, max_length=8)
    deals: int = Field(default=150, ge=10, le=500)


@router.post("/tournament")
def start_tournament(req: TournamentRequest) -> dict[str, Any]:
    known = {a["kind"] for a in available_agents()}
    bad = [a for a in req.agents if a not in known]
    if bad or len(set(req.agents)) != len(req.agents):
        raise HTTPException(400, f"invalid agents: {bad or 'duplicates'}")
    if any(j["state"] == "running" for j in _jobs.values()):
        raise HTTPException(409, "a tournament is already running")
    RUNS.mkdir(exist_ok=True)
    run_id = uuid.uuid4().hex[:8]
    out = RUNS / f"tournament_{run_id}.json"
    proc = subprocess.Popen([sys.executable, "-m", "poker_ai.sim.tournament", "--agents", *req.agents,
                             "--deals", str(req.deals), "--workers", "2", "--out", str(out)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    _jobs[run_id] = {"id": run_id, "state": "running", "proc": proc, "out": out, "started_at": time.time(),
                     "agents": req.agents, "deals": req.deals}
    return job_status(run_id)


@router.get("/tournament/{run_id}/status")
def job_status(run_id: str) -> dict[str, Any]:
    job = _jobs.get(run_id)
    if job is None:
        raise HTTPException(404, "unknown tournament")
    proc: subprocess.Popen = job["proc"]
    if job["state"] == "running" and proc.poll() is not None:
        if proc.returncode == 0 and job["out"].exists():
            job["state"] = "finished"
            job["finished_at"] = time.time()
        else:
            job["state"] = "failed"
            job["error"] = (proc.stderr.read().decode()[-500:] if proc.stderr else "") or f"exit code {proc.returncode}"
    progress = _read(job["out"].with_suffix(".progress.json")) or {"done": 0, "total": 0}
    return {"id": run_id, "state": job["state"], "agents": job["agents"], "deals": job["deals"],
            "progress": progress, "elapsed": round(time.time() - job["started_at"], 1), "error": job.get("error")}

"""Local web demo: Ohio 留一法轨迹与代理决策说明."""


import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.demo_payload import list_demo_subjects, run_demo_trajectory_for_subject

app = FastAPI(title="Agentic CGM demo")

STATIC = Path(__file__).parent / "static"
CKPT_DEFAULT = ROOT / "artifacts" / "lstm.pt"


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "repo": str(ROOT)}


@app.get("/api/subjects")
def api_subjects() -> dict:
    return list_demo_subjects(ROOT)


@app.get("/api/trajectory")
def api_trajectory(
    subject: str = Query(..., description="Holdout subject id, e.g. 540"),
    max_steps: int = Query(
        2500, description="Cap rollout length for browser responsiveness"
    ),
    lookback: int = 24,
    horizon: int = 6,
) -> dict:
    if not CKPT_DEFAULT.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Checkpoint not found: {CKPT_DEFAULT}. Train with --training_dir and --holdout_subject first.",
        )
    try:
        return run_demo_trajectory_for_subject(
            ROOT,
            subject.strip(),
            CKPT_DEFAULT,
            lookback=lookback,
            horizon=horizon,
            max_test_steps=max_steps,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


app.mount(
    "/static",
    StaticFiles(directory=str(STATIC)),
    name="static",
)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")

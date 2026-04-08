# Agentic AI for health time series

Design and results: **`REPORT.md`**.

## Setup

```bash
conda env create -f environment.yml
conda activate agentic_medical_ai
cd /path/to/agentic_medical_ai
export PYTHONPATH=.
```

Scripts add the repo root to `sys.path`; set `PYTHONPATH` if you import `src` from a notebook or `python -c`.

## Data

Unpack public **OhioT1DM** into:

- `data/Training/*-ws-training.xml`
- `data/Testing/*-ws-testing.xml`

Default workflow uses subject **540**: LSTM trains on the other 11 Training files; the agent fits anomaly detection on `540-ws-training.xml` and rolls out on `540-ws-testing.xml` (see `REPORT.md`).

## Run

**1. Train LSTM** (writes `artifacts/lstm.pt` for eval and the web UI):

```bash
python scripts/train_lstm.py \
  --training_dir data/Training --testing_dir data/Testing \
  --holdout_subject 540 --epochs 40 --out artifacts/lstm.pt
```

**2. Agent evaluation** (default `--subject 540`):

```bash
python scripts/run_agent_eval.py \
  --train_dir data/Training --test_dir data/Testing --ckpt artifacts/lstm.pt
```

**3. Web demo** (needs the checkpoint above):

```bash
uvicorn demo_web.app:app --reload --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000/ and load a trajectory. API: `demo_web/app.py` → `GET /api/trajectory` → `src/demo_payload.py` (same rollout logic as `run_agent_eval.py`).

## Layout

| Path | Role |
|------|------|
| `src/data/` | Ohio XML → `GlucoseSeries` |
| `src/models/` | `GlucoseLSTM` |
| `src/tools/` | Forecaster, anomaly, guidelines |
| `src/agent/` | Memory, policy, loop |
| `scripts/` | `train_lstm.py`, `run_agent_eval.py` |
| `demo_web/` | FastAPI UI |

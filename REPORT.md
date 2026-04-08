# Short report: Agentic multivariate glucose time series (OhioT1DM-oriented)

## 1. Problem formulation

We address **short-horizon blood glucose forecasting** from multivariate CGM-style streams (glucose every 5 minutes, optional insulin and carb signals) as an **early-warning** primitive: accurate **30-minute-ahead** prediction (here `horizon=6` steps) supports alerting before clinical thresholds. The setting matches OhioT1DM-style ambulatory Type 1 diabetes data (CGM + pump/meal proxies). A static regressor is insufficient for the assignment: we wrap the predictor in an **agent loop** that **plans** when to invoke heavy models, **detects OOD windows**, **retrieves guideline snippets**, and **adapts** call frequency from recent error.

## 2. Model + agent design

- **Predictor:** `GlucoseLSTM` — stacked LSTM + MLP head, trained with z-normalized windows; checkpoint stores `norm_mean` / `norm_std` for consistent inference.
- **Agent loop:** At each holdout time \(t\), observe lag window \(x_{t-L:t}\), true future label \(g_{t+H}\) (for evaluation only). Components:
  - **Policy (rule-based):** Escalates to the LSTM + MC dropout when (i) isolation-forest **OOD**, (ii) **volatile** short-term rate of change, (iii) **critical** last CGM (\(<80\) or \(>240\) mg/dL) with **guideline retrieval**, (iv) **reflection** condition (rolling MAE above an adaptive trigger), or (v) **periodic refresh** after many cheap steps.
  - **Cheap default:** persistence baseline (last CGM) when the regime looks stable—demonstrates **non-uniform** tool use.
  - **Memory:** rolling absolute errors; **reflection** tightens the MAE trigger after sustained poor performance (see `src/agent/memory.py`).

Optional LLM policy can replace `plan_step` without changing tools.

## 3. Tool design and usage strategy

| Tool | Role | When called (examples) |
|------|------|------------------------|
| `LstmForecastTool` | Deep forecaster + optional MC dropout uncertainty | OOD, volatility, critical glucose, poor average error, periodic refresh |
| `MultivariateAnomalyTool` | IsolationForest on hand-crafted window stats | Every step (cheap); drives OOD escalation |
| `GuidelineRetrievalTool` | Rule-based “retrieval” over threshold bands | Critical predicted/last glucose branch |

**Non-trivial pattern:** the agent often skips the LSTM when the anomaly score is in-distribution and variability is low, but **forces** deep forecasting and uncertainty when the signal is erratic or historically miscalibrated—this matches the requirement to tie tool use to **uncertainty / distribution shift proxies**.

## 4. Results and analysis

After `python scripts/train_lstm.py ...` and `python scripts/run_agent_eval.py ...` on OhioT1DM XML:

- **RMSE / MAE** on the holdout Testing trajectory (mg/dL scale in agent eval).
- **AUROC (hypo below 70 mg/dL on future actual):** may be **undefined** if the holdout has no positive class.
- **Behaviour:** LSTM call rate, OOD flags, and guideline usage depend on policy thresholds (see `src/agent/policy.py`).

Report numbers using the **same** leave-one-out id and `data/Training` / `data/Testing` paths as in training to avoid leakage.

## 5. OhioT1DM data hookup

The repository layout `data/Training/*-ws-training.xml` and `data/Testing/*-ws-testing.xml` matches the public OhioT1DM split. `load_ohio_ws_xml_segments` reads `<glucose_level>` CGM streams (`ts` as `DD-MM-YYYY HH:MM:SS`), **splits** streams when consecutive CGM samples are more than **60 minutes** apart, builds a **separate** 5-minute grid per contiguous segment (glucose interpolated only within the segment), and aligns **basal** (U/h → U per step), **temp_basal** intervals, **bolus** totals per bucket, and **meal** carbs. Training and sliding windows use **per-segment** time splits and never cross segment or subject boundaries.

**Environment (conda):** `conda env create -f environment.yml` then `conda activate agentic_medical_ai`. Project root should be on `PYTHONPATH` when invoking scripts.

Training reads Ohio XML directly (`train_lstm.py --training_dir ... --testing_dir ...`).

**11 vs 1 (demo subject):** train on eleven `*-ws-training.xml` files by holding out one id, then serve the web demo for that same id on `*-ws-testing.xml`:

```bash
export PYTHONPATH=.
python scripts/train_lstm.py --training_dir data/Training --holdout_subject 540 --epochs 40 --out artifacts/lstm.pt
```

**HTML demo:** interactive timeline (past CGM window → current reading → prediction at \(t+h\)), tool flags, rationale, and guideline snippets:

```bash
uvicorn demo_web.app:app --reload --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/`, select the **same** holdout id (e.g. 540), click 加载轨迹.

Agent evaluation (same holdout as training):

```bash
python scripts/run_agent_eval.py --train_dir data/Training --test_dir data/Testing --holdout_subject 540 --ckpt artifacts/lstm.pt
# optional: --max_test_steps 4000
```

## 6. Limitations and future work

- **Regulatory / clinical:** research prototype only; not validated for dosing decisions.
- **OhioT1DM only:** loaders target the public `*-ws-training.xml` / `*-ws-testing.xml` schema.
- **Policies:** rules are interpretable but suboptimal; learn a policy from logged tool outcomes, or add an LLM planner with structured tool JSON.
- **Models:** Transformers (e.g., patch TST) and explicit probabilistic heads would strengthen uncertainty quantification used for routing.

## 7. Code map

- `environment.yml` — conda 环境
- `src/data/dataset.py` — OhioT1DM XML, `load_ohio_training_segments` / `load_ohio_ws_xml_segments` (gap-split, no cross-subject windows)
- `src/models/lstm_predictor.py` — LSTM forecaster
- `src/tools/` — forecaster, anomaly, guideline tools
- `src/agent/` — memory, policy, loop
- `src/demo_payload.py` — 组装网页用轨迹 JSON（调用现有 `run_agent_on_train_test`）
- `demo_web/` — FastAPI + 静态 HTML 演示
- `scripts/train_lstm.py`, `scripts/run_agent_eval.py`
- `notebooks/demo.ipynb` — trajectory plot

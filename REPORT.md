# Short report: Agentic multivariate glucose time series (OhioT1DM-oriented)

## 1. Problem formulation

**Task:** **Early warning** for **hypoglycemia and hyperglycemia**—i.e. supporting awareness of **glycemic risk** (dangerously low or high glucose) in time for preventive action. This is a **course / simulation** exercise, **not** a real clinical trial or a certified medical alerting system.

**Implementation approach:** We **design and implement** a **prototype agentic AI system** (per *Agentic AI for Health Time Series*, `task.txt`): an **agent loop** with a **policy**, **tools** (here: **anomaly detection**, **LSTM forecaster**, and **guideline-style retrieval**—the latter is a lightweight demo helper, **not** a live clinical information system or medical API), **memory**, and **reflection**, with **non-trivial tool use**. Concrete models, tools, and the Ohio pipeline are in §§2–5.

**Problem setting:** **Dataset:** public **OhioT1DM** (`data/Training/*-ws-training.xml`, `data/Testing/*-ws-testing.xml`). **Environment:** **ambulatory wearable-style** multivariate streams—**continuous CGM** with **pump and meal–aligned** signals, 5-minute sampling, segments split at long gaps—typical of **outpatient Type 1 diabetes** monitoring.

## 2. Model + agent design

- **Predictor:** `GlucoseLSTM` — stacked LSTM + MLP head, trained with z-normalized windows; checkpoint stores `norm_mean` / `norm_std` for consistent inference.

- **Agent loop:** At each **test** time \(t\) (rollout on official Testing), observe lag window \(x_{t-L:t}\); the realized future glucose \(g_{t+H}\) is used **only** to score error and metrics, not as a policy input. In `plan_step`, rules are evaluated **top-to-bottom**; the **first matching row** fixes the action (`src/agent/policy.py`).

| Priority | Trigger | LSTM | MC dropout | Guideline |
|:--------:|---------|:----:|:----------:|:---------:|
| 1 | **Critical** last CGM (\(<80\) or \(>240\) mg/dL) | ✓ | ✓ | ✓ |
| 2 | IsolationForest **OOD** on the lag window | ✓ | ✓ | ✓ |
| 3 | **Volatile:** abs. mean of one-step glucose diffs on the **last 4** CGM points exceeds **1.4** mg/dL per step (default in `policy.py`) | ✓ | ✓ | — |
| 4 | **Reflection:** rolling MAE \(>\) adaptive `mae_trigger_mgdl` | ✓ | ✓ | — |
| 5 | **Periodic refresh:** cheap steps since last deep call \(>\) `deep_refresh_interval_steps` | ✓ | — | — |
| default | none of the above | no; **persistence** baseline (last CGM in window) | — | — |

Each step still runs the **lightweight** anomaly scorer for OOD. **Memory** keeps a deque of recent absolute errors; their mean is the **rolling MAE** for the policy. **`reflect()`** adjusts `mae_trigger_mgdl` (typically **lowers** it after sustained high MAE; can **relax** slightly when MAE is very low—see `src/agent/memory.py`).

Optional LLM policy can replace `plan_step` without changing tools.

## 3. Tool design and usage strategy

| Tool | Role | When called (examples) |
|------|------|------------------------|
| `LstmForecastTool` | Deep forecaster + optional MC dropout uncertainty | OOD, volatility, critical glucose, poor average error, periodic refresh |
| `MultivariateAnomalyTool` | IsolationForest on hand-crafted window stats | Every step (cheap); drives OOD escalation |
| `GuidelineRetrievalTool` | Rule-based “retrieval” over threshold bands | Only when policy enables guideline (priority 1 or 2); `query` uses step prediction |

**Non-trivial pattern:** the agent often skips the LSTM when the anomaly score is in-distribution and variability is low, but **forces** deep forecasting and uncertainty when the signal is erratic or historically miscalibrated—this matches the requirement to tie tool use to **uncertainty / distribution shift proxies**.

## 4. Results and analysis

**Setup:** Subject **540**; LSTM trained with `--holdout_subject 540` (11 other Training files; held-out test during training: `540-ws-testing.xml`). Agent: anomaly fit on `540-ws-training.xml`, rollout on `540-ws-testing.xml` (default `run_agent_eval.py` with `--ckpt artifacts/lstm.pt`). Numbers below are one **Testing** run; they vary with checkpoint, seeds, and device.

**Forecast quality (agent rollout):** Predictions are compared to realized future glucose at the configured horizon on every evaluated step. On this run:

| Metric | Value |
|--------|------:|
| RMSE | **25.08** mg/dL |
| MAE | **18.52** mg/dL |
| AUROC (future actual \(< 70\) mg/dL hypo event) | **0.959** |

The AUROC is well-defined here because the Testing segment contains both hypo and non-hypo future outcomes at \(t+H\). The score uses \(-\hat{g}\) vs. the binary label (see `run_agent_eval.py`).

**Agent behaviour (same run):** Tool-use rates summarize how often each branch fired:

| Quantity | Value |
|----------|------:|
| LSTM tool rate | **82.0%** |
| MC dropout rate | **81.9%** |
| Guideline retrieval rate | **26.7%** |
| OOD flag rate (IsolationForest) | **14.0%** |
| Adaptive MAE trigger at end of trajectory | **22.0** mg/dL |

**Interpretation:** ~**18%** of steps use the cheap baseline (non-trivial tool skipping). High LSTM use is consistent with volatility (priority 3) and periodic refresh; MC tracks most LSTM calls; guidelines only on **critical** / **OOD** branches.

If there are **no** future hypo events at \(t+H\), AUROC is **undefined** (`nan`); rates still depend on `policy.py` thresholds.

## 5. Web demo

**FastAPI** app: `demo_web/app.py`, static files in `demo_web/static/`, trajectory JSON via `src/demo_payload.py` (`GET /api/trajectory` → same rollout as `run_agent_eval.py`, default `artifacts/lstm.pt`, 404 if missing). Open the app, **load trajectory** for subject **540** (or another id in `data/Training`) to see the timeline, tool flags, rationale, and guideline snippets.

```bash
uvicorn demo_web.app:app --reload --host 127.0.0.1 --port 8000
```

Train/eval commands and data layout: **`README.md`**.

## 6. Limitations and future work

### 6.1 Clinical and deployment

This is a **course / research prototype**, not a regulated medical device or validated for insulin dosing, treatment changes, or emergency alerting. We do **not** evaluate fairness across demographic groups, sensors, or care settings. Any real deployment would need prospective studies, explicit human-in-the-loop protocols, safety monitoring, and compliance with local software-as-a-medical-device rules—none of which are in scope here.

### 6.2 Data and generalization

Pipelines target OhioT1DM-style `*-ws-training.xml` / `*-ws-testing.xml`; other CGM vendors, sampling rates, or missing-data patterns would require new loaders and retraining. Models trained on historical Ohio cohorts may **fail** under domain shift (new devices, geography, protocols). The split in §4 (LSTM on other subjects’ Training; anomaly on the same subject’s Training before Testing) limits obvious leakage but is **not** a substitute for independent external validation.

### 6.3 Evaluation and metrics

RMSE/MAE treat all errors symmetrically and do **not** encode clinical priorities (e.g., missing a severe hypo may be worse than a moderate error elsewhere). The hypo AUROC uses a fixed **70 mg/dL** threshold at \(t+H\) and predicted glucose as a score (`run_agent_eval.py`); it is one **screening-style** metric, not a full calibration or decision-curve analysis, and is **undefined** if there are no hypo positives. Reported numbers vary with MC dropout, IsolationForest subsampling, seeds, and hardware.

### 6.4 Agent and tools

The policy is **interpretable** but **hand-tuned**; it does not learn when to escalate from logged data. The “guideline” tool is **rule-based** over fixed bands, not a live clinical knowledge base. IsolationForest on engineered window statistics is a lightweight OOD signal, not a guarantee of robustness to adversarial or novel regimes. Natural extensions: **offline RL / imitation learning** on trajectories, or an **LLM planner** with strict tool JSON and guardrails—still subject to §6.1.

### 6.5 Models and uncertainty

`GlucoseLSTM` with MC dropout gives a **coarse** uncertainty proxy for routing; there is no full distributional calibration of multi-step forecasts. Richer sequence models (**Transformers**, patch-based encoders), **deep ensembles**, or **conformal / quantile** heads could better support **risk-aware** tool selection (when to trust the forecaster vs. the cheap baseline). Causal or mechanistic structure (insulin–glucose dynamics) is largely **implicit** in the LSTM and could be made explicit in future work.

## 7. Code map

- `environment.yml` — conda environment
- `src/data/dataset.py` — OhioT1DM XML → `GlucoseSeries` (gap-split, no cross-subject windows)
- `src/data/windows.py` — z-score on train segments + sliding-window `(X, y)` (no cross-segment windows)
- `src/models/lstm_predictor.py` — LSTM forecaster
- `src/tools/` — forecaster, anomaly, guideline tools
- `src/agent/` — memory, policy, loop
- `src/demo_payload.py` — trajectory JSON for the web UI (wraps `run_agent_on_train_test`)
- `demo_web/` — FastAPI + static HTML demo
- `scripts/train_lstm.py`, `scripts/run_agent_eval.py`

# Agentic AI for health time series

作业要求见 **`task.txt`**；设计与结果见 **`REPORT.md`**。

## 环境

```bash
conda env create -f environment.yml
conda activate agentic_medical_ai
```

在项目根目录执行脚本（脚本内会将根目录加入 `sys.path`）。

## 入口

```bash
python scripts/train_lstm.py --training_dir data/Training --testing_dir data/Testing --holdout_subject 540 --epochs 40 --out artifacts/lstm.pt
python scripts/run_agent_eval.py --train_dir data/Training --test_dir data/Testing --holdout_subject 540 --ckpt artifacts/lstm.pt
```

| | |
|--|--|
| Web 演示 | `uvicorn demo_web.app:app --reload --host 127.0.0.1 --port 8000` |

## 布局

- `src/data/` — OhioT1DM XML → `GlucoseSeries`  
- `src/models/` — `GlucoseLSTM`  
- `src/tools/` — 预测、异常、指南  
- `src/agent/` — 记忆、策略、循环  
- `scripts/` — `train_lstm.py`、`run_agent_eval.py`  
- `demo_web/` — FastAPI  

数据：`data/Training/`、`data/Testing/` 下的 `*-ws-*.xml`。

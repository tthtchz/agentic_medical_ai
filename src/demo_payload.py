"""Build JSON payloads for the HTML demo (no change to model/agent core logic)."""


from pathlib import Path

from src.agent.loop import AgentConfig, run_agent_on_train_test
from src.data.dataset import (
    concat_glucose_series,
    load_ohio_testing_subject,
    load_ohio_training_segments,
    ohio_subject_ids,
)


def run_demo_trajectory_for_subject(
    repo_root: Path,
    holdout_subject: str,
    ckpt: Path,
    lookback: int = 24,
    horizon: int = 6,
    max_test_steps: int | None = 2500,
) -> dict:
    train_dir = repo_root / "data" / "Training"
    test_dir = repo_root / "data" / "Testing"
    train_s = load_ohio_training_segments(train_dir, holdout_subject=holdout_subject)
    test_segs = load_ohio_testing_subject(test_dir, holdout_subject)
    test_s = concat_glucose_series(test_segs)
    cfg = AgentConfig(ckpt_path=Path(ckpt))
    traj, mem = run_agent_on_train_test(
        train_s,
        test_s,
        cfg,
        lookback,
        horizon,
        max_test_steps=max_test_steps,
    )
    g = test_s.glucose.astype(float).tolist()
    steps = [
        {
            "t": s.t_index,
            "predicted_glucose": s.predicted_glucose,
            "actual_glucose": s.actual_glucose,
            "used_lstm": s.used_lstm,
            "used_mc": s.used_mc,
            "used_guideline": s.used_guideline,
            "anomaly_ood": s.anomaly_ood,
            "rationale": s.rationale,
            "guideline_snippet": s.guideline_snippet,
        }
        for s in traj
    ]
    return {
        "subject": holdout_subject,
        "lookback": lookback,
        "horizon": horizon,
        "glucose_mgdl": g,
        "steps": steps,
        "adaptive_mae_trigger_mgdl": mem.mae_trigger_mgdl,
    }


def list_demo_subjects(repo_root: Path) -> dict:
    td = repo_root / "data" / "Training"
    ids = ohio_subject_ids(td)
    return {
        "subject_ids": ids,
        "default_subject": ids[0] if ids else None,
    }

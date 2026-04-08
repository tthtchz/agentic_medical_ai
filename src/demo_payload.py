"""Build JSON payloads for the HTML demo (no change to model/agent core logic)."""


from pathlib import Path

from src.agent.loop import AgentConfig, run_agent_on_train_test
from src.data.dataset import (
    concat_glucose_series,
    load_ohio_testing_subject,
    load_ohio_training_subject,
    ohio_subject_ids,
)


def run_demo_trajectory_for_subject(
    repo_root: Path,
    subject: str,
    ckpt: Path,
    lookback: int = 24,
    horizon: int = 6,
    max_test_steps: int | None = 2500,
) -> dict:
    """
    Fit the anomaly tool on **this subject's** ``{id}-ws-training.xml`` windows only;
    run the agent on ``{id}-ws-testing.xml`` (same id).

    Uses pre-split Ohio files under ``data/Training`` and ``data/Testing``.
    """
    train_dir = repo_root / "data" / "Training"
    test_dir = repo_root / "data" / "Testing"
    train_s = load_ohio_training_subject(train_dir, subject)
    test_segs = load_ohio_testing_subject(test_dir, subject)
    test_s = concat_glucose_series(test_segs)
    cfg = AgentConfig(ckpt_path=Path(ckpt))
    traj, mem = run_agent_on_train_test(
        train_s,
        test_segs,
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
        "subject": subject,
        "lookback": lookback,
        "horizon": horizon,
        "glucose_mgdl": g,
        "steps": steps,
        "adaptive_mae_trigger_mgdl": mem.mae_trigger_mgdl,
    }


def list_demo_subjects(repo_root: Path) -> dict:
    td = repo_root / "data" / "Training"
    ids = ohio_subject_ids(td)
    preferred = "540"
    default = preferred if preferred in ids else (ids[0] if ids else None)
    return {
        "subject_ids": ids,
        "default_subject": default,
    }

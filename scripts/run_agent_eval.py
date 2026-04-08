#!/usr/bin/env python3
"""Evaluate agent on Ohio XML (default) or synthetic (dev only)."""


import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sklearn.metrics import roc_auc_score

from src.agent.loop import AgentConfig, run_agent_on_series, run_agent_on_train_test
from src.data.dataset import (
    load_ohio_testing_subject,
    load_ohio_training_segments,
    load_series,
    ohio_subject_ids,
)

DEFAULT_TRAIN = ROOT / "data" / "Training"
DEFAULT_TEST = ROOT / "data" / "Testing"


def _first_training_subject(training_dir: Path) -> str:
    ids = ohio_subject_ids(training_dir)
    if not ids:
        raise SystemExit(
            f"No *-ws-training.xml under {training_dir}. "
            "Place OhioT1DM data under data/Training and data/Testing, or use --synthetic."
        )
    return ids[0]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Default: same leave-one-out as training — 11 Training XML + 1 Testing XML. "
            "Use --synthetic for development without XML files."
        )
    )
    ap.add_argument(
        "--synthetic",
        action="store_true",
        help="Development only: single synthetic series (no Ohio files).",
    )
    ap.add_argument(
        "--pooled",
        action="store_true",
        help="Concatenate all XML under train_dir and test_dir (no per-subject holdout).",
    )
    ap.add_argument("--source", choices=["synthetic", "csv", "dir"], default="synthetic")
    ap.add_argument("--path", type=str, default=None)
    ap.add_argument(
        "--train_dir",
        type=str,
        default=str(DEFAULT_TRAIN),
        help=f"Ohio Training folder (default: {DEFAULT_TRAIN})",
    )
    ap.add_argument(
        "--test_dir",
        type=str,
        default=str(DEFAULT_TEST),
        help=f"Ohio Testing folder (default: {DEFAULT_TEST})",
    )
    ap.add_argument(
        "--holdout_subject",
        type=str,
        default=None,
        help="Holdout id for leave-one-out (default: first id under train_dir). Ignored with --pooled.",
    )
    ap.add_argument("--ckpt", type=str, default="artifacts/lstm.pt")
    ap.add_argument("--lookback", type=int, default=24)
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument(
        "--max_test_steps",
        type=int,
        default=None,
        help="Cap rollout length on the test segment.",
    )
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        print("Checkpoint missing; run scripts/train_lstm.py first:", ckpt)
        sys.exit(1)

    cfg = AgentConfig(ckpt_path=ckpt)
    holdout_used: str | None = None

    if args.synthetic:
        if args.pooled:
            raise SystemExit("Do not combine --synthetic with --pooled.")
        series = load_series(args.source, Path(args.path) if args.path else None)
        traj, mem = run_agent_on_series(series, cfg, args.lookback, args.horizon)
    elif args.pooled:
        train_s = load_series("dir", Path(args.train_dir))
        test_s = load_series("dir", Path(args.test_dir))
        traj, mem = run_agent_on_train_test(
            train_s,
            test_s,
            cfg,
            args.lookback,
            args.horizon,
            max_test_steps=args.max_test_steps,
        )
    else:
        td, tst = Path(args.train_dir), Path(args.test_dir)
        holdout_used = args.holdout_subject or _first_training_subject(td)
        train_s = load_ohio_training_segments(td, holdout_subject=holdout_used)
        test_s = load_ohio_testing_subject(tst, holdout_used)
        traj, mem = run_agent_on_train_test(
            train_s,
            test_s,
            cfg,
            args.lookback,
            args.horizon,
            max_test_steps=args.max_test_steps,
        )

    if not traj:
        print("No trajectory steps (series too short?)")
        sys.exit(1)

    pred = np.array([s.predicted_glucose for s in traj])
    act = np.array([s.actual_glucose for s in traj])
    rmse = float(np.sqrt(np.mean((pred - act) ** 2)))
    mae = float(np.mean(np.abs(pred - act)))

    y_hypo = (act < 70).astype(int)
    if y_hypo.sum() > 0 and (1 - y_hypo).sum() > 0:
        scores = -pred
        auroc = float(roc_auc_score(y_hypo, scores))
    else:
        auroc = float("nan")

    n = len(traj)
    lstm_rate = sum(s.used_lstm for s in traj) / n
    mc_rate = sum(s.used_mc for s in traj) / n
    guide_rate = sum(s.used_guideline for s in traj) / n
    ood_rate = sum(s.anomaly_ood for s in traj) / n

    if args.synthetic:
        split_note = "synthetic series (development)"
    elif args.pooled:
        split_note = "Ohio pooled: all XML concatenated in train_dir / test_dir"
    else:
        split_note = f"Ohio leave-one-out holdout={holdout_used}"

    print(f"--- metrics ({split_note}) ---")
    print(f"RMSE mg/dL: {rmse:.2f}")
    print(f"MAE mg/dL:  {mae:.2f}")
    print(f"AUROC hypo (70 mg/dL, future actual): {auroc:.3f}")
    print("--- agent behaviour ---")
    print(f"LSTM tool rate:      {lstm_rate:.2%}")
    print(f"MC dropout rate:     {mc_rate:.2%}")
    print(f"Guideline retrieval: {guide_rate:.2%}")
    print(f"OOD flag rate:       {ood_rate:.2%}")
    print(f"adaptive MAE trigger: {mem.mae_trigger_mgdl:.1f} mg/dL")

    sample = traj[:5] + traj[-3:]
    print("--- sample decisions ---")
    for s in sample:
        print(
            f"t={s.t_index} pred={s.predicted_glucose:.1f} act={s.actual_glucose:.1f} "
            f"LSTM={s.used_lstm} OOD={s.anomaly_ood} :: {s.rationale}"
        )


if __name__ == "__main__":
    main()

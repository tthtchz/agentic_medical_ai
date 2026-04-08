#!/usr/bin/env python3
"""Train GlucoseLSTM on OhioT1DM *-ws-*.xml."""


import argparse
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.dataset import (
    load_ohio_testing_subject,
    load_ohio_training_segments,
    ohio_subject_ids,
    time_split_segments,
)
from src.data.windows import (
    build_arrays_with_stats_segments,
    zscore_stats_segments,
)
from src.models.lstm_predictor import GlucoseLSTM

DEFAULT_TRAINING = ROOT / "data" / "Training"
DEFAULT_TESTING = ROOT / "data" / "Testing"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "OhioT1DM: 11 *-ws-training.xml (holdout excluded), CGM gap segments, per-segment train/val; "
            "windows never cross subjects or gaps. Z-score from train portions. Official *-ws-testing.xml "
            "only for final test RMSE (not for epoch selection)."
        )
    )
    ap.add_argument(
        "--training_dir",
        type=str,
        default=str(DEFAULT_TRAINING),
        help=f"Directory with *-ws-training.xml (default: {DEFAULT_TRAINING})",
    )
    ap.add_argument(
        "--testing_dir",
        type=str,
        default=str(DEFAULT_TESTING),
        help=f"Directory with *-ws-testing.xml (default: {DEFAULT_TESTING})",
    )
    ap.add_argument(
        "--holdout_subject",
        type=str,
        default="540",
        help="Leave-one-out subject id: train on all other *-ws-training.xml; test on this id's Testing XML (default 540).",
    )
    ap.add_argument(
        "--train_only",
        action="store_true",
        help="Use only Training XML (optional holdout exclusion) with 80/20 time split; no Testing XML.",
    )
    ap.add_argument(
        "--train_val_frac",
        type=float,
        default=0.8,
        help="Ohio: per contiguous segment, first fraction of time = train; rest = validation (default 0.8).",
    )
    ap.add_argument("--lookback", type=int, default=24)
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--out", type=str, default="artifacts/lstm.pt")
    ap.add_argument(
        "--save_npz",
        type=str,
        default=None,
        help="Optional path to save train/val arrays + norm stats (e.g. artifacts/ohio_xy.npz)",
    )
    ap.add_argument(
        "--continuous_sgd",
        action="store_true",
        help=(
            "Classic mode: each epoch continues SGD from the previous epoch's weights. "
            "Default is the opposite: each epoch reloads the best-so-far weights and retrains one epoch from there "
            "(fresh Adam each epoch)."
        ),
    )
    args = ap.parse_args()

    if not (0.5 < args.train_val_frac < 1.0):
        raise SystemExit("--train_val_frac must be between 0.5 and 1.0 (exclusive).")

    X_test: np.ndarray | None = None
    y_test: np.ndarray | None = None
    holdout_id: str | None = None

    if args.train_only:
        train_segs = load_ohio_training_segments(
            Path(args.training_dir),
            holdout_subject=args.holdout_subject,
        )
        mean, std = zscore_stats_segments(train_segs)
        X_np, y_np = build_arrays_with_stats_segments(
            train_segs, args.lookback, args.horizon, mean, std
        )
        norm = (mean, std)
        n = len(X_np)
        split = int(0.8 * n)
        X_tr, y_tr = X_np[:split], y_np[:split]
        X_va, y_va = X_np[split:], y_np[split:]
        split_note = f"Ohio Training XML only, window-level 80/20 ({n} windows)"
    else:
        td, tst = Path(args.training_dir), Path(args.testing_dir)
        holdout = args.holdout_subject
        holdout_id = str(holdout)
        all_sids = ohio_subject_ids(td)
        train_sids = sorted(s for s in all_sids if str(s) != str(holdout))
        print(
            f"Leave-one-out: holdout subject = {holdout}  "
            f"(does not appear in training concat; official test file: {holdout}-ws-testing.xml)"
        )
        print(f"Training segments from these {len(train_sids)} Training XML subjects: {', '.join(train_sids)}")
        train_segments_full = load_ohio_training_segments(td, holdout_subject=holdout)
        train_segs, val_segs = time_split_segments(
            train_segments_full,
            train_frac=args.train_val_frac,
            min_tail_steps=0,
        )
        mean, std = zscore_stats_segments(train_segs)
        X_tr, y_tr = build_arrays_with_stats_segments(
            train_segs, args.lookback, args.horizon, mean, std
        )
        X_va, y_va = build_arrays_with_stats_segments(
            val_segs, args.lookback, args.horizon, mean, std
        )
        norm = (mean, std)
        test_segs = load_ohio_testing_subject(tst, holdout)
        X_test, y_test = build_arrays_with_stats_segments(
            test_segs, args.lookback, args.horizon, mean, std
        )
        split_note = (
            f"Ohio: train windows={len(X_tr)} (per-segment time split, first {args.train_val_frac:.0%} of each segment), "
            f"val windows={len(X_va)}; z-score from train portions only. "
            f"Official test windows={len(X_test)} (held out; not used for epoch selection)."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    F = X_tr.shape[-1]
    model = GlucoseLSTM(
        n_features=F,
        hidden=args.hidden,
        horizon_steps=args.horizon,
    ).to(device)
    lr = 1e-3
    loss_fn = nn.MSELoss()

    ds_tr = TensorDataset(
        torch.from_numpy(X_tr).float(),
        torch.from_numpy(y_tr).float(),
    )
    dl = DataLoader(ds_tr, batch_size=args.batch, shuffle=True)

    X_va_t = torch.from_numpy(X_va).float().to(device)
    y_va_t = torch.from_numpy(y_va).float().to(device)

    best_val = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    print(split_note)
    if args.continuous_sgd:
        print(
            "Training mode: **continuous SGD** — each epoch continues from the previous epoch's weights.\n"
        )
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        print(
            "Training mode: **each epoch starts from best-so-far** — load best weights, fresh Adam, "
            "train one full pass; update best only if val improves. Next epoch always reloads best (not last epoch).\n"
        )

    for ep in range(args.epochs):
        if not args.continuous_sgd:
            model.load_state_dict(best_state)
            opt = torch.optim.Adam(model.parameters(), lr=lr)
        model.train()
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(X_va_t)
            val = float(loss_fn(pv, y_va_t).sqrt().item())

        improved = val < best_val
        if improved:
            best_val = val
            best_epoch = ep + 1
            best_state = copy.deepcopy(model.state_dict())

        extra = "  (improved best)" if improved else ""
        print(
            f"epoch {ep+1}/{args.epochs}  val_rmse_norm={val:.4f}  |  "
            f"best_val={best_val:.4f} @ epoch {best_epoch}{extra}"
        )

    state_to_save = best_state
    official_test_rmse: float | None = None
    if X_test is not None and y_test is not None:
        model.load_state_dict(state_to_save)
        model.eval()
        X_te_t = torch.from_numpy(X_test).float().to(device)
        y_te_t = torch.from_numpy(y_test).float().to(device)
        with torch.no_grad():
            pt = model(X_te_t)
            official_test_rmse = float(loss_fn(pt, y_te_t).sqrt().item())
        print(
            f"\nOfficial test (held-out subject {holdout_id} Testing XML, "
            f"same norm as train segment): test_rmse_norm={official_test_rmse:.4f}"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state": state_to_save,
        "horizon_steps": args.horizon,
        "lookback": args.lookback,
        "n_features": F,
        "norm_mean": norm[0] if norm else None,
        "norm_std": norm[1] if norm else None,
        "best_val_rmse_norm": best_val,
        "best_epoch": best_epoch,
        "official_test_rmse_norm": official_test_rmse,
    }
    torch.save(payload, out)

    if args.save_npz:
        npz_path = Path(args.save_npz)
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        save_kw: dict = {
            "X_train": X_tr,
            "y_train": y_tr,
            "X_val": X_va,
            "y_val": y_va,
            "norm_mean": norm[0],
            "norm_std": norm[1],
            "lookback": np.array([args.lookback]),
            "horizon": np.array([args.horizon]),
        }
        if X_test is not None:
            save_kw["X_official_test"] = X_test
            save_kw["y_official_test"] = y_test
        np.savez(npz_path, **save_kw)
        print(f"saved arrays: {npz_path}")

    mode = "continuous SGD" if args.continuous_sgd else "train-from-best each epoch"
    print(
        f"\nsaved {out}  [{mode}] validation best: epoch {best_epoch}, val_rmse_norm={best_val:.4f}"
    )


if __name__ == "__main__":
    main()

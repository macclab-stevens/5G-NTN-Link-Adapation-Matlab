import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import polars as pl
import xgboost as xgb


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def get_feature_list(df_cols: List[str]) -> List[str]:
    # Use only the specified context inputs plus MCS
    base = [
        "slot_percent",
        "slot",
        "ele_angle",
        "pathloss",
        "snr",
        "cqi",
        "window",
        "target_bler",
    ]
    feats = [c for c in base if c in df_cols]
    feats.append("mcs")  # include MCS for conditional pass modeling
    return feats


def train_test_split_idx(n: int, test_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = int(n * test_frac)
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return train_idx, test_idx


def try_train(params: dict, dtrain: xgb.DMatrix, dvalid: xgb.DMatrix, num_rounds: int, early_stopping: int) -> xgb.Booster:
    watchlist = [(dtrain, "train"), (dvalid, "valid")]
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=num_rounds,
        evals=watchlist,
        early_stopping_rounds=early_stopping,
        verbose_eval=50,
    )
    return bst


def main() -> None:
    ap = argparse.ArgumentParser(description="Train XGBoost to predict pass/fail and choose highest feasible MCS")
    ap.add_argument("--data", type=str, default="", help="(Optional) Combined Parquet features path")
    ap.add_argument("--train", type=str, default="features/train.parquet", help="Train Parquet features path")
    ap.add_argument("--test", type=str, default="features/test.parquet", help="Test Parquet features path")
    ap.add_argument("--output-dir", type=str, default="models", help="Directory to write model artifacts")
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Training device")
    ap.add_argument("--max-rows", type=int, default=2_000_000, help="Optional cap on number of rows for training")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--test-frac", type=float, default=0.2, help="Test fraction")
    ap.add_argument("--num-rounds", type=int, default=1500, help="Boosting rounds")
    ap.add_argument("--early-stopping", type=int, default=100, help="Early stopping rounds")
    ap.add_argument("--export-recs", action="store_true", help="Export per-row recommended MCS on test set")
    ap.add_argument("--threshold", type=float, default=0.5, help="Minimum pass probability to accept an MCS")
    ap.add_argument("--calibrate-target", type=float, default=None, help="Target violation rate on validation (e.g., 0.1). If set, calibrates and saves threshold.")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    # Determine input mode (combined vs separate)
    if args.data:
        lf = pl.scan_parquet(args.data)
        cols = lf.columns
        feats = get_feature_list(cols)
        needed = feats + ["label_pass", "tbs"]
        lf = lf.select([c for c in needed if c in cols])
        lf = lf.filter(pl.all_horizontal([pl.col(c).is_not_null() for c in feats + ["label_pass"]]))
        if args.max_rows and args.max_rows > 0:
            lf = lf.head(args.max_rows)
        df = lf.collect(streaming=True)
        if df.height == 0:
            print("No data available after filtering")
            return
        X = np.column_stack([df[c].to_numpy().astype(np.float32, copy=False) for c in feats])
        y = df["label_pass"].to_numpy().astype(np.float32, copy=False)
        train_idx, test_idx = train_test_split_idx(df.height, args.test_frac, args.seed)
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
    else:
        lf_tr = pl.scan_parquet(args.train)
        lf_te = pl.scan_parquet(args.test)
        cols = lf_tr.columns
        feats = get_feature_list(cols)
        needed = feats + ["label_pass", "tbs"]
        lf_tr = lf_tr.select([c for c in needed if c in cols]).filter(pl.all_horizontal([pl.col(c).is_not_null() for c in feats + ["label_pass"]]))
        lf_te = lf_te.select([c for c in needed if c in lf_te.columns]).filter(pl.all_horizontal([pl.col(c).is_not_null() for c in feats + ["label_pass"]]))
        if args.max_rows and args.max_rows > 0:
            lf_tr = lf_tr.head(args.max_rows)
            lf_te = lf_te.head(max(args.max_rows // 4, 1))
        df_tr = lf_tr.collect(streaming=True)
        df_te = lf_te.collect(streaming=True)
        if df_tr.height == 0 or df_te.height == 0:
            print("Empty train or test after filtering")
            return
        X_train = np.column_stack([df_tr[c].to_numpy().astype(np.float32, copy=False) for c in feats])
        y_train = df_tr["label_pass"].to_numpy().astype(np.float32, copy=False)
        X_test = np.column_stack([df_te[c].to_numpy().astype(np.float32, copy=False) for c in feats])
        y_test = df_te["label_pass"].to_numpy().astype(np.float32, copy=False)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_test, label=y_test)

    # Params
    params = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss"],
        "max_depth": 8,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "max_bin": 256,
        "nthread": os.cpu_count() or 8,
        "tree_method": "hist",
    }

    # Device auto-detect: try CUDA if requested or auto
    tried_cuda = False
    if args.device in ("auto", "cuda"):
        try:
            params_cuda = params.copy()
            params_cuda.update({"device": "cuda", "tree_method": "hist"})
            bst = try_train(params_cuda, dtrain, dvalid, args.num_rounds, args.early_stopping)
            tried_cuda = True
        except Exception as e:
            print(f"CUDA training failed, falling back to CPU: {e}")
            bst = try_train(params, dtrain, dvalid, args.num_rounds, args.early_stopping)
    else:
        bst = try_train(params, dtrain, dvalid, args.num_rounds, args.early_stopping)

    # Save model and metadata
    model_path = out_dir / "xgb_mcs_pass.json"
    bst.save_model(model_path)

    meta = {
        "features": feats,
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "device_used": "cuda" if tried_cuda and "cuda" in bst.attributes().get("device", "") else "cpu",
        "best_iteration": bst.best_iteration if hasattr(bst, "best_iteration") else None,
    }
    (out_dir / "model_meta.json").write_text(json.dumps(meta, indent=2))

    # Simple evaluation on test set
    y_pred = bst.predict(dvalid)
    # metrics
    eps = 1e-9
    logloss = float(-(y_test * np.log(y_pred + eps) + (1 - y_test) * np.log(1 - y_pred + eps)).mean())
    acc = float(((y_pred >= 0.5) == (y_test >= 0.5)).mean())
    (out_dir / "metrics.json").write_text(json.dumps({"logloss": logloss, "accuracy": acc}, indent=2))
    print(f"Saved model to {model_path}, logloss={logloss:.4f}, acc={acc:.4f}")

    # Threshold calibration on validation (using actual MCS predictions)
    chosen_threshold = None
    if args.calibrate_target is not None and y_test is not None:
        targets = y_test
        preds = y_pred
        grid = np.linspace(0.99, 0.01, 99, dtype=np.float32)
        rows = []
        best_thr = None
        best_obj = (-1.0, 0.0)  # prioritize meeting constraint, then acceptance rate
        for thr in grid:
            accept = preds >= thr
            n_acc = int(accept.sum())
            if n_acc == 0:
                viol = 0.0
            else:
                viol = float(((targets[accept] < 0.5).sum()) / n_acc)
            acc_rate = float(n_acc / preds.shape[0])
            rows.append({"threshold": float(thr), "violation": viol, "accept_rate": acc_rate})
            # objective: violation <= target, maximize accept_rate; else minimize violation gap
            if viol <= args.calibrate_target:
                # pick highest threshold that meets target (stricter is safer) with highest acceptance
                score = (1.0, acc_rate)
            else:
                score = (0.0, -abs(viol - args.calibrate_target))
            if score > best_obj:
                best_obj = score
                best_thr = float(thr)
        chosen_threshold = best_thr
        # Save calibration curve
        pl.DataFrame(rows).write_csv(out_dir / "threshold_calibration.csv")
        print(f"Calibrated threshold: {chosen_threshold} for target violation {args.calibrate_target}")

    # Update meta with calibration/threshold
    meta_updated = json.loads((out_dir / "model_meta.json").read_text())
    if chosen_threshold is not None:
        meta_updated["threshold"] = float(chosen_threshold)
        meta_updated["calibrate_target"] = float(args.calibrate_target)
    else:
        meta_updated.setdefault("threshold", float(args.threshold))
    (out_dir / "model_meta.json").write_text(json.dumps(meta_updated, indent=2))

    # Optional: export recommendations on test set
    if args.export_recs:
        # Determine candidate MCS range from data
        mcs_vals = df["mcs"].to_numpy().astype(int, copy=False)
        m_min, m_max = int(mcs_vals.min()), int(mcs_vals.max())
        candidates = np.arange(m_min, m_max + 1, dtype=np.int32)

        base_feat_idx = [feats.index(c) for c in feats if c != "mcs"]
        mcs_pos = feats.index("mcs")

        # Build augmented matrices in batches to control memory
        batch = 20000
        preds_best = np.zeros(X_test.shape[0], dtype=np.float32)
        mcs_best = np.full(X_test.shape[0], m_min, dtype=np.int32)
        for start in range(0, X_test.shape[0], batch):
            end = min(start + batch, X_test.shape[0])
            Xb = X_test[start:end]
            best_prob = np.zeros(Xb.shape[0], dtype=np.float32)
            best_mcs = np.full(Xb.shape[0], m_min, dtype=np.int32)
            for m in candidates:
                Xa = Xb.copy()
                Xa[:, mcs_pos] = float(m)
                dp = xgb.DMatrix(Xa)
                pr = bst.predict(dp)
                # Accept if >= threshold, keep highest mcs that passes
                mask = pr >= args.threshold
                # Update where this candidate is accepted and is >= current best mcs
                update = mask & (m >= best_mcs)
                best_prob[update] = pr[update]
                best_mcs[update] = m
            preds_best[start:end] = best_prob
            mcs_best[start:end] = best_mcs

        rec_path = out_dir / "recommendations.csv"
        out_df = pl.DataFrame(
            {
                "actual_mcs": df["mcs"].to_numpy()[test_idx],
                "pred_mcs": mcs_best,
                "pred_prob": preds_best,
                "label_pass": y_test.astype(int),
                "tbs": df["tbs"].to_numpy()[test_idx] if "tbs" in df.columns else None,
            }
        )
        out_df.write_csv(rec_path)
        print(f"Wrote recommendations to {rec_path}")


if __name__ == "__main__":
    main()

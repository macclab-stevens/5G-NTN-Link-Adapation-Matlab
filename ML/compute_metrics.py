#!/usr/bin/env python3
"""
Compute basic error metrics and calibration for the pass‑probability model on
the test set. Also produces per‑slice metrics (e.g., by `cqi` or `snr_round`).

Metrics reported (predicted probability vs. observed label_pass):
  - MSE  (aka Brier score for probabilities)
  - RMSE
  - MAE
  - ECE  (expected calibration error, from reliability diagram)

Notes:
  - The model predicts p = P(pass | context, mcs). We compare p to the
    observed label_pass in the logs (0/1) for the actually used MCS.
  - MSE here is the Brier score, a standard metric for probabilistic
    classifiers. We focus on Brier/MAE and calibration quality (ECE + plots).

Outputs under --output-dir (default: reports/):
  - metrics_prob.json    : JSON with overall metrics above
  - metrics_by_<slice>.csv: Per‑slice metrics when --slice-by is provided
  - calibration_overall.csv/.png: Reliability curve (overall)
  - calibration_by_<slice>.csv/.png: Reliability curves per slice (top-k)
  - prob_predictions.csv : Optional CSV with columns [pred_prob, label_pass]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import polars as pl
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_features_list(model_meta_path: Path) -> List[str]:
    meta = json.loads(model_meta_path.read_text())
    return meta["features"]


def _metrics(y: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y
    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    return {"rows": int(y.shape[0]), "mae": mae, "mse": mse, "rmse": rmse}


def _calibration(y: np.ndarray, y_pred: np.ndarray, n_bins: int = 15, with_ci: bool = False) -> tuple[pl.DataFrame, float]:
    """Return reliability table and ECE.

    Table columns: bin_lower, bin_upper, pred_mean, true_rate, count
    ECE = sum(|true_rate - pred_mean| * count) / N
    """
    y_pred = np.clip(y_pred, 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(y_pred, bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    rows = []
    N = y.shape[0]
    ece_num = 0.0
    for b in range(n_bins):
        mask = idx == b
        cnt = int(mask.sum())
        if cnt == 0:
            pred_mean = 0.0
            true_rate = 0.0
        else:
            pred_mean = float(y_pred[mask].mean())
            true_rate = float(y[mask].mean())
        ece_num += abs(true_rate - pred_mean) * cnt
        rec = {
            "bin_lower": float(bins[b]),
            "bin_upper": float(bins[b + 1]),
            "pred_mean": pred_mean,
            "true_rate": true_rate,
            "count": cnt,
        }
        if with_ci:
            # 95% Wilson score interval for binomial proportion
            z = 1.96
            if cnt > 0:
                p = true_rate
                denom = 1.0 + (z * z) / cnt
                center = p + (z * z) / (2 * cnt)
                rad = z * np.sqrt((p * (1 - p)) / cnt + (z * z) / (4 * cnt * cnt))
                lo = max(0.0, (center - rad) / denom)
                hi = min(1.0, (center + rad) / denom)
            else:
                lo, hi = 0.0, 0.0
            rec["ci_lo"] = float(lo)
            rec["ci_hi"] = float(hi)
        rows.append(rec)
    ece = float(ece_num / N)
    return pl.DataFrame(rows), ece


def _plot_calibration(df: pl.DataFrame, title: str, out_png: Path, show_ci: bool = False) -> None:
    order = np.argsort(df["pred_mean"].to_numpy())
    x = df["pred_mean"].to_numpy()[order]
    y = df["true_rate"].to_numpy()[order]
    plt.figure(figsize=(5.2, 4.2))
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="ideal")
    plt.plot(x, y, marker="o", color="#1f77b4", label="calibration")
    if show_ci and "ci_lo" in df.columns and "ci_hi" in df.columns:
        lo = df["ci_lo"].to_numpy()[order]
        hi = df["ci_hi"].to_numpy()[order]
        plt.fill_between(x, lo, hi, color="#1f77b4", alpha=0.18, linewidth=0, label="95% CI")
    plt.xlabel("Predicted probability")
    plt.ylabel("Empirical pass rate")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute error metrics for probability predictions vs label_pass")
    ap.add_argument("--model", type=str, default="models/xgb_mcs_pass.json", help="Path to saved XGBoost model")
    ap.add_argument("--meta", type=str, default="models/model_meta.json", help="Path to model metadata JSON (contains feature order)")
    ap.add_argument("--test", type=str, default="features/test.parquet", help="Parquet with test rows (must contain label_pass)")
    ap.add_argument("--output-dir", type=str, default="reports", help="Directory to write outputs")
    ap.add_argument("--sample", type=int, default=200_000, help="Max rows from test set (0 = all)")
    ap.add_argument("--export-csv", action="store_true", help="Write per-row predictions to CSV")
    # Slicing & calibration
    ap.add_argument("--slice-by", type=str, default="cqi", help="Column to slice metrics by (e.g., 'cqi' or 'snr_round')")
    ap.add_argument("--max-slices", type=int, default=8, help="Max slices to include in per-slice plots")
    ap.add_argument("--min-slice-count", type=int, default=1000, help="Minimum rows per slice to include")
    ap.add_argument("--calibration-bins", type=int, default=15, help="Number of bins for calibration diagrams")
    ap.add_argument("--calibration-ci", action="store_true", help="Overlay 95% CI band for empirical pass rate in calibration plots")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    feats = load_features_list(Path(args.meta))

    # Load features in the correct order + label
    lf = pl.scan_parquet(args.test)
    have_cols = lf.columns
    need = [c for c in feats if c in have_cols]
    if "label_pass" not in have_cols:
        raise SystemExit("label_pass column not found in test data; re-run featurization/training.")
    sel = need + ["label_pass"]
    slice_col: Optional[str] = None
    if args.slice_by and args.slice_by in have_cols:
        slice_col = args.slice_by
        if slice_col not in sel:
            sel.append(slice_col)
    if args.sample and args.sample > 0:
        lf = lf.head(args.sample)
    df = lf.select(sel).collect(streaming=True)
    if df.height == 0:
        raise SystemExit("No rows available from test data")

    # Model + predictions
    X = np.column_stack([df[c].to_numpy().astype(np.float32, copy=False) for c in need])
    y = df["label_pass"].to_numpy().astype(np.float32, copy=False)
    dtest = xgb.DMatrix(X)
    bst = xgb.Booster()
    bst.load_model(args.model)
    y_pred = bst.predict(dtest).astype(np.float32)

    # Metrics (overall)
    metrics = _metrics(y, y_pred)

    # Calibration (overall)
    calib_df, ece = _calibration(y, y_pred, n_bins=args.calibration_bins, with_ci=args.calibration_ci)
    metrics["ece"] = float(ece)
    calib_df.write_csv(out_dir / "calibration_overall.csv")
    _plot_calibration(calib_df, "Calibration (overall)", out_dir / "calibration_overall.png", show_ci=args.calibration_ci)
    (out_dir / "metrics_prob.json").write_text(json.dumps(metrics, indent=2))

    # Per-slice metrics and calibration
    if slice_col is not None and slice_col in df.columns:
        # Determine most frequent slices
        counts = df[slice_col].value_counts().sort("count", descending=True)
        counts = counts.filter(pl.col("count") >= args.min_slice_count)
        top = counts.head(args.max_slices)
        records = []
        # Prepare plot of calibration per slice
        plt.figure(figsize=(6.8, 4.6))
        plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="ideal")
        for val in top[slice_col].to_list():
            mask = (df[slice_col] == val).to_numpy()
            cur_m = _metrics(y[mask], y_pred[mask])
            cur_m[slice_col] = float(val) if isinstance(val, (int, float)) else val
            records.append(cur_m)
            cal_df, _ = _calibration(y[mask], y_pred[mask], n_bins=max(8, args.calibration_bins // 2))
            plt.plot(cal_df["pred_mean"].to_numpy(), cal_df["true_rate"].to_numpy(), marker="o", linewidth=1.2, label=f"{slice_col}={val}")
        if records:
            pl.DataFrame(records).write_csv(out_dir / f"metrics_by_{slice_col}.csv")
            plt.xlabel("Predicted probability")
            plt.ylabel("Empirical pass rate")
            plt.title(f"Calibration by {slice_col}")
            plt.grid(True, alpha=0.3)
            plt.legend(loc="best", fontsize=8)
            plt.tight_layout()
            plt.savefig(out_dir / f"calibration_by_{slice_col}.png", dpi=150)
            plt.close()

    if args.export_csv:
        pl.DataFrame({"pred_prob": y_pred, "label_pass": y.astype(np.int32)}).write_csv(out_dir / "prob_predictions.csv")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

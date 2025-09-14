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


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate XGBoost model, export feature importance and interpretable outputs")
    ap.add_argument("--model", type=str, default="models/xgb_mcs_pass.json", help="Path to saved XGBoost model")
    ap.add_argument("--test", type=str, default="features/test.parquet", help="Path to test Parquet dataset")
    ap.add_argument("--meta", type=str, default="models/model_meta.json", help="Path to model metadata JSON")
    ap.add_argument("--output-dir", type=str, default="reports", help="Directory to write reports")
    ap.add_argument("--sample", type=int, default=50000, help="Max rows to load for evaluation")
    ap.add_argument("--shap-sample", type=int, default=10000, help="Rows to use for SHAP pred_contribs")
    ap.add_argument("--topk", type=int, default=20, help="Top K features for importance plot")
    # Tradeoff analysis
    ap.add_argument("--tradeoff", action="store_true", help="Compute throughput vs violation tradeoffs across thresholds")
    ap.add_argument("--slice-by", type=str, default="snr_round", help="Slice column (e.g., 'snr_round' or 'cqi') for per-slice tradeoffs")
    ap.add_argument("--min-slice-count", type=int, default=3000, help="Minimum rows per slice to include in plots")
    ap.add_argument("--max-slices", type=int, default=8, help="Max number of slices to plot (most frequent)")
    ap.add_argument("--grid-steps", type=int, default=99, help="Number of thresholds from 0.99..0.01 to evaluate")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    feats = load_features_list(Path(args.meta))

    # Load test data (include requested slice column if present). Avoid double scan by using schema.
    schema_names = pl.scan_parquet(args.test).collect_schema().names()
    sel = [c for c in feats + ["label_pass", "tbs"] if c in schema_names]
    if args.slice_by and args.slice_by in schema_names and args.slice_by not in sel:
        sel.append(args.slice_by)
    lf = pl.scan_parquet(args.test).select(sel)
    if args.sample and args.sample > 0:
        lf = lf.head(args.sample)
    df = lf.collect()
    # Derive common slice columns if requested but absent
    if args.slice_by == "snr_round" and "snr_round" not in df.columns and "snr" in df.columns:
        df = df.with_columns(pl.col("snr").round(0).alias("snr_round"))
    if df.height == 0:
        print("No test rows loaded.")
        return

    X = np.column_stack([df[c].to_numpy().astype(np.float32, copy=False) for c in feats])
    y = df["label_pass"].to_numpy().astype(np.float32, copy=False) if "label_pass" in df.columns else None
    tbs = df["tbs"].to_numpy().astype(np.float32, copy=False) if "tbs" in df.columns else None

    dtest = xgb.DMatrix(X, label=y)

    # Load model
    bst = xgb.Booster()
    bst.load_model(args.model)

    # Predictions
    y_pred = bst.predict(dtest)
    pred_path = out_dir / "predictions.csv"
    out_cols = {"pred_prob": y_pred}
    if y is not None:
        out_cols["label_pass"] = y.astype(np.int32)
    if "tbs" in df.columns:
        out_cols["tbs"] = df["tbs"].to_numpy()
    for i, f in enumerate(feats):
        out_cols[f] = X[:, i]
    pl.DataFrame(out_cols).write_csv(pred_path)

    # Basic metrics
    eps = 1e-9
    metrics = {}
    if y is not None:
        logloss = float(-(y * np.log(y_pred + eps) + (1 - y) * np.log(1 - y_pred + eps)).mean())
        acc = float(((y_pred >= 0.5) == (y >= 0.5)).mean())
        metrics.update({"logloss": logloss, "accuracy": acc})
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Feature importance
    fmap_gain = bst.get_score(importance_type="gain")
    fmap_weight = bst.get_score(importance_type="weight")
    # Map indices back to feature names (XGBoost uses f0, f1 ... if unset)
    # If the model stored real names, use them directly; else fallback to feats order
    def normalize_map(m):
        out = {}
        for k, v in m.items():
            if k.startswith("f") and k[1:].isdigit():
                idx = int(k[1:])
                name = feats[idx] if idx < len(feats) else k
            else:
                name = k
            out[name] = float(v)
        return out

    imp_gain = normalize_map(fmap_gain)
    imp_weight = normalize_map(fmap_weight)
    pl.DataFrame({"feature": list(imp_gain.keys()), "gain": list(imp_gain.values())}).sort("gain", descending=True).write_csv(out_dir / "feature_importance_gain.csv")
    pl.DataFrame({"feature": list(imp_weight.keys()), "weight": list(imp_weight.values())}).sort("weight", descending=True).write_csv(out_dir / "feature_importance_weight.csv")

    # Plot top-K by gain
    top = sorted(imp_gain.items(), key=lambda x: x[1], reverse=True)[: args.topk]
    if top:
        labels, vals = zip(*top)
        plt.figure(figsize=(max(6, len(labels) * 0.5), 4))
        plt.bar(range(len(labels)), vals, color="#4C78A8")
        plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
        plt.ylabel("Gain")
        plt.title("Feature importance (gain)")
        plt.tight_layout()
        plt.savefig(out_dir / "feature_importance_gain.png", dpi=130)
        plt.close()

    # SHAP contributions (pred_contribs) on a sample
    ns = min(args.shap_sample, X.shape[0])
    dtest_s = xgb.DMatrix(X[:ns])
    contribs = bst.predict(dtest_s, pred_contribs=True)
    # contribs shape: (ns, n_features + 1) last column is bias
    shap = np.abs(contribs[:, :-1])  # exclude bias
    shap_mean = shap.mean(axis=0)
    shap_df = pl.DataFrame({"feature": feats, "mean_abs_shap": shap_mean})
    shap_df.sort("mean_abs_shap", descending=True).write_csv(out_dir / "shap_summary.csv")

    # Per-sample top contributions (top 5 features)
    k = min(5, len(feats))
    records = []
    for i in range(ns):
        row = shap[i]
        idxs = np.argsort(-np.abs(row))[:k]
        for j in idxs:
            records.append({"row_index": i, "feature": feats[j], "shap_value": float(row[j])})
    pl.DataFrame(records).write_csv(out_dir / "shap_top_contribs.csv")

    # Tradeoff analysis (overall and by slice)
    if args.tradeoff:
        thresholds = np.linspace(0.99, 0.01, max(5, args.grid_steps), dtype=np.float32)

        def tradeoff_curves(y_pred: np.ndarray, y_true: Optional[np.ndarray], tbs_arr: Optional[np.ndarray]) -> pl.DataFrame:
            rows = []
            n = y_pred.shape[0]
            for thr in thresholds:
                acc_mask = y_pred >= thr
                n_acc = int(acc_mask.sum())
                acc_rate = float(n_acc / n)
                if n_acc == 0:
                    viol = 0.0
                    thr_obs_tput = 0.0
                    thr_exp_tput = 0.0
                else:
                    if y_true is not None:
                        viol = float(((y_true[acc_mask] < 0.5).sum()) / n_acc)
                    else:
                        viol = float((1.0 - y_pred[acc_mask]).mean())
                    if tbs_arr is not None:
                        # Observed throughput among accepted, counting only passes
                        if y_true is not None:
                            thr_obs_tput = float((tbs_arr[acc_mask] * (y_true[acc_mask] >= 0.5)).mean())
                        else:
                            thr_obs_tput = float((tbs_arr[acc_mask] * (y_pred[acc_mask] >= thr)).mean())
                        # Expected throughput among accepted
                        thr_exp_tput = float((tbs_arr[acc_mask] * y_pred[acc_mask]).mean())
                    else:
                        thr_obs_tput = 0.0
                        thr_exp_tput = 0.0
                rows.append({
                    "threshold": float(thr),
                    "accept_rate": acc_rate,
                    "violation": viol,
                    "obs_throughput": thr_obs_tput,
                    "exp_throughput": thr_exp_tput,
                })
            return pl.DataFrame(rows)

        # Overall tradeoff
        overall = tradeoff_curves(y_pred, y, tbs)
        overall.write_csv(out_dir / "tradeoff_overall.csv")

        def plot_xy(x, ys, labels, ylabel, fname):
            plt.figure(figsize=(7, 4))
            for yv, lab in zip(ys, labels):
                plt.plot(x, yv, label=lab)
            plt.xlabel("Threshold τ")
            plt.ylabel(ylabel)
            plt.grid(True, alpha=0.3)
            if len(labels) > 1:
                plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / fname, dpi=130)
            plt.close()

        plot_xy(overall["threshold"].to_numpy(),
                [overall["violation"].to_numpy()],
                ["overall"],
                "Violation rate",
                "tradeoff_overall_violation.png")
        plot_xy(overall["threshold"].to_numpy(),
                [overall["obs_throughput"].to_numpy(), overall["exp_throughput"].to_numpy()],
                ["observed", "expected"],
                "Throughput",
                "tradeoff_overall_throughput.png")

        # Per-slice tradeoff
        slice_col = args.slice_by
        if slice_col in df.columns:
            counts = df[slice_col].value_counts().sort("count", descending=True)
            counts = counts.filter(pl.col("count") >= args.min_slice_count)
            if counts.height > 0:
                top_slices = counts.head(args.max_slices)[slice_col].to_list()
                rows_all = []
                for s in top_slices:
                    mask = (df[slice_col] == s).to_numpy()
                    yps = y_pred[mask]
                    ys = y[mask] if y is not None else None
                    tb = tbs[mask] if tbs is not None else None
                    cur = tradeoff_curves(yps, ys, tb).with_columns(pl.lit(s).alias("slice"))
                    rows_all.append(cur)
                by_slice = pl.concat(rows_all, how="diagonal")
                by_slice.write_csv(out_dir / f"tradeoff_by_{slice_col}.csv")

                # Plots per slice
                thresh = by_slice["threshold"].unique().sort().to_list()
                xs = np.array(thresh, dtype=float)
                viol_series = []
                obs_tput_series = []
                labels = []
                for s in top_slices:
                    sub = by_slice.filter(pl.col("slice") == s).sort("threshold")
                    if sub.height == 0:
                        continue
                    labels.append(str(s))
                    viol_series.append(sub["violation"].to_numpy())
                    obs_tput_series.append(sub["obs_throughput"].to_numpy())
                if labels:
                    plot_xy(xs, viol_series, labels, f"Violation rate (by {slice_col})", f"tradeoff_violation_by_{slice_col}.png")
                    plot_xy(xs, obs_tput_series, labels, f"Observed throughput (by {slice_col})", f"tradeoff_throughput_by_{slice_col}.png")
            else:
                print(f"No slices met min count for {slice_col}; skipping slice plots")
        else:
            print(f"Slice column '{slice_col}' not in test data; skipping slice plots")

    print(f"Wrote predictions, metrics, and interpretation artifacts to {out_dir}")


if __name__ == "__main__":
    main()

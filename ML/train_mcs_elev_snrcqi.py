"""Train an XGBoost classifier mapping (elevation angle, SNR, CQI) to MCS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import xgboost as xgb


FEATURES = ["ele_angle", "snr", "cqi"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a 3-feature XGBoost MCS classifier")
    parser.add_argument("--data", type=Path, default=Path("features/all.parquet"), help="Input Parquet dataset with ele_angle, snr, cqi, mcs")
    parser.add_argument("--output-root", type=Path, default=Path("models/mcs_elev_snrcqi"), help="Directory to store model artifacts")
    parser.add_argument("--report-dir", type=Path, default=Path("reports/mcs_elev_snrcqi"), help="Directory to store evaluation plots")
    parser.add_argument("--test-frac", type=float, default=0.2, help="Fraction used for validation")
    parser.add_argument("--seed", type=int, default=52, help="Random seed")
    parser.add_argument("--max-depth", type=int, default=4, help="Max tree depth")
    parser.add_argument("--eta", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--rounds", type=int, default=300, help="Boosting rounds")
    parser.add_argument("--sample", type=int, default=1_000_000, help="Optional row cap (0=all)")
    parser.add_argument("--mcs-min", type=int, default=0, help="Minimum MCS to keep")
    parser.add_argument("--mcs-max", type=int, default=27, help="Maximum MCS to keep")
    parser.add_argument("--monotone", action="store_true", help="Apply monotone constraints (+snr,+cqi,0 for ele_angle)")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def train_test_split(features: np.ndarray, labels: np.ndarray, test_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = features.shape[0]
    order = rng.permutation(n)
    n_test = max(1, int(n * test_frac))
    test_idx = order[:n_test]
    train_idx = order[n_test:]
    return features[train_idx], features[test_idx], labels[train_idx], labels[test_idx]


def encode_labels(y: np.ndarray) -> Tuple[np.ndarray, Dict[int, int], List[int]]:
    classes = sorted(int(v) for v in np.unique(y))
    mapping = {c: i for i, c in enumerate(classes)}
    encoded = np.array([mapping[int(v)] for v in y], dtype=np.int32)
    return encoded, mapping, classes


def decode(pred_idx: np.ndarray, classes: List[int]) -> np.ndarray:
    arr = np.array(classes, dtype=np.int32)
    return arr[pred_idx]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, classes: List[int]) -> Dict[str, float]:
    acc = float(np.mean(y_true == y_pred))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    top2 = 0.0
    if y_proba.ndim == 2 and y_proba.shape[1] > 1:
        top2_idx = np.argsort(y_proba, axis=1)[:, -2:]
        class_arr = np.array(classes, dtype=np.int32)
        top2_classes = class_arr[top2_idx]
        matches = (top2_classes == y_true[:, None]).any(axis=1)
        top2 = float(np.mean(matches))
    return {"accuracy": acc, "mae": mae, "top2_accuracy": top2}


def main() -> None:
    args = parse_args()

    lf = pl.scan_parquet(args.data)
    lf = lf.select([*(c for c in FEATURES if c in lf.columns), "mcs"])
    lf = lf.filter(pl.all_horizontal([pl.col(c).is_not_null() for c in FEATURES + ["mcs"]]))
    if args.mcs_min is not None:
        lf = lf.filter(pl.col("mcs") >= args.mcs_min)
    if args.mcs_max is not None:
        lf = lf.filter(pl.col("mcs") <= args.mcs_max)
    if args.sample and args.sample > 0:
        lf = lf.head(args.sample)
    df = lf.collect()
    if df.height == 0:
        raise SystemExit("No rows available after filtering")

    X = np.column_stack([df[c].to_numpy().astype(np.float32, copy=False) for c in FEATURES])
    y_raw = df["mcs"].to_numpy().astype(np.int32, copy=False)

    y_encoded, mapping, classes = encode_labels(y_raw)
    X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, args.test_frac, args.seed)

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURES)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=FEATURES)

    monotone = "(0,1,1)" if args.monotone else "(0,0,0)"
    params = {
        "objective": "multi:softprob",
        "num_class": len(classes),
        "eval_metric": ["mlogloss"],
        "eta": args.eta,
        "max_depth": args.max_depth,
        "subsample": 0.9,
        "colsample_bytree": 1.0,
        "tree_method": "hist",
        "seed": args.seed,
        "monotone_constraints": monotone,
    }

    booster = xgb.train(params, dtrain, num_boost_round=args.rounds, evals=[(dtrain, "train"), (dval, "val")], verbose_eval=False)

    proba_val = booster.predict(dval)
    pred_idx = np.argmax(proba_val, axis=1)
    y_pred = decode(pred_idx, classes)
    y_true = decode(y_val, classes)

    metrics = compute_metrics(y_true, y_pred, proba_val, classes)

    ensure_dir(args.output_root)
    ensure_dir(args.report_dir)

    metrics_path = args.report_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    class_index_path = args.output_root / "class_index.json"
    with class_index_path.open("w", encoding="utf-8") as f:
        json.dump({"classes": classes}, f, indent=2)

    model_path = args.output_root / "model.json"
    booster.save_model(model_path)

    # Scatter plot: actual vs predicted for a sample
    rng = np.random.default_rng(args.seed)
    sample_idx = rng.choice(X_val.shape[0], size=min(5000, X_val.shape[0]), replace=False)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true[sample_idx], y_pred[sample_idx], alpha=0.3, s=10)
    ax.plot([classes[0], classes[-1]], [classes[0], classes[-1]], color="red", linewidth=1)
    ax.set_xlabel("True MCS")
    ax.set_ylabel("Predicted MCS")
    ax.set_title("True vs Predicted MCS")
    fig.tight_layout()
    fig.savefig(args.report_dir / "true_vs_pred.png", dpi=200)
    plt.close(fig)

    # Confusion matrix heatmap
    conf = np.zeros((len(classes), len(classes)), dtype=np.int32)
    for t, p in zip(y_true, y_pred, strict=False):
        ti = classes.index(int(t))
        pi = classes.index(int(p))
        conf[ti, pi] += 1
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(conf, origin="lower", cmap="Blues")
    ax.set_xticks(range(len(classes)), labels=[str(c) for c in classes])
    ax.set_yticks(range(len(classes)), labels=[str(c) for c in classes])
    ax.set_xlabel("Predicted MCS")
    ax.set_ylabel("True MCS")
    ax.set_title("Confusion Matrix")
    fig.colorbar(im, ax=ax, label="Count")
    fig.tight_layout()
    fig.savefig(args.report_dir / "confusion_matrix.png", dpi=200)
    plt.close(fig)

    print(f"Metrics written to {metrics_path}")
    print(f"Model saved to {model_path}")
    print(f"Plots saved under {args.report_dir}")


if __name__ == "__main__":
    main()

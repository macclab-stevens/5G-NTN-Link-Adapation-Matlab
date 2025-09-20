"""Train an XGBoost model that maps (SNR, CQI) to MCS indices capped at 15."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train monotonic XGBoost model for MCS prediction.")
    parser.add_argument("--lut", type=Path, default=Path("data/snr_cqi_lut_mcs15.csv"), help="Input LUT capped at desired MCS range.")
    parser.add_argument("--output-root", type=Path, default=Path("models/mcs15_xgb"), help="Directory for model artifacts.")
    parser.add_argument("--report-dir", type=Path, default=Path("reports/mcs15_xgb"), help="Directory for plots and metrics.")
    parser.add_argument("--test-frac", type=float, default=0.2, help="Fraction of data used for evaluation.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for train/test split.")
    parser.add_argument("--max-depth", type=int, default=4, help="Maximum tree depth.")
    parser.add_argument("--eta", type=float, default=0.1, help="Learning rate for boosting.")
    parser.add_argument("--rounds", type=int, default=300, help="Number of boosting rounds.")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def train_test_split(features: np.ndarray, labels: np.ndarray, test_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = features.shape[0]
    indices = rng.permutation(n)
    n_test = max(1, int(n * test_frac))
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    return features[train_idx], features[test_idx], labels[train_idx], labels[test_idx]


def encode_labels(y: np.ndarray) -> Tuple[np.ndarray, Dict[int, int], List[int]]:
    classes = sorted(int(v) for v in np.unique(y))
    to_idx = {c: i for i, c in enumerate(classes)}
    encoded = np.array([to_idx[int(v)] for v in y], dtype=np.int32)
    return encoded, to_idx, classes


def decode_predictions(pred_idx: np.ndarray, classes: List[int]) -> np.ndarray:
    class_arr = np.array(classes, dtype=np.int32)
    return class_arr[pred_idx]


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, classes: List[int]) -> Dict[str, float]:
    acc = float(np.mean(y_true == y_pred))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    # Top-2 accuracy
    top2 = 0.0
    if y_proba.ndim == 2 and y_proba.shape[1] >= 2:
        top2_idx = np.argsort(y_proba, axis=1)[:, -2:]
        class_arr = np.array(classes)
        top2_classes = class_arr[top2_idx]
        matches = (top2_classes == y_true[:, None]).any(axis=1)
        top2 = float(np.mean(matches))
    return {"accuracy": acc, "mae": mae, "top2_accuracy": top2}


def main() -> None:
    args = parse_args()

    if not args.lut.exists():
        raise FileNotFoundError(f"Cannot find LUT: {args.lut}")

    df = pd.read_csv(args.lut)
    for col in ("snr", "cqi", "mcs"):
        if col not in df.columns:
            raise ValueError(f"LUT {args.lut} missing required column '{col}'")

    df["cqi"] = df["cqi"].astype(int)
    df["mcs"] = df["mcs"].astype(int)

    features = df[["snr", "cqi"]].to_numpy(dtype=np.float32)
    labels = df["mcs"].to_numpy(dtype=np.int32)

    y_encoded, label_to_idx, classes = encode_labels(labels)

    X_train, X_test, y_train, y_test = train_test_split(features, y_encoded, args.test_frac, args.seed)

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=["snr", "cqi"])
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=["snr", "cqi"])

    monotone_constraints = "(1,1)"  # Increase in SNR or CQI should not lower the chosen MCS
    params = {
        "objective": "multi:softprob",
        "num_class": len(classes),
        "eval_metric": ["mlogloss"],
        "eta": args.eta,
        "max_depth": args.max_depth,
        "subsample": 0.9,
        "colsample_bytree": 1.0,
        "tree_method": "hist",
        "monotone_constraints": monotone_constraints,
        "seed": args.seed,
    }

    watchlist = [(dtrain, "train"), (dtest, "test")]
    booster = xgb.train(params, dtrain, num_boost_round=args.rounds, evals=watchlist, verbose_eval=False)

    proba_test = booster.predict(dtest)
    pred_idx = np.argmax(proba_test, axis=1)
    y_pred = decode_predictions(pred_idx, classes)
    y_true = decode_predictions(y_test, classes)

    metrics = classification_metrics(y_true, y_pred, proba_test, classes)

    ensure_dir(args.report_dir)
    ensure_dir(args.output_root)

    metrics_path = args.report_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    mapping_path = args.output_root / "class_index.json"
    with mapping_path.open("w", encoding="utf-8") as f:
        json.dump({"classes": classes}, f, indent=2)

    model_path = args.output_root / "model.json"
    booster.save_model(model_path)

    # Visualization prep
    snr_vals = np.sort(df["snr"].unique())
    cqi_vals = np.sort(df["cqi"].unique())

    pivot = df.pivot_table(values="mcs", index="cqi", columns="snr", aggfunc="mean")
    pivot = pivot.reindex(index=cqi_vals, columns=snr_vals)
    actual_grid = pivot.to_numpy()

    mesh_snr, mesh_cqi = np.meshgrid(snr_vals, cqi_vals)
    grid_features = np.column_stack([mesh_snr.ravel(), mesh_cqi.ravel()]).astype(np.float32)
    dgrid = xgb.DMatrix(grid_features, feature_names=["snr", "cqi"])
    grid_pred_idx = np.argmax(booster.predict(dgrid), axis=1)
    grid_pred = decode_predictions(grid_pred_idx, classes).reshape(mesh_snr.shape)

    # Plot 1: LUT vs Model heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    im0 = axes[0].pcolormesh(snr_vals, cqi_vals, actual_grid, shading="auto", cmap="viridis")
    axes[0].set_title("Filtered LUT MCS")
    axes[0].set_xlabel("SNR (dB)")
    axes[0].set_ylabel("CQI")
    fig.colorbar(im0, ax=axes[0], label="MCS")

    im1 = axes[1].pcolormesh(snr_vals, cqi_vals, grid_pred, shading="auto", cmap="viridis")
    axes[1].set_title("Model Predicted MCS")
    axes[1].set_xlabel("SNR (dB)")
    fig.colorbar(im1, ax=axes[1], label="MCS")
    heatmap_path = args.report_dir / "heatmap_comparison.png"
    fig.tight_layout()
    fig.savefig(heatmap_path, dpi=200)
    plt.close(fig)

    # Plot 2: Reliability curves per CQI subset
    fig_rel, ax_rel = plt.subplots(figsize=(8, 5))
    selected_cqi = [c for c in (0, 4, 8, 12, 15) if c in set(cqi_vals)]
    for cqi in selected_cqi:
        idx = np.where(cqi_vals == cqi)[0]
        if idx.size == 0:
            continue
        row = idx[0]
        ax_rel.plot(snr_vals, actual_grid[row, :], label=f"CQI {cqi} LUT", linewidth=1.8)
        ax_rel.plot(snr_vals, grid_pred[row, :], linestyle="--", label=f"CQI {cqi} Model", linewidth=1.8)
    ax_rel.set_xlabel("SNR (dB)")
    ax_rel.set_ylabel("MCS")
    ax_rel.set_title("Model vs LUT by CQI")
    ax_rel.legend(ncol=2, fontsize="small")
    fig_rel.tight_layout()
    reliability_path = args.report_dir / "reliability_curves.png"
    fig_rel.savefig(reliability_path, dpi=200)
    plt.close(fig_rel)

    # Plot 3: Confusion matrix heatmap (absolute counts)
    conf = np.zeros((len(classes), len(classes)), dtype=np.int32)
    for true_c, pred_c in zip(y_true, y_pred, strict=False):
        true_idx = classes.index(int(true_c))
        pred_idx_cls = classes.index(int(pred_c))
        conf[true_idx, pred_idx_cls] += 1

    fig_conf, ax_conf = plt.subplots(figsize=(6, 5))
    im_conf = ax_conf.imshow(conf, origin="lower", cmap="Blues")
    ax_conf.set_xticks(range(len(classes)), labels=[str(c) for c in classes])
    ax_conf.set_yticks(range(len(classes)), labels=[str(c) for c in classes])
    ax_conf.set_xlabel("Predicted MCS")
    ax_conf.set_ylabel("True MCS")
    ax_conf.set_title("Confusion Matrix")
    fig_conf.colorbar(im_conf, ax=ax_conf, label="Count")
    fig_conf.tight_layout()
    confusion_path = args.report_dir / "confusion_matrix.png"
    fig_conf.savefig(confusion_path, dpi=200)
    plt.close(fig_conf)

    print(f"Metrics written to {metrics_path}")
    print(f"Model saved to {model_path}")
    print(f"Plots saved under {args.report_dir}")


if __name__ == "__main__":
    main()

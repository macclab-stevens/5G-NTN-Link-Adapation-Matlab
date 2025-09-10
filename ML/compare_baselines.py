#!/usr/bin/env python3
"""
Baseline CQI mapping and OLLA comparison against the trained model.

This script simulates per-row decisions for several policies and compares them
on reliability and throughput. It uses the trained model as a counterfactual
oracle to estimate P(pass | context, MCS), and optionally samples outcomes to
approximate realized performance.

Policies:
  - model_threshold: highest MCS with P(pass) ≥ τ (τ from model_meta.json)
  - model_throughput: argmax_m P(pass) × spectral_efficiency(m)
  - baseline_cqi: round(CQI) clipped to [0, 27]
  - baseline_cqi_olla: like baseline_cqi but with a running offset (OLLA)

Outputs:
  - reports/baseline_summary.json: overall metrics per policy
  - reports/baseline_summary.png: bar chart (throughput, violation)
  - reports/baseline_vs_model_elev_blerw<window>.png: BLER & Avg MCS vs elevation
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import polars as pl
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from demo import MCSRecommender
from mcs_tables import spectral_efficiency


BASE_COLS = [
    "slot_percent","slot","ele_angle","pathloss","snr","cqi","window","target_bler",
]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def cqi_to_mcs(cqi: float) -> int:
    return int(max(0, min(27, round(float(cqi)))))


def choose_model_threshold(rec: MCSRecommender, ctx: Dict[str, float]) -> Tuple[int, float, float]:
    tau = float(rec.threshold)
    best = 0
    best_p = 0.0
    best_score = -1.0
    for m in range(28):
        p = rec.predict_pass_probability(ctx, m)
        if p >= tau and m >= best:
            best = m
            best_p = p
            best_score = p * spectral_efficiency(m)
        elif best_score < 0.0:
            # remember argmax p for fallback if nothing meets tau
            if p > best_p:
                best = m
                best_p = p
    if best_score < 0.0:
        best_score = best_p * spectral_efficiency(best)
    return best, best_p, best_score


def choose_model_throughput(rec: MCSRecommender, ctx: Dict[str, float]) -> Tuple[int, float, float]:
    best = 0
    best_p = 0.0
    best_score = 0.0
    for m in range(28):
        p = rec.predict_pass_probability(ctx, m)
        s = p * spectral_efficiency(m)
        if s > best_score:
            best = m
            best_p = p
            best_score = s
    return best, best_p, best_score


def simulate_policies(
    df: pl.DataFrame,
    rec: MCSRecommender,
    step: float,
    target_bler: float,
    rng: np.random.Generator,
    sample: bool,
) -> Dict[str, Dict[str, float]]:
    """Vectorized simulation using a single batch of model predictions.

    This avoids per-row x per-MCS calls which are extremely slow.
    """
    # Prepare base feature matrix in model feature order (without mcs column)
    feats = rec.features
    base_cols = [c for c in feats if c != "mcs"]
    n = df.height
    d = len(feats)
    Xbase = np.zeros((n, d), dtype=np.float32)
    for j, c in enumerate(feats):
        if c == "mcs":
            continue
        if c in df.columns:
            Xbase[:, j] = df[c].to_numpy().astype(np.float32, copy=False)
        else:
            Xbase[:, j] = 0.0

    # Predict probabilities for all MCS in one sweep
    mcs_pos = feats.index("mcs")
    preds = np.zeros((n, 28), dtype=np.float32)
    for m in range(28):
        Xa = Xbase.copy()
        Xa[:, mcs_pos] = float(m)
        dmat = xgb.DMatrix(Xa)
        preds[:, m] = rec.model.predict(dmat).astype(np.float32)

    eff = np.array([spectral_efficiency(m) for m in range(28)], dtype=np.float32)

    # Model threshold policy (with fallback to argmax p if none meet tau)
    tau = float(rec.threshold)
    m_idx = np.arange(28, dtype=np.int32)[None, :]
    mask = preds >= tau
    best = np.where(mask, m_idx, -1).max(axis=1)
    fallback = preds.argmax(axis=1)
    best = np.where(best < 0, fallback, best).astype(np.int32)
    p_best = preds[np.arange(n), best]
    s_best = p_best * eff[best]

    # Model throughput policy
    scores = preds * eff[None, :]
    tp_best = scores.argmax(axis=1).astype(np.int32)
    p_tp = preds[np.arange(n), tp_best]
    s_tp = scores[np.arange(n), tp_best]

    # Baseline CQI policy
    cqi_vals = df["cqi"].to_numpy().astype(np.float32, copy=False) if "cqi" in df.columns else np.full(n, 9.0, dtype=np.float32)
    m_cqi = np.clip(np.rint(cqi_vals), 0, 27).astype(np.int32)
    p_cqi = preds[np.arange(n), m_cqi]
    s_cqi = p_cqi * eff[m_cqi]

    # OLLA policy (sequential offset)
    delta = 0.0
    up = step * target_bler
    down = step * (1.0 - target_bler)
    m_olla = np.empty(n, dtype=np.int32)
    p_olla = np.empty(n, dtype=np.float32)
    for i in range(n):
        m = int(np.clip(round((cqi_vals[i] if i < cqi_vals.shape[0] else 9.0) + delta), 0, 27))
        m_olla[i] = m
        p = preds[i, m]
        p_olla[i] = p
        succ = (rng.random() < p) if sample else p >= (1.0 - target_bler)
        if succ:
            delta += up
        else:
            delta -= down
    s_olla = p_olla * eff[m_olla]

    def summarize(p: np.ndarray, s: np.ndarray, m: np.ndarray) -> Dict[str, float]:
        return {
            "accept_rate": 1.0,
            "violation": float(1.0 - p.mean()),
            "throughput_expected": float(s.mean()),
            "throughput_realized": float(s.mean()) if not sample else float((eff[m] * (rng.random(n) < p)).mean()),
            "avg_mcs": float(m.mean()),
        }

    return {
        "model_threshold": summarize(p_best, s_best, best),
        "model_throughput": summarize(p_tp, s_tp, tp_best),
        "baseline_cqi": summarize(p_cqi, s_cqi, m_cqi),
        "baseline_cqi_olla": summarize(p_olla, s_olla, m_olla),
    }


def upd(store: Dict[str, float], m: int, p: float, s: float, rng: np.random.Generator, sample: bool, ret_success: bool=False):
    store["n"] += 1
    store["mcs_avg"] += m
    store["throughput_exp"] += s
    if sample:
        succ = float(rng.random() < p)
    else:
        succ = p  # use expectation if not sampling
    store["succ"] += succ
    store["throughput_real"] += succ * spectral_efficiency(m)
    if ret_success:
        return bool(succ)


def elevation_series(
    df: pl.DataFrame,
    rec: MCSRecommender,
    step: float,
    target_bler: float,
    rng: np.random.Generator,
    sample: bool,
    bin_width: float,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    # Vectorized per-row policy decisions using one prediction matrix
    feats = rec.features
    n = df.height
    d = len(feats)
    Xbase = np.zeros((n, d), dtype=np.float32)
    for j, c in enumerate(feats):
        if c == "mcs":
            continue
        if c in df.columns:
            Xbase[:, j] = df[c].to_numpy().astype(np.float32, copy=False)
        else:
            Xbase[:, j] = 0.0

    mcs_pos = feats.index("mcs")
    preds = np.zeros((n, 28), dtype=np.float32)
    for m in range(28):
        Xa = Xbase.copy()
        Xa[:, mcs_pos] = float(m)
        preds[:, m] = rec.model.predict(xgb.DMatrix(Xa)).astype(np.float32)

    eff = np.array([spectral_efficiency(m) for m in range(28)], dtype=np.float32)
    tau = float(rec.threshold)

    # Model threshold
    m_idx = np.arange(28, dtype=np.int32)[None, :]
    mask = preds >= tau
    best = np.where(mask, m_idx, -1).max(axis=1)
    fallback = preds.argmax(axis=1)
    best = np.where(best < 0, fallback, best).astype(np.int32)
    p_tau = preds[np.arange(n), best]
    m_tau = best.astype(np.float32)

    # Model throughput
    scores = preds * eff[None, :]
    tp_best = scores.argmax(axis=1).astype(np.int32)
    p_tp = preds[np.arange(n), tp_best]
    m_tp = tp_best.astype(np.float32)

    # Baseline CQI
    cqi_vals = df["cqi"].to_numpy().astype(np.float32, copy=False) if "cqi" in df.columns else np.full(n, 9.0, dtype=np.float32)
    m_cqi = np.clip(np.rint(cqi_vals), 0, 27).astype(np.int32)
    p_cqi = preds[np.arange(n), m_cqi]

    # OLLA (sequential offset, look up p from preds)
    delta = 0.0
    up = step * target_bler
    down = step * (1.0 - target_bler)
    m_olla = np.empty(n, dtype=np.int32)
    p_olla = np.empty(n, dtype=np.float32)
    for i in range(n):
        m = int(np.clip(round((cqi_vals[i] if i < cqi_vals.shape[0] else 9.0) + delta), 0, 27))
        m_olla[i] = m
        p = preds[i, m]
        p_olla[i] = p
        succ = (rng.random() < p) if sample else p >= (1.0 - target_bler)
        if succ:
            delta += up
        else:
            delta -= down

    # Build DataFrame for binning
    ele = df["ele_angle"].to_numpy().astype(np.float32, copy=False) if "ele_angle" in df.columns else np.zeros(n, dtype=np.float32)
    plot_df = pl.DataFrame({
        "ele": ele,
        "p_model_tau": p_tau, "m_model_tau": m_tau,
        "p_model_tp": p_tp,  "m_model_tp": m_tp,
        "p_cqi": p_cqi,      "m_cqi": m_cqi.astype(np.float32),
        "p_olla": p_olla,    "m_olla": m_olla.astype(np.float32),
    }).with_columns(((pl.col("ele") / bin_width).floor() * bin_width).alias("ele_bin"))

    def agg(prefix: str) -> List[pl.Expr]:
        return [
            (1 - pl.col(f"p_{prefix}")).mean().alias(f"bler_{prefix}"),
            pl.col(f"m_{prefix}").mean().alias(f"mcs_{prefix}"),
        ]

    gb = plot_df.group_by("ele_bin").agg(
        agg("model_tau") + agg("model_tp") + agg("cqi") + agg("olla")
    ).sort("ele_bin")

    elev = gb["ele_bin"].to_numpy()
    bler = {k: gb[f"bler_{k}"].to_numpy() for k in ["model_tau","model_tp","cqi","olla"]}
    mcs  = {k: gb[f"mcs_{k}"].to_numpy() for k in ["model_tau","model_tp","cqi","olla"]}
    return elev, bler, mcs


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare CQI/OLLA baselines vs model")
    ap.add_argument("--data", type=str, default="features/test.parquet")
    ap.add_argument("--output-dir", type=str, default="reports")
    ap.add_argument("--sample", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--step", type=float, default=0.25, help="OLLA step (CQI units). Up=step×target, down=step×(1-target)")
    ap.add_argument("--target-bler", type=float, default=0.1, help="Target BLER for OLLA")
    ap.add_argument("--sample-outcomes", action="store_true", help="Sample success with Bernoulli(p) instead of using expectations")
    ap.add_argument("--bin-width", type=float, default=1.0, help="Elevation bin width (degrees)")
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "gpu"], help="Inference device for XGBoost model")
    args = ap.parse_args()

    out = Path(args.output_dir)
    ensure_dir(out)

    # Load data
    lf = pl.scan_parquet(args.data)
    cols = lf.columns
    need = [c for c in BASE_COLS if c in cols]
    if args.sample and args.sample > 0:
        lf = lf.head(args.sample)
    df = lf.select(need).collect(streaming=True)
    if df.height == 0:
        raise SystemExit("No rows available in input data")

    # Recommender/model
    # Recommender/model with device hint
    if args.device == "gpu":
        rec = MCSRecommender(device="cuda")
    elif args.device == "cpu":
        rec = MCSRecommender(device="cpu")
    else:
        # auto: try CUDA, fall back to CPU inside MCSRecommender
        rec = MCSRecommender()
        try:
            rec.set_device("cuda")
        except Exception:
            rec.set_device("cpu")
    rng = np.random.default_rng(args.seed)

    # Overall metrics
    summary = simulate_policies(df, rec, args.step, args.target_bler, rng, args.sample_outcomes)
    (out / "baseline_summary.json").write_text(json.dumps(summary, indent=2))

    # Summary bar plot
    labels = list(summary.keys())
    thr_exp = [summary[k]["throughput_expected"] for k in labels]
    viol = [summary[k]["violation"] for k in labels]
    fig, ax1 = plt.subplots(figsize=(9.0, 4.2))
    x = np.arange(len(labels))
    bars1 = ax1.bar(x - 0.2, thr_exp, width=0.4, label="Expected Throughput", color="#1f77b4")
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + 0.2, viol, width=0.4, label="Violation", color="#d62728")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right")
    ax1.set_xlabel("Policy")
    ax1.set_ylabel("Expected Throughput")
    ax2.set_ylabel("Violation Rate")
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.set_title("Policy Comparison: Throughput vs Violation")
    # Combined legend for both axes
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="upper center", ncols=2, framealpha=0.9, bbox_to_anchor=(0.5, 1.02))

    # Add value labels on bars
    def annotate(ax, bars, fmt="{:.2f}"):
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2.0, h, fmt.format(h),
                    ha='center', va='bottom', fontsize=8)
    annotate(ax1, bars1, fmt="{:.2f}")
    annotate(ax2, bars2, fmt="{:.2f}")

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "baseline_summary.png", dpi=160)
    plt.close(fig)

    # Elevation series (0..90) — note this still does a single pass over rows
    elev, bler, mcs = elevation_series(df, rec, args.step, args.target_bler, rng, args.sample_outcomes, args.bin_width)
    fig, (ax_t, ax_b) = plt.subplots(2, 1, figsize=(9.8, 6.0), sharex=True, gridspec_kw={"hspace": 0.08})
    colors = {
        "model_tau": "#2ca02c",
        "model_tp": "#9467bd",
        "cqi": "#8c564b",
        "olla": "#ff7f0e",
    }
    names = {
        "model_tau": "Model τ",
        "model_tp": "Model Throughput",
        "cqi": "Baseline CQI",
        "olla": "CQI + OLLA",
    }
    for k in ["model_tau","model_tp","cqi","olla"]:
        ax_t.plot(elev, bler[k], color=colors[k], label=names[k], linewidth=1.5)
        ax_b.plot(elev, mcs[k], color=colors[k], label=names[k], linewidth=1.5, linestyle='--')
    ax_t.set_ylabel("Avg BLER (1 − p)")
    ax_b.set_ylabel("Avg MCS")
    ax_b.set_xlabel("Elevation Angle")
    ax_t.grid(True, alpha=0.3)
    ax_b.grid(True, alpha=0.3)
    ax_t.set_xlim(0.0, 90.0)
    ax_b.set_xlim(0.0, 90.0)
    leg1 = ax_t.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), framealpha=0.9)
    leg2 = ax_b.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), framealpha=0.9)
    fig.subplots_adjust(right=0.78, top=0.92)
    fig.suptitle("Policies vs Elevation (Expected)")
    fig.savefig(out / "baseline_vs_model_elev.png", dpi=160)
    plt.close(fig)

    print(f"Wrote baseline comparison to {out}")


if __name__ == "__main__":
    main()

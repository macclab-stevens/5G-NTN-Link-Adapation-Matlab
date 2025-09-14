#!/usr/bin/env python3
"""
Run a full evaluation pipeline for the trained XGBoost pass-probability model
and produce a concise summary.

Steps:
  1) Evaluate + interpret: predictions, logloss/accuracy, feature importance, SHAP,
     and threshold tradeoffs (overall and by slice).
  2) Probability metrics + calibration: Brier/MAE/RMSE/ECE and reliability plots.
  3) Threshold/throughput sweeps: helper plots for acceptance/throughput Pareto.
  4) Summary: gather key metrics and (optionally) acceptance/violation at the
     model threshold into reports/summary.json and summary.txt.

Usage example:
  uv run python evaluate_all.py \
    --model models/xgb_mcs_pass.json --meta models/model_meta.json \
    --test features/test.parquet --output-dir reports \
    --slice-by cqi --sample 100000 --shap-sample 20000 --grid-steps 99
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import polars as pl


def run(cmd: list[str]) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def pick_row_at_threshold(df: pl.DataFrame, thr: float, thr_col: str = "threshold") -> Optional[dict]:
    if thr_col not in df.columns:
        return None
    vals = df[thr_col].to_numpy()
    if len(vals) == 0:
        return None
    import numpy as np
    i = int(np.argmin(np.abs(vals - thr)))
    if i < 0 or i >= len(vals):
        return None
    return {c: (float(df[c][i]) if df[c].dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64) else df[c][i]) for c in df.columns}


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate all: run model evaluation + metrics + plots and summarize")
    ap.add_argument("--model", type=str, default="models/xgb_mcs_pass.json")
    ap.add_argument("--meta", type=str, default="models/model_meta.json")
    ap.add_argument("--test", type=str, default="features/test.parquet")
    ap.add_argument("--output-dir", type=str, default="reports")
    ap.add_argument("--slice-by", type=str, default="snr_round")
    ap.add_argument("--sample", type=int, default=100_000)
    ap.add_argument("--shap-sample", type=int, default=20_000)
    ap.add_argument("--grid-steps", type=int, default=99)
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "gpu"], help="Device hint for helper plots")
    args = ap.parse_args()

    here = Path(__file__).parent.resolve()
    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    py = sys.executable or "python"

    # 1) Evaluate + interpret (+ tradeoff curves)
    run([
        py, str(here / "evaluate_xgb.py"),
        "--model", args.model,
        "--meta", args.meta,
        "--test", args.test,
        "--output-dir", args.output_dir,
        "--sample", str(args.sample),
        "--shap-sample", str(args.shap_sample),
        "--tradeoff",
        "--slice-by", args.slice_by,
        "--grid-steps", str(args.grid_steps),
    ])

    # 2) Probability metrics + calibration (overall and by slice)
    run([
        py, str(here / "compute_metrics.py"),
        "--model", args.model,
        "--meta", args.meta,
        "--test", args.test,
        "--output-dir", args.output_dir,
        "--slice-by", args.slice_by,
        "--calibration-ci",
    ])

    # 3) Helper sweeps/plots
    run([
        py, str(here / "plot_threshold_sweep.py"),
        "--test-data", args.test,
        "--device", args.device,
        "--grid", str(max(25, args.grid_steps // 4)),
    ])
    run([
        py, str(here / "plot_throughput_pareto.py"),
        "--test-data", args.test,
        "--device", args.device,
        "--grid", str(max(25, args.grid_steps // 4)),
    ])

    # 4) Summarize
    summary = {}

    # Base metrics
    metrics_path = out_dir / "metrics.json"
    if metrics_path.exists():
        try:
            summary.update({"predictive": json.loads(metrics_path.read_text())})
        except Exception:
            pass

    prob_metrics_path = out_dir / "metrics_prob.json"
    if prob_metrics_path.exists():
        try:
            summary.update({"probability": json.loads(prob_metrics_path.read_text())})
        except Exception:
            pass

    # Model meta + thresholded stats from tradeoff curves
    thr = 0.5
    meta = {}
    meta_path = Path(args.meta)
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            thr = float(meta.get("threshold", thr))
        except Exception:
            pass
    summary["meta"] = meta

    # Overall tradeoff at threshold
    tradeoff_overall = out_dir / "tradeoff_overall.csv"
    if tradeoff_overall.exists():
        try:
            df = pl.read_csv(tradeoff_overall)
            row = pick_row_at_threshold(df, thr)
            if row is None:
                row = {}
            summary["at_threshold"] = {
                "threshold": thr,
                "accept_rate": row.get("accept_rate"),
                "violation": row.get("violation"),
                "obs_throughput": row.get("obs_throughput"),
                "exp_throughput": row.get("exp_throughput"),
            }
        except Exception:
            pass

    # By-slice worst violation at threshold
    tradeoff_by_c = out_dir / f"tradeoff_by_{args.slice_by}.csv"
    if tradeoff_by_c.exists():
        try:
            df = pl.read_csv(tradeoff_by_c)
            # nearest threshold rows per slice
            slices = df["slice"].unique().to_list()
            worst = None
            for s in slices:
                sub = df.filter(pl.col("slice") == s).sort("threshold")
                row = pick_row_at_threshold(sub, thr)
                if row is None:
                    continue
                v = float(row.get("violation", 0.0))
                if (worst is None) or (v > worst["violation"]):
                    worst = {"slice": s, "violation": v, "accept_rate": float(row.get("accept_rate", 0.0))}
            if worst is not None:
                summary["worst_slice_at_threshold"] = worst
        except Exception:
            pass

    # Write summary files
    out_json = out_dir / "summary.json"
    out_txt = out_dir / "summary.txt"
    out_json.write_text(json.dumps(summary, indent=2))

    # Human-readable brief
    lines = []
    pred = summary.get("predictive", {})
    prob = summary.get("probability", {})
    at_thr = summary.get("at_threshold", {})
    worst = summary.get("worst_slice_at_threshold", {})
    lines.append(f"LogLoss={pred.get('logloss', 'N/A')}, Accuracy={pred.get('accuracy', 'N/A')}")
    if prob:
        lines.append(
            f"Brier(MSE)={prob.get('mse', prob.get('brier', 'N/A'))}, RMSE={prob.get('rmse', 'N/A')}, MAE={prob.get('mae', 'N/A')}, ECE={prob.get('ece', 'N/A')}"
        )
    if at_thr:
        lines.append(
            f"At τ={at_thr.get('threshold', 'N/A')}: accept_rate={at_thr.get('accept_rate', 'N/A')}, violation={at_thr.get('violation', 'N/A')}, exp_tput={at_thr.get('exp_throughput', 'N/A')}"
        )
    if worst:
        lines.append(
            f"Worst slice at τ: {worst.get('slice', 'N/A')} with violation={worst.get('violation', 'N/A')} (accept_rate={worst.get('accept_rate', 'N/A')})"
        )
    out_txt.write_text("\n".join(lines) + "\n")

    print(f"Wrote summary to {out_json} and {out_txt}")


if __name__ == "__main__":
    main()

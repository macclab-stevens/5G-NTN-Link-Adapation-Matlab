#!/usr/bin/env python3
"""
EDA for features/all.parquet

Computes summary statistics, basic distributions, correlations, and saves
figures + a JSON report. Designed to handle large files via Polars lazy I/O
with optional sampling.

Outputs (default ML/reports):
  - eda_summary.json
  - hist_snr.png, hist_cqi.png, hist_mcs.png, hist_bler.png
  - corr_heatmap.png (selected numeric features)
  - bler_vs_snr.png (scatter/hexbin on sample)
  - pass_rate_by_cqi.png, pass_rate_by_mcs.png (binned rates)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


NUMERIC_PREF = [
    "snr",
    "pathloss",
    "cqi",
    "mcs",
    "tbs",
    "tcr",
    "bler",
    "target_bler",
    "slot",
    "slot_percent",
    "ele_angle",
    "window",
    "blkerr",
    "label_pass",
    "snr_round",
    "snr_bin05",
    "snr_clip",
    "pathloss_round",
    "mod_code",
    "snr_cqi",
    "snr_pathloss",
]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def detect_numeric_cols(schema: dict[str, pl.DataType]) -> List[str]:
    numeric_cols: List[str] = []
    for name, dt in schema.items():
        if pl.datatypes.is_numeric(dt):
            numeric_cols.append(name)
    # prioritize known columns first, then the rest
    pref = [c for c in NUMERIC_PREF if c in numeric_cols]
    rest = [c for c in numeric_cols if c not in pref]
    return pref + rest


def summarize_columns(df: pl.DataFrame) -> Dict[str, Any]:
    n = df.height
    out: Dict[str, Any] = {"rows": int(n), "cols": int(df.width)}
    # Missing rates
    nulls = df.null_count()
    miss = {}
    for c in df.columns:
        try:
            # null_count returns a single-row Series per column
            miss[c] = float(pl.Series(nulls[c]).to_numpy()[0]) / max(1, n)
        except Exception:
            miss[c] = float("nan")
    out["missing_fraction"] = miss

    # Basic stats for numeric columns
    try:
        num_cols = df.select(pl.selectors.numeric()).columns
    except Exception:
        # Fallback: try to infer by dtypes name
        num_cols = [c for c in df.columns if str(df[c].dtype).lower() not in ("string", "str", "categorical", "object")] 
    stats: Dict[str, Dict[str, float]] = {}
    qtiles = [0.01, 0.05, 0.5, 0.95, 0.99]
    for c in num_cols:
        s = df[c]
        try:
            desc = {
                "count": float(s.len()),
                "mean": float(s.mean()),
                "std": float(s.std() if hasattr(s, "std") else np.nan),
                "min": float(s.min()),
                "max": float(s.max()),
            }
            for q in qtiles:
                desc[f"q{int(q*100)}"] = float(s.quantile(q, interpolation="nearest"))
        except Exception:
            desc = {"count": float(s.len())}
        stats[c] = desc
    out["numeric_stats"] = stats

    # Simple rates if present
    if "label_pass" in df.columns:
        out["pass_rate"] = float(df["label_pass"].mean())
    if "blkerr" in df.columns:
        out["ack_rate"] = float((df["blkerr"] == 1).mean())
    if "bler" in df.columns and "target_bler" in df.columns:
        out["violation_rate"] = float((df["bler"] > df["target_bler"]).mean())
    if "tbs" in df.columns and "blkerr" in df.columns:
        out["throughput_bytes_per_row_mean"] = float((df["tbs"] * (df["blkerr"] == 1)).mean())
    return out


def plot_hist(ax, data: np.ndarray, title: str, bins: int = 60, rng: tuple[float, float] | None = None):
    if rng is not None:
        ax.hist(np.clip(data, rng[0], rng[1]), bins=bins, color="#4c78a8", alpha=0.9)
    else:
        ax.hist(data, bins=bins, color="#4c78a8", alpha=0.9)
    ax.set_title(title)
    ax.grid(True, alpha=0.2)


def make_plots(df: pl.DataFrame, out_dir: Path) -> None:
    # Histograms
    figs = []
    if "snr" in df.columns:
        f, ax = plt.subplots(figsize=(6,4))
        plot_hist(ax, df["snr"].to_numpy(), "SNR distribution", bins=80, rng=(-20, 50))
        f.tight_layout(); f.savefig(out_dir / "hist_snr.png", dpi=150); plt.close(f)
        figs.append("hist_snr.png")
    if "cqi" in df.columns:
        f, ax = plt.subplots(figsize=(6,4))
        plot_hist(ax, df["cqi"].to_numpy(), "CQI distribution", bins=16, rng=(0, 15))
        f.tight_layout(); f.savefig(out_dir / "hist_cqi.png", dpi=150); plt.close(f)
        figs.append("hist_cqi.png")
    if "mcs" in df.columns:
        f, ax = plt.subplots(figsize=(6,4))
        plot_hist(ax, df["mcs"].to_numpy(), "MCS distribution", bins=28, rng=(0, 27))
        f.tight_layout(); f.savefig(out_dir / "hist_mcs.png", dpi=150); plt.close(f)
        figs.append("hist_mcs.png")
    if "bler" in df.columns:
        f, ax = plt.subplots(figsize=(6,4))
        plot_hist(ax, df["bler"].to_numpy(), "BLER distribution", bins=60, rng=(0, 1))
        f.tight_layout(); f.savefig(out_dir / "hist_bler.png", dpi=150); plt.close(f)
        figs.append("hist_bler.png")

    # Correlation heatmap for selected numeric columns
    sel = [c for c in [
        "label_pass","blkerr","bler","snr","cqi","mcs","tbs","tcr","pathloss","ele_angle","slot_percent","window","target_bler",
        "snr_cqi","snr_pathloss"
    ] if c in df.columns]
    if len(sel) >= 2:
        arr = df.select(sel).to_pandas()
        try:
            corr = arr.corr(numeric_only=True)
            f, ax = plt.subplots(figsize=(max(6, 0.5*len(sel)), max(5, 0.5*len(sel))))
            im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
            ax.set_xticks(range(len(sel))); ax.set_xticklabels(sel, rotation=90)
            ax.set_yticks(range(len(sel))); ax.set_yticklabels(sel)
            f.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title("Correlation heatmap")
            f.tight_layout(); f.savefig(out_dir / "corr_heatmap.png", dpi=150); plt.close(f)
        except Exception:
            pass

    # BLER vs SNR (sampled scatter/hexbin)
    if "snr" in df.columns and "bler" in df.columns:
        try:
            f, ax = plt.subplots(figsize=(6,4))
            x = df["snr"].to_numpy()
            y = np.clip(df["bler"].to_numpy(), 0, 1)
            hb = ax.hexbin(x, y, gridsize=60, cmap="viridis", bins="log")
            ax.set_xlabel("SNR (dB)"); ax.set_ylabel("BLER")
            ax.set_title("BLER vs SNR (hexbin)")
            f.colorbar(hb, ax=ax)
            f.tight_layout(); f.savefig(out_dir / "bler_vs_snr.png", dpi=150); plt.close(f)
        except Exception:
            pass

    # Pass rate by CQI / MCS (binned)
    if "label_pass" in df.columns and "cqi" in df.columns:
        grp = df.group_by("cqi").agg(pl.col("label_pass").mean().alias("pass_rate"))
        pdf = grp.sort("cqi").to_pandas()
        f, ax = plt.subplots(figsize=(6,4))
        ax.plot(pdf["cqi"], pdf["pass_rate"], marker="o")
        ax.set_xlabel("CQI"); ax.set_ylabel("Pass rate"); ax.set_ylim(0,1)
        ax.grid(True, alpha=0.2); ax.set_title("Pass rate by CQI")
        f.tight_layout(); f.savefig(out_dir / "pass_rate_by_cqi.png", dpi=150); plt.close(f)
    if "label_pass" in df.columns and "mcs" in df.columns:
        grp = df.group_by("mcs").agg(pl.col("label_pass").mean().alias("pass_rate"))
        pdf = grp.sort("mcs").to_pandas()
        f, ax = plt.subplots(figsize=(6,4))
        ax.plot(pdf["mcs"], pdf["pass_rate"], marker="o")
        ax.set_xlabel("MCS"); ax.set_ylabel("Pass rate"); ax.set_ylim(0,1)
        ax.grid(True, alpha=0.2); ax.set_title("Pass rate by MCS")
        f.tight_layout(); f.savefig(out_dir / "pass_rate_by_mcs.png", dpi=150); plt.close(f)


def main() -> int:
    ap = argparse.ArgumentParser(description="EDA for features/*.parquet")
    ap.add_argument("--path", type=str, default="ML/features/all.parquet", help="Input Parquet path")
    ap.add_argument("--out-dir", type=str, default="ML/reports", help="Output directory for figures & JSON")
    ap.add_argument("--sample", type=int, default=300_000, help="Rows to sample for EDA (0=all; caution large)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    in_path = Path(args.path)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    lf = pl.scan_parquet(in_path)
    # Total rows (fast)
    try:
        total_rows = int(lf.select(pl.len()).collect(streaming=True).item())
    except Exception:
        total_rows = -1

    # Decide sample
    df_sample: pl.DataFrame
    if args.sample and args.sample > 0:
        try:
            df_sample = lf.sample(n=args.sample, with_replacement=False, seed=args.seed).collect(streaming=True)
        except Exception:
            # Fallback: head
            df_sample = lf.head(args.sample).collect(streaming=True)
    else:
        df_sample = lf.collect(streaming=True)

    # Summaries
    summary = summarize_columns(df_sample)
    summary["path"] = str(in_path)
    summary["total_rows"] = total_rows
    summary["sample_rows"] = int(df_sample.height)

    # Save JSON
    (out_dir / "eda_summary.json").write_text(json.dumps(summary, indent=2))

    # Make plots
    make_plots(df_sample, out_dir)

    # Print concise console summary
    print("=== EDA Summary ===")
    print(f"file: {in_path}")
    print(f"rows: total={total_rows} sample={df_sample.height}")
    for k in ["pass_rate","ack_rate","violation_rate","throughput_bytes_per_row_mean"]:
        if k in summary:
            print(f"{k}: {summary[k]:.4f}")
    # Key ranges
    for c in ["snr","cqi","mcs","bler"]:
        if c in df_sample.columns:
            s = df_sample[c]
            try:
                vmin = float(s.min()); vmax = float(s.max())
                print(f"range {c}: {vmin:.4g} .. {vmax:.4g}")
            except Exception:
                pass
    print(f"Saved JSON and figures to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

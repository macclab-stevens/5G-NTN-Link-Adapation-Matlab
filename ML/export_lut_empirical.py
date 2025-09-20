#!/usr/bin/env python3
"""
Export a 2D SNR×CQI → MCS lookup table directly from logged data (no model).

Method per (snr_bin, cqi_bin):
  - Aggregate observed pass rate and throughput for each MCS used in the bin
  - threshold objective: choose highest MCS with pass_rate ≥ τ (± margin)
  - throughput objective: choose argmax observed throughput; optional guardrail on pass_rate

Notes and caveats:
  - This is observational (bandit) data. Only MCS values actually used in a bin
    can be evaluated. If the logging policy rarely tried higher MCS, this LUT will
    be conservative. Use --min-count to filter sparse MCS and --fill-cqi-grid to
    fill missing CQIs per SNR with nearest neighbor choice.
  - Observed throughput can be computed from TBS if present; otherwise a proxy
    based on spectral efficiency is used (pass_rate * spectral_efficiency).

Usage examples:
  uv run python export_lut_empirical.py \
    --data features/all.parquet --out data/snr_cqi_lut_empirical.csv \
    --objective threshold --threshold 0.9 --snr-bin 0.1 --cqi-bin 1 \
    --min-count 50 --fill-cqi-grid --cqi-min 0 --cqi-max 15

  uv run python export_lut_empirical.py \
    --data features/all.parquet --out data/snr_cqi_lut_empirical_tput.csv \
    --objective throughput --min-pass-guardrail 0.8 --snr-bin 0.1 --cqi-bin 1 \
    --min-count 50 --fill-cqi-grid --cqi-min 0 --cqi-max 15
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl

from mcs_tables import spectral_efficiency


def main() -> None:
    ap = argparse.ArgumentParser(description="Export empirical SNR×CQI→MCS LUT from logged data")
    ap.add_argument("--data", type=str, default="features/all.parquet")
    ap.add_argument("--out", type=str, default="data/snr_cqi_lut_empirical.csv")
    ap.add_argument("--objective", type=str, default="threshold", choices=["threshold", "throughput"], help="Selection objective per bin")
    ap.add_argument("--threshold", type=float, default=0.9, help="Pass-rate threshold for threshold objective")
    ap.add_argument("--min-pass-guardrail", type=float, default=-1.0, help="If >=0, restrict throughput objective to pass_rate ≥ guardrail")
    ap.add_argument(
        "--select-rule",
        type=str,
        default="highest_feasible",
        choices=["highest_feasible", "max_prob_feasible", "lowest_feasible"],
        help="Tie-breaker among feasible MCS (by index, by pass rate, or by lowest index)",
    )
    ap.add_argument("--prob-margin", type=float, default=0.0, help="Extra margin added to threshold/guardrail feasibility")
    ap.add_argument("--snr-bin", type=float, default=0.1)
    ap.add_argument("--cqi-bin", type=float, default=1.0)
    ap.add_argument("--min-count", type=int, default=50, help="Minimum rows for an (snr,cqi,mcs) cell to include")
    ap.add_argument("--fill-cqi-grid", action="store_true", help="Fill missing CQIs per SNR via nearest-neighbor CQI choice")
    ap.add_argument("--cqi-min", type=float, default=0.0)
    ap.add_argument("--cqi-max", type=float, default=15.0)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load columns
    lf = pl.scan_parquet(args.data)
    have = lf.columns
    need = ["snr","cqi","mcs","label_pass"]
    if not all(c in have for c in need):
        missing = [c for c in need if c not in have]
        raise SystemExit(f"Input missing required columns: {missing}. Re-run featurize.py to include them.")
    sel = ["snr","cqi","mcs","label_pass"]
    use_tbs = "tbs" in have
    if use_tbs:
        sel.append("tbs")

    df = lf.select(sel).collect(streaming=True)
    if df.height == 0:
        raise SystemExit("No rows available in input data")

    # Bin SNR/CQI
    snr_bin = float(args.snr_bin)
    cqi_bin = float(args.cqi_bin)
    df = df.with_columns([
        (pl.col("snr")/snr_bin).round(0).cast(pl.Int64).alias("snr_code"),
        (pl.col("cqi")/cqi_bin).round(0).cast(pl.Int64).alias("cqi_code"),
    ])

    # Aggregate per (snr,cqi,mcs)
    aggs = [
        pl.len().alias("n"),
        pl.col("label_pass").mean().alias("pass_rate"),
    ]
    if use_tbs:
        aggs.append((pl.col("tbs") * pl.col("label_pass").cast(pl.Float64)).mean().alias("obs_tput"))
    g = df.group_by(["snr_code","cqi_code","mcs"]).agg(aggs)
    g = g.filter(pl.col("n") >= args.min_count)
    if g.height == 0:
        raise SystemExit("No (snr,cqi,mcs) cells met min-count; lower --min-count or adjust bins")

    # If no TBS, compute throughput proxy via spectral efficiency
    if not use_tbs:
        # Map mcs→eff and compute expected throughput = pass_rate * eff
        m_arr = g["mcs"].to_numpy()
        eff = np.array([spectral_efficiency(int(m)) for m in m_arr], dtype=np.float64)
        g = g.with_columns((pl.col("pass_rate") * pl.Series("eff", eff)).alias("obs_tput"))

    # Pick a recommendation per (snr,cqi)
    # We'll do this in Python for clarity
    recs = []
    thr_req = float(args.threshold) + float(max(0.0, args.prob_margin))
    guard = float(args.min_pass_guardrail) + float(max(0.0, args.prob_margin)) if args.min_pass_guardrail is not None and args.min_pass_guardrail >= 0 else None
    gdf = g.sort(["snr_code","cqi_code","mcs"]).to_pandas()
    import pandas as pd
    for (s, c), sub in gdf.groupby(["snr_code","cqi_code"], sort=False):
        # sub has columns: n, pass_rate, obs_tput, mcs
        if args.objective == "throughput":
            if guard is not None:
                feas = sub[sub["pass_rate"] >= guard]
                if not feas.empty:
                    if args.select_rule == "max_prob_feasible":
                        m = int(feas.loc[feas["pass_rate"].idxmax(), "mcs"])
                    elif args.select_rule == "lowest_feasible":
                        m = int(feas["mcs"].min())
                    else:
                        m = int(feas.loc[feas["obs_tput"].idxmax(), "mcs"])
                else:
                    m = int(sub.loc[sub["pass_rate"].idxmax(), "mcs"])  # fallback: max pass rate
            else:
                m = int(sub.loc[sub["obs_tput"].idxmax(), "mcs"])  # pure throughput
        else:
            feas = sub[sub["pass_rate"] >= thr_req]
            if not feas.empty:
                if args.select_rule == "max_prob_feasible":
                    m = int(feas.loc[feas["pass_rate"].idxmax(), "mcs"])  # most reliable among feasible
                elif args.select_rule == "lowest_feasible":
                    m = int(feas["mcs"].min())
                else:
                    m = int(feas["mcs"].max())  # highest index among feasible
            else:
                m = int(sub.loc[sub["pass_rate"].idxmax(), "mcs"])  # fallback
        recs.append({
            "snr": float(s) * snr_bin,
            "cqi": float(c) * cqi_bin,
            "mcs": m,
        })

    # Optional: fill CQI grid per SNR by nearest neighbor
    if args.fill_cqi_grid and recs:
        # group by snr, nn fill cqi in [cqi_min, cqi_max]
        from collections import defaultdict
        by_snr: dict[float, dict[float, int]] = defaultdict(dict)
        for r in recs:
            by_snr[round(r["snr"], 6)][round(r["cqi"], 6)] = int(r["mcs"])
        filled = []
        cmin = args.cqi_min
        cmax = args.cqi_max
        step = cqi_bin
        def frange(a,b,step):
            n = int(round((b - a)/step))
            return [round(a + i*step, 10) for i in range(n+1)]
        for s, cmap in by_snr.items():
            if not cmap:
                continue
            c_present = np.array(sorted(cmap.keys()), dtype=np.float64)
            for c in frange(cmin, cmax, step):
                if c in cmap:
                    continue
                pos = np.searchsorted(c_present, c)
                pos = np.clip(pos, 0, len(c_present)-1)
                left = max(pos-1, 0)
                right = pos
                dl = abs(c - c_present[left])
                dr = abs(c_present[right] - c)
                choose = c_present[right] if dr < dl else c_present[left]
                filled.append({"snr": s, "cqi": c, "mcs": int(cmap[float(choose)])})
        if filled:
            # merge + dedup
            all_rows = recs + filled
            seen = set()
            dedup = []
            for r in all_rows:
                key = (round(r["snr"], 6), round(r["cqi"], 6))
                if key in seen:
                    continue
                seen.add(key)
                dedup.append(r)
            recs = dedup

    pl.DataFrame(recs).sort(["snr","cqi"]).write_csv(out_path)
    print(f"Wrote empirical LUT with {len(recs)} entries to {out_path}")


if __name__ == "__main__":
    main()

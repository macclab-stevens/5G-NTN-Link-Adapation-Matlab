#!/usr/bin/env python3
"""
Export a 2D SNR×CQI → MCS lookup table derived from a trained model and a
dataset. The LUT can then be evaluated against the model using
compare_baselines.py via --snr-cqi-lut.

Strategy per (snr_bin, cqi_bin):
  - Compute average P(pass | context, mcs) across rows within the bin
  - Objective=throughput: choose argmax_m E[P(pass)] * spectral_efficiency(m)
  - Objective=threshold: choose highest m with E[P(pass)] ≥ τ; fallback to argmax E[P(pass)]

Usage:
  uv run python export_lut.py \
    --data features/test.parquet --out data/snr_cqi_lut.csv \
    --objective throughput --snr-bin 0.1 --cqi-bin 1 --sample 200000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
import xgboost as xgb

from mcs_tables import spectral_efficiency


def load_features_list(model_meta_path: Path) -> list[str]:
    meta = json.loads(model_meta_path.read_text())
    return meta["features"]


def batch_predict_all_mcs(df: pl.DataFrame, feats: list[str], bst: xgb.Booster) -> np.ndarray:
    """Predict P(pass) for all rows and all 0..27 MCS in one batch.

    Returns: preds (n_rows, 28)
    """
    n = df.height
    d = len(feats)
    # Base matrix in feature order
    Xbase = np.zeros((n, d), dtype=np.float32)
    mcs_pos = feats.index("mcs")
    for j, c in enumerate(feats):
        if c == "mcs":
            continue
        if c in df.columns:
            Xbase[:, j] = df[c].to_numpy().astype(np.float32, copy=False)
        else:
            Xbase[:, j] = 0.0
    Xall = np.repeat(Xbase, 28, axis=0)
    Xall[:, mcs_pos] = np.tile(np.arange(28, dtype=np.float32), n)
    dmat = xgb.DMatrix(Xall)
    y_all = bst.predict(dmat).astype(np.float32)
    return y_all.reshape(n, 28)


def main() -> None:
    ap = argparse.ArgumentParser(description="Export SNR×CQI→MCS LUT from model and data")
    ap.add_argument("--data", type=str, default="features/test.parquet")
    ap.add_argument("--out", type=str, default="data/snr_cqi_lut.csv")
    ap.add_argument("--model", type=str, default="models/xgb_mcs_pass.json")
    ap.add_argument("--meta", type=str, default="models/model_meta.json")
    ap.add_argument("--objective", type=str, default="threshold", choices=["threshold", "throughput"], help="LUT selection objective per (snr,cqi) bin")
    ap.add_argument("--threshold", type=float, default=-1.0, help="Threshold for threshold objective. If <0, use model meta or 0.5")
    ap.add_argument("--pass-quantile", type=float, default=-1.0, help="If >=0, aggregate P(pass) by this quantile (e.g., 0.1 for q10) instead of mean")
    ap.add_argument("--min-pass-guardrail", type=float, default=-1.0, help="If >=0, restrict candidates to P(pass)≥guardrail (mainly for throughput objective)")
    ap.add_argument("--select-rule", type=str, default="highest_feasible", choices=["highest_feasible", "max_prob_feasible", "lowest_feasible"], help="When multiple MCS meet feasibility, choose largest, highest P(pass), or lowest index")
    ap.add_argument("--prob-margin", type=float, default=0.0, help="Extra probability margin added to threshold/guardrail (e.g., 0.05 requires p ≥ τ+0.05)")
    ap.add_argument("--mcs-min", type=int, default=0, help="Minimum allowed MCS in LUT output")
    ap.add_argument("--mcs-max", type=int, default=27, help="Maximum allowed MCS in LUT output (hard cap)")
    ap.add_argument("--snr-bin", type=float, default=0.1)
    ap.add_argument("--cqi-bin", type=float, default=1.0)
    ap.add_argument("--min-count", type=int, default=500, help="Minimum rows per (snr,cqi) bin to include")
    ap.add_argument("--sample", type=int, default=200_000, help="Max rows to use (0=all)")
    ap.add_argument("--fill-cqi-grid", action="store_true", help="Fill missing CQI bins per SNR using nearest CQI neighbor (defines a dense CQI grid)")
    ap.add_argument("--cqi-min", type=float, default=0.0, help="Minimum CQI for grid fill (inclusive)")
    ap.add_argument("--cqi-max", type=float, default=15.0, help="Maximum CQI for grid fill (inclusive)")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    feats = load_features_list(Path(args.meta))
    lf = pl.scan_parquet(args.data)
    have = lf.columns
    need = [c for c in feats if c in have]
    # Ensure snr/cqi present for binning
    if "snr" not in have or "cqi" not in have:
        raise SystemExit("Input data must contain 'snr' and 'cqi' columns to export SNR×CQI LUT")
    sel = list(dict.fromkeys(need + ["snr", "cqi"]))
    if args.sample and args.sample > 0:
        lf = lf.head(args.sample)
    df = lf.select(sel).collect(streaming=True)
    if df.height == 0:
        raise SystemExit("No rows available in input data")

    # Model
    bst = xgb.Booster()
    bst.load_model(args.model)

    # Override/resolve threshold
    thr = args.threshold
    if thr < 0:
        try:
            meta = json.loads(Path(args.meta).read_text())
            thr = float(meta.get("threshold", 0.5))
        except Exception:
            thr = 0.5

    # Predictions for all MCS
    preds = batch_predict_all_mcs(df, feats, bst)  # shape: (n, 28)
    eff = np.array([spectral_efficiency(m) for m in range(28)], dtype=np.float32)

    # Bin snr and cqi
    snr_bin = float(args.snr_bin)
    cqi_bin = float(args.cqi_bin)
    snr_codes = np.rint(df["snr"].to_numpy() / snr_bin)
    cqi_codes = np.rint(df["cqi"].to_numpy() / cqi_bin)
    # Group rows by (snr_code, cqi_code) using numpy for compatibility
    sc = snr_codes.astype(np.int64)
    cc = cqi_codes.astype(np.int64)
    order = np.lexsort((cc, sc))
    sc_sorted = sc[order]
    cc_sorted = cc[order]
    # Find boundaries where (snr,cqi) changes
    change = (np.diff(sc_sorted) != 0) | (np.diff(cc_sorted) != 0)
    starts = np.concatenate(([0], np.nonzero(change)[0] + 1))
    ends = np.concatenate((starts[1:], [sc_sorted.shape[0]]))
    records = []
    for a, b in zip(starts, ends):
        idx = order[a:b]
        if idx.size < args.min_count:
            continue
        # Aggregate per-MCS probability across rows in this bin
        if args.pass_quantile is not None and args.pass_quantile >= 0.0:
            p_stat = np.quantile(preds[idx], float(args.pass_quantile), axis=0)
        else:
            p_stat = preds[idx].mean(axis=0)
        if args.objective == "throughput":
            # Optional guardrail: only consider candidates meeting min-pass
            if args.min_pass_guardrail is not None and args.min_pass_guardrail >= 0.0:
                guard = float(args.min_pass_guardrail) + float(max(0.0, args.prob_margin))
                feas = np.where(p_stat >= guard)[0]
                if feas.size > 0:
                    score = p_stat * eff
                    if args.select_rule == "max_prob_feasible":
                        m = int(feas[np.argmax(p_stat[feas])])
                    elif args.select_rule == "lowest_feasible":
                        m = int(feas.min())
                    else:
                        m = int(feas[np.argmax(score[feas])])
                else:
                    # Fallback: pick best by P(pass)
                    m = int(np.argmax(p_stat))
            else:
                score = p_stat * eff
                m = int(np.argmax(score))
        else:
            req = float(thr) + float(max(0.0, args.prob_margin))
            feas = np.where(p_stat >= req)[0]
            if feas.size > 0:
                if args.select_rule == "max_prob_feasible":
                    m = int(feas[np.argmax(p_stat[feas])])
                elif args.select_rule == "lowest_feasible":
                    m = int(feas.min())
                else:
                    m = int(feas.max())
            else:
                m = int(np.argmax(p_stat))
        # Clamp to allowed range
        m = int(max(args.mcs_min, min(args.mcs_max, m)))
        records.append({
            "snr": float(sc_sorted[a]) * snr_bin,
            "cqi": float(cc_sorted[a]) * cqi_bin,
            "mcs": int(m),
        })

    # Optional: fill CQI grid per SNR using nearest-neighbor along CQI
    if args.fill_cqi_grid and records:
        # Build map snr_code -> {cqi_code: mcs}
        snr_code_map: dict[int, dict[int, int]] = {}
        for r in records:
            s_code = int(round(r["snr"] / snr_bin))
            c_code = int(round(r["cqi"] / cqi_bin))
            snr_code_map.setdefault(s_code, {})[c_code] = int(r["mcs"])
        cmin_code = int(round(args.cqi_min / cqi_bin))
        cmax_code = int(round(args.cqi_max / cqi_bin))
        filled = []
        for s_code, cmap in snr_code_map.items():
            if not cmap:
                continue
            present = np.array(sorted(cmap.keys()), dtype=np.int64)
            for c_code in range(cmin_code, cmax_code + 1):
                if c_code in cmap:
                    continue
                # nearest present CQI code
                pos = np.searchsorted(present, c_code)
                pos = np.clip(pos, 0, present.size - 1)
                left = max(pos - 1, 0)
                right = pos
                dl = abs(c_code - int(present[left]))
                dr = abs(int(present[right]) - c_code)
                choose = int(present[right]) if dr < dl else int(present[left])
                mcs_fill = cmap[choose]
                filled.append({
                    "snr": float(s_code) * snr_bin,
                    "cqi": float(c_code) * cqi_bin,
                    "mcs": int(mcs_fill),
                })
        if filled:
            # Merge and drop duplicates (if any)
            all_rows = records + filled
            # Deduplicate by keys
            seen = set()
            dedup = []
            for r in all_rows:
                key = (round(r["snr"], 6), round(r["cqi"], 6))
                if key in seen:
                    continue
                seen.add(key)
                dedup.append(r)
            records = dedup

    if not records:
        raise SystemExit("No bins met min-count; lower --min-count or adjust bins")

    pl.DataFrame(records).sort(["snr", "cqi"]).write_csv(out_path)
    print(f"Wrote LUT with {len(records)} entries to {out_path}")


if __name__ == "__main__":
    main()

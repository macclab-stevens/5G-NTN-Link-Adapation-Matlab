#!/usr/bin/env python3
"""
Threshold sweep panels vs τ:
- Acceptance rate
- Expected throughput among accepted
- Aggregate throughput (accept_rate x expected among accepted)

Saves a 3-panel figure and CSV.
"""

import argparse
from pathlib import Path
from typing import List

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


def load_contexts(path: str, n: int) -> pl.DataFrame:
    df = pl.read_parquet(path)
    if n and n < df.height:
        df = df.sample(n=n, seed=42)
    return df.select([c for c in BASE_COLS if c in df.columns])


def batch_predict(rec: MCSRecommender, ctx_df: pl.DataFrame, mcs_values: List[int], batch: int = 5000) -> np.ndarray:
    feats = rec.features
    base_order = [feats.index(c) for c in BASE_COLS]
    mcs_pos = feats.index("mcs")
    Xb = np.column_stack([ctx_df[c].to_numpy().astype(np.float32, copy=False) for c in BASE_COLS])
    n = Xb.shape[0]
    d = len(feats)
    out = np.zeros((n, len(mcs_values)), dtype=np.float32)
    Xa = np.zeros((min(batch, n), d), dtype=np.float32)
    for start in range(0, n, batch):
        end = min(start + batch, n)
        rows = end - start
        Xa[:rows, :] = 0.0
        for src, dst in enumerate(base_order):
            Xa[:rows, dst] = Xb[start:end, src]
        for j, m in enumerate(mcs_values):
            Xa[:rows, mcs_pos] = float(m)
            pr = rec.model.predict(xgb.DMatrix(Xa[:rows]))
            out[start:end, j] = pr.astype(np.float32)
    return out


def main():
    ap = argparse.ArgumentParser(description="Threshold sweep panels vs tau")
    ap.add_argument("--test-data", type=str, default="features/test.parquet")
    ap.add_argument("--sample-size", type=int, default=10000)
    ap.add_argument("--output", type=str, default="reports/threshold_sweep.png")
    ap.add_argument("--csv-out", type=str, default="reports/threshold_sweep.csv")
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--grid", type=int, default=25)
    args = ap.parse_args()

    ctx = load_contexts(args.test_data, args.sample_size)
    rec = MCSRecommender(device=('cuda' if args.device == 'gpu' else 'cpu'))
    mcs_vals = list(range(28))
    preds = batch_predict(rec, ctx, mcs_vals)
    eff = np.array([spectral_efficiency(m) for m in mcs_vals], dtype=np.float32)
    taus = np.linspace(0.1, 0.95, max(5, args.grid), dtype=np.float32)

    n, k = preds.shape
    m_idx = np.arange(k)[None, :]
    acc_rate = []
    tput_acc = []
    tput_agg = []
    rel_acc = []
    for tau in taus:
        mask = preds >= tau
        best = np.where(mask, m_idx, -1).max(axis=1)
        best = np.where(best < 0, 0, best)
        p = preds[np.arange(n), best]
        accepted = p >= tau
        acc = float(accepted.mean())
        acc_rate.append(acc)
        if acc > 0:
            rel_acc.append(float(p[accepted].mean()))
            ea = float((p[accepted] * eff[best[accepted]]).mean())
            tput_acc.append(ea)
            tput_agg.append(acc * ea)
        else:
            rel_acc.append(0.0)
            tput_acc.append(0.0)
            tput_agg.append(0.0)

    # CSV
    out_csv = Path(args.csv_out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['tau', 'accept_rate', 'reliability_acc', 'tput_acc', 'tput_agg'])
        for i in range(len(taus)):
            w.writerow([float(taus[i]), float(acc_rate[i]), float(rel_acc[i]), float(tput_acc[i]), float(tput_agg[i])])

    # Plot panels
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
    ax1.plot(taus, acc_rate, '-o', color='#1f77b4', markersize=3)
    ax1.set_ylabel('Acceptance rate')
    ax1.grid(True, alpha=0.3)
    ax2.plot(taus, tput_acc, '-o', color='#2ca02c', markersize=3)
    ax2.set_ylabel('Expected throughput (accepted)')
    ax2.grid(True, alpha=0.3)
    ax3.plot(taus, tput_agg, '-o', color='#d62728', markersize=3)
    ax3.set_ylabel('Aggregate throughput')
    ax3.set_xlabel('Threshold τ')
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    out_png = Path(args.output)
    plt.savefig(out_png, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"Saved threshold sweep panels to {out_png} and data to {out_csv}")


if __name__ == '__main__':
    main()


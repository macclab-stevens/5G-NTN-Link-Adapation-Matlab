"""
Throughput–Reliability Pareto plot.

Shows only:
- Threshold policy sweep over τ: aggregate throughput vs reliability (among accepted)
- Model throughput policy point

Outputs PNG and CSV to reports/.
"""

import argparse
from pathlib import Path
from typing import List, Dict

import numpy as np
import polars as pl
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from demo import MCSRecommender
from mcs_tables import spectral_efficiency


BASE_COLS = [
    "slot_percent",
    "slot",
    "ele_angle",
    "pathloss",
    "snr",
    "cqi",
    "window",
    "target_bler",
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


def threshold_metrics(preds: np.ndarray, eff: np.ndarray, taus: np.ndarray) -> Dict[str, np.ndarray]:
    n, k = preds.shape
    m_idx = np.arange(k)[None, :]
    rel = np.zeros_like(taus, dtype=np.float32)
    acc_rate = np.zeros_like(taus, dtype=np.float32)
    tput_acc = np.zeros_like(taus, dtype=np.float32)
    tput_agg = np.zeros_like(taus, dtype=np.float32)
    for i, tau in enumerate(taus):
        mask = preds >= tau
        best = np.where(mask, m_idx, -1).max(axis=1)
        best = np.where(best < 0, 0, best)
        p = preds[np.arange(n), best]
        accepted = p >= tau
        acc = float(accepted.mean())
        acc_rate[i] = acc
        if acc > 0:
            rel[i] = float(p[accepted].mean())
            tput_acc[i] = float((p[accepted] * eff[best[accepted]]).mean())
            tput_agg[i] = acc * tput_acc[i]
        else:
            rel[i] = 0.0
            tput_acc[i] = 0.0
            tput_agg[i] = 0.0
    return {
        'reliability_acc': rel,
        'accept_rate': acc_rate,
        'tput_acc': tput_acc,
        'tput_agg': tput_agg,
    }


def main():
    ap = argparse.ArgumentParser(description="Throughput–Reliability Pareto plot")
    ap.add_argument("--test-data", type=str, default="features/test.parquet")
    ap.add_argument("--sample-size", type=int, default=10000)
    ap.add_argument("--output", type=str, default="reports/throughput_pareto.png")
    ap.add_argument("--csv-out", type=str, default="reports/throughput_pareto_data.csv")
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--grid", type=int, default=25, help="# of τ values from 0.1..0.95")
    args = ap.parse_args()

    ctx = load_contexts(args.test_data, args.sample_size)
    device_hint = 'cuda' if args.device == 'gpu' else 'cpu'
    rec = MCSRecommender(device=device_hint)
    mcs_vals = list(range(28))
    preds = batch_predict(rec, ctx, mcs_vals)
    eff = np.array([spectral_efficiency(m) for m in mcs_vals], dtype=np.float32)

    # Threshold sweep
    taus = np.linspace(0.1, 0.95, max(5, args.grid), dtype=np.float32)
    thr = threshold_metrics(preds, eff, taus)

    # Throughput policy point
    scores = preds * eff[None, :]
    best_idx = scores.argmax(axis=1)
    p_best = preds[np.arange(preds.shape[0]), best_idx]
    tput_best = float(scores.max(axis=1).mean())
    rel_best = float(p_best.mean())

    # CSV output
    out_csv = Path(args.csv_out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['tau', 'reliability_acc', 'accept_rate', 'tput_acc', 'tput_agg'])
        for i in range(len(taus)):
            w.writerow([float(taus[i]), float(thr['reliability_acc'][i]), float(thr['accept_rate'][i]), float(thr['tput_acc'][i]), float(thr['tput_agg'][i])])

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(thr['reliability_acc'], thr['tput_agg'], '-o', color='#1f77b4', label='Threshold sweep (aggregate throughput)', markersize=3)
    ax.scatter([rel_best], [tput_best], color='#2ca02c', marker='^', s=60, label='Model throughput policy')
    ax.set_xlabel('Reliability (success rate among accepted)')
    ax.set_ylabel('Aggregate Expected Throughput (per context) [acc × ⟨p⟩×⟨E⟩]')
    ax.grid(True, alpha=0.3)

    # Annotate a few τ points
    m_idx = np.arange(len(mcs_vals))[None, :]
    pick_ids = [int(0.2 * (len(taus)-1)), int(0.5 * (len(taus)-1)), int(0.8 * (len(taus)-1))]
    for i in pick_ids:
        tau = float(taus[i])
        mask = preds >= tau
        best = np.where(mask, m_idx, -1).max(axis=1)
        best = np.where(best < 0, 0, best)
        p = preds[np.arange(preds.shape[0]), best]
        acc = p >= tau
        if not np.any(acc):
            continue
        acc_rate = float(acc.mean())
        p_mean = float(p[acc].mean())
        e_mean = float(eff[best[acc]].mean())
        x = float(thr['reliability_acc'][i])
        y = float(thr['tput_agg'][i])
        prod = acc_rate * p_mean * e_mean
        txt = (
            f"τ={tau:.2f}, acc={acc_rate:.2f}\n"
            f"⟨p⟩={p_mean:.2f}, ⟨E⟩={e_mean:.2f}\n"
            f"acc×⟨p⟩×⟨E⟩≈{prod:.2f} (Y≈{y:.2f})"
        )
        ax.annotate(txt, xy=(x, y), xytext=(x+0.03, y+0.03*max(thr['tput_agg']) if max(thr['tput_agg'])>0 else y+0.03),
                    arrowprops=dict(arrowstyle='->', color='#555'), fontsize=8, bbox=dict(boxstyle='round,pad=0.2', fc='w', ec='#999', alpha=0.8))

    # Annotate throughput policy
    try:
        e_best_mean = float(eff[best_idx].mean())
        ax.annotate(
            f"Throughput policy\n⟨p⟩={rel_best:.2f}, ⟨E⟩={e_best_mean:.2f}\n⟨p⟩×⟨E⟩≈{(rel_best*e_best_mean):.2f}",
            xy=(rel_best, tput_best), xytext=(rel_best-0.25, tput_best),
            arrowprops=dict(arrowstyle='->', color='#2ca02c'), fontsize=8,
            bbox=dict(boxstyle='round,pad=0.2', fc='w', ec='#999', alpha=0.85)
        )
    except Exception:
        pass

    ax.legend()
    plt.tight_layout()
    out_png = Path(args.output)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"Saved Pareto plot to {out_png} and data to {out_csv}")


if __name__ == '__main__':
    main()


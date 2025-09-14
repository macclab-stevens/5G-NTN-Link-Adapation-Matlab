#!/usr/bin/env python3
"""
Plot Avg BLER vs Elevation with Avg MCS overlays for different BLER targets.

Left Y-axis: Avg BLER (from data, optionally smoothed)
Right Y-axis: Avg recommended MCS (using the trained model) for guardrails:
  BLER_target ∈ {0.1%, 1%, 5%, 10%} ⇒ τ = 1 − target

Input:
  - A raw Case9 CSV (columns like slotPercnt, eleAnge, PathLoss, SNR, CQI, window, Targetbler, BLER, MCS, ...)
  - Trained model in ML/models/

Output: reports/bler_vs_elevation.png (and optional CSV)
Notes:
- Uses elevation bins (configurable width) and aggregates within bins instead of
  index-based rolling averages to avoid chaotic oscillations.
"""

import argparse
import glob
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import polars as pl
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from demo import MCSRecommender


BASE_COLS = [
    "slot_percent", "slot", "ele_angle", "pathloss", "snr", "cqi", "window", "target_bler"
]


def canonical_name(name: str) -> str:
    import re
    n = name.strip()
    n = re.sub(r"[^0-9A-Za-z]+", "_", n).lower().strip("_")
    fixes = {
        "slotpercnt": "slot_percent",
        "eleange": "ele_angle",
        "pathloss": "pathloss",
        "snr": "snr",
        "cqi": "cqi",
        "targetbler": "target_bler",
        "bler": "bler",
        "blker r": "blkerr",
        "bl kerr": "blkerr",
    }
    return fixes.get(n, n)


def build_rename_map(cols: List[str]) -> Dict[str, str]:
    base = {c: canonical_name(c) for c in cols}
    values = set(base.values())
    # Promote SINR columns to snr if needed
    if "snr" not in values:
        prefer = ["dl_sinr", "sinr", "ul_sinr"]
        for orig, canon in list(base.items()):
            if canon in prefer:
                base[orig] = "snr"
                break
    return base


def read_case9_csv(path: Path, sample: Optional[int] = None) -> pl.DataFrame:
    df = pl.read_csv(path)
    df = df.rename(build_rename_map(df.columns))
    # Coerce expected columns
    casts = {}
    for c, dt in (
        ("slot_percent", pl.Float64),
        ("slot", pl.Float64),
        ("ele_angle", pl.Float64),
        ("pathloss", pl.Float64),
        ("snr", pl.Float64),
        ("cqi", pl.Float64),
        ("window", pl.Float64),
        ("target_bler", pl.Float64),
        ("bler", pl.Float64),
        ("mcs", pl.Int32),
    ):
        if c in df.columns:
            casts[c] = pl.col(c).cast(dt, strict=False)
    if casts:
        df = df.with_columns(list(casts.values()))
    if sample and df.height > sample:
        df = df.sample(n=sample, seed=42)
    return df


def batch_predict(rec: MCSRecommender, ctx_df: pl.DataFrame, mcs_values: List[int], batch: int = 5000) -> np.ndarray:
    feats = rec.features
    # Build order mapping BASE_COLS -> feature indices
    base_order: List[int] = []
    for c in BASE_COLS:
        if c not in feats:
            continue
        base_order.append(feats.index(c))
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


def rolling_mean(arr: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return arr
    w = int(w)
    if w < 1:
        w = 1
    c = np.cumsum(np.insert(arr, 0, 0.0))
    out = (c[w:] - c[:-w]) / w
    # Pad to original length by preprending first value
    pad = np.full(w - 1, out[0] if len(out) > 0 else arr[0] if len(arr) else 0.0, dtype=np.float64)
    return np.concatenate([pad, out])


def main():
    ap = argparse.ArgumentParser(description="Plot BLER vs Elevation with model MCS overlays")
    ap.add_argument("--input", type=str, default="", help="Case9 CSV path (single file mode)")
    ap.add_argument("--input-dir", type=str, default="data", help="Directory to search when using --all-in-dir")
    ap.add_argument("--include-glob", type=str, default="Case9_MCS_ThroughputCalulation_*.csv", help="Glob pattern inside input-dir for --all-in-dir")
    ap.add_argument("--all-in-dir", action="store_true", help="Process all CSVs matching pattern in --input-dir")
    ap.add_argument("--output", type=str, default="reports/bler_vs_elevation.png")
    ap.add_argument("--csv-out", type=str, default="reports/bler_vs_elevation_data.csv")
    ap.add_argument("--output-dir", type=str, default="reports", help="Output directory when --all-in-dir is set")
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--sample-size", type=int, default=60000)
    ap.add_argument("--bin-width", type=float, default=1.0, help="Elevation bin width (degrees)")
    ap.add_argument("--smooth", type=int, default=1, help="Optional rolling window (in bins) applied after binning")
    args = ap.parse_args()

    # Determine input list
    paths: List[Path]
    if args.all_in_dir:
        pat = str(Path(args.input_dir) / args.include_glob)
        paths = [Path(p) for p in sorted(glob.glob(pat))]
        if not paths:
            raise FileNotFoundError(f"No CSV files found for pattern: {pat}")
    else:
        if args.input:
            paths = [Path(args.input)]
        else:
            cand = sorted(glob.glob("data/Case9_MCS_ThroughputCalulation_BLERw2000Tbler0.*.csv"))
            if not cand:
                cand = sorted(glob.glob("data/Case9_MCS_ThroughputCalulation_*.csv"))
            if not cand:
                raise FileNotFoundError("No Case9 CSV files found under data/")
            paths = [Path(cand[0])]

    # When in multi mode, ensure output dir exists
    if args.all_in_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        run_merge_groups(paths, args)
    else:
        run_single(paths[0], args, single_output=(Path(args.output), Path(args.csv_out)))

def run_single(path: Path, args, single_output: Tuple[Path, Path]) -> None:
    out_png_path, out_csv_path = single_output

    df_raw = read_case9_csv(path, sample=args.sample_size)
    if not all(c in df_raw.columns for c in ("ele_angle", "bler")):
        raise ValueError("Input CSV missing required columns: ele_angle and bler")

    # Sort by elevation for nicer curves
    df_raw = df_raw.sort("ele_angle")

    # Build contexts for model
    ctx = df_raw.select([c for c in BASE_COLS if c in df_raw.columns])
    # Fill missing feature columns with zeros (model expects all)
    for c in BASE_COLS:
        if c not in ctx.columns:
            ctx = ctx.with_columns(pl.lit(0.0).alias(c))

    device_hint = 'cuda' if args.device == 'gpu' else 'cpu'
    rec = MCSRecommender(device=device_hint)
    mcs_vals = list(range(28))
    preds = batch_predict(rec, ctx, mcs_vals)

    # Targets and guardrails (τ = 1 − target)
    targets = [0.001, 0.01, 0.05, 0.10]
    taus = [1.0 - t for t in targets]
    rec_mcs = {}
    m_idx = np.arange(len(mcs_vals))[None, :]
    for t, tau in zip(targets, taus):
        mask = preds >= float(tau)
        best = np.where(mask, m_idx, -1).max(axis=1)
        best = np.where(best < 0, 0, best)
        rec_mcs[t] = best.astype(np.float32)

    # Elevation binning
    bw = float(args.bin_width)
    # Build a plotting frame with predictions
    n_rows = preds.shape[0]
    row_idx = np.arange(n_rows)
    p_t001 = preds[row_idx, rec_mcs[0.001].astype(int)]
    p_t010 = preds[row_idx, rec_mcs[0.01].astype(int)]
    p_t050 = preds[row_idx, rec_mcs[0.05].astype(int)]
    p_t100 = preds[row_idx, rec_mcs[0.10].astype(int)]

    plot_df = df_raw.select(["ele_angle", "bler"]).with_columns([
        pl.lit(rec_mcs[0.001]).alias("mcs_t001"),
        pl.lit(rec_mcs[0.01]).alias("mcs_t010"),
        pl.lit(rec_mcs[0.05]).alias("mcs_t050"),
        pl.lit(rec_mcs[0.10]).alias("mcs_t100"),
        pl.lit(p_t001).alias("p_t001"),
        pl.lit(p_t010).alias("p_t010"),
        pl.lit(p_t050).alias("p_t050"),
        pl.lit(p_t100).alias("p_t100"),
    ])
    plot_df = plot_df.with_columns(((pl.col("ele_angle") / bw).floor() * bw).alias("ele_bin"))
    gb = plot_df.group_by("ele_bin").agg([
        pl.col("bler").mean().alias("bler_avg_base"),
        (1 - pl.col("p_t001")).mean().alias("bler_avg_t001"),
        (1 - pl.col("p_t010")).mean().alias("bler_avg_t010"),
        (1 - pl.col("p_t050")).mean().alias("bler_avg_t050"),
        (1 - pl.col("p_t100")).mean().alias("bler_avg_t100"),
        pl.col("mcs_t001").mean().alias("mcs_avg_t001"),
        pl.col("mcs_t010").mean().alias("mcs_avg_t010"),
        pl.col("mcs_t050").mean().alias("mcs_avg_t050"),
        pl.col("mcs_t100").mean().alias("mcs_avg_t100"),
    ]).sort("ele_bin")

    elev = gb["ele_bin"].to_numpy()
    bler_base = gb["bler_avg_base"].to_numpy()
    mcs_avgs = {
        0.001: gb["mcs_avg_t001"].to_numpy(),
        0.01: gb["mcs_avg_t010"].to_numpy(),
        0.05: gb["mcs_avg_t050"].to_numpy(),
        0.10: gb["mcs_avg_t100"].to_numpy(),
    }
    bler_avgs = {
        0.001: gb["bler_avg_t001"].to_numpy(),
        0.01: gb["bler_avg_t010"].to_numpy(),
        0.05: gb["bler_avg_t050"].to_numpy(),
        0.10: gb["bler_avg_t100"].to_numpy(),
    }

    # Optional smoothing on the binned series
    win_bins = max(1, int(args.smooth))
    if win_bins > 1:
        bler_base = rolling_mean(bler_base.astype(np.float64), win_bins)
        for k in list(mcs_avgs.keys()):
            mcs_avgs[k] = rolling_mean(mcs_avgs[k].astype(np.float64), win_bins)
            bler_avgs[k] = rolling_mean(bler_avgs[k].astype(np.float64), win_bins)

    # Determine window label (prefer file's window; else use smoothing/bw label)
    wval = None
    if "window" in df_raw.columns:
        try:
            wvals = df_raw["window"].unique().to_list()
            if len(wvals) == 1:
                wval = int(float(wvals[0]))
        except Exception:
            pass
    if wval is None:
        # Derive from smoothing/bins for transparency
        wval = f"bins{int(bw)}_smooth{int(win_bins)}"

    # Update output filenames to include window
    tag = f"blerw{int(wval)}" if isinstance(wval, (int, float)) else f"blerw_{wval}"
    def with_tag(pth: str) -> Path:
        p = Path(pth)
        return p.with_name(f"{p.stem}_{tag}{p.suffix}")

    out_csv = with_tag(str(out_csv_path))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df = pl.DataFrame({
        "ele_angle": elev,
        "bler_avg_base": bler_base,
        **{f"bler_avg_t{int(t*1000):03d}": bler_avgs[t] for t in targets},
        **{f"mcs_avg_t{int(t*1000):03d}": mcs_avgs[t] for t in targets},
    })
    out_df.write_csv(out_csv)

    # Plot: two panels sharing X (top = BLER, bottom = Avg MCS)
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(9.8, 6.2), sharex=True, gridspec_kw={"hspace": 0.08})

    # Top: BLER
    ax_top.plot(elev, bler_base, color="#1f77b4", label="Baseline", linewidth=1.6)
    ax_top.set_ylabel("Avg BLER")
    ax_top.grid(True, alpha=0.3)
    colors = {
        0.001: "#ff7f0e",  # orange
        0.01: "#2ca02c",   # green
        0.05: "#9467bd",  # purple
        0.10: "#8c564b",  # brown
    }
    bler_handles = []
    bler_labels = []
    mcs_handles = []
    mcs_labels = []
    for t in targets:
        # Same color across panels; solid for BLER, dashed for MCS
        h1, = ax_top.plot(elev, bler_avgs[t], color=colors[t], linewidth=1.6, linestyle='-',
                          label=f"BLER_Target = {t*100:.1f}%")
        h2, = ax_bot.plot(elev, mcs_avgs[t], color=colors[t], linewidth=1.6, linestyle='--',
                          label=f"Avg MCS ({t*100:.1f}%)")
        bler_handles.append(h1); bler_labels.append(h1.get_label())
        mcs_handles.append(h2); mcs_labels.append(h2.get_label())

    # Legends (outside the axes)
    leg1 = ax_top.legend([ax_top.lines[0]] + bler_handles, ["Baseline"] + bler_labels,
                         loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, framealpha=0.9)
    ax_top.add_artist(leg1)
    ax_bot.legend(mcs_handles, mcs_labels, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, framealpha=0.9, title="Avg MCS")

    # Bottom panel labels
    ax_bot.set_ylabel("Avg MCS")
    ax_bot.set_xlabel("Elevation Angle")
    ax_bot.grid(True, alpha=0.3)

    # Limit x to 0..90 degrees and adjust layout for outside legends
    ax_top.set_xlim(0.0, 90.0)
    ax_bot.set_xlim(0.0, 90.0)
    fig.suptitle(f"BLER Window = {wval}")
    fig.subplots_adjust(right=0.78, top=0.90)
    out_png = with_tag(str(out_png_path))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {out_png} and data to {out_csv}")
def parse_tbler_and_window(name: str) -> Tuple[Optional[float], Optional[int]]:
    tb = None
    win = None
    m = re.search(r"Tbler([0-9.]+)", name, re.IGNORECASE)
    if m:
        try:
            tb = float(m.group(1))
        except Exception:
            tb = None
    w = re.search(r"BLERw(\d+)", name, re.IGNORECASE)
    if w:
        try:
            win = int(w.group(1))
        except Exception:
            win = None
    return tb, win


def run_merge_groups(paths: List[Path], args) -> None:
    """Merge multiple CSVs by target BLER parsed from filename and plot one figure.

    - Groups files by their `Tbler` in the filename (e.g., Tbler0.1 → 0.1)
    - For each target group, aggregates rows and computes per-elevation averages
    - Produces a single output using the legacy naming convention `bler_vs_elevation_blerw<window>.png`
    """
    # Build groups by target BLER and collect windows
    groups: Dict[float, List[Path]] = {}
    windows: List[int] = []
    for p in paths:
        t, w = parse_tbler_and_window(p.name)
        if w is not None:
            windows.append(w)
        if t is None:
            # Skip files without a parsable target
            continue
        groups.setdefault(t, []).append(p)

    if not groups:
        raise FileNotFoundError("No files with Tbler<val> in filename found for merging")

    # Determine window tag (mode if mixed)
    wval: Optional[int]
    if windows:
        from collections import Counter
        wval = Counter(windows).most_common(1)[0][0]
    else:
        wval = None

    # Helper to tag outputs
    tag = f"blerw{int(wval)}" if isinstance(wval, (int, float)) else "blerw"
    def with_tag(pth: str) -> Path:
        p = Path(pth)
        return p.with_name(f"{p.stem}_{tag}{p.suffix}")

    # Load model
    device_hint = 'cuda' if args.device == 'gpu' else 'cpu'
    rec = MCSRecommender(device=device_hint)
    mcs_vals = list(range(28))

    # Colors assigned by sorted targets
    targets = sorted(groups.keys())
    palette = ["#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#1f77b4", "#d62728", "#17becf"]
    colors = {t: palette[i % len(palette)] for i, t in enumerate(targets)}

    # Baseline across all files (average BLER by elevation)
    dfs_all = []
    for ps in paths:
        dfs_all.append(read_case9_csv(ps, sample=args.sample_size))
    df_all = pl.concat(dfs_all, how="diagonal")
    bw = float(args.bin_width)
    base = df_all.select(["ele_angle", "bler"]).with_columns(((pl.col("ele_angle") / bw).floor() * bw).alias("ele_bin"))
    base_g = base.group_by("ele_bin").agg([pl.col("bler").mean().alias("bler_avg_base")]).sort("ele_bin")
    elev_base = base_g["ele_bin"].to_numpy()
    bler_base = base_g["bler_avg_base"].to_numpy()

    # Plot setup (two panels)
    import matplotlib.pyplot as plt
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(9.8, 6.2), sharex=True, gridspec_kw={"hspace": 0.08})
    # Top baseline
    ax_top.plot(elev_base, bler_base, color="#1f77b4", linewidth=1.6, label="Baseline")
    ax_top.set_ylabel("Avg BLER")
    ax_top.grid(True, alpha=0.3)

    # Iterate groups per target
    bler_handles = []
    bler_labels = []
    mcs_handles = []
    mcs_labels = []
    for t in targets:
        # Combine group's frames
        dfs_t = [read_case9_csv(p, sample=args.sample_size) for p in groups[t]]
        df_t = pl.concat(dfs_t, how="diagonal")
        # Build contexts
        ctx = df_t.select([c for c in BASE_COLS if c in df_t.columns])
        for c in BASE_COLS:
            if c not in ctx.columns:
                ctx = ctx.with_columns(pl.lit(0.0).alias(c))
        preds = batch_predict(rec, ctx, mcs_vals)

        tau = float(1.0 - t)
        m_idx = np.arange(len(mcs_vals))[None, :]
        mask = preds >= tau
        best = np.where(mask, m_idx, -1).max(axis=1)
        best = np.where(best < 0, 0, best)
        p_sel = preds[np.arange(preds.shape[0]), best]
        m_sel = best.astype(np.float32)

        # Bin by elevation
        plot_df = df_t.select(["ele_angle"]).with_columns([
            pl.lit(p_sel).alias("p_sel"),
            pl.lit(m_sel).alias("m_sel"),
        ])
        plot_df = plot_df.with_columns(((pl.col("ele_angle") / bw).floor() * bw).alias("ele_bin"))
        gb = plot_df.group_by("ele_bin").agg([
            (1 - pl.col("p_sel")).mean().alias("bler_avg_t"),
            pl.col("m_sel").mean().alias("mcs_avg_t"),
        ]).sort("ele_bin")
        elev = gb["ele_bin"].to_numpy()
        bler_avg_t = gb["bler_avg_t"].to_numpy()
        mcs_avg_t = gb["mcs_avg_t"].to_numpy()

        color = colors[t]
        h1, = ax_top.plot(elev, bler_avg_t, color=color, linewidth=1.6, linestyle='-', label=f"BLER_Target = {t*100:.1f}%")
        h2, = ax_bot.plot(elev, mcs_avg_t, color=color, linewidth=1.6, linestyle='--', label=f"Avg MCS ({t*100:.1f}%)")
        bler_handles.append(h1); bler_labels.append(h1.get_label())
        mcs_handles.append(h2); mcs_labels.append(h2.get_label())

    # Legends outside
    leg1 = ax_top.legend([ax_top.lines[0]] + bler_handles, ["Baseline"] + bler_labels,
                         loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, framealpha=0.9)
    ax_top.add_artist(leg1)
    ax_bot.legend(mcs_handles, mcs_labels, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, framealpha=0.9, title="Avg MCS")
    ax_bot.set_ylabel("Avg MCS")
    ax_bot.set_xlabel("Elevation Angle")
    ax_bot.grid(True, alpha=0.3)

    # Limit to 0..90 degrees and adjust layout for outside legends
    ax_top.set_xlim(0.0, 90.0)
    ax_bot.set_xlim(0.0, 90.0)
    fig.suptitle(f"BLER Window = {wval if wval is not None else ''}")
    fig.subplots_adjust(right=0.78, top=0.90)
    out_png = with_tag(str(Path(args.output_dir) / "bler_vs_elevation.png"))
    out_csv = with_tag(str(Path(args.output_dir) / "bler_vs_elevation_data.csv"))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close()

    # Minimal long-form CSV export
    rows = []
    for x, y in zip(elev_base.tolist(), bler_base.tolist()):
        rows.append({"ele_bin": float(x), "series": "baseline_bler", "value": float(y)})
    # Note: target lines are plotted only in PNG; CSV contains baseline bins here.
    pl.DataFrame(rows).write_csv(out_csv)
    print(f"Saved plot to {out_png} and data to {out_csv}")


if __name__ == "__main__":
    main()

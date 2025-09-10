import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import polars as pl
import xgboost as xgb
from mcs_tables import spectral_efficiency


def load_model(model_path: Path) -> xgb.Booster:
    bst = xgb.Booster()
    bst.load_model(model_path)
    return bst


def load_features_list(meta_path: Path) -> List[str]:
    import json

    meta = json.loads(meta_path.read_text())
    return meta["features"]


def recommend_for_frame(
    df: pl.DataFrame,
    feats: List[str],
    bst: xgb.Booster,
    mcs_min: int,
    mcs_max: int,
    threshold: float,
    batch: int = 50000,
    objective: str = "threshold",  # "threshold" or "throughput"
) -> pl.DataFrame:
    # Expect df to contain only context features (without mcs). We'll inject mcs values per candidate.
    # Build base array with placeholders for mcs feature appended at end.
    base_feats = [c for c in feats if c != "mcs"]
    X_base = np.column_stack([df[c].to_numpy().astype(np.float32, copy=False) for c in base_feats])
    mcs_pos = feats.index("mcs")
    # If mcs is not last in feats, we need to interleave columns; build a template array
    d = len(feats)
    n = X_base.shape[0]
    # Map base feature indices into the full array positions
    base_order = [feats.index(c) for c in base_feats]

    preds_best = np.zeros(n, dtype=np.float32)
    mcs_best = np.full(n, mcs_min, dtype=np.int32)

    for start in range(0, n, batch):
        end = min(start + batch, n)
        Xb = X_base[start:end]
        # Initialize augmented matrix with zeros
        Xa = np.zeros((Xb.shape[0], d), dtype=np.float32)
        # Place base features into their positions
        for src_idx, dst_idx in enumerate(base_order):
            Xa[:, dst_idx] = Xb[:, src_idx]

        best_prob = np.zeros(Xb.shape[0], dtype=np.float32)
        best_mcs = np.full(Xb.shape[0], mcs_min, dtype=np.int32)
        best_score = np.zeros(Xb.shape[0], dtype=np.float32)
        for m in range(mcs_min, mcs_max + 1):
            Xa[:, mcs_pos] = float(m)
            dmat = xgb.DMatrix(Xa)
            pr = bst.predict(dmat)
            if objective == "threshold":
                mask = pr >= threshold
                update = mask & (m >= best_mcs)
                best_prob[update] = pr[update]
                best_mcs[update] = m
            else:
                # throughput proxy score: spectral efficiency * probability
                eff = spectral_efficiency(m)
                score = pr * eff
                update = score > best_score
                best_score[update] = score[update]
                best_prob[update] = pr[update]
                best_mcs[update] = m
        preds_best[start:end] = best_prob
        mcs_best[start:end] = best_mcs

    out = df.with_columns([
        pl.Series("rec_mcs", mcs_best),
        pl.Series("rec_prob", preds_best),
    ])
    return out


def canonical_name(name: str) -> str:
    import re

    n = name.strip()
    n = re.sub(r"[^0-9A-Za-z]+", "_", n).lower().strip("_")
    fixes = {
        "slotpercnt": "slot_percent",
        "eleange": "ele_angle",
        "targetbler": "target_bler",
    }
    return fixes.get(n, n)


def main() -> None:
    ap = argparse.ArgumentParser(description="Recommend optimal MCS from context variables using trained pass-prob model")
    ap.add_argument("--input", type=str, default="features/test.parquet", help="Input Parquet/CSV with context variables")
    ap.add_argument("--output", type=str, default="reports/recommendations.csv", help="Output CSV with recommendations")
    ap.add_argument("--model", type=str, default="models/xgb_mcs_pass.json", help="Trained model path")
    ap.add_argument("--meta", type=str, default="models/model_meta.json", help="Model metadata (for feature order)")
    ap.add_argument("--threshold", type=float, default=-1.0, help="Minimum pass probability (threshold objective). If <0, load from model meta if available.")
    ap.add_argument("--objective", type=str, default="threshold", choices=["threshold", "throughput"], help="Recommendation objective")
    ap.add_argument("--baseline", type=str, default="none", choices=["none", "cqi"], help="Use a baseline (e.g., CQI->MCS) instead of the model")
    ap.add_argument("--cqi-offset", type=int, default=0, help="Offset to add to CQI when baseline=cqi")
    ap.add_argument("--mcs-min", type=int, default=None, help="Min MCS to consider (default: from data)")
    ap.add_argument("--mcs-max", type=int, default=None, help="Max MCS to consider (default: from data)")
    args = ap.parse_args()

    model_path = Path(args.model)
    meta_path = Path(args.meta)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bst = load_model(model_path)
    feats = load_features_list(meta_path)
    # Load threshold from meta if not provided
    if args.threshold < 0:
        try:
            import json
            meta = json.loads(meta_path.read_text())
            thr = meta.get("threshold", None)
            if thr is not None:
                args.threshold = float(thr)
        except Exception:
            args.threshold = 0.5

    # Load input context
    path = Path(args.input)
    if path.suffix.lower() == ".parquet":
        lf = pl.scan_parquet(path)
    else:
        lf = pl.scan_csv(path)
    # Only keep context features (without mcs)
    base_feats = [c for c in feats if c != "mcs"]
    lf = lf.select([c for c in base_feats if c in lf.columns])

    # Collect
    df = lf.collect(streaming=True)
    if df.height == 0:
        print("No rows in input to recommend for")
        return

    # Determine MCS range
    mcs_min: Optional[int] = args.mcs_min
    mcs_max: Optional[int] = args.mcs_max
    if mcs_min is None or mcs_max is None:
        # Fallback: infer from train/test features if present in file
        print("Fallback: Inferring MCS range from data")
        if "mcs" in df.columns:
            arr = df["mcs"].to_numpy()
            mcs_min = int(np.nanmin(arr)) if mcs_min is None else mcs_min
            mcs_max = int(np.nanmax(arr)) if mcs_max is None else mcs_max
        else:
            # Common 5G NR range (0..27)
            mcs_min = 0 if mcs_min is None else mcs_min
            mcs_max = 27 if mcs_max is None else mcs_max

    if args.baseline == "cqi":
        # Baseline: map CQI to MCS via simple offset and clip
        if "cqi" not in df.columns:
            raise SystemExit("baseline=cqi requires 'cqi' column in input")
        cqi_vals = df["cqi"].to_numpy()
        rec = np.clip(np.rint(cqi_vals).astype(int) + int(args.cqi_offset), mcs_min, mcs_max).astype(int)
        out = df.with_columns([
            pl.Series("rec_mcs", rec),
        ])
        # Optionally attach model probability for the recommended MCS
        try:
            base_feats = [c for c in feats if c != "mcs"]
            X_base = np.column_stack([df[c].to_numpy().astype(np.float32, copy=False) for c in base_feats])
            d = len(feats)
            n = X_base.shape[0]
            base_order = [feats.index(c) for c in base_feats]
            Xa = np.zeros((n, d), dtype=np.float32)
            for src_idx, dst_idx in enumerate(base_order):
                Xa[:, dst_idx] = X_base[:, src_idx]
            mcs_pos = feats.index("mcs")
            Xa[:, mcs_pos] = rec.astype(np.float32)
            pr = bst.predict(xgb.DMatrix(Xa))
            out = out.with_columns([pl.Series("rec_prob", pr)])
        except Exception:
            pass
    else:
        out = recommend_for_frame(
            df,
            feats,
            bst,
            mcs_min,
            mcs_max,
            args.threshold,
            objective=args.objective,
        )
    out.write_csv(out_path)
    print(f"Wrote recommendations to {out_path}")


if __name__ == "__main__":
    main()

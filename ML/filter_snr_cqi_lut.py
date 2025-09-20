"""Utilities for filtering and exporting SNR/CQI lookup tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter an SNR/CQI LUT to a capped MCS range.")
    parser.add_argument("--input", type=Path, default=Path("data/snr_cqi_lut_dense.csv"), help="Path to the raw LUT CSV.")
    parser.add_argument("--output", type=Path, default=Path("data/snr_cqi_lut_mcs15.csv"), help="Destination for the filtered LUT.")
    parser.add_argument("--max-mcs", type=int, default=15, help="Maximum MCS value to retain (values above are clipped).")
    parser.add_argument("--min-mcs", type=int, default=0, help="Minimum MCS value to retain (values below are raised).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Missing LUT input: {args.input}")

    df = pd.read_csv(args.input)

    required_cols = {"snr", "cqi", "mcs"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Input LUT missing required columns: {sorted(missing)}")

    df["cqi"] = df["cqi"].round().astype(int)
    df["mcs"] = df["mcs"].round().astype(int)
    df["mcs"] = df["mcs"].clip(lower=args.min_mcs, upper=args.max_mcs)

    df = df.sort_values(["snr", "cqi"]).reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    min_mcs = df["mcs"].min()
    max_mcs = df["mcs"].max()
    print(f"Wrote {len(df)} rows to {args.output} (mcs in [{min_mcs}, {max_mcs}]).")


if __name__ == "__main__":
    main()

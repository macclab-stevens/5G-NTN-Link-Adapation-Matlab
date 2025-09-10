import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def canonical_name(name: str) -> str:
    n = name.strip()
    n = re.sub(r"[^0-9A-Za-z]+", "_", n).lower().strip("_")
    # Specific fixes
    fixes = {
        "slotpercnt": "slot_percent",
        "eleange": "ele_angle",
        "targetbler": "target_bler",
        "blkerr": "blkerr",
    }
    return fixes.get(n, n)


def build_rename_map(cols: List[str]) -> Dict[str, str]:
    return {c: canonical_name(c) for c in cols}


def engineer_features(lf: pl.LazyFrame) -> pl.LazyFrame:
    # Assumes columns have been canonicalized already
    return (
        lf.with_columns(
            [
                # Clean strings: strip whitespace and quotes, then cast
                pl.col("snr").cast(pl.Utf8, strict=False).str.strip_chars().str.strip_chars("'").cast(pl.Float64, strict=False),
                pl.col("pathloss").cast(pl.Utf8, strict=False).str.strip_chars().str.strip_chars("'").cast(pl.Float64, strict=False),
                pl.col("cqi").cast(pl.Utf8, strict=False).str.strip_chars().str.strip_chars("'").cast(pl.Int32, strict=False),
                pl.col("mcs").cast(pl.Utf8, strict=False).str.strip_chars().str.strip_chars("'").cast(pl.Int32, strict=False),
                pl.col("tbs").cast(pl.Utf8, strict=False).str.strip_chars().str.strip_chars("'").cast(pl.Float64, strict=False),
                pl.col("tcr").cast(pl.Utf8, strict=False).str.strip_chars().str.strip_chars("'").cast(pl.Float64, strict=False),
                pl.col("bler").cast(pl.Utf8, strict=False).str.strip_chars().str.strip_chars("'").cast(pl.Float64, strict=False),
                pl.col("target_bler").cast(pl.Utf8, strict=False).str.strip_chars().str.strip_chars("'").cast(pl.Float64, strict=False),
                pl.col("slot").cast(pl.Utf8, strict=False).str.strip_chars().str.strip_chars("'").cast(pl.Int64, strict=False),
                pl.col("slot_percent").cast(pl.Utf8, strict=False).str.strip_chars().str.strip_chars("'").cast(pl.Float64, strict=False),
                pl.col("ele_angle").cast(pl.Utf8, strict=False).str.strip_chars().str.strip_chars("'").cast(pl.Float64, strict=False),
                pl.col("window").cast(pl.Utf8, strict=False).str.strip_chars().str.strip_chars("'").cast(pl.Float64, strict=False),
                pl.col("blkerr").cast(pl.Utf8, strict=False).str.strip_chars().str.strip_chars("'").cast(pl.Int8, strict=False),
                pl.col("mod").cast(pl.Utf8, strict=False).str.strip_chars().str.strip_chars("'").alias("mod"),
            ]
        )
        .with_columns(
            [
                # Derived features
                pl.col("snr").round(0).alias("snr_round"),
                ((pl.col("snr") * 2).floor() / 2).alias("snr_bin05"),
                pl.col("snr").clip(-20, 50).alias("snr_clip"),
                pl.col("pathloss").round(0).alias("pathloss_round"),
                pl.col("mod").cast(pl.Categorical).to_physical().alias("mod_code"),
                (pl.col("snr") * pl.col("cqi").cast(pl.Float64)).alias("snr_cqi"),
                (pl.col("snr") * pl.col("pathloss")).alias("snr_pathloss"),
                # Label: pass when BLER <= Target and BLKErr == 1 (ACK)
                (
                    (pl.col("bler") <= pl.col("target_bler"))
                    & (pl.col("blkerr") == 1)
                )
                .cast(pl.Int8)
                .alias("label_pass"),
            ]
        )
    )


def finalize_columns(cols: List[str]) -> List[str]:
    # Feature set focused on predicting pass given context + MCS.
    core = [
        "snr",
        "snr_round",
        "snr_bin05",
        "snr_clip",
        "pathloss",
        "pathloss_round",
        "cqi",
        "ele_angle",
        "slot_percent",
        "slot",
        "window",
        "target_bler",
        "mod_code",
        "snr_cqi",
        "snr_pathloss",
        # include mcs as a feature for conditional pass modeling
        "mcs",
    ]
    # Always keep identifiers/labels/optionals if present
    keep_also = ["tbs", "tcr", "bler", "blkerr", "label_pass"]
    final = [c for c in core + keep_also if c in cols]
    return final


def _align_select(df: pl.DataFrame, schema_cols: List[str]) -> pl.DataFrame:
    # Add missing columns as nulls and reorder to schema
    cols_to_add = [c for c in schema_cols if c not in df.columns]
    if cols_to_add:
        df = df.with_columns([pl.lit(None).alias(c) for c in cols_to_add])
    return df.select(schema_cols)


def process_file_split(
    csv_path: Path,
    writer_train: pq.ParquetWriter,
    writer_test: pq.ParquetWriter,
    chunksize: int,
    schema_cols: List[str],
    rng: "np.random.Generator",
    test_frac: float,
    start_offset: int = 0,
    writer_combined: Optional[pq.ParquetWriter] = None,
) -> None:
    header_df = pl.read_csv(csv_path, n_rows=0)
    rename_map = build_rename_map(header_df.columns)

    offset = int(start_offset)
    total = 0
    while True:
        lf = pl.scan_csv(csv_path).slice(offset, chunksize).rename(rename_map)
        lf = engineer_features(lf)
        df = lf.collect()
        if df.height == 0:
            break
        total += df.height
        offset += df.height

        # Select final columns and align to schema
        used_columns = finalize_columns(df.columns)
        df = df.select([c for c in used_columns if c in schema_cols])
        df = _align_select(df, schema_cols)

        # Random split mask
        mask = rng.random(df.height) < test_frac
        if writer_combined is not None:
            writer_combined.write_table(df.to_arrow())
        if mask.any():
            writer_test.write_table(df.filter(mask).to_arrow())
        if (~mask).any():
            writer_train.write_table(df.filter(~mask).to_arrow())


def main() -> None:
    ap = argparse.ArgumentParser(description="Feature engineering and dataset split to Parquet train/test")
    ap.add_argument("--data-dir", type=str, default="data", help="Directory with CSV files")
    ap.add_argument("--out", type=str, default="", help="(Optional) Combined Parquet output path")
    ap.add_argument("--train-out", type=str, default="features/train.parquet", help="Train Parquet output path")
    ap.add_argument("--test-out", type=str, default="features/test.parquet", help="Test Parquet output path")
    ap.add_argument("--test-frac", type=float, default=0.2, help="Test split fraction")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    ap.add_argument("--chunksize", type=int, default=200_000, help="Chunk rows to stream per write")
    ap.add_argument("--include-glob", type=str, default="*.csv", help="Glob pattern to include files (default: *.csv)")
    ap.add_argument("--max-files", type=int, default=0, help="Optional cap on number of files to process (0 = all)")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    combined_path = Path(args.out) if args.out else None
    train_path = Path(args.train_out) if args.train_out else None
    test_path = Path(args.test_out) if args.test_out else None
    if combined_path:
        ensure_dir(combined_path.parent)
    if train_path:
        ensure_dir(train_path.parent)
    if test_path:
        ensure_dir(test_path.parent)

    csvs = sorted([p for p in data_dir.glob(args.include_glob) if p.is_file()])
    if args.max_files and args.max_files > 0:
        csvs = csvs[: args.max_files]
    if not csvs:
        print(f"No CSV files found in {data_dir}")
        return

    # Initialize writer lazily after first chunk to get schema
    writer_train: Optional[pq.ParquetWriter] = None
    writer_test: Optional[pq.ParquetWriter] = None
    writer_combined: Optional[pq.ParquetWriter] = None
    schema_cols: Optional[List[str]] = None
    rng = __import__("numpy").random.default_rng(args.seed)
    try:
        for p in tqdm(csvs, desc="Featurizing CSVs"):
            # For the first file/chunk, create writers with that schema
            if writer_train is None or writer_test is None:
                # Peek a small slice to get engineered schema
                lf0 = engineer_features(pl.scan_csv(p).slice(0, min(10_000, args.chunksize)).rename(build_rename_map(pl.read_csv(p, n_rows=0).columns)))
                df0 = lf0.collect()
                df0 = df0.select(finalize_columns(df0.columns))
                tbl0 = df0.to_arrow()
                schema_cols = df0.columns
                if train_path:
                    writer_train = pq.ParquetWriter(train_path, tbl0.schema, compression="snappy")
                if test_path:
                    writer_test = pq.ParquetWriter(test_path, tbl0.schema, compression="snappy")
                if combined_path:
                    writer_combined = pq.ParquetWriter(combined_path, tbl0.schema, compression="snappy")

                # Write the peeked rows split across train/test (+ combined)
                written_rows = 0
                if df0.height > 0 and (writer_train is not None and writer_test is not None):
                    mask0 = rng.random(df0.height) < args.test_frac
                    if writer_combined is not None:
                        writer_combined.write_table(tbl0)
                    if mask0.any():
                        writer_test.write_table(df0.filter(mask0).to_arrow())
                    if (~mask0).any():
                        writer_train.write_table(df0.filter(~mask0).to_arrow())
                    written_rows = df0.height
                # Continue from after the peeked rows
                process_file_split(
                    p,
                    writer_train,
                    writer_test,
                    args.chunksize,
                    schema_cols,
                    rng,
                    args.test_frac,
                    start_offset=written_rows,
                    writer_combined=writer_combined,
                )
            else:
                process_file_split(
                    p,
                    writer_train,
                    writer_test,
                    args.chunksize,
                    schema_cols or [],
                    rng,
                    args.test_frac,
                    writer_combined=writer_combined,
                )
    finally:
        if writer_train is not None:
            writer_train.close()
        if writer_test is not None:
            writer_test.close()
        if writer_combined is not None:
            writer_combined.close()

    if train_path:
        print(f"Wrote train to {train_path}")
    if test_path:
        print(f"Wrote test to {test_path}")
    if combined_path:
        print(f"Wrote combined to {combined_path}")


if __name__ == "__main__":
    main()

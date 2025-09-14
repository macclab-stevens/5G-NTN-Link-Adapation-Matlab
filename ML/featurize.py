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
        # Common variants -> canonical
        "slot_index": "slot",
        "nslot": "slot",
        "slotidx": "slot",
        "elevation": "ele_angle",
        "elevation_angle": "ele_angle",
        "elevation_deg": "ele_angle",
        "elev": "ele_angle",
        "path_loss": "pathloss",
        "dl_cqi": "cqi",
        "cqi_dl": "cqi",
        "mcs_index": "mcs",
        "mcs_idx": "mcs",
        "mcsidx": "mcs",
        "transport_block_size": "tbs",
        "transportblocksize": "tbs",
        "tb_size": "tbs",
        "tbs_bytes": "tbs",
        "window_len": "window",
        "filter_window": "window",
        "win": "window",
        "ackresult": "ack_result",
        "ackstatus": "ack_status",
        "ack_nack": "ack_result",
        "harq_ack": "ack_status",
        "ack": "ack_status",
        "block_error_rate": "bler",
        "blerrate": "bler",
        "bler_percent": "bler",
    }
    return fixes.get(n, n)


def build_rename_map(cols: List[str]) -> Dict[str, str]:
    """Build a robust rename map to canonical names.

    Additionally, map common SINR column names to `snr` when `snr` is absent
    (e.g., logs that expose `dl_sinr`/`ul_sinr`).
    """
    base = {c: canonical_name(c) for c in cols}

    # If there is no explicit `snr` column, promote a SINR column to `snr`.
    values = set(base.values())
    if "snr" not in values:
        # Prefer downlink SINR, then generic SINR, then uplink SINR
        prefer = ["dl_sinr", "dl_sinr_db", "sinr", "sinr_db", "ul_sinr", "ul_sinr_db"]
        for orig, canon in list(base.items()):
            if canon in prefer:
                base[orig] = "snr"
                break

    # Normalize common alias canonical names to our targets
    alias_map = {
        "mcs_index": "mcs",
        "mcs_idx": "mcs",
        "mcsidx": "mcs",
        "path_loss": "pathloss",
        "elevation": "ele_angle",
        "elevation_angle": "ele_angle",
        "elevation_deg": "ele_angle",
        "cqi_dl": "cqi",
        "dl_cqi": "cqi",
        "transport_block_size": "tbs",
        "transportblocksize": "tbs",
        "tb_size": "tbs",
        "tbs_bytes": "tbs",
        "window_len": "window",
        "filter_window": "window",
        "win": "window",
        "ack": "ack_status",
        "ack_nack": "ack_result",
        "ackresult": "ack_result",
        "ackstatus": "ack_status",
        "block_error_rate": "bler",
        "blerrate": "bler",
        "bler_percent": "bler",
    }
    for orig, canon in list(base.items()):
        if canon in alias_map:
            base[orig] = alias_map[canon]

    return base


def engineer_features(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Engineer features robustly with optional columns.

    Assumes columns have been canonicalized already. Safely handles datasets that
    provide SINR but not SNR, or ACK fields instead of BLKErr, etc.
    """
    cols = set(lf.columns)

    clean_exprs: List[pl.Expr] = []

    # Helper to add cast/clean only if column exists
    def add_clean(col: str, dtype) -> None:
        if col in cols:
            clean_exprs.append(
                pl.col(col)
                .cast(pl.Utf8, strict=False)
                .str.strip_chars()
                .str.strip_chars("'")
                .cast(dtype, strict=False)
            )

    # Numeric/string cleaning for known columns when present
    for name, dtype in (
        ("snr", pl.Float64),
        ("pathloss", pl.Float64),
        ("cqi", pl.Int32),
        ("mcs", pl.Int32),
        ("tbs", pl.Float64),
        ("tcr", pl.Float64),
        ("bler", pl.Float64),
        ("target_bler", pl.Float64),
        ("slot", pl.Int64),
        ("slot_percent", pl.Float64),
        ("ele_angle", pl.Float64),
        ("window", pl.Float64),
        ("blkerr", pl.Int8),
    ):
        add_clean(name, dtype)

    # Map ACK/NACK variants to blkerr ∈ {0,1} if `blkerr` not present
    if "blkerr" not in cols:
        if "ack_result" in cols:
            clean_exprs.append(
                pl.when(pl.col("ack_result").cast(pl.Utf8).str.to_uppercase() == pl.lit("ACK"))
                .then(pl.lit(1))
                .otherwise(pl.lit(0))
                .cast(pl.Int8)
                .alias("blkerr")
            )
        elif "ack_status" in cols:
            # Heuristic: treat status == 1 as ACK, else 0
            clean_exprs.append(
                (pl.col("ack_status").cast(pl.Int8, strict=False) == 1)
                .cast(pl.Int8)
                .alias("blkerr")
            )
        elif "ack" in cols:
            # Generic ack column; try numeric 1/0 first, else parse text
            clean_exprs.append(
                pl.when(pl.col("ack").cast(pl.Int8, strict=False).is_not_null())
                .then((pl.col("ack").cast(pl.Int8, strict=False) == 1).cast(pl.Int8))
                .otherwise(
                    (pl.col("ack").cast(pl.Utf8, strict=False).str.to_uppercase() == pl.lit("ACK")).cast(pl.Int8)
                )
                .alias("blkerr")
            )

    # Provide a default target_bler if missing (10% expressed as 10.0)
    if "target_bler" not in cols:
        clean_exprs.append(pl.lit(10.0).alias("target_bler"))

    # Keep `mod` as string if present
    if "mod" in cols:
        clean_exprs.append(pl.col("mod").cast(pl.Utf8, strict=False).str.strip_chars().str.strip_chars("'").alias("mod"))

    lf = lf.with_columns(clean_exprs) if clean_exprs else lf

    # Derived features (only add when inputs exist)
    derived_exprs: List[pl.Expr] = []

    if "snr" in lf.columns:
        derived_exprs.extend(
            [
                pl.col("snr").round(0).alias("snr_round"),
                ((pl.col("snr") * 2).floor() / 2).alias("snr_bin05"),
                pl.col("snr").clip(-20, 50).alias("snr_clip"),
            ]
        )

    if "pathloss" in lf.columns:
        derived_exprs.append(pl.col("pathloss").round(0).alias("pathloss_round"))

    if "mod" in lf.columns:
        derived_exprs.append(pl.col("mod").cast(pl.Categorical).to_physical().alias("mod_code"))

    if "snr" in lf.columns and "cqi" in lf.columns:
        derived_exprs.append((pl.col("snr") * pl.col("cqi").cast(pl.Float64)).alias("snr_cqi"))

    if "snr" in lf.columns and "pathloss" in lf.columns:
        derived_exprs.append((pl.col("snr") * pl.col("pathloss")).alias("snr_pathloss"))

    # Label: pass when BLER <= Target and BLKErr == 1 (ACK)
    if all(c in lf.columns for c in ("bler", "target_bler", "blkerr")):
        derived_exprs.append(
            (
                (pl.col("bler") <= pl.col("target_bler")) & (pl.col("blkerr") == 1)
            )
            .cast(pl.Int8)
            .alias("label_pass")
        )

    return lf.with_columns(derived_exprs) if derived_exprs else lf


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


def _arrow_to_polars_dtype(t: pa.DataType):
    import pyarrow as pa  # local for clarity

    if pa.types.is_float64(t):
        return pl.Float64
    if pa.types.is_float32(t):
        return pl.Float32
    if pa.types.is_int64(t):
        return pl.Int64
    if pa.types.is_int32(t):
        return pl.Int32
    if pa.types.is_int16(t):
        return pl.Int16
    if pa.types.is_int8(t):
        return pl.Int8
    if pa.types.is_uint64(t):
        return pl.UInt64
    if pa.types.is_uint32(t):
        return pl.UInt32
    if pa.types.is_uint16(t):
        return pl.UInt16
    if pa.types.is_uint8(t):
        return pl.UInt8
    if pa.types.is_boolean(t):
        return pl.Boolean
    # Fallback to Utf8
    return pl.Utf8


def _align_select(df: pl.DataFrame, schema_cols: List[str], schema_arrow: pa.Schema) -> pl.DataFrame:
    """Ensure df has all schema columns with correct dtypes and order.

    - Adds missing columns as nulls cast to expected dtype
    - Casts existing columns to expected dtype when they differ (incl. Null)
    - Reorders columns to match writer schema order
    """
    # Add missing columns with correct dtype
    missing_exprs: List[pl.Expr] = []
    for field in schema_arrow:
        name = field.name
        if name not in df.columns:
            pdt = _arrow_to_polars_dtype(field.type)
            missing_exprs.append(pl.lit(None).cast(pdt).alias(name))
    if missing_exprs:
        df = df.with_columns(missing_exprs)

    # Cast columns to expected dtype when necessary
    cast_exprs: List[pl.Expr] = []
    for field in schema_arrow:
        name = field.name
        if name in df.columns:
            expected = _arrow_to_polars_dtype(field.type)
            # Polars Null/other mismatches should be cast
            try:
                cur = df.schema[name]
            except Exception:
                cur = None
            if cur != expected:
                cast_exprs.append(pl.col(name).cast(expected, strict=False))
    if cast_exprs:
        df = df.with_columns(cast_exprs)

    # Select and order
    return df.select(schema_cols)


def _detect_separator(path: Path) -> str:
    """Robustly detect delimiter by sniffing the header.

    Many .tab files in the wild are actually comma-separated. We inspect the
    first chunk to choose between tab, comma, or semicolon. Fallback to comma.
    """
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            head = f.read(65536)
    except Exception:
        # Fall back to extension heuristic if read fails
        ext = path.suffix.lower()
        return "\t" if ext in {".tsv", ".tab"} else ","
    # Count common delimiters
    cnt_tab = head.count("\t")
    cnt_comma = head.count(",")
    cnt_semicolon = head.count(";")
    # Prefer the delimiter with the highest count
    if max(cnt_tab, cnt_comma, cnt_semicolon) == 0:
        # No obvious delimiter; default by extension
        ext = path.suffix.lower()
        return "\t" if ext in {".tsv", ".tab"} else ","
    if cnt_tab >= cnt_comma and cnt_tab >= cnt_semicolon and cnt_tab > 0:
        return "\t"
    if cnt_comma >= cnt_semicolon and cnt_comma > 0:
        return ","
    if cnt_semicolon > 0:
        return ";"
    return ","


def _read_header(path: Path) -> pl.DataFrame:
    sep = _detect_separator(path)
    return pl.read_csv(path, n_rows=0, separator=sep)


def _scan_file(path: Path) -> pl.LazyFrame:
    sep = _detect_separator(path)
    return pl.scan_csv(path, separator=sep)


def process_file_split(
    csv_path: Path,
    writer_train: pq.ParquetWriter,
    writer_test: pq.ParquetWriter,
    chunksize: int,
    schema_cols: List[str],
    schema_arrow: pa.Schema,
    rng: "np.random.Generator",
    test_frac: float,
    start_offset: int = 0,
    writer_combined: Optional[pq.ParquetWriter] = None,
) -> None:
    header_df = _read_header(csv_path)
    rename_map = build_rename_map(header_df.columns)

    offset = int(start_offset)
    total = 0
    while True:
        lf = _scan_file(csv_path).slice(offset, chunksize).rename(rename_map)
        lf = engineer_features(lf)
        df = lf.collect()
        if df.height == 0:
            break
        total += df.height
        offset += df.height

        # Select final columns and align to schema
        used_columns = finalize_columns(df.columns)
        df = df.select([c for c in used_columns if c in schema_cols])
        df = _align_select(df, schema_cols, schema_arrow)

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
    ap.add_argument(
        "--include-glob",
        type=str,
        default="*",
        help="Glob pattern to include files (CSV/TAB; default: *). Only .csv/.tab/.tsv are processed.",
    )
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

    allowed_exts = {".csv", ".tsv", ".tab", ".CSV", ".TSV", ".TAB"}
    csvs = sorted([p for p in data_dir.glob(args.include_glob) if p.is_file() and p.suffix in allowed_exts])
    if args.max_files and args.max_files > 0:
        csvs = csvs[: args.max_files]
    if not csvs:
        print(f"No CSV/TAB files found in {data_dir} matching '{args.include_glob}'")
        return

    # Initialize writer lazily after first chunk to get schema
    writer_train: Optional[pq.ParquetWriter] = None
    writer_test: Optional[pq.ParquetWriter] = None
    writer_combined: Optional[pq.ParquetWriter] = None
    schema_cols: Optional[List[str]] = None
    rng = __import__("numpy").random.default_rng(args.seed)
    try:
        for p in tqdm(csvs, desc="Featurizing CSV/TAB"):
            # For the first file/chunk, create writers with that schema
            if writer_train is None or writer_test is None:
                # Peek a small slice to get engineered schema
                lf0 = engineer_features(
                    _scan_file(p)
                    .slice(0, min(10_000, args.chunksize))
                    .rename(build_rename_map(_read_header(p).columns))
                )
                df0 = lf0.collect()
                df0 = df0.select(finalize_columns(df0.columns))
                tbl0 = df0.to_arrow()
                schema_cols = df0.columns
                schema_arrow = tbl0.schema
                if train_path:
                    writer_train = pq.ParquetWriter(train_path, schema_arrow, compression="snappy")
                if test_path:
                    writer_test = pq.ParquetWriter(test_path, schema_arrow, compression="snappy")
                if combined_path:
                    writer_combined = pq.ParquetWriter(combined_path, schema_arrow, compression="snappy")

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
                    schema_arrow,
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
                    schema_arrow,  # type: ignore[arg-type]
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

"""
CyberForeSight AI — Ingestion Pipeline (Phase 1)
Project: SIH26153 · AI-based Network Attack Forecasting from Network Traffic Data (NTRO)

Module: src/ingestion.py
Description:
    Loads raw CIC-IDS2018 CSV data from `data/raw/`, cleans whitespace from column
    names, handles repeated header rows, and drops rows containing infinite or NaN values.
    Adheres to RULES.md Rule 3: all drops are logged with exact counts and reasons.
"""

from __future__ import annotations

import argparse
import csv as _csv_module
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Setup module logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ingestion")

# Common representations of infinite strings in CICFlowMeter output
INF_STRINGS = [
    "Infinity",
    "-Infinity",
    "infinity",
    "-infinity",
    "inf",
    "-inf",
    "INF",
    "-INF",
]

DEFAULT_RAW_DIR = Path("data/raw")

# Values that are considered benign/normal in label columns (case-insensitive)
BENIGN_LABELS = {
    "benign",
    "normal",
    "normal_traffic",
    "genuine_traffic",
    "legitimate",
    "none",
    "no_attack",
    "clean",
}

# Canonical snake_case feature names. Incoming columns are normalized via these
# aliases so that any CIC-IDS2018-like schema (or a partial/renamed variant)
# maps onto the pipeline's expected feature set. Unknown columns are kept as-is.
CANONICAL_COLUMN_ALIASES: Dict[str, str] = {
    "timestamp": "timestamp",
    "time": "timestamp",
    "date": "timestamp",
    "flow_start_time": "timestamp",
    "src_ip": "src_ip",
    "source_ip": "src_ip",
    "ip_src": "src_ip",
    "dst_ip": "dst_ip",
    "destination_ip": "dst_ip",
    "ip_dst": "dst_ip",
    "src_port": "src_port",
    "source_port": "src_port",
    "sport": "src_port",
    "dst_port": "dst_port",
    "destination_port": "dst_port",
    "dport": "dst_port",
    "protocol": "protocol",
    "proto": "protocol",
    "flow_duration": "flow_duration",
    "duration": "flow_duration",
    "tot_fwd_pkts": "tot_fwd_pkts",
    "total_fwd_pkts": "tot_fwd_pkts",
    "total_number_of_fwd_packets": "tot_fwd_pkts",
    "fwd_packets_total": "tot_fwd_pkts",
    "tot_bwd_pkts": "tot_bwd_pkts",
    "total_bwd_pkts": "tot_bwd_pkts",
    "total_number_of_bwd_packets": "tot_bwd_pkts",
    "bwd_packets_total": "tot_bwd_pkts",
    "totlen_fwd_pkts": "totlen_fwd_pkts",
    "total_len_fwd_pkts": "totlen_fwd_pkts",
    "total_length_of_fwd_packets": "totlen_fwd_pkts",
    "fwd_header_length": "totlen_fwd_pkts",
    "totlen_bwd_pkts": "totlen_bwd_pkts",
    "total_len_bwd_pkts": "totlen_bwd_pkts",
    "total_length_of_bwd_packets": "totlen_bwd_pkts",
    "bwd_header_length": "totlen_bwd_pkts",
    "flow_byts_s": "flow_byts_s",
    "flow_bytes_s": "flow_byts_s",
    "flow_pkts_s": "flow_pkts_s",
    "flow_packets_s": "flow_pkts_s",
    "flow_iat_mean": "flow_iat_mean",
    "flow_iat_std": "flow_iat_std",
    "flow_iat_max": "flow_iat_max",
    "flow_iat_min": "flow_iat_min",
    "fwd_iat_total": "fwd_iat_total",
    "fwd_iat_mean": "fwd_iat_mean",
    "fwd_iat_std": "fwd_iat_std",
    "fwd_iat_max": "fwd_iat_max",
    "fwd_iat_min": "fwd_iat_min",
    "bwd_iat_total": "bwd_iat_total",
    "bwd_iat_mean": "bwd_iat_mean",
    "bwd_iat_std": "bwd_iat_std",
    "bwd_iat_max": "bwd_iat_max",
    "bwd_iat_min": "bwd_iat_min",
    "fwd_pkt_len_max": "fwd_pkt_len_max",
    "fwd_pkt_len_min": "fwd_pkt_len_min",
    "fwd_pkt_len_mean": "fwd_pkt_len_mean",
    "fwd_pkt_len_std": "fwd_pkt_len_std",
    "bwd_pkt_len_max": "bwd_pkt_len_max",
    "bwd_pkt_len_min": "bwd_pkt_len_min",
    "bwd_pkt_len_mean": "bwd_pkt_len_mean",
    "bwd_pkt_len_std": "bwd_pkt_len_std",
    "pkt_len_max": "pkt_len_max",
    "pkt_len_min": "pkt_len_min",
    "pkt_len_mean": "pkt_len_mean",
    "pkt_len_std": "pkt_len_std",
    "fin_flag_cnt": "fin_flag_cnt",
    "fin_flag_count": "fin_flag_cnt",
    "syn_flag_cnt": "syn_flag_cnt",
    "syn_flag_count": "syn_flag_cnt",
    "syn_flags": "syn_flag_cnt",
    "rst_flag_cnt": "rst_flag_cnt",
    "rst_flag_count": "rst_flag_cnt",
    "psh_flag_cnt": "psh_flag_cnt",
    "psh_flag_count": "psh_flag_cnt",
    "ack_flag_cnt": "ack_flag_cnt",
    "ack_flag_count": "ack_flag_cnt",
    "urg_flag_cnt": "urg_flag_cnt",
    "urg_flag_count": "urg_flag_cnt",
    "init_fwd_win_byts": "init_fwd_win_byts",
    "initial_window_bytes_fwd": "init_fwd_win_byts",
    "init_bwd_win_byts": "init_bwd_win_byts",
    "initial_window_bytes_bwd": "init_bwd_win_byts",
    "active_mean": "active_mean",
    "active_std": "active_std",
    "active_max": "active_max",
    "active_min": "active_min",
    "idle_mean": "idle_mean",
    "idle_std": "idle_std",
    "idle_max": "idle_max",
    "idle_min": "idle_min",
    "label": "label",
    "labels": "label",
    "class": "label",
    "category": "label",
    "attack_type": "label",
    "attack_stage": "attack_stage",
    "stage": "attack_stage",
    "tactics": "attack_stage",
}

# Non-numeric / identifier columns that must never be numeric-coerced.
NON_NUMERIC_COLUMNS = {
    "timestamp",
    "label",
    "attack_stage",
    "src_ip",
    "dst_ip",
    "src_port",
}


def _snake_case(name: str) -> str:
    """Converts an arbitrary column name to a compact snake_case key."""
    s = str(name).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes column names to canonical snake_case using the alias map.
    Duplicate columns created by aliasing are collapsed (first kept).
    """
    if df is None or df.empty or len(df.columns) == 0:
        return df
    renamed = {}
    for col in df.columns:
        key = _snake_case(col)
        canonical = CANONICAL_COLUMN_ALIASES.get(key, key)
        renamed[col] = canonical
    df = df.rename(columns=renamed)
    df = df.loc[:, ~df.columns.duplicated()]
    return df


def detect_encoding(path: Union[str, Path]) -> str:
    """Detects a readable text encoding for a CSV file."""
    sample = Path(path).read_bytes()[:8192]
    for enc in ("utf-8-sig", "utf-8", "utf-16", "utf-16-le", "cp1252", "latin-1"):
        try:
            sample.decode(enc)
            return enc
        except (UnicodeDecodeError, LookupError):
            continue
    return "latin-1"


def detect_separator(path: Union[str, Path], encoding: str = "utf-8-sig") -> str:
    """Detects the CSV field separator using csv.Sniffer."""
    try:
        with open(path, encoding=encoding, errors="ignore") as f:
            sample = f.read(4096)
        dialect = _csv_module.Sniffer().sniff(sample, delimiters=",;\t|")
        return dialect.delimiter
    except Exception:
        return ","


def read_csv_robust(
    path: Union[str, Path],
    nrows: Optional[int] = None,
    chunksize: Optional[int] = None,
) -> Union[pd.DataFrame, pd.io.parsers.TextFileReader]:
    """
    Reads a CSV using detected encoding and separator. Falls back gracefully
    if inference fails. Preserves string columns that are not numeric.
    """
    p = Path(path)
    enc = detect_encoding(p)
    sep = detect_separator(p, enc)

    try:
        reader = pd.read_csv(
            p,
            nrows=nrows,
            chunksize=chunksize,
            sep=sep,
            encoding=enc,
            engine="python",
            low_memory=False,
        )
        return reader
    except Exception:
        logger.warning(f"Robust CSV read failed ({sep=} {enc=}); retrying with defaults.")
        return pd.read_csv(
            p,
            nrows=nrows,
            chunksize=chunksize,
            low_memory=False,
        )


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strips leading and trailing whitespace from all column names.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with stripped column names.
    """
    initial_cols = list(df.columns)
    df.columns = df.columns.astype(str).str.strip()
    
    renamed_count = sum(1 for old, new in zip(initial_cols, df.columns) if old != new)
    if renamed_count > 0:
        logger.debug(f"Stripped whitespace from {renamed_count} column names.")
    return df


def remove_repeated_headers(
    df: pd.DataFrame, header_indicator_col: str = "dst_port"
) -> Tuple[pd.DataFrame, int]:
    """
    Identifies and removes duplicate CSV header rows embedded within data rows.
    In CIC-IDS2018, file concatenations often introduce lines where column values
    repeat the header names (e.g., Dst Port == 'Dst Port'). Falls back to the
    first column if the preferred indicator column is absent.

    Args:
        df: Input DataFrame.
        header_indicator_col: Column name to check for repetition.

    Returns:
        Tuple of (cleaned DataFrame, count of header rows removed).
    """
    if df is None or len(df) == 0:
        return df, 0

    indicator = header_indicator_col
    if indicator not in df.columns:
        indicator = df.columns[0]

    header_mask = df[indicator].astype(str).str.strip() == str(indicator)
    dropped_headers = int(header_mask.sum())
    if dropped_headers > 0:
        df = df[~header_mask].copy()
        logger.info(
            f"Dropped {dropped_headers:,} duplicate header row(s) matching '{indicator}'."
        )
    return df, dropped_headers


def clean_data_with_report(
    df: pd.DataFrame,
    source_name: str = "dataset",
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Cleans DataFrame according to Phase 1 specifications and RULES.md, returning
    a human-readable drop report:

      1. Strips whitespace from column headers.
      2. Normalizes column names to canonical snake_case via alias mapping.
      3. Removes embedded duplicate header rows.
      4. Coerces numeric columns appropriately (string/numeric infinities -> NaN).
      5. Imputes remaining NaN values with column medians instead of dropping
         valid rows (NaN is handled safely and logged, per RULES.md).
      6. Drops only rows that are truly invalid: empty rows or rows where the
         timestamp cannot be parsed later are handled downstream with counts.

    Args:
        df: Raw DataFrame to clean.
        source_name: Identifier for logging/audit purposes (e.g. filename).

    Returns:
        Tuple of (cleaned DataFrame, report dictionary with drop counts/reasons).
    """
    report: Dict[str, object] = {
        "source": source_name,
        "rows_loaded": int(len(df)),
        "rows_kept": 0,
        "rows_dropped_headers": 0,
        "rows_dropped_empty": 0,
        "rows_dropped_inf_nan": 0,
        "rows_imputed_nan": 0,
        "columns_removed_all_nan": 0,
        "reasons": [],
    }

    if df is None:
        report["reasons"].append("No DataFrame provided.")
        report["rows_kept"] = 0
        return df, report

    initial_row_count = len(df)
    logger.info(f"[{source_name}] Starting data cleaning on {initial_row_count:,} raw rows...")

    # Step 1: Clean + normalize column names
    df = clean_column_names(df)
    df = normalize_columns(df)
    report["columns"] = list(df.columns)

    # Step 2: Remove duplicate header rows embedded in the data
    df, dropped_headers = remove_repeated_headers(df, header_indicator_col="dst_port")
    report["rows_dropped_headers"] = int(dropped_headers)
    if dropped_headers:
        report["reasons"].append(f"Dropped {dropped_headers:,} duplicate CSV header row(s).")

    # Step 3: Replace string representations of infinity with np.nan
    df = df.replace(INF_STRINGS, np.nan)

    # Step 4: Identify numeric columns and coerce non-numeric columns to numeric.
    # Non-numeric identifier/string columns are preserved as-is.
    for col in df.columns:
        if col in NON_NUMERIC_COLUMNS:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Replace remaining inf/-inf with NaN
    if len(df) > 0:
        inf_mask = np.isinf(df.select_dtypes(include=[np.number]))
        if inf_mask.any().any():
            df = df.replace([np.inf, -np.inf], np.nan)

    # Step 5: Drop columns that became all-NaN (unusable)
    if len(df) > 0:
        numeric_df = df.select_dtypes(include=[np.number])
        all_nan_cols = numeric_df.columns[numeric_df.isna().all()]
        if len(all_nan_cols) > 0:
            df = df.drop(columns=list(all_nan_cols))
            report["columns_removed_all_nan"] = int(len(all_nan_cols))
            report["reasons"].append(
                f"Removed {len(all_nan_cols)} column(s) that were entirely non-numeric/NaN: "
                f"{', '.join(list(all_nan_cols))}."
            )

    # Step 6: Drop empty rows (all fields blank)
    empty_mask = df.isna().all(axis=1) | (df.astype(str).apply(lambda r: r.str.strip().eq("").all(), axis=1))
    if len(df) > 0:
        df = df[~empty_mask]
        report["rows_dropped_empty"] = int(empty_mask.sum())
        if int(empty_mask.sum()) > 0:
            report["reasons"].append(f"Dropped {int(empty_mask.sum()):,} empty row(s).")

    # Step 7: Impute remaining NaN in numeric columns with column median instead
    # of dropping rows (safe, logged per RULES.md).
    if len(df) > 0:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        nan_counts = int(df[numeric_cols].isna().sum().sum()) if len(numeric_cols) else 0
        if nan_counts > 0:
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            report["rows_imputed_nan"] = nan_counts
            report["reasons"].append(
                f"Imputed {nan_counts:,} NaN value(s) across numeric columns using column medians."
            )

    report["rows_kept"] = int(len(df))
    dropped_total = report["rows_dropped_headers"] + report["rows_dropped_empty"]
    retention_pct = (report["rows_kept"] / initial_row_count * 100) if initial_row_count > 0 else 0.0

    logger.info(
        f"[{source_name}] Cleaning complete: retained {report['rows_kept']:,}/{initial_row_count:,} "
        f"rows ({retention_pct:.2f}%). Total dropped: {dropped_total:,} "
        f"(headers: {report['rows_dropped_headers']:,}, empty: {report['rows_dropped_empty']:,})."
    )

    return df, report


def clean_data(
    df: pd.DataFrame,
    source_name: str = "dataset",
) -> pd.DataFrame:
    """
    Backward-compatible wrapper around clean_data_with_report.

    Args:
        df: Raw DataFrame to clean.
        source_name: Identifier for logging/audit purposes.

    Returns:
        Cleaned DataFrame.
    """
    df_clean, _ = clean_data_with_report(df, source_name=source_name)
    return df_clean


def load_raw_csv(
    file_path: Union[str, Path],
    nrows: Optional[int] = None,
    chunksize: Optional[int] = None,
) -> pd.DataFrame:
    """
    Loads and cleans a single raw CSV file from disk.
    Uses separator/encoding auto-detection for robustness.

    Args:
        file_path: Path to the raw CSV file.
        nrows: Optional number of rows to read (useful for inspection/testing).
        chunksize: Optional chunk size for large files to reduce memory pressure.

    Returns:
        Cleaned pandas DataFrame.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Raw CSV file does not exist: {path}")

    logger.info(f"Loading raw file: {path.name} (size: {path.stat().st_size / (1024 * 1024):.1f} MB)")

    if chunksize is not None and chunksize > 0:
        chunks: List[pd.DataFrame] = []
        chunk_idx = 0
        reader = read_csv_robust(path, nrows=nrows, chunksize=chunksize)
        for chunk in reader:
            cleaned_chunk = clean_data(chunk, source_name=f"{path.name}#chunk_{chunk_idx}")
            chunks.append(cleaned_chunk)
            chunk_idx += 1
        df_cleaned = pd.concat(chunks, ignore_index=True)
    else:
        raw_df = read_csv_robust(path, nrows=nrows)
        df_cleaned = clean_data(raw_df, source_name=path.name)

    return df_cleaned


def load_raw_csv_with_report(
    file_path: Union[str, Path],
    nrows: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Loads and cleans a single raw CSV with a drop/clean report for auditing.

    Args:
        file_path: Path to the raw CSV file.
        nrows: Optional number of rows to read.

    Returns:
        Tuple of (cleaned DataFrame, drop report).
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Raw CSV file does not exist: {path}")

    logger.info(f"Loading raw file with report: {path.name} (size: {path.stat().st_size / (1024 * 1024):.1f} MB)")

    raw_df = read_csv_robust(path, nrows=nrows)
    if isinstance(raw_df, pd.io.parsers.TextFileReader):
        raw_df = pd.concat(list(raw_df), ignore_index=True)

    df_cleaned, report = clean_data_with_report(raw_df, source_name=path.name)
    return df_cleaned, report


def load_raw_data(
    data_dir: Union[str, Path] = DEFAULT_RAW_DIR,
    pattern: str = "*.csv",
    nrows_per_file: Optional[int] = None,
    chunksize: Optional[int] = None,
    concatenate: bool = True,
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Discovers, loads, and cleans all raw CSV files in data/raw/.

    Args:
        data_dir: Path to directory containing raw CSVs (default: data/raw).
        pattern: Glob pattern to identify raw CSV files.
        nrows_per_file: Optional limit on rows per file (for testing/quick evaluation).
        chunksize: Optional chunk size for processing large CSV files.
        concatenate: If True, concatenates all cleaned files into one DataFrame.
                     If False, returns a dictionary mapping filename to DataFrame.

    Returns:
        Single unified DataFrame or dictionary of DataFrames keyed by filename.
    """
    dir_path = Path(data_dir)
    if not dir_path.is_dir():
        raise FileNotFoundError(f"Raw data directory does not exist: {dir_path}")

    csv_files = sorted(list(dir_path.glob(pattern)))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {dir_path} matching '{pattern}'")

    logger.info(f"Found {len(csv_files)} raw CSV file(s) in '{dir_path}'.")

    cleaned_dfs: Dict[str, pd.DataFrame] = {}
    for file_path in csv_files:
        df = load_raw_csv(file_path, nrows=nrows_per_file, chunksize=chunksize)
        cleaned_dfs[file_path.name] = df

    if concatenate:
        logger.info(f"Concatenating {len(cleaned_dfs)} cleaned DataFrames into a single dataset...")
        unified_df = pd.concat(list(cleaned_dfs.values()), ignore_index=True)
        logger.info(
            f"Unified dataset shape: {unified_df.shape[0]:,} rows x {unified_df.shape[1]} columns."
        )
        return unified_df

    return cleaned_dfs


# Alias for backward compatibility / explicit naming
load_all_raw_data = load_raw_data


def inspect_dataset(df: pd.DataFrame, title: str = "Dataset Inspection") -> Dict[str, object]:
    """
    Inspects columns, missing value counts, and label distribution
    per Phase 1 Step 13.

    Args:
        df: The DataFrame to inspect.
        title: Title for the log output.

    Returns:
        Dictionary containing summary statistics.
    """
    summary: Dict[str, object] = {
        "rows": len(df),
        "columns": len(df.columns),
        "nan_count": int(df.isna().sum().sum()),
        "labels": {},
    }

    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")
    print(f"Total Rows:     {summary['rows']:,}")
    print(f"Total Columns:  {summary['columns']}")
    print(f"Total NaN/Null: {summary['nan_count']}")

    if "Label" in df.columns:
        label_dist = df["Label"].value_counts().to_dict()
        summary["labels"] = label_dist
        print("\nLabel Distribution:")
        for lbl, count in label_dist.items():
            pct = (count / len(df)) * 100
            print(f"  - {lbl:<25}: {count:>10,} ({pct:6.2f}%)")
    else:
        print("Warning: 'Label' column not present in DataFrame.")

    print(f"{'=' * 60}\n")
    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="CyberForeSight AI — Ingestion Pipeline (Phase 1)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DEFAULT_RAW_DIR),
        help="Path to raw CSV directory (default: data/raw)",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Specific raw CSV filename in data-dir to load. If omitted, loads all CSVs.",
    )
    parser.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="Number of rows to read per file (useful for fast verification).",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=None,
        help="Optional chunksize for chunked processing.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI execution entrypoint."""
    args = parse_args()

    data_dir = Path(args.data_dir)

    if args.file:
        target_file = data_dir / args.file if not Path(args.file).is_absolute() else Path(args.file)
        df = load_raw_csv(target_file, nrows=args.nrows, chunksize=args.chunksize)
        inspect_dataset(df, title=f"Summary for {target_file.name}")
    else:
        df = load_all_raw_data(
            data_dir=data_dir,
            nrows_per_file=args.nrows,
            chunksize=args.chunksize,
            concatenate=True,
        )
        inspect_dataset(df, title="Summary for All Raw CSV Files")


if __name__ == "__main__":
    main()

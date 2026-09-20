"""
CyberForeSight AI — Timestamp Audit Module
Phase 2: Validates timestamp quality before windowing.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.constants_v2 import RAW_DATA_FILES, WINDOW_SIZE_SECONDS
from src.ingestion import load_raw_csv, clean_data

logger = logging.getLogger("timestamp_audit")

def audit_timestamps(
    raw_dir: Path,
    date: str,
    nrows: Optional[int] = None,
) -> Dict:
    """
    Audits timestamp quality for a specific date's raw CSV.
    
    Returns detailed counts and statistics.
    """
    filename = RAW_DATA_FILES.get(date)
    if not filename:
        raise ValueError(f"No raw file configured for date: {date}")
    
    file_path = raw_dir / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Raw file not found: {file_path}")
    
    logger.info(f"Auditing timestamps for {date} ({filename})")
    
    # Load raw data
    df = load_raw_csv(file_path, nrows=nrows)
    df = clean_data(df, source_name=filename)
    
    total_rows = len(df)
    
    # Resolve timestamp column
    timestamp_col = None
    for cand in ("timestamp", "Timestamp", "time", "date"):
        if cand in df.columns:
            timestamp_col = cand
            break
    
    if timestamp_col is None:
        raise ValueError(f"No timestamp column found in {filename}")
    
    # Parse with detailed tracking
    parsed_ts = pd.to_datetime(
        df[timestamp_col], format="%d/%m/%Y %H:%M:%S", errors="coerce"
    )
    
    # Count 1970-01-01 timestamps (epoch default from failed parsing)
    epoch_ts = pd.Timestamp("1970-01-01")
    epoch_mask = parsed_ts == epoch_ts
    epoch_count = int(epoch_mask.sum())
    
    # If format parsing left nulls, try generic parser
    if parsed_ts.isna().any():
        null_mask = parsed_ts.isna()
        parsed_ts.loc[null_mask] = pd.to_datetime(
            df.loc[null_mask, timestamp_col], dayfirst=True, errors="coerce"
        )
    
    invalid_count = int(parsed_ts.isna().sum())
    valid_count = total_rows - invalid_count
    
    valid_timestamps = parsed_ts[~parsed_ts.isna() & (parsed_ts != epoch_ts)]
    earliest_valid = valid_timestamps.min() if len(valid_timestamps) > 0 else None
    latest_valid = valid_timestamps.max() if len(valid_timestamps) > 0 else None
    
    # Check for 1970 dates after generic parsing
    epoch_after_generic = int((parsed_ts == epoch_ts).sum())
    
    result = {
        "date": date,
        "file": filename,
        "total_rows": total_rows,
        "timestamp_column": timestamp_col,
        "invalid_timestamps": invalid_count,
        "epoch_1970_count_initial": epoch_count,
        "epoch_1970_count_after_generic": epoch_after_generic,
        "valid_timestamps": valid_count,
        "earliest_valid": str(earliest_valid) if earliest_valid is not None else None,
        "latest_valid": str(latest_valid) if latest_valid is not None else None,
        "retention_rate": (valid_count / total_rows * 100) if total_rows > 0 else 0.0,
    }
    
    logger.info(
        f"Timestamp audit {date}: total={total_rows:,}, "
        f"valid={valid_count:,}, invalid={invalid_count:,}, "
        f"epoch_1970={epoch_count:,}, range={result['earliest_valid']} to {result['latest_valid']}"
    )
    
    return result


def audit_all_dates(
    raw_dir: Path,
    dates: List[str],
    nrows: Optional[int] = None,
) -> List[Dict]:
    """Audit timestamps for multiple dates."""
    results = []
    for date in dates:
        results.append(audit_timestamps(raw_dir, date, nrows=nrows))
    return results


def print_timestamp_audit_report(results: List[Dict]) -> None:
    """Print a formatted timestamp audit report."""
    print("\n" + "=" * 80)
    print("TIMESTAMP QUALITY AUDIT REPORT")
    print("=" * 80)
    for r in results:
        print(f"\nDate: {r['date']} ({r['file']})")
        print(f"  Total rows:           {r['total_rows']:,}")
        print(f"  Timestamp column:     {r['timestamp_column']}")
        print(f"  Valid timestamps:     {r['valid_timestamps']:,} ({r['retention_rate']:.2f}%)")
        print(f"  Invalid timestamps:   {r['invalid_timestamps']:,}")
        print(f"  1970-01-01 (initial): {r['epoch_1970_count_initial']:,}")
        print(f"  1970-01-01 (final):   {r['epoch_1970_count_after_generic']:,}")
        print(f"  Earliest valid:       {r['earliest_valid']}")
        print(f"  Latest valid:         {r['latest_valid']}")
    print("=" * 80 + "\n")
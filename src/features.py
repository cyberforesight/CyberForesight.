"""
CyberForeSight AI — Feature Engineering & Time-Window Pipeline (Phase 1)
Project: SIH26153 · AI-based Network Attack Forecasting from Network Traffic Data (NTRO)

Module: src/features.py
Description:
    Implements Phase 1 (Steps 15-18) of PHASES.md:
      1. Parses timestamp columns and floors them to 60-second non-overlapping buckets.
      2. Groups flows by time windows and aggregates flow and packet-level metrics (state vector S_t).
      3. Labels each window with a binary target `is_attack_window` (1 if any non-Benign flow, 0 otherwise).
      4. Normalizes feature metrics using scikit-learn's StandardScaler.
      5. Saves the fitted scaler to `models/scaler.joblib` for inference reuse.
      6. Exports the normalized feature matrix to `data/windows/window_features_normalized.csv`.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.ingestion import BENIGN_LABELS, clean_data, load_raw_csv, read_csv_robust
except ImportError:
    from ingestion import BENIGN_LABELS, clean_data, load_raw_csv, read_csv_robust

# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("features")

DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_WINDOWS_DIR = Path("data/windows")
DEFAULT_OUTPUT_CSV = DEFAULT_WINDOWS_DIR / "window_features_normalized.csv"
DEFAULT_MODELS_DIR = Path("models")
DEFAULT_SCALER_PATH = DEFAULT_MODELS_DIR / "scaler.joblib"

# Feature aggregation mapping for 60-second window state vector S_t
# Keys are canonical snake_case names produced by ingestion.normalize_columns,
# matched via WINDOW_AGGREGATION_FEATURES (alias table) below.
WINDOW_AGGREGATIONS: Dict[str, List[str]] = {
    # Flow durations
    "flow_duration": ["mean", "max"],
    # Packet volumes
    "tot_fwd_pkts": ["sum", "mean"],
    "tot_bwd_pkts": ["sum", "mean"],
    # Byte volumes
    "totlen_fwd_pkts": ["sum", "mean"],
    "totlen_bwd_pkts": ["sum", "mean"],
    # Packet length statistics
    "fwd_pkt_len_max": ["max"],
    "fwd_pkt_len_mean": ["mean"],
    "bwd_pkt_len_max": ["max"],
    "bwd_pkt_len_mean": ["mean"],
    "pkt_len_max": ["max"],
    "pkt_len_mean": ["mean"],
    "pkt_len_min": ["min"],
    # Rates and throughput
    "flow_byts_s": ["mean", "max"],
    "flow_pkts_s": ["mean", "max"],
    # Inter-arrival times
    "flow_iat_mean": ["mean"],
    "flow_iat_max": ["max"],
    # TCP flags (SYN flood, teardown, scan signatures)
    "fin_flag_cnt": ["sum"],
    "syn_flag_cnt": ["sum"],
    "rst_flag_cnt": ["sum"],
    "psh_flag_cnt": ["sum"],
    "ack_flag_cnt": ["sum"],
    "urg_flag_cnt": ["sum"],
    # TCP window sizes
    "init_fwd_win_byts": ["mean"],
    "init_bwd_win_byts": ["mean"],
    # Active & Idle timings
    "active_mean": ["mean"],
    "idle_mean": ["mean"],
    # Port & Protocol diversity (reconnaissance / scan detection)
    "dst_port": ["nunique"],
    "protocol": ["nunique"],
}

# Aliases for incoming CIC-IDS2018 column names -> canonical aggregation columns.
# During build_feature_pipeline, any raw files are normalized to canonical
# snake_case names via ingestion.normalize_columns, so this table is a fallback
# for legacy in-memory DataFrames that bypass ingestion.
FEATURE_COLUMN_ALIASES: Dict[str, str] = {
    "tot_fwd_pkts": "tot_fwd_pkts",
    "tot_bwd_pkts": "tot_bwd_pkts",
    "totlen_fwd_pkts": "totlen_fwd_pkts",
    "totlen_bwd_pkts": "totlen_bwd_pkts",
    "fwd_pkt_len_max": "fwd_pkt_len_max",
    "fwd_pkt_len_mean": "fwd_pkt_len_mean",
    "bwd_pkt_len_max": "bwd_pkt_len_max",
    "bwd_pkt_len_mean": "bwd_pkt_len_mean",
    "pkt_len_max": "pkt_len_max",
    "pkt_len_mean": "pkt_len_mean",
    "pkt_len_min": "pkt_len_min",
    "flow_byts_s": "flow_byts_s",
    "flow_pkts_s": "flow_pkts_s",
    "flow_iat_mean": "flow_iat_mean",
    "flow_iat_max": "flow_iat_max",
    "fin_flag_cnt": "fin_flag_cnt",
    "syn_flag_cnt": "syn_flag_cnt",
    "rst_flag_cnt": "rst_flag_cnt",
    "psh_flag_cnt": "psh_flag_cnt",
    "ack_flag_cnt": "ack_flag_cnt",
    "urg_flag_cnt": "urg_flag_cnt",
    "init_fwd_win_byts": "init_fwd_win_byts",
    "init_bwd_win_byts": "init_bwd_win_byts",
    "active_mean": "active_mean",
    "idle_mean": "idle_mean",
    "dst_port": "dst_port",
    "protocol": "protocol",
    "flow_duration": "flow_duration",
}


def resolve_timestamp_col(df: pd.DataFrame, preferred: Optional[str] = None) -> str:
    """Resolves the timestamp column name from a DataFrame (case/alias tolerant)."""
    if preferred and preferred in df.columns:
        return preferred
    for cand in ("timestamp", "Timestamp", "time", "date", "window_timestamp"):
        if cand in df.columns:
            return cand
    return preferred or "timestamp"


def resolve_label_col(df: pd.DataFrame, preferred: Optional[str] = None) -> Optional[str]:
    """Resolves the ground-truth label column (label or attack_stage)."""
    if preferred and preferred in df.columns:
        return preferred
    for cand in ("label", "attack_stage", "labels", "class", "category", "attack_type"):
        if cand in df.columns:
            return cand
    return None


def parse_and_bucket_timestamps(
    df: pd.DataFrame,
    window_size_seconds: int = 60,
    timestamp_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Parses timestamp strings into datetime objects and floors them to
    non-overlapping time buckets of the specified window size.

    Args:
        df: Input DataFrame containing timestamp column.
        window_size_seconds: Window length in seconds (default: 60).
        timestamp_col: Name of the timestamp column (auto-detected if None).

    Returns:
        DataFrame with an added 'window_timestamp' column and invalid timestamps dropped.
    """
    if timestamp_col is None:
        timestamp_col = resolve_timestamp_col(df)

    if timestamp_col not in df.columns:
        raise ValueError(f"Required timestamp column '{timestamp_col}' not found in DataFrame.")

    logger.debug(f"Parsing '{timestamp_col}' and flooring to {window_size_seconds}s windows...")
    
    # Attempt high-performance format parsing first with fallback
    try:
        parsed_ts = pd.to_datetime(
            df[timestamp_col], format="%d/%m/%Y %H:%M:%S", errors="coerce"
        )
    except Exception:
        parsed_ts = pd.to_datetime(df[timestamp_col], dayfirst=True, errors="coerce")

    # If format parsing left nulls, try generic parser on remaining
    if parsed_ts.isna().any():
        null_mask = parsed_ts.isna()
        parsed_ts.loc[null_mask] = pd.to_datetime(
            df.loc[null_mask, timestamp_col], dayfirst=True, errors="coerce"
        )

    invalid_count = int(parsed_ts.isna().sum())
    if invalid_count > 0:
        logger.warning(
            f"Dropping {invalid_count:,} row(s) with unparseable timestamps."
        )
        valid_mask = parsed_ts.notna()
        df = df[valid_mask].copy()
        parsed_ts = parsed_ts[valid_mask]

    df["window_timestamp"] = parsed_ts.dt.floor(f"{window_size_seconds}s")
    return df


def aggregate_window_features(
    df: pd.DataFrame,
    window_col: str = "window_timestamp",
    label_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Groups flows by time window and computes aggregate metrics for flow and packet behavior,
    along with the binary ground truth `is_attack_window`.

    The label column is resolved automatically from 'label', 'attack_stage', or
    common synonyms and uses a benign set match (case-insensitive).

    Args:
        df: DataFrame containing flow records with window_timestamp and Label.
        window_col: Grouping column for time buckets.
        label_col: Ground truth flow label column (auto-resolved if None).

    Returns:
        Window-level DataFrame with state vector features and target label.
    """
    if window_col not in df.columns:
        raise ValueError(f"Grouping column '{window_col}' missing from DataFrame.")

    if label_col is None:
        label_col = resolve_label_col(df)
        if label_col:
            logger.info(f"Resolved label column: '{label_col}'")

    # Filter aggregation dict to columns present in df
    available_aggs = {
        col: aggs for col, aggs in WINDOW_AGGREGATIONS.items() if col in df.columns
    }
    missing_cols = set(WINDOW_AGGREGATIONS.keys()) - set(available_aggs.keys())
    if missing_cols:
        logger.warning(
            f"Columns not found in dataset for aggregation: {sorted(list(missing_cols))}"
        )

    logger.debug(f"Aggregating {len(df):,} flows across available metrics...")
    grouped = df.groupby(window_col, sort=True)

    # Perform multi-metric aggregation
    features_df = grouped.agg(available_aggs)

    # Flatten hierarchical MultiIndex column names into clean snake_case
    clean_columns = [
        f"{col}_{func}".strip().lower().replace(" ", "_").replace("/", "_")
        for col, func in features_df.columns
    ]
    features_df.columns = clean_columns

    # Add flow count per window
    features_df["flow_count"] = grouped.size()

    # Determine binary target is_attack_window:
    # 1 if any flow in this 60s window is not benign, 0 otherwise
    if label_col is not None and label_col in df.columns:
        def _is_attack(s: pd.Series) -> bool:
            return any(
                str(v).strip().lower() not in BENIGN_LABELS for v in s
            )

        features_df["is_attack_window"] = (
            grouped[label_col].apply(_is_attack)
        ).astype(int)
    else:
        logger.warning("No label column found; is_attack_window set to 0.")
        features_df["is_attack_window"] = 0

    # Fill any aggregation NaNs (e.g. variance or division edges) with 0.0
    features_df = features_df.fillna(0.0)

    # Reset index so window_timestamp becomes a standard column
    features_df = features_df.reset_index()

    # Ensure float dtypes for feature columns
    feature_cols = [c for c in features_df.columns if c not in {window_col, "is_attack_window"}]
    for c in feature_cols:
        features_df[c] = pd.to_numeric(features_df[c], errors="coerce").fillna(0.0)

    return features_df


def scale_feature_matrix(
    window_df: pd.DataFrame,
    scaler_path: Optional[Union[str, Path]] = DEFAULT_SCALER_PATH,
    fit: bool = True,
    existing_scaler: Optional[StandardScaler] = None,
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Normalizes numerical state vector features with StandardScaler.
    Excludes non-feature columns ('window_timestamp', 'is_attack_window').
    When fit=True, fits scaler and saves to scaler_path.
    When fit=False (inference mode), loads existing scaler and uses transform().
    NEVER overwrites scaler_path when fit=False.

    Args:
        window_df: Window-aggregated DataFrame.
        scaler_path: Filepath where scaler should be saved or loaded.
        fit: If True, fits scaler on window_df. If False, transforms with existing scaler.
        existing_scaler: Pre-fitted StandardScaler instance (if fit=False).

    Returns:
        Tuple of (normalized DataFrame, fitted/loaded StandardScaler).
    """
    non_feature_cols = {
        "window_timestamp",
        "is_attack_window",
        "label",
        "attack_stage",
        "src_ip",
        "dst_ip",
        "src_port",
    }
    feature_cols = [c for c in window_df.columns if c not in non_feature_cols]

    if not feature_cols:
        raise ValueError("No feature columns available for scaling.")

    logger.info(f"Scaling {len(feature_cols)} state-vector features with StandardScaler (fit={fit})...")

    if fit:
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(window_df[feature_cols])

        if scaler_path is not None:
            path = Path(scaler_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(scaler, path)
            logger.info(f"Fitted scaler saved successfully to: {path}")
        df_normalized = pd.DataFrame(scaled_values, columns=feature_cols)
    else:
        if existing_scaler is not None:
            scaler = existing_scaler
        elif scaler_path is not None and Path(scaler_path).exists():
            scaler = joblib.load(scaler_path)
            logger.info(f"Loaded existing scaler from: {scaler_path}")
        else:
            raise ValueError("fit=False but no scaler provided or found on disk.")

        # Check expected features from scaler if available
        if hasattr(scaler, "feature_names_in_"):
            scaler_cols = list(scaler.feature_names_in_)
            # If all 36 scaler columns are present in window_df:
            if all(c in window_df.columns for c in scaler_cols):
                scaled_values = scaler.transform(window_df[scaler_cols])
                df_normalized = pd.DataFrame(scaled_values, columns=scaler_cols)
            else:
                # Validate that all 26 required LSTM features are present
                meta_path = Path("models/lstm_metadata.json")
                if meta_path.exists():
                    import json
                    with open(meta_path) as f:
                        meta = json.load(f)
                    required_lstm_cols = meta.get("feature_names", [])
                    missing_lstm = [c for c in required_lstm_cols if c not in window_df.columns]
                    if missing_lstm:
                        raise ValueError(
                            f"Inference schema error: Required LSTM feature(s) missing from window features: {missing_lstm}"
                        )

                # Apply scaler parameters directly for available columns without inventing data
                scaled_dict = {}
                col_to_idx = {name: i for i, name in enumerate(scaler_cols)}
                for c in feature_cols:
                    if c in col_to_idx:
                        idx = col_to_idx[c]
                        mean_val = float(scaler.mean_[idx])
                        scale_val = float(scaler.scale_[idx])
                        scaled_dict[c] = (window_df[c].values - mean_val) / (scale_val if scale_val != 0.0 else 1.0)
                    else:
                        scaled_dict[c] = window_df[c].values
                df_normalized = pd.DataFrame(scaled_dict)
        else:
            scaled_values = scaler.transform(window_df[feature_cols])
            df_normalized = pd.DataFrame(scaled_values, columns=feature_cols)

    # Construct normalized DataFrame retaining timestamp and target
    if "window_timestamp" in window_df.columns:
        df_normalized.insert(0, "window_timestamp", window_df["window_timestamp"].values)
    if "is_attack_window" in window_df.columns:
        df_normalized["is_attack_window"] = window_df["is_attack_window"].values

    return df_normalized, scaler


def process_raw_file_to_windows(
    file_path: Union[str, Path],
    nrows: Optional[int] = None,
    chunksize: Optional[int] = None,
    window_size_seconds: int = 60,
) -> pd.DataFrame:
    """
    Loads a single raw CSV file, cleans it, parses timestamps, and aggregates
    flows into 60-second window records.

    Args:
        file_path: Path to the raw CSV file.
        nrows: Optional row limit for fast testing.
        chunksize: Optional chunk size for memory-constrained execution.
        window_size_seconds: Window size in seconds.

    Returns:
        Window-aggregated DataFrame for the file.
    """
    path = Path(file_path)
    logger.info(f"Processing raw file for window aggregation: {path.name}")

    if chunksize is not None and chunksize > 0:
        window_chunks: List[pd.DataFrame] = []
        for chunk in pd.read_csv(path, nrows=nrows, chunksize=chunksize, low_memory=False):
            cleaned_chunk = clean_data(chunk, source_name=path.name)
            bucketed_chunk = parse_and_bucket_timestamps(
                cleaned_chunk, window_size_seconds=window_size_seconds
            )
            agg_chunk = aggregate_window_features(bucketed_chunk)
            window_chunks.append(agg_chunk)
        
        # Combine chunks and re-aggregate any boundary overlaps
        combined = pd.concat(window_chunks, ignore_index=True)
        # Re-aggregate by window_timestamp if chunks split a window
        window_df = combined.groupby("window_timestamp", as_index=False).mean()
        # Ensure target remains binary max
        target_series = combined.groupby("window_timestamp")["is_attack_window"].max()
        window_df["is_attack_window"] = target_series.values
    else:
        cleaned_df = load_raw_csv(path, nrows=nrows)
        bucketed_df = parse_and_bucket_timestamps(
            cleaned_df, window_size_seconds=window_size_seconds
        )
        window_df = aggregate_window_features(bucketed_df)

    logger.info(
        f"Generated {len(window_df):,} windows from {path.name} "
        f"(Attacks: {window_df['is_attack_window'].sum():,}, Benign: {(window_df['is_attack_window'] == 0).sum():,})."
    )
    return window_df


def build_feature_pipeline(
    raw_data_dir: Optional[Union[str, Path]] = DEFAULT_RAW_DIR,
    output_path: Union[str, Path] = DEFAULT_OUTPUT_CSV,
    scaler_path: Union[str, Path] = DEFAULT_SCALER_PATH,
    pattern: str = "*.csv",
    nrows_per_file: Optional[int] = None,
    chunksize: Optional[int] = None,
    window_size_seconds: int = 60,
    fit_scaler: bool = False,
    file_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Execution of Feature Pipeline:
      If file_path is provided:
        Processes ONLY that single raw CSV file (isolated inference path).
      Otherwise:
        Discovers and processes all raw CSV files in raw_data_dir.
      Cleans and aggregates into 60s windows with is_attack_window target.
      Normalizes features with StandardScaler:
        - fit_scaler=False (default): Uses existing saved scaler without refitting or saving.
        - fit_scaler=True: Fits scaler on feature matrix and saves to scaler_path.
      Writes normalized window features to output_path.

    Args:
        raw_data_dir: Directory containing raw CSV files (ignored if file_path is provided).
        output_path: Target path for normalized feature matrix CSV.
        scaler_path: Target path for joblib scaler.
        pattern: Pattern matching raw files.
        nrows_per_file: Optional row limit per file.
        chunksize: Optional chunk size for large files.
        window_size_seconds: Window duration in seconds.
        fit_scaler: If True, fits scaler and saves to scaler_path. If False, transforms with existing scaler.
        file_path: Optional explicit single raw CSV path for isolated processing.

    Returns:
        Normalized window feature DataFrame.
    """
    if file_path is not None:
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"Specified raw file does not exist: {p}")
        logger.info(f"Isolated feature extraction on single file: {p.name}")
        file_windows = process_raw_file_to_windows(
            file_path=p,
            nrows=nrows_per_file,
            chunksize=chunksize,
            window_size_seconds=window_size_seconds,
        )
        combined_windows = file_windows.sort_values("window_timestamp").reset_index(drop=True)
    else:
        if raw_data_dir is None:
            raw_data_dir = DEFAULT_RAW_DIR
        dir_path = Path(raw_data_dir)
        files = sorted(list(dir_path.glob(pattern)))
        if not files:
            raise FileNotFoundError(f"No raw CSV files found in {dir_path} matching '{pattern}'")

        logger.info(f"Found {len(files)} raw CSV file(s) for feature extraction.")

        all_windows: List[pd.DataFrame] = []
        for f_path in files:
            file_windows = process_raw_file_to_windows(
                file_path=f_path,
                nrows=nrows_per_file,
                chunksize=chunksize,
                window_size_seconds=window_size_seconds,
            )
            all_windows.append(file_windows)

        # Concatenate all window datasets and sort chronologically
        combined_windows = pd.concat(all_windows, ignore_index=True)
        combined_windows = combined_windows.sort_values("window_timestamp").reset_index(drop=True)

    total_windows = len(combined_windows)
    attack_windows = int(combined_windows["is_attack_window"].sum()) if "is_attack_window" in combined_windows.columns else 0
    benign_windows = total_windows - attack_windows
    attack_ratio = (attack_windows / total_windows * 100) if total_windows > 0 else 0.0

    logger.info(
        f"Total unified windows: {total_windows:,} "
        f"| Attack windows: {attack_windows:,} ({attack_ratio:.2f}%) "
        f"| Benign windows: {benign_windows:,}."
    )

    # Scale feature matrix without overwriting unless fit_scaler=True
    normalized_df, scaler = scale_feature_matrix(
        combined_windows, scaler_path=scaler_path, fit=fit_scaler
    )

    # Save normalized dataset
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    normalized_df.to_csv(out_file, index=False)
    logger.info(f"Normalized feature matrix saved to: {out_file} (shape: {normalized_df.shape})")

    return normalized_df


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="CyberForeSight AI — Feature Pipeline & Time-Window Aggregator (Phase 1)"
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=str(DEFAULT_RAW_DIR),
        help="Directory containing raw CSV files (default: data/raw)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=str(DEFAULT_OUTPUT_CSV),
        help="Output path for normalized CSV (default: data/windows/window_features_normalized.csv)",
    )
    parser.add_argument(
        "--scaler-path",
        type=str,
        default=str(DEFAULT_SCALER_PATH),
        help="Output path for StandardScaler joblib file (default: models/scaler.joblib)",
    )
    parser.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="Optional row limit per raw CSV file (useful for fast verification).",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=None,
        help="Optional chunksize for reading large CSV files.",
    )
    parser.add_argument(
        "--window-seconds",
        type=int,
        default=60,
        help="Time window size in seconds (default: 60).",
    )
    return parser.parse_args()


def main() -> None:
    """CLI execution entrypoint."""
    args = parse_args()

    df_normalized = build_feature_pipeline(
        raw_data_dir=args.raw_dir,
        output_path=args.output_path,
        scaler_path=args.scaler_path,
        nrows_per_file=args.nrows,
        chunksize=args.chunksize,
        window_size_seconds=args.window_seconds,
    )

    print(f"\n{'=' * 60}")
    print("PHASE 1 FEATURE PIPELINE COMPLETE")
    print(f"{'=' * 60}")
    print(f"Total Windows:         {len(df_normalized):,}")
    print(f"Total Features Scaled: {len(df_normalized.columns) - 2}")
    print(f"Attack Windows:        {int(df_normalized['is_attack_window'].sum()):,}")
    print(f"Benign Windows:        {int((df_normalized['is_attack_window'] == 0).sum()):,}")
    print(f"Output File:           {args.output_path}")
    print(f"Scaler File:           {args.scaler_path}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()

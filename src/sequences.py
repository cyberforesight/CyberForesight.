"""
CyberForeSight AI — Partition-Specific Sequence Construction
Phase 2: Creates sequences independently within each partition.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.constants_v2 import CANONICAL_36_FEATURES, SEQ_LEN, WINDOW_SIZE_SECONDS
from src.features import parse_and_bucket_timestamps, aggregate_window_features, scale_feature_matrix
from src.ingestion import load_raw_csv, clean_data

logger = logging.getLogger("sequences")

def load_raw_file_to_windows(
    file_path: Union[str, Path],
    nrows: Optional[int] = None,
    chunksize: Optional[int] = None,
    window_size_seconds: int = WINDOW_SIZE_SECONDS,
) -> pd.DataFrame:
    """
    Loads a single raw CSV file and produces window-aggregated DataFrame.
    Does NOT scale features - returns raw window features with is_attack_window.
    """
    path = Path(file_path)
    logger.info(f"Loading raw file to windows: {path.name}")
    
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
        window_df = combined.groupby("window_timestamp", as_index=False).mean()
        target_series = combined.groupby("window_timestamp")["is_attack_window"].max()
        window_df["is_attack_window"] = target_series.values
    else:
        cleaned_df = load_raw_csv(path, nrows=nrows)
        bucketed_df = parse_and_bucket_timestamps(
            cleaned_df, window_size_seconds=window_size_seconds
        )
        window_df = aggregate_window_features(bucketed_df)
    
    # Ensure canonical feature order
    missing_features = [f for f in CANONICAL_36_FEATURES if f not in window_df.columns]
    if missing_features:
        raise ValueError(f"Missing canonical features in {path.name}: {missing_features}")
    
    # Select only canonical features + timestamp + target
    required_cols = ["window_timestamp"] + CANONICAL_36_FEATURES + ["is_attack_window"]
    window_df = window_df[required_cols].copy()
    
    logger.info(
        f"Generated {len(window_df):,} windows from {path.name} "
        f"(Attacks: {int(window_df['is_attack_window'].sum()):,}, "
        f"Benign: {int((window_df['is_attack_window'] == 0).sum()):,})."
    )
    return window_df


def load_windows_for_dates(
    raw_dir: Union[str, Path],
    dates: List[str],
    nrows_per_file: Optional[int] = None,
    chunksize: Optional[int] = None,
) -> pd.DataFrame:
    """
    Loads and concatenates window DataFrames for multiple dates.
    Returns combined DataFrame sorted chronologically.
    """
    raw_dir = Path(raw_dir)
    all_windows: List[pd.DataFrame] = []
    
    from src.constants_v2 import RAW_DATA_FILES
    
    for date in dates:
        filename = RAW_DATA_FILES.get(date)
        if not filename:
            raise ValueError(f"No raw file configured for date: {date}")
        file_path = raw_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Raw file not found: {file_path}")
        
        window_df = load_raw_file_to_windows(
            file_path, nrows=nrows_per_file, chunksize=chunksize
        )
        all_windows.append(window_df)
    
    combined = pd.concat(all_windows, ignore_index=True)
    combined = combined.sort_values("window_timestamp").reset_index(drop=True)
    
    logger.info(
        f"Loaded {len(combined):,} total windows for dates {dates} "
        f"(Attacks: {int(combined['is_attack_window'].sum()):,})"
    )
    return combined


def split_train_windows_chronologically(
    train_windows: pd.DataFrame,
    val_fraction: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits training windows chronologically into train and supplemental validation.
    Last val_fraction of windows go to supplemental validation.
    """
    n = len(train_windows)
    split_idx = int(n * (1.0 - val_fraction))
    
    train_part = train_windows.iloc[:split_idx].reset_index(drop=True)
    supp_val_part = train_windows.iloc[split_idx:].reset_index(drop=True)
    
    logger.info(
        f"Train split: train={len(train_part):,} windows "
        f"(Attacks: {int(train_part['is_attack_window'].sum()):,}), "
        f"supp_val={len(supp_val_part):,} windows "
        f"(Attacks: {int(supp_val_part['is_attack_window'].sum()):,})"
    )
    return train_part, supp_val_part


def create_sequences_for_partition(
    windows_df: pd.DataFrame,
    feature_cols: List[str],
    seq_len: int = SEQ_LEN,
    target_col: str = "is_attack_window",
) -> Tuple[np.ndarray, np.ndarray, List]:
    """
    Creates rolling sequences within a SINGLE partition.
    
    Input: windows_df sorted by window_timestamp for ONE partition only
    Output: X (N-seq_len, seq_len, n_features), y (N-seq_len,), target_timestamps
    
    The first seq_len windows cannot produce a valid target and are excluded.
    """
    if len(windows_df) <= seq_len:
        logger.warning(f"Partition has only {len(windows_df)} windows, need > {seq_len} for sequences")
        return np.array([]), np.array([]), []
    
    features_matrix = windows_df[feature_cols].values.astype(np.float32)
    targets_vector = windows_df[target_col].values.astype(np.float32)
    timestamps = pd.to_datetime(windows_df["window_timestamp"]).tolist()
    
    n_windows = len(windows_df)
    X_seq: List[np.ndarray] = []
    y_seq: List[float] = []
    target_ts: List = []
    
    for i in range(n_windows - seq_len):
        # Input: windows [t-4, t-3, t-2, t-1, t] (seq_len windows)
        X_seq.append(features_matrix[i : i + seq_len])
        # Target: window t+1 (next window after the sequence)
        y_seq.append(targets_vector[i + seq_len])
        # Target timestamp is the timestamp of the target window (t+1)
        target_ts.append(timestamps[i + seq_len])
    
    X_arr = np.array(X_seq, dtype=np.float32)
    y_arr = np.array(y_seq, dtype=np.float32)
    
    logger.info(
        f"Created {len(X_arr):,} sequences from {n_windows} windows "
        f"(dropped first {seq_len} windows, shape: {X_arr.shape})"
    )
    return X_arr, y_arr, target_ts


def prepare_all_partition_sequences(
    raw_dir: Union[str, Path],
    train_dates: List[str],
    supp_val_fraction: float,
    monitoring_val_dates: List[str],
    test_dates: List[str],
    nrows_per_file: Optional[int] = None,
    chunksize: Optional[int] = None,
) -> Dict:
    """
    Prepares sequences for all partitions independently.
    
    Returns dict with:
    - train_windows, train_X, train_y, train_ts
    - supp_val_windows, supp_val_X, supp_val_y, supp_val_ts
    - monitoring_val_windows, monitoring_val_X, monitoring_val_y, monitoring_val_ts
    - test_windows, test_X, test_y, test_ts
    """
    # Load train windows (Feb 14 + 15)
    train_windows = load_windows_for_dates(raw_dir, train_dates, nrows_per_file, chunksize)
    
    # Split train into train + supplemental validation
    train_part, supp_val_part = split_train_windows_chronologically(train_windows, supp_val_fraction)
    
    # Load monitoring validation (Feb 22)
    monitoring_val_windows = load_windows_for_dates(raw_dir, monitoring_val_dates, nrows_per_file, chunksize)
    
    # Load test (Feb 28) - STRICTLY HELD OUT
    test_windows = load_windows_for_dates(raw_dir, test_dates, nrows_per_file, chunksize)
    
    # Create sequences for each partition independently
    train_X, train_y, train_ts = create_sequences_for_partition(train_part, CANONICAL_36_FEATURES)
    supp_val_X, supp_val_y, supp_val_ts = create_sequences_for_partition(supp_val_part, CANONICAL_36_FEATURES)
    monitoring_val_X, monitoring_val_y, monitoring_val_ts = create_sequences_for_partition(
        monitoring_val_windows, CANONICAL_36_FEATURES
    )
    test_X, test_y, test_ts = create_sequences_for_partition(test_windows, CANONICAL_36_FEATURES)
    
    return {
        "train": {
            "windows": train_part,
            "X": train_X, "y": train_y, "timestamps": train_ts,
        },
        "supp_val": {
            "windows": supp_val_part,
            "X": supp_val_X, "y": supp_val_y, "timestamps": supp_val_ts,
        },
        "monitoring_val": {
            "windows": monitoring_val_windows,
            "X": monitoring_val_X, "y": monitoring_val_y, "timestamps": monitoring_val_ts,
        },
        "test": {
            "windows": test_windows,
            "X": test_X, "y": test_y, "timestamps": test_ts,
        },
    }
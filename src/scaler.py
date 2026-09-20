"""
CyberForeSight AI — V2 Scaler Implementation
Phase 2: Train-only StandardScaler with strict fit/transform separation.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.constants_v2 import CANONICAL_36_FEATURES, V2_SCALER_PATH

logger = logging.getLogger("scaler_v2")

def fit_scaler_v2(
    train_windows: pd.DataFrame,
    scaler_path: Union[str, Path] = V2_SCALER_PATH,
) -> StandardScaler:
    """
    Fits StandardScaler on training windows only.
    
    Args:
        train_windows: Training partition DataFrame with canonical 36 features
        scaler_path: Where to save the fitted scaler
    
    Returns:
        Fitted StandardScaler
    """
    # Validate feature columns
    missing = [f for f in CANONICAL_36_FEATURES if f not in train_windows.columns]
    if missing:
        raise ValueError(f"Training windows missing canonical features: {missing}")
    
    logger.info(f"Fitting V2 scaler on {len(train_windows):,} training windows...")
    
    scaler = StandardScaler()
    scaler.fit(train_windows[CANONICAL_36_FEATURES])
    
    # Verify feature order matches
    fitted_names = list(scaler.feature_names_in_)
    if fitted_names != CANONICAL_36_FEATURES:
        raise ValueError(
            f"Scaler feature order mismatch!\n"
            f"Expected: {CANONICAL_36_FEATURES}\n"
            f"Got:      {fitted_names}"
        )
    
    # Save scaler
    path = Path(scaler_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path)
    logger.info(f"V2 scaler saved to: {path}")
    logger.info(f"Scaler mean_: {scaler.mean_[:5]}... (first 5)")
    logger.info(f"Scaler scale_: {scaler.scale_[:5]}... (first 5)")
    
    return scaler


def transform_with_scaler_v2(
    windows_df: pd.DataFrame,
    scaler: StandardScaler,
    scaler_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Transforms window features using a pre-fitted V2 scaler.
    NEVER fits the scaler - only transforms.
    
    Args:
        windows_df: Partition DataFrame with canonical 36 features
        scaler: Pre-fitted StandardScaler (from fit_scaler_v2)
        scaler_path: Optional path for logging
    
    Returns:
        DataFrame with normalized features (same columns + timestamp + target)
    """
    # Validate feature columns
    missing = [f for f in CANONICAL_36_FEATURES if f not in windows_df.columns]
    if missing:
        raise ValueError(f"Windows missing canonical features: {missing}")
    
    # Verify scaler feature order
    if hasattr(scaler, "feature_names_in_"):
        fitted_names = list(scaler.feature_names_in_)
        if fitted_names != CANONICAL_36_FEATURES:
            raise ValueError(
                f"Scaler feature order mismatch during transform!\n"
                f"Expected: {CANONICAL_36_FEATURES}\n"
                f"Got:      {fitted_names}"
            )
    
    logger.info(f"Transforming {len(windows_df):,} windows with V2 scaler...")
    
    # Transform only the 36 canonical features
    scaled_values = scaler.transform(windows_df[CANONICAL_36_FEATURES])
    
    # Build normalized DataFrame
    df_normalized = pd.DataFrame(scaled_values, columns=CANONICAL_36_FEATURES)
    
    # Preserve timestamp and target
    if "window_timestamp" in windows_df.columns:
        df_normalized.insert(0, "window_timestamp", windows_df["window_timestamp"].values)
    if "is_attack_window" in windows_df.columns:
        df_normalized["is_attack_window"] = windows_df["is_attack_window"].values
    
    return df_normalized


def load_scaler_v2(scaler_path: Union[str, Path] = V2_SCALER_PATH) -> StandardScaler:
    """
    Loads the V2 scaler from disk.
    """
    path = Path(scaler_path)
    if not path.exists():
        raise FileNotFoundError(f"V2 scaler not found at: {path}")
    
    scaler = joblib.load(path)
    logger.info(f"Loaded V2 scaler from: {path}")
    
    # Verify feature order
    if hasattr(scaler, "feature_names_in_"):
        fitted_names = list(scaler.feature_names_in_)
        if fitted_names != CANONICAL_36_FEATURES:
            raise ValueError(
                f"Loaded scaler feature order mismatch!\n"
                f"Expected: {CANONICAL_36_FEATURES}\n"
                f"Got:      {fitted_names}"
            )
    
    return scaler


def scaler_v2_inference_contract() -> str:
    """
    Returns the inference contract documentation for V2 scaler.
    """
    return """
V2 SCALER INFERENCE CONTRACT
=============================

Training:
    scaler = StandardScaler()
    scaler.fit(train_windows[CANONICAL_36_FEATURES])  # ONLY on train
    joblib.dump(scaler, "models/scaler_v2.joblib")

Inference (uploaded CSV):
    1. Load scaler_v2 = joblib.load("models/scaler_v2.joblib")
    2. Extract 36 canonical features from uploaded data
    3. scaled = scaler_v2.transform(features[CANONICAL_36_FEATURES])  # TRANSFORM ONLY
    4. NEVER call fit() or fit_transform() on inference data
    5. NEVER overwrite models/scaler_v2.joblib during inference

Validation:
    - scaler.feature_names_in_ must exactly match CANONICAL_36_FEATURES
    - scaler must be fitted only on training partition
    - All partitions (val, test, inference) use transform() only
"""
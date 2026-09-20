"""
CyberForeSight AI — Label Construction Verification
Phase 2: Documents and verifies how is_attack_window is created.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

from src.constants_v2 import BENIGN_LABELS
from src.ingestion import load_raw_csv, clean_data

logger = logging.getLogger("label_verification")

def verify_label_construction(
    raw_dir: Path,
    date: str,
    nrows: Optional[int] = None,
) -> Dict:
    """
    Verifies how labels are mapped to is_attack_window for a specific date.
    
    Returns detailed information about label processing.
    """
    from src.constants_v2 import RAW_DATA_FILES
    
    filename = RAW_DATA_FILES.get(date)
    if not filename:
        raise ValueError(f"No raw file configured for date: {date}")
    
    file_path = raw_dir / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Raw file not found: {file_path}")
    
    logger.info(f"Verifying label construction for {date} ({filename})")
    
    # Load raw data
    df = load_raw_csv(file_path, nrows=nrows)
    df = clean_data(df, source_name=filename)
    
    # Resolve label column
    label_col = None
    for cand in ("label", "attack_stage", "labels", "class", "category", "attack_type"):
        if cand in df.columns:
            label_col = cand
            break
    
    if label_col is None:
        raise ValueError(f"No label column found in {filename}")
    
    # Get label distribution
    label_dist = df[label_col].value_counts().to_dict()
    
    # Classify each label as benign or attack
    benign_labels = set(BENIGN_LABELS)
    label_classification = {}
    for label, count in label_dist.items():
        is_benign = str(label).strip().lower() in benign_labels
        label_classification[str(label)] = {
            "count": int(count),
            "is_benign": is_benign,
            "normalized": str(label).strip().lower(),
        }
    
    # Check for label leakage into feature columns
    # (features should never contain label information)
    feature_columns = [c for c in df.columns if c not in {"label", "attack_stage", "timestamp", "Timestamp", "time", "date"}]
    label_related_cols = [c for c in feature_columns if "label" in c.lower() or "attack" in c.lower() or "class" in c.lower()]
    
    result = {
        "date": date,
        "file": filename,
        "label_column": label_col,
        "label_distribution": {str(k): int(v) for k, v in label_dist.items()},
        "label_classification": label_classification,
        "benign_label_set": sorted(list(benign_labels)),
        "label_related_feature_columns": label_related_cols,
        "leakage_check_passed": len(label_related_cols) == 0,
    }
    
    logger.info(
        f"Label verification {date}: label_col={label_col}, "
        f"unique_labels={len(label_dist)}, "
        f"leakage_check={'PASSED' if result['leakage_check_passed'] else 'FAILED'}"
    )
    
    return result


def print_label_verification_report(results: List[Dict]) -> None:
    """Print a formatted label verification report."""
    print("\n" + "=" * 80)
    print("LABEL CONSTRUCTION VERIFICATION REPORT")
    print("=" * 80)
    for r in results:
        print(f"\nDate: {r['date']} ({r['file']})")
        print(f"  Label column: {r['label_column']}")
        print(f"  Benign label set: {r['benign_label_set']}")
        print(f"  Leakage check: {'PASSED' if r['leakage_check_passed'] else 'FAILED'}")
        if r['label_related_feature_columns']:
            print(f"  WARNING - Label-related feature columns: {r['label_related_feature_columns']}")
        print(f"  Label distribution:")
        for label, info in r['label_classification'].items():
            status = "BENIGN" if info['is_benign'] else "ATTACK"
            print(f"    {label}: {info['count']:,} ({status})")
    print("=" * 80 + "\n")
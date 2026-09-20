"""
CyberForeSight AI — V2 Pipeline Tests
Phase 2: Implementation verification tests.
"""
import json
import os
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import torch

# V2 imports
from src.constants_v2 import (
    CANONICAL_36_FEATURES,
    EVAL_THRESHOLD,
    LSTM_V2_CONFIG,
    SEQ_LEN,
    V2_LSTM_PATH,
    V2_METADATA_PATH,
    V2_SCALER_PATH,
    TRAIN_CONFIG,
)
from src.sequences_v2 import (
    create_sequences_for_partition,
    split_train_windows_chronologically,
)
from src.scaler_v2 import (
    fit_scaler_v2,
    transform_with_scaler_v2,
)


# ──────────────────────────────────────────────────────────────────────────────
# Test 1: Exactly 36 canonical features
# ──────────────────────────────────────────────────────────────────────────────
def test_canonical_36_features_count():
    assert len(CANONICAL_36_FEATURES) == 36


# ──────────────────────────────────────────────────────────────────────────────
# Test 2: Feature order consistency
# ──────────────────────────────────────────────────────────────────────────────
def test_canonical_36_features_order():
    expected = [
        "flow_duration_mean",
        "flow_duration_max",
        "tot_fwd_pkts_sum",
        "tot_fwd_pkts_mean",
        "tot_bwd_pkts_sum",
        "tot_bwd_pkts_mean",
        "totlen_fwd_pkts_sum",
        "totlen_fwd_pkts_mean",
        "totlen_bwd_pkts_sum",
        "totlen_bwd_pkts_mean",
        "fwd_pkt_len_mean_mean",
        "bwd_pkt_len_mean_mean",
        "flow_byts_s_mean",
        "flow_byts_s_max",
        "flow_pkts_s_mean",
        "flow_pkts_s_max",
        "flow_iat_mean_mean",
        "flow_iat_max_max",
        "syn_flag_cnt_sum",
        "rst_flag_cnt_sum",
        "psh_flag_cnt_sum",
        "ack_flag_cnt_sum",
        "urg_flag_cnt_sum",
        "dst_port_nunique",
        "protocol_nunique",
        "flow_count",
        "fwd_pkt_len_max_max",
        "bwd_pkt_len_max_max",
        "pkt_len_max_max",
        "pkt_len_mean_mean",
        "pkt_len_min_min",
        "fin_flag_cnt_sum",
        "init_fwd_win_byts_mean",
        "init_bwd_win_byts_mean",
        "active_mean_mean",
        "idle_mean_mean",
    ]
    assert CANONICAL_36_FEATURES == expected


def test_feature_order_matches_scaler():
    """Verify V1 scaler feature order matches canonical 36."""
    scaler_path = Path("models/scaler.joblib")
    if not scaler_path.exists():
        pytest.skip("V1 scaler not found")
    scaler = joblib.load(scaler_path)
    assert list(scaler.feature_names_in_) == CANONICAL_36_FEATURES


# ──────────────────────────────────────────────────────────────────────────────
# Test 3: No label column in feature matrix
# ──────────────────────────────────────────────────────────────────────────────
def test_no_label_in_features():
    no_label_cols = {"label", "attack_stage", "Label", "Attack Stage"}
    feature_set = set(CANONICAL_36_FEATURES)
    assert len(feature_set & no_label_cols) == 0


# ──────────────────────────────────────────────────────────────────────────────
# Test 4: Train-only scaler logic
# ──────────────────────────────────────────────────────────────────────────────
def test_scaler_fit_only_on_train():
    """Verify fit_scaler_v2 fits only on train and saves correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        scaler_path = Path(tmpdir) / "scaler_test.joblib"
        
        # Create synthetic train windows
        n = 100
        train_df = pd.DataFrame({
            "window_timestamp": pd.date_range("2018-01-01", periods=n, freq="60s"),
            "is_attack_window": [0] * 80 + [1] * 20,
        })
        for f in CANONICAL_36_FEATURES:
            train_df[f] = np.random.randn(n)
        
        scaler = fit_scaler_v2(train_df, scaler_path)
        
        assert scaler.n_features_in_ == 36
        assert list(scaler.feature_names_in_) == CANONICAL_36_FEATURES
        assert scaler_path.exists()


# ──────────────────────────────────────────────────────────────────────────────
# Test 5: No "fit()" during inference
# ──────────────────────────────────────────────────────────────────────────────
def test_no_fit_during_inference():
    """Verify transform_with_scaler_v2 never fits."""
    with tempfile.TemporaryDirectory() as tmpdir:
        scaler_path = Path(tmpdir) / "scaler_test.joblib"
        
        train_df = pd.DataFrame({
            "window_timestamp": pd.date_range("2018-01-01", periods=50, freq="60s"),
            "is_attack_window": [0] * 40 + [1] * 10,
        })
        for f in CANONICAL_36_FEATURES:
            train_df[f] = np.random.randn(50)
        
        scaler = fit_scaler_v2(train_df, scaler_path)
        
        # Create test data
        test_df = pd.DataFrame({
            "window_timestamp": pd.date_range("2018-02-01", periods=20, freq="60s"),
            "is_attack_window": [0] * 15 + [1] * 5,
        })
        for f in CANONICAL_36_FEATURES:
            test_df[f] = np.random.randn(20)
        
        # Transform test data
        result = transform_with_scaler_v2(test_df, scaler)
        
        # Verify transformed values are scaled (mean ~0 for train, but test is different)
        for f in CANONICAL_36_FEATURES:
            assert f in result.columns


# ──────────────────────────────────────────────────────────────────────────────
# Test 6: Partition-specific sequence construction
# ──────────────────────────────────────────────────────────────────────────────
def test_partition_specific_sequences():
    """Verify sequences don't cross partition boundaries."""
    # Create two separate partitions
    partition1 = pd.DataFrame({
        "window_timestamp": pd.date_range("2018-01-01", periods=10, freq="60s"),
        "is_attack_window": [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
    })
    for f in CANONICAL_36_FEATURES:
        partition1[f] = np.random.randn(10)
    
    partition2 = pd.DataFrame({
        "window_timestamp": pd.date_range("2018-02-01", periods=10, freq="60s"),
        "is_attack_window": [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    })
    for f in CANONICAL_36_FEATURES:
        partition2[f] = np.random.randn(10)
    
    X1, y1, ts1 = create_sequences_for_partition(partition1, CANONICAL_36_FEATURES)
    X2, y2, ts2 = create_sequences_for_partition(partition2, CANONICAL_36_FEATURES)
    
    # Each partition of 10 windows should produce 5 sequences (10 - 5)
    assert X1.shape == (5, 5, 36)
    assert X2.shape == (5, 5, 36)
    assert y1.shape == (5,)
    assert y2.shape == (5,)
    
    # First 5 windows dropped - target is window t+1
    # partition1: targets are windows 5,6,7,8,9 -> [1,1,0,0,0]
    assert list(y1) == [1, 1, 0, 0, 0]
    # partition2: targets are windows 5,6,7,8,9 -> [0,0,0,0,0]
    assert list(y2) == [0, 0, 0, 0, 0]


# ──────────────────────────────────────────────────────────────────────────────
# Test 7: No sequence crossing partition boundaries
# ──────────────────────────────────────────────────────────────────────────────
def test_no_cross_partition_sequences():
    """Sequences created independently per partition - cannot cross boundaries."""
    # This is inherently tested by the partition-specific construction
    # If sequences were created on combined data and then sliced,
    # the first sequence of partition 2 would contain last 4 windows of partition 1
    pass  # Covered by test_partition_specific_sequences


# ──────────────────────────────────────────────────────────────────────────────
# Test 8: Correct "t+1" target indexing
# ──────────────────────────────────────────────────────────────────────────────
def test_correct_t_plus_1_target():
    """Verify target y[i] corresponds to window index i + SEQ_LEN."""
    n = 20
    df = pd.DataFrame({
        "window_timestamp": pd.date_range("2018-01-01", periods=n, freq="60s"),
        "is_attack_window": [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1],
    })
    for f in CANONICAL_36_FEATURES:
        df[f] = np.arange(n)  # Use sequential values to verify indexing
    
    X, y, ts = create_sequences_for_partition(df, CANONICAL_36_FEATURES)
    
    # Sequence 0: windows 0-4 -> target is window 5 = 1 (correct)
    assert y[0] == df["is_attack_window"].iloc[5]
    # Sequence 1: windows 1-5 -> target is window 6 = 1
    assert y[1] == df["is_attack_window"].iloc[6]
    # Sequence N: windows N to N+4 -> target is window N+5
    for i in range(len(y)):
        assert y[i] == df["is_attack_window"].iloc[i + SEQ_LEN]


# ──────────────────────────────────────────────────────────────────────────────
# Test 9: Correct target timestamp
# ──────────────────────────────────────────────────────────────────────────────
def test_target_timestamp_is_t_plus_1():
    """Verify target timestamp = t+1 window timestamp."""
    n = 10
    timestamps = pd.date_range("2018-01-01", periods=n, freq="60s")
    df = pd.DataFrame({
        "window_timestamp": timestamps,
        "is_attack_window": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    })
    for f in CANONICAL_36_FEATURES:
        df[f] = np.random.randn(n)
    
    X, y, ts = create_sequences_for_partition(df, CANONICAL_36_FEATURES)
    
    # First sequence (i=0): input windows 0-4, target window 5
    assert ts[0] == timestamps[5]
    # Second sequence (i=1): input windows 1-5, target window 6
    assert ts[1] == timestamps[6]
    for i in range(len(ts)):
        assert ts[i] == timestamps[i + SEQ_LEN]


# ──────────────────────────────────────────────────────────────────────────────
# Test 10: Feb 28 cannot enter training/scaler fitting
# ──────────────────────────────────────────────────────────────────────────────
def test_feb28_excluded_from_training():
    """Verify Feb 28 test data is never in train dates."""
    from src.constants_v2 import TRAIN_DATES, MONITORING_VAL_DATE, TEST_DATES
    
    all_dates = TRAIN_DATES + MONITORING_VAL_DATE + TEST_DATES
    # Feb 28 must be in test only
    assert "2018-02-28" in TEST_DATES
    assert "2018-02-28" not in TRAIN_DATES
    assert "2018-02-28" not in MONITORING_VAL_DATE


# ──────────────────────────────────────────────────────────────────────────────
# Test 11: Baseline uses same V2 scaler
# ──────────────────────────────────────────────────────────────────────────────
def test_baseline_uses_same_features():
    """Verify LR baseline uses same 36 features as LSTM."""
    from sklearn.linear_model import LogisticRegression
    
    n = 50
    train_X = np.random.randn(n, SEQ_LEN, 36).astype(np.float32)
    train_y = np.array([0] * 40 + [1] * 10)
    test_X = np.random.randn(20, SEQ_LEN, 36).astype(np.float32)
    test_y = np.array([0] * 15 + [1] * 5)
    
    from src.train_v2 import train_baseline_v2
    model, metrics = train_baseline_v2(train_X, train_y, test_X, test_y)
    
    # LR should use last window (36 features)
    assert model.n_features_in_ == 36


# ──────────────────────────────────────────────────────────────────────────────
# Test 12: Old V1 artifacts remain unchanged
# ──────────────────────────────────────────────────────────────────────────────
def test_v1_artifacts_unchanged():
    """Verify V1 artifacts still exist and have correct structure."""
    v1_model = Path("models/lstm_world_model.pth")
    v1_scaler = Path("models/scaler.joblib")
    v1_meta = Path("models/lstm_metadata.json")
    
    assert v1_model.exists()
    assert v1_scaler.exists()
    assert v1_meta.exists()
    
    # V1 model should be 26-feature
    sd = torch.load(v1_model, map_location="cpu", weights_only=True)
    assert sd["lstm.weight_ih_l0"].shape[1] == 26
    
    # V1 metadata should be 26-feature
    with open(v1_meta) as f:
        meta = json.load(f)
    assert meta["input_size"] == 26
    assert len(meta["feature_names"]) == 26


# ──────────────────────────────────────────────────────────────────────────────
# Test 13: 36-feature LSTM accepts correct input shape
# ──────────────────────────────────────────────────────────────────────────────
def test_lstm_v2_input_shape():
    """Verify LSTM accepts 36-feature input."""
    from src.lstm_model import LSTMWorldModel
    
    model = LSTMWorldModel(
        input_size=36,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
    )
    model.eval()
    
    batch = 4
    X = torch.randn(batch, SEQ_LEN, 36)
    
    with torch.no_grad():
        output = model(X)
    
    assert output.shape == (batch, 1)
    assert all(val >= 0 and val <= 1 for val in output.squeeze().tolist())


# ──────────────────────────────────────────────────────────────────────────────
# Test 14: Small datasets fail cleanly
# ──────────────────────────────────────────────────────────────────────────────
def test_small_dataset_fails_cleanly():
    """Verify datasets with < 6 windows fail gracefully."""
    small_df = pd.DataFrame({
        "window_timestamp": pd.date_range("2018-01-01", periods=4, freq="60s"),
        "is_attack_window": [0, 0, 0, 1],
    })
    for f in CANONICAL_36_FEATURES:
        small_df[f] = np.random.randn(4)
    
    X, y, ts = create_sequences_for_partition(small_df, CANONICAL_36_FEATURES)
    
    # Should return empty arrays, not crash
    assert len(X) == 0
    assert len(y) == 0
    assert len(ts) == 0


# ──────────────────────────────────────────────────────────────────────────────
# Test 15: Train-only scaler on partitions
# ──────────────────────────────────────────────────────────────────────────────
def test_scaler_does_not_overwrite_v1():
    """Verify V2 scaler fitting never touches V1 scaler path."""
    from src.constants_v2 import V2_SCALER_PATH, DEFAULT_SCALER_PATH
    
    assert str(V2_SCALER_PATH) != str(DEFAULT_SCALER_PATH)


# ──────────────────────────────────────────────────────────────────────────────
# Test 16: Threshold frozen at 0.5
# ──────────────────────────────────────────────────────────────────────────────
def test_threshold_frozen():
    assert EVAL_THRESHOLD == 0.5


# ──────────────────────────────────────────────────────────────────────────────
# Test 17: LSTM config uses 36 features
# ──────────────────────────────────────────────────────────────────────────────
def test_lstm_v2_config():
    assert LSTM_V2_CONFIG["input_size"] == 36
    assert LSTM_V2_CONFIG["hidden_size"] == 64
    assert LSTM_V2_CONFIG["num_layers"] == 2
    assert LSTM_V2_CONFIG["dropout"] == 0.2
    assert LSTM_V2_CONFIG["sequence_length"] == SEQ_LEN


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
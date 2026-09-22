"""
tests/test_pipeline.py
-----------------------
Comprehensive Automated Test Suite for Crypto & P2P Forensic Intelligence Dashboard.
Covers:
- Pydantic validation schemas & IPv4 address validation
- Heterogeneous graph construction & CIOH wallet clustering
- Multi-modal feature engineering matrix extraction
- Isolation Forest anomaly detection & 0-100 risk score calibration
- SHAP TreeExplainer local attribution & Compliance Reason Code mapping
- Alert ticket schema validation
"""

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import numpy as np
import pandas as pd
from pydantic import ValidationError

from ingestion_pipeline import IngestionPipeline, RawTransaction, EnrichedTransaction, GeoIPEnrichment
from graph_engine import HeterogeneousGraphEngine
from feature_engineering import FeatureEngineeringMatrix
from train_model import ForensicRiskModel
from explainability_alerts import ForensicExplainabilityEngine, AlertTicket, REASON_CODE_CATALOG


@pytest.fixture
def sample_raw_tx():
    return {
        "tx_id": "0x4a1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sender_address": "0x1111111111111111111111111111111111111111",
        "receiver_address": "0x2222222222222222222222222222222222222222",
        "amount": 10.5,
        "fee": 0.002,
        "network_protocol": "Bitcoin",
        "src_ip": "192.168.1.100",
        "dst_ip": "10.0.0.50",
        "src_asn": "AS15169",
        "dst_asn": "AS16509",
        "payload_size_bytes": 520,
        "pattern_label": "TEST_TX"
    }


def test_pydantic_valid_transaction(sample_raw_tx):
    """Test valid transaction parsing through Pydantic."""
    tx = RawTransaction(**sample_raw_tx)
    assert tx.tx_id.startswith("0x")
    assert tx.amount == 10.5
    assert tx.src_ip == "192.168.1.100"


def test_pydantic_invalid_ipv4(sample_raw_tx):
    """Test that invalid IPv4 formats are strictly rejected."""
    bad_tx = sample_raw_tx.copy()
    bad_tx["src_ip"] = "999.999.999.999"  # Invalid IP
    with pytest.raises(ValidationError):
        RawTransaction(**bad_tx)


def test_pydantic_negative_amount(sample_raw_tx):
    """Test that non-positive amounts are rejected."""
    bad_tx = sample_raw_tx.copy()
    bad_tx["amount"] = -5.0
    with pytest.raises(ValidationError):
        RawTransaction(**bad_tx)


def test_ingestion_pipeline_enrichment(sample_raw_tx):
    """Test offline GeoIP and ASN threat enrichment."""
    pipeline = IngestionPipeline()
    enriched, malformed = pipeline.parse_record(sample_raw_tx)
    assert malformed is None
    assert enriched is not None
    assert enriched.src_enrichment.asn == "AS15169"
    assert enriched.src_enrichment.tier in ["LOW_RISK", "MEDIUM_RISK", "HIGH_RISK", "CRITICAL_RISK"]
    assert enriched.fee_to_amount_ratio == pytest.approx(0.002 / 10.5, rel=1e-3)


def test_common_input_ownership_heuristic():
    """Test CIOH wallet clustering across co-spending transactions."""
    pipeline = IngestionPipeline()
    graph_eng = HeterogeneousGraphEngine()

    now = datetime.now(timezone.utc)
    # 2 transactions with co-spending inputs (w1+w2) and (w2+w3) -> should cluster w1, w2, w3 into single Entity
    tx1_dict = {
        "tx_id": "0x1111111111111111111111111111111111111111111111111111111111111111",
        "timestamp": now.isoformat(),
        "sender_address": "0xwalletA;0xwalletB",
        "receiver_address": "0xwalletDest1",
        "amount": 5.0,
        "fee": 0.001,
        "network_protocol": "Bitcoin",
        "src_ip": "1.1.1.1",
        "dst_ip": "2.2.2.2",
        "src_asn": "AS15169",
        "dst_asn": "AS15169",
        "payload_size_bytes": 400
    }
    tx2_dict = {
        "tx_id": "0x2222222222222222222222222222222222222222222222222222222222222222",
        "timestamp": now.isoformat(),
        "sender_address": "0xwalletB;0xwalletC",
        "receiver_address": "0xwalletDest2",
        "amount": 8.0,
        "fee": 0.001,
        "network_protocol": "Bitcoin",
        "src_ip": "1.1.1.1",
        "dst_ip": "2.2.2.2",
        "src_asn": "AS15169",
        "dst_asn": "AS15169",
        "payload_size_bytes": 400
    }

    e_tx1, _ = pipeline.parse_record(tx1_dict)
    e_tx2, _ = pipeline.parse_record(tx2_dict)

    wallet_map = graph_eng.run_common_input_ownership_heuristic([e_tx1, e_tx2])
    # WalletA, WalletB, WalletC must belong to the exact same Entity ID
    assert wallet_map["0xwalletA"] == wallet_map["0xwalletB"]
    assert wallet_map["0xwalletB"] == wallet_map["0xwalletC"]


def test_heterogeneous_graph_construction(sample_raw_tx):
    """Test Heterogeneous Graph construction with IP, Wallet, and TXID nodes."""
    pipeline = IngestionPipeline()
    e_tx, _ = pipeline.parse_record(sample_raw_tx)
    graph_eng = HeterogeneousGraphEngine()
    G = graph_eng.build_heterogeneous_graph([e_tx])

    assert G.has_node(e_tx.tx_id)
    assert G.has_node(f"IP:{e_tx.src_ip}")
    assert G.has_node(f"WALLET:{e_tx.sender_address}")
    assert G.has_node(f"WALLET:{e_tx.receiver_address}")

    metrics = graph_eng.compute_graph_metrics()
    assert e_tx.tx_id in metrics
    assert "pagerank" in metrics[e_tx.tx_id]
    assert "betweenness" in metrics[e_tx.tx_id]


def test_feature_engineering_and_model_scoring():
    """Test full feature extraction and Isolation Forest risk scoring calibration."""
    pipeline = IngestionPipeline()
    txs = pipeline.ingest_file("data/transactions.csv")
    assert len(txs) > 0

    graph_eng = HeterogeneousGraphEngine()
    fe_matrix = FeatureEngineeringMatrix(graph_eng)
    X, full_df = fe_matrix.extract_features(txs, fit_embedder=True)

    assert X.shape[0] == len(txs)
    assert X.shape[1] >= 30  # Comprehensive multi-modal features
    assert not X.isnull().values.any()

    # Risk model test
    model = ForensicRiskModel(contamination=0.10)
    scored_df, X_scaled = model.fit_and_score(X, full_df)

    assert "risk_score" in scored_df.columns
    assert "risk_tier" in scored_df.columns
    assert scored_df["risk_score"].min() >= 0
    assert scored_df["risk_score"].max() <= 100
    assert set(scored_df["risk_tier"].unique()).issubset({"CRITICAL", "HIGH", "MEDIUM", "LOW"})


def test_shap_explainability_and_reason_codes():
    """Test SHAP TreeExplainer local attribution and compliance reason code mapping."""
    pipeline = IngestionPipeline()
    txs = pipeline.ingest_file("data/transactions.csv")

    graph_eng = HeterogeneousGraphEngine()
    fe_matrix = FeatureEngineeringMatrix(graph_eng)
    X, full_df = fe_matrix.extract_features(txs, fit_embedder=True)

    model = ForensicRiskModel(contamination=0.10)
    scored_df, X_scaled = model.fit_and_score(X, full_df)

    exp_engine = ForensicExplainabilityEngine(model)
    alerts = exp_engine.generate_alert_tickets(scored_df, X, min_risk_score=75)

    assert len(alerts) > 0
    top_alert = alerts[0]
    assert isinstance(top_alert, AlertTicket)
    assert top_alert.risk_score >= 75
    assert len(top_alert.reason_codes) > 0
    assert top_alert.reason_codes[0].code.startswith("RC_")


if __name__ == "__main__":
    pytest.main(["-v", "tests/test_pipeline.py"])

"""
explainability_alerts.py
-------------------------
SHAP Explainability & Compliance Investigator Alerting Engine.
Features:
- SHAP TreeExplainer integration for local feature attribution on Isolation Forest.
- Maps top anomaly-driving features to regulatory Compliance Reason Codes.
- Structured Pydantic AlertTicket schema with rank-ordered reasons.
- Automated Suspicious Activity Report (SAR) narrative generation.
- High-resolution SHAP waterfall & feature attribution bar chart generator.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from pydantic import BaseModel, Field

from train_model import ForensicRiskModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ForensicExplainability")

MODELS_DIR = Path("models")
DATA_DIR = Path("data")

# Comprehensive Compliance Reason Code Catalog
REASON_CODE_CATALOG: Dict[str, Dict[str, str]] = {
    "peel_chain_heuristic_score": {
        "code": "RC_PEEL_CHAIN",
        "title": "Peel Chain Laundering Pattern",
        "description": "Rapid sequential peel-off transaction identified with return change address structuring.",
        "severity": "CRITICAL",
        "regulatory_ref": "FATF Rec. 16 / FinCEN SAR Typology: Layering via Peel Chains"
    },
    "fan_out_count": {
        "code": "RC_HIGH_FAN_OUT",
        "title": "High Fan-Out Dispersal",
        "description": "Structuring dispersal across multiple discrete receiver wallets within a single event.",
        "severity": "HIGH",
        "regulatory_ref": "AML Structuring / Smurfing Fan-Out"
    },
    "fan_in_count": {
        "code": "RC_HIGH_FAN_IN",
        "title": "Smurfing Multi-Input Aggregation",
        "description": "Coordinated aggregation of funds from multiple independent input wallets.",
        "severity": "HIGH",
        "regulatory_ref": "FATF Common-Input-Ownership Smurfing"
    },
    "fan_in_fan_out_ratio": {
        "code": "RC_FAN_RATIO_ANOMALY",
        "title": "Abnormal Flow Ratio",
        "description": "Severe imbalance between input funding sources and output distribution targets.",
        "severity": "MEDIUM",
        "regulatory_ref": "Graph Flow Topology Heuristic"
    },
    "ip_connection_freq_src": {
        "code": "RC_HIGH_CONN_FREQ",
        "title": "Source IP Burst Frequency Anomaly",
        "description": "High-density transaction flood broadcasted from a single IP address (automated script/bot).",
        "severity": "HIGH",
        "regulatory_ref": "Cyber Forensics: Automated Drainage Bot"
    },
    "tx_velocity_seconds": {
        "code": "RC_VELOCITY_DRAIN",
        "title": "Sub-Second Velocity Drain",
        "description": "Inter-transaction latency below human threshold, characteristic of automated wallet draining.",
        "severity": "CRITICAL",
        "regulatory_ref": "High-Frequency Automated Sweeper Detection"
    },
    "log_velocity": {
        "code": "RC_VELOCITY_DRAIN",
        "title": "Sub-Second Velocity Drain",
        "description": "Inter-transaction latency below human threshold, characteristic of automated wallet draining.",
        "severity": "CRITICAL",
        "regulatory_ref": "High-Frequency Automated Sweeper Detection"
    },
    "is_bulletproof_asn": {
        "code": "RC_BULLETPROOF_HOST",
        "title": "Bulletproof Hosting ASN",
        "description": "Transaction broadcasted via known bulletproof host or illicit infrastructure provider.",
        "severity": "CRITICAL",
        "regulatory_ref": "OFAC Cyber Sanctions / High-Risk Infrastructure"
    },
    "src_asn_risk_weight": {
        "code": "RC_HIGH_RISK_ASN",
        "title": "High-Risk Source ASN",
        "description": "Broadcast origin Autonomous System exhibits extreme malicious traffic concentration.",
        "severity": "HIGH",
        "regulatory_ref": "Geo-Network Risk Intelligence Rating"
    },
    "is_tor_relay": {
        "code": "RC_TOR_EXIT_NODE",
        "title": "Darknet / Tor Exit Relay",
        "description": "Network broadcast routed through an anonymizing Tor exit node or VPN darknet gateway.",
        "severity": "HIGH",
        "regulatory_ref": "Anonymity-Enhanced Cryptocurrency Routing"
    },
    "is_round_number_amount": {
        "code": "RC_ROUND_NUMBER_STRUCT",
        "title": "Round-Number Structuring",
        "description": "Transaction amount matches calibrated round threshold structuring heuristics.",
        "severity": "MEDIUM",
        "regulatory_ref": "Bank Secrecy Act (BSA) Smurfing Thresholds"
    },
    "fee_to_amount_ratio": {
        "code": "RC_FEE_ANOMALY",
        "title": "Priority Fee Bribery / Fee Anomaly",
        "description": "Disproportionate mining/gas fee relative to principal transfer amount.",
        "severity": "MEDIUM",
        "regulatory_ref": "Miner Extractable Value (MEV) / Priority Laundering Tip"
    },
    "payload_zscore": {
        "code": "RC_PAYLOAD_ANOMALY",
        "title": "Anomalous P2P Packet Size",
        "description": "Network protocol payload length deviates significantly from standard transaction templates.",
        "severity": "MEDIUM",
        "regulatory_ref": "P2P Network Protocol Inspection"
    },
    "payload_size_bytes": {
        "code": "RC_PAYLOAD_ANOMALY",
        "title": "Anomalous P2P Packet Size",
        "description": "Network protocol payload length deviates significantly from standard transaction templates.",
        "severity": "MEDIUM",
        "regulatory_ref": "P2P Network Protocol Inspection"
    },
    "tx_pagerank": {
        "code": "RC_GRAPH_CENTRALITY_SPIKE",
        "title": "High Network Topology Hub",
        "description": "Node sits at a critical graph crossroads with extreme PageRank authority.",
        "severity": "MEDIUM",
        "regulatory_ref": "Graph Centrality & Hub Identification"
    },
    "tx_betweenness": {
        "code": "RC_BETWEENNESS_BRIDGE",
        "title": "Inter-Cluster Bridge Transaction",
        "description": "Transaction serves as a topological conduit bridging discrete wallet clusters.",
        "severity": "HIGH",
        "regulatory_ref": "Multi-Cluster Funnel / Conduit Detection"
    },
    "entity_wallet_cluster_size": {
        "code": "RC_MULTI_INPUT_COSPEND",
        "title": "Clustered Entity Co-Spending Ring",
        "description": "Associated with an entity spanning multiple co-spending wallet addresses.",
        "severity": "HIGH",
        "regulatory_ref": "Common-Input-Ownership Heuristic Attribution"
    }
}


class ReasonCodeDetail(BaseModel):
    """Structured compliance reason code detail."""
    code: str
    title: str
    description: str
    severity: str
    regulatory_ref: str
    shap_weight: float


class AlertTicket(BaseModel):
    """Production-grade Forensic Compliance Alert Ticket."""
    alert_id: str
    tx_id: str
    risk_score: int = Field(..., ge=0, le=100)
    priority: str  # CRITICAL, HIGH, MEDIUM, LOW
    timestamp: datetime
    sender_address: str
    receiver_address: str
    amount: float
    fee: float
    protocol: str
    src_ip: str
    src_asn: str
    src_org: str
    reason_codes: List[ReasonCodeDetail]
    top_features: Dict[str, float]
    investigator_notes: str
    pattern_label: Optional[str] = "UNKNOWN"


class ForensicExplainabilityEngine:
    """
    SHAP TreeExplainer engine providing local feature attribution,
    reason code extraction, and visual artifact generation.
    """

    def __init__(self, risk_model: ForensicRiskModel):
        self.risk_model = risk_model
        self.explainer: Optional[shap.TreeExplainer] = None
        self._init_explainer()

    def _init_explainer(self):
        """Initialize SHAP TreeExplainer on Isolation Forest."""
        if self.risk_model.is_trained:
            logger.info("Initializing SHAP TreeExplainer on Isolation Forest...")
            self.explainer = shap.TreeExplainer(self.risk_model.model)

    def compute_shap_values(self, X_scaled: np.ndarray) -> np.ndarray:
        """Compute SHAP values for feature matrix."""
        if self.explainer is None:
            self._init_explainer()
        logger.info(f"Computing SHAP values for {len(X_scaled)} records...")
        shap_vals = self.explainer.shap_values(X_scaled)
        return shap_vals

    def explain_transaction(
        self,
        tx_row: pd.Series,
        shap_vector: np.ndarray,
        feature_names: List[str],
        top_k: int = 5
    ) -> List[ReasonCodeDetail]:
        """
        Extracts top anomaly-driving features from SHAP attribution and maps to Reason Codes.
        Note: In Isolation Forest, more negative SHAP pushes towards anomaly (or positive magnitude).
        We sort by absolute impact on the anomaly decision.
        """
        # Pair feature names with their SHAP impact
        # Negative SHAP value in IsolationForest means feature reduces normality -> pushes towards anomaly
        feature_impacts = []
        for feat_name, s_val in zip(feature_names, shap_vector):
            # Invert sign so positive impact = pushing towards anomaly
            anomaly_contribution = -float(s_val)
            feature_impacts.append((feat_name, anomaly_contribution))

        # Sort by highest positive anomaly contribution
        feature_impacts.sort(key=lambda x: x[1], reverse=True)

        reason_codes: List[ReasonCodeDetail] = []
        seen_codes = set()

        for feat_name, impact in feature_impacts:
            if feat_name in REASON_CODE_CATALOG and impact > 0.001:
                cat_info = REASON_CODE_CATALOG[feat_name]
                if cat_info["code"] not in seen_codes:
                    seen_codes.add(cat_info["code"])
                    reason_codes.append(ReasonCodeDetail(
                        code=cat_info["code"],
                        title=cat_info["title"],
                        description=cat_info["description"],
                        severity=cat_info["severity"],
                        regulatory_ref=cat_info["regulatory_ref"],
                        shap_weight=round(impact, 4)
                    ))
            if len(reason_codes) >= top_k:
                break

        # Fallback if no specific code mapped
        if not reason_codes:
            reason_codes.append(ReasonCodeDetail(
                code="RC_STATISTICAL_OUTLIER",
                title="Multi-Dimensional Topological Outlier",
                description="Transaction deviates significantly across combined graph and network dimensions.",
                severity="MEDIUM",
                regulatory_ref="Unsupervised Multi-Variate Isolation Heuristic",
                shap_weight=0.1
            ))

        return reason_codes

    def generate_alert_tickets(
        self,
        scored_df: pd.DataFrame,
        X_features: pd.DataFrame,
        min_risk_score: int = 60
    ) -> List[AlertTicket]:
        """
        Generates prioritized AlertTicket objects for all transactions meeting the risk threshold.
        """
        logger.info(f"Generating Alert Tickets for transactions with Risk Score >= {min_risk_score}...")
        X_scaled = self.risk_model.scaler.transform(X_features)
        shap_values = self.compute_shap_values(X_scaled)

        alerts: List[AlertTicket] = []
        feature_names = self.risk_model.feature_names

        for idx, (row_idx, row) in enumerate(scored_df.iterrows()):
            risk_score = int(row["risk_score"])
            if risk_score < min_risk_score:
                continue

            s_vec = shap_values[idx]
            reasons = self.explain_transaction(row, s_vec, feature_names)

            # Top 5 feature names and their values
            top_feats = {}
            sorted_indices = np.argsort(-s_vec)[::-1]  # Highest anomaly push
            for fi in sorted_indices[:5]:
                fname = feature_names[fi]
                top_feats[fname] = float(row.get(fname, 0.0))

            # Generate compliance narrative
            ts_str = str(row["timestamp"])
            narrative = (
                f"Suspicious activity detected on {ts_str}. Transaction {row['tx_id'][:16]}... "
                f"exhibited Risk Score {risk_score}/100 ({row['risk_tier']} Priority). "
                f"Primary indicators: {', '.join([r.code for r in reasons])}. "
                f"Source entity originating from ASN {row.get('src_asn', 'N/A')} ({row.get('src_org', 'N/A')}). "
                f"Recommended Action: Immediate compliance review and enhanced forensic graph tracing."
            )

            ticket = AlertTicket(
                alert_id=f"ALT-{row['timestamp'].strftime('%Y%m%d') if hasattr(row['timestamp'], 'strftime') else '2026'}-{idx+1:04d}",
                tx_id=str(row["tx_id"]),
                risk_score=risk_score,
                priority=str(row["risk_tier"]),
                timestamp=row["timestamp"] if isinstance(row["timestamp"], datetime) else datetime.fromisoformat(str(row["timestamp"])),
                sender_address=str(row["sender_address"]),
                receiver_address=str(row["receiver_address"]),
                amount=float(row["amount"]),
                fee=float(row["fee"]),
                protocol=str(row["network_protocol"]),
                src_ip=str(row["src_ip"]),
                src_asn=str(row["src_asn"]),
                src_org=str(row.get("src_org", "Unknown")),
                reason_codes=reasons,
                top_features=top_feats,
                investigator_notes=narrative,
                pattern_label=str(row.get("pattern_label", "UNKNOWN"))
            )
            alerts.append(ticket)

        # Sort by Risk Score descending
        alerts.sort(key=lambda a: a.risk_score, reverse=True)
        logger.info(f"[+] Generated {len(alerts)} prioritized alert tickets.")
        return alerts

    def plot_shap_waterfall_or_bar(
        self,
        tx_row: pd.Series,
        shap_vector: np.ndarray,
        feature_names: List[str],
        top_n: int = 8
    ) -> plt.Figure:
        """
        Renders a high-resolution dark-themed SHAP feature attribution bar chart.
        """
        # Extract impacts (positive = anomaly push)
        impacts = []
        for feat_name, s_val in zip(feature_names, shap_vector):
            impact = -float(s_val)  # Inverted so >0 means higher anomaly
            val = tx_row.get(feat_name, 0.0)
            impacts.append((feat_name, impact, val))

        impacts.sort(key=lambda x: abs(x[1]), reverse=True)
        top_impacts = impacts[:top_n]
        top_impacts.reverse()  # For bottom-to-top horizontal bar chart

        labels = [f"{x[0]} ({x[2]:.2f})" if isinstance(x[2], (int, float)) else str(x[0]) for x in top_impacts]
        values = [x[1] for x in top_impacts]
        colors = ["#ff0054" if v > 0 else "#00f0ff" for v in values]

        fig, ax = plt.subplots(figsize=(9, 4.8), facecolor="#0e1117")
        ax.set_facecolor("#161b22")

        bars = ax.barh(labels, values, color=colors, edgecolor="#30363d", height=0.6)

        ax.axvline(0, color="#8b949e", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_title(
            f"SHAP Anomaly Attribution: {str(tx_row['tx_id'])[:14]}... (Risk: {int(tx_row.get('risk_score', 0))}/100)",
            color="#f0f6fc", fontsize=12, fontweight="bold", pad=12
        )
        ax.set_xlabel("Contribution to Anomaly Risk (Inverted SHAP Value)", color="#8b949e", fontsize=10)
        ax.tick_params(colors="#c9d1d9", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#30363d")

        plt.tight_layout()
        return fig


if __name__ == "__main__":
    from train_model import run_training_pipeline
    scored_df, risk_model, X_features = run_training_pipeline()

    exp_engine = ForensicExplainabilityEngine(risk_model)
    alerts = exp_engine.generate_alert_tickets(scored_df, X_features, min_risk_score=70)

    print(f"\n[+] Total Alerts Generated (Risk >= 70): {len(alerts)}")
    if alerts:
        top_alert = alerts[0]
        print(f"\n=== Top Alert: {top_alert.alert_id} ===")
        print(f"TXID: {top_alert.tx_id}")
        print(f"Risk Score: {top_alert.risk_score} [{top_alert.priority}]")
        print(f"Reason Codes: {[r.code for r in top_alert.reason_codes]}")
        print(f"Investigator Narrative:\n{top_alert.investigator_notes}")

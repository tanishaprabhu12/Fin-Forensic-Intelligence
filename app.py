"""
app.py
------
Crypto & P2P Forensic Intelligence Dashboard.
Offline-First, Production-Grade Streamlit Application.
Features:
- Dark cyber forensic theme with glassmorphism KPI cards and glowing risk badges.
- Tab 1: Alerts Feed (Interactive triage queue with multi-filter search and export).
- Tab 2: Graph View (PyVis interactive physics network with CIOH entity cluster explorer).
- Tab 3: Explainability Card (SHAP waterfall/bar feature attribution and automated SAR generator).
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from ingestion_pipeline import IngestionPipeline, EnrichedTransaction
from graph_engine import HeterogeneousGraphEngine, NODE_COLORS
from feature_engineering import FeatureEngineeringMatrix
from train_model import ForensicRiskModel, run_training_pipeline
from explainability_alerts import ForensicExplainabilityEngine, AlertTicket, REASON_CODE_CATALOG

# Streamlit Page Config
st.set_page_config(
    page_title="CryptoForensics | P2P Intelligence Ops",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Dark Forensic Theme CSS
st.markdown("""
<style>
    /* Main Background and Fonts */
    .stApp {
        background-color: #0b0f19;
        color: #e6edf3;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Headers */
    h1, h2, h3, h4 {
        color: #f0f6fc;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(22, 27, 34, 0.9) 0%, rgba(13, 17, 23, 0.95) 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 18px 22px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
        margin-bottom: 12px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: #58a6ff;
        transform: translateY(-2px);
    }
    .metric-title {
        color: #8b949e;
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 6px;
    }
    .metric-value {
        color: #ffffff;
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    .metric-sub {
        font-size: 0.78rem;
        margin-top: 4px;
    }
    
    /* Risk Badges */
    .badge-critical {
        background-color: rgba(255, 0, 84, 0.18);
        color: #ff0054;
        border: 1px solid #ff0054;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 700;
        display: inline-block;
    }
    .badge-high {
        background-color: rgba(255, 159, 28, 0.18);
        color: #ff9f1c;
        border: 1px solid #ff9f1c;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 700;
        display: inline-block;
    }
    .badge-medium {
        background-color: rgba(255, 230, 0, 0.18);
        color: #ffe600;
        border: 1px solid #ffe600;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 700;
        display: inline-block;
    }
    .badge-low {
        background-color: rgba(0, 240, 255, 0.15);
        color: #00f0ff;
        border: 1px solid #00f0ff;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 700;
        display: inline-block;
    }
    .badge-code {
        background-color: rgba(157, 78, 221, 0.18);
        color: #d8b4fe;
        border: 1px solid #9d4edd;
        padding: 2px 7px;
        border-radius: 4px;
        font-size: 0.72rem;
        font-family: monospace;
        margin-right: 4px;
        margin-bottom: 2px;
        display: inline-block;
    }

    /* Telemetry Card */
    .telemetry-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
    }
    
    /* Code/Monospace format */
    .mono-hash {
        font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, Courier, monospace;
        font-size: 0.82rem;
        color: #79c0ff;
    }
</style>
""", unsafe_allow_html=True)

DATA_DIR = Path("data")
MODELS_DIR = Path("models")


# ---------------------------------------------------------
# Cached Data & Model Loaders
# ---------------------------------------------------------

@st.cache_resource(show_spinner="Initializing Forensic Graph & AI Models...")
def load_or_train_pipeline() -> Tuple[pd.DataFrame, pd.DataFrame, HeterogeneousGraphEngine, ForensicRiskModel, ForensicExplainabilityEngine, List[EnrichedTransaction]]:
    """Loads transactions, builds heterogeneous graph, extracts features, trains/loads model, and SHAP engine."""
    pipeline = IngestionPipeline()
    csv_path = DATA_DIR / "transactions.csv"
    if not csv_path.exists():
        from generate_synthetic_data import export_synthetic_data
        export_synthetic_data()

    txs = pipeline.ingest_file(csv_path)

    graph_eng = HeterogeneousGraphEngine()
    G = graph_eng.build_heterogeneous_graph(txs)

    fe_matrix = FeatureEngineeringMatrix(graph_eng)
    X, full_df = fe_matrix.extract_features(txs, fit_embedder=True)

    risk_model = ForensicRiskModel(contamination=0.10, random_state=42)
    scored_df, X_scaled = risk_model.fit_and_score(X, full_df)
    risk_model.save(MODELS_DIR)

    exp_engine = ForensicExplainabilityEngine(risk_model)

    return scored_df, X, graph_eng, risk_model, exp_engine, txs


# Load Core Engines
try:
    scored_df, X_features, graph_engine, risk_model, exp_engine, enriched_txs = load_or_train_pipeline()
except Exception as e:
    st.error(f"Initialization error: {e}")
    st.stop()


# ---------------------------------------------------------
# Header & Navigation
# ---------------------------------------------------------

st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: center; padding-bottom: 15px; border-bottom: 1px solid #30363d; margin-bottom: 20px;">
    <div>
        <h1 style="margin: 0; font-size: 2.1rem; display: flex; align-items: center; gap: 12px;">
            <span>🛡️</span> Crypto & P2P Forensic Intelligence Dashboard
        </h1>
        <div style="color: #8b949e; font-size: 0.9rem; margin-top: 4px;">
            Offline-First Forensic Telemetry &bull; Heterogeneous Graph Correlation &bull; Isolation Forest Anomaly Scoring &bull; SHAP Explainability
        </div>
    </div>
    <div style="text-align: right;">
        <span style="background: rgba(6, 214, 160, 0.15); color: #06d6a0; border: 1px solid #06d6a0; padding: 5px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 700;">
            ● SYSTEM ONLINE (OFFLINE-FIRST)
        </span>
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Top Executive Metrics Row
# ---------------------------------------------------------

total_tx = len(scored_df)
crit_alerts = int((scored_df["risk_tier"] == "CRITICAL").sum())
high_alerts = int((scored_df["risk_tier"] == "HIGH").sum())
total_val_at_risk = float(scored_df[scored_df["risk_tier"].isin(["CRITICAL", "HIGH"])]["amount"].sum())
unique_entities = len(graph_engine.entity_metadata)
suspicious_asn_count = int((scored_df["is_bulletproof_asn"] == 1).sum())

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Total Transactions</div>
        <div class="metric-value">{total_tx:,}</div>
        <div class="metric-sub" style="color: #58a6ff;">Ingested & Enriched</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card" style="border-left: 4px solid #ff0054;">
        <div class="metric-title">Critical & High Alerts</div>
        <div class="metric-value" style="color: #ff0054;">{crit_alerts + high_alerts}</div>
        <div class="metric-sub" style="color: #ff9f1c;">{crit_alerts} Critical &bull; {high_alerts} High</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Value at Risk</div>
        <div class="metric-value">{total_val_at_risk:,.2f} <span style="font-size: 1.1rem; color: #8b949e;">ETH/BTC</span></div>
        <div class="metric-sub" style="color: #06d6a0;">Flagged Volume</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Clustered Entities</div>
        <div class="metric-value">{unique_entities}</div>
        <div class="metric-sub" style="color: #a855f7;">CIOH Wallet Groups</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Bulletproof / Tor ASN</div>
        <div class="metric-value" style="color: #ffb703;">{suspicious_asn_count}</div>
        <div class="metric-sub" style="color: #8b949e;">High-Risk Relays</div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------

st.sidebar.header("🔍 Forensic Filters")

# Search query
search_query = st.sidebar.text_input(
    "Search TXID, Wallet, IP, or Entity",
    placeholder="0x... or 192.168... or Entity-001"
).strip()

# Priority Filter
priority_filter = st.sidebar.multiselect(
    "Filter by Risk Priority",
    options=["CRITICAL", "HIGH", "MEDIUM", "LOW"],
    default=["CRITICAL", "HIGH", "MEDIUM", "LOW"]
)

# Protocol Filter
all_protocols = sorted(list(scored_df["network_protocol"].unique()))
protocol_filter = st.sidebar.multiselect(
    "Network Protocol",
    options=all_protocols,
    default=all_protocols
)

# Risk Score Slider
min_score = st.sidebar.slider("Minimum Risk Score", min_value=0, max_value=100, value=0, step=5)

# High-Risk Network Flag
only_suspicious_network = st.sidebar.checkbox("Only Bulletproof / Tor / Darknet ASNs", value=False)

# Filter Dataframe
filtered_df = scored_df.copy()
if priority_filter:
    filtered_df = filtered_df[filtered_df["risk_tier"].isin(priority_filter)]
if protocol_filter:
    filtered_df = filtered_df[filtered_df["network_protocol"].isin(protocol_filter)]
if min_score > 0:
    filtered_df = filtered_df[filtered_df["risk_score"] >= min_score]
if only_suspicious_network:
    filtered_df = filtered_df[filtered_df["is_high_risk_network"] == 1]
if search_query:
    q_lower = search_query.lower()
    filtered_df = filtered_df[
        filtered_df["tx_id"].str.lower().str.contains(q_lower) |
        filtered_df["sender_address"].str.lower().str.contains(q_lower) |
        filtered_df["receiver_address"].str.lower().str.contains(q_lower) |
        filtered_df["src_ip"].str.contains(q_lower) |
        filtered_df["dst_ip"].str.contains(q_lower) |
        filtered_df["entity_id"].str.lower().str.contains(q_lower)
    ]


# ---------------------------------------------------------
# Main Tabs UI
# ---------------------------------------------------------

tab_alerts, tab_graph, tab_explain = st.tabs([
    "🚨 Tab 1: Alerts Feed (Triage Queue)",
    "🕸️ Tab 2: Graph View (Heterogeneous Topology)",
    "🧬 Tab 3: Explainability & Forensic Dossier"
])


# =========================================================
# TAB 1: ALERTS FEED
# =========================================================

with tab_alerts:
    st.subheader(f"Triage Queue ({len(filtered_df)} Transactions Matching Filters)")

    # Action bar
    col_a1, col_a2, col_a3 = st.columns([3, 1, 1])
    with col_a1:
        st.caption("Sorted by calibrated Risk Score (0-100). Click any transaction hash to copy or inspect in Tab 3.")
    with col_a2:
        csv_data = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Queue CSV",
            data=csv_data,
            file_name=f"forensic_alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    with col_a3:
        json_data = filtered_df.to_json(orient="records", indent=2).encode('utf-8')
        st.download_button(
            label="📦 Export JSON",
            data=json_data,
            file_name=f"forensic_alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )

    # Display Styled Table
    table_display = filtered_df[[
        "risk_score", "risk_tier", "tx_id", "timestamp", "amount", "network_protocol",
        "entity_id", "src_ip", "src_org", "pattern_label"
    ]].sort_values(by="risk_score", ascending=False).copy()

    # Rename for professional presentation
    table_display.columns = [
        "Risk", "Priority", "Transaction Hash", "Timestamp (UTC)", "Amount",
        "Protocol", "Clustered Entity", "Source IP", "ISP / Autonomous System", "Typology"
    ]

    st.dataframe(
        table_display,
        use_container_width=True,
        height=520,
        column_config={
            "Risk": st.column_config.ProgressColumn(
                "Risk Score",
                help="Normalized 0-100 Anomaly Risk Score",
                format="%d",
                min_value=0,
                max_value=100
            ),
            "Amount": st.column_config.NumberColumn(
                "Amount",
                format="%.4f"
            ),
            "Transaction Hash": st.column_config.TextColumn(
                "TXID",
                width="medium"
            )
        }
    )

    # Top Alerts Drilldown Card Preview
    st.markdown("---")
    st.markdown("### ⚡ Top 3 Critical Incidents Requiring Immediate Compliance Review")
    top_critical = filtered_df.sort_values(by="risk_score", ascending=False).head(3)

    for _, row in top_critical.iterrows():
        tier = row["risk_tier"]
        badge_cls = f"badge-{tier.lower()}"
        st.markdown(f"""
        <div class="telemetry-card" style="border-left: 4px solid {'#ff0054' if tier=='CRITICAL' else '#ff9f1c'};">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <div>
                    <span class="{badge_cls}">{tier} PRIORITY &bull; RISK {row['risk_score']}/100</span>
                    <span style="color: #8b949e; margin-left: 10px; font-size: 0.85rem;">Pattern: <b>{row['pattern_label']}</b></span>
                </div>
                <div style="color: #8b949e; font-size: 0.8rem;">{row['timestamp']}</div>
            </div>
            <div style="margin-bottom: 6px;">
                <span style="color: #8b949e;">TX Hash:</span> <span class="mono-hash">{row['tx_id']}</span>
            </div>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; font-size: 0.85rem; color: #c9d1d9;">
                <div><b>Amount:</b> {row['amount']:.4f} {row['network_protocol']}</div>
                <div><b>Entity:</b> {row['entity_id']}</div>
                <div><b>Source IP:</b> {row['src_ip']}</div>
                <div><b>ISP:</b> {row['src_org']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# =========================================================
# TAB 2: GRAPH VIEW (HETEROGENEOUS TOPOLOGY)
# =========================================================

with tab_graph:
    st.subheader("Heterogeneous Network Topology & Common-Input-Ownership Cluster Graph")
    st.caption("Interactive multi-modal graph visualizing IP Nodes (Amber/Red), Wallet Addresses (Cyan), Transaction Hashes (Purple), and Co-Spending Entities (Green).")

    col_g1, col_g2, col_g3, col_g4 = st.columns(4)

    with col_g1:
        graph_mode = st.selectbox(
            "Graph Focus Mode",
            ["High-Risk Subgraph", "Specific Entity Cluster", "Transaction Ego-Network", "Full Network Sample"]
        )

    with col_g2:
        if graph_mode == "Specific Entity Cluster":
            # Multi-wallet entities first
            multi_entities = [k for k, v in graph_engine.entity_metadata.items() if v["wallet_count"] > 1]
            all_entities = multi_entities + [k for k, v in graph_engine.entity_metadata.items() if v["wallet_count"] == 1]
            selected_entity = st.selectbox("Select Clustered Entity", all_entities)
        elif graph_mode == "Transaction Ego-Network":
            top_tx_ids = list(filtered_df.sort_values(by="risk_score", ascending=False)["tx_id"].head(50))
            selected_tx = st.selectbox("Select Target TXID", top_tx_ids)
        else:
            selected_entity = None
            selected_tx = None

    with col_g3:
        graph_hops = st.slider("Graph Traversal Depth (Hops)", min_value=1, max_value=3, value=2)

    with col_g4:
        enable_physics = st.checkbox("Enable Physics Simulation", value=True)

    # Subgraph Extraction Logic
    if graph_mode == "Specific Entity Cluster" and selected_entity:
        entity_wallets = graph_engine.entity_metadata.get(selected_entity, {}).get("wallets", [])
        cluster_nodes = set()
        for w in entity_wallets:
            cluster_nodes.add(f"WALLET:{w}")
        
        # Expand 1 hop
        expanded = set(cluster_nodes)
        for cn in cluster_nodes:
            if graph_engine.graph.has_node(cn):
                expanded.update(graph_engine.graph.neighbors(cn))
                expanded.update(graph_engine.graph.predecessors(cn))
        sub_g = graph_engine.graph.subgraph(expanded).copy()

    elif graph_mode == "Transaction Ego-Network" and selected_tx:
        sub_g = graph_engine.extract_subgraph_for_entity_or_tx(selected_tx, hops=graph_hops, max_nodes=70)

    elif graph_mode == "High-Risk Subgraph":
        # Extract top 35 high-risk transactions and their neighbors
        high_risk_txs = list(scored_df[scored_df["risk_score"] >= 60]["tx_id"].head(25))
        sub_nodes = set(high_risk_txs)
        for tx_id in high_risk_txs:
            if graph_engine.graph.has_node(tx_id):
                sub_nodes.update(graph_engine.graph.neighbors(tx_id))
                sub_nodes.update(graph_engine.graph.predecessors(tx_id))
        sub_g = graph_engine.graph.subgraph(sub_nodes).copy()

    else:
        # Full network sample
        sample_nodes = sorted(graph_engine.graph.nodes(), key=lambda n: graph_engine.graph.degree(n), reverse=True)[:80]
        sub_g = graph_engine.graph.subgraph(sample_nodes).copy()

    # Render PyVis Graph
    html_graph = graph_engine.export_to_pyvis(sub_g, height="620px", width="100%", physics=enable_physics)
    components.html(html_graph, height=640, scrolling=False)

    # Graph Legend & Entity Profile Breakdown
    col_leg, col_ent = st.columns([1, 1])

    with col_leg:
        st.markdown("#### 🎨 Graph Topology Legend")
        st.markdown(f"""
        - 🟣 **Transaction Node (Diamond):** `{NODE_COLORS['TXID']}` &bull; Broadcasted on-chain event
        - 🔵 **Wallet Address (Circle):** `{NODE_COLORS['WALLET']}` &bull; Discrete cryptocurrency address
        - 🟡 **Standard Relay IP (Square):** `{NODE_COLORS['IP']}` &bull; Normal ISP / Residential relay
        - 🔴 **High-Risk IP (Square):** `{NODE_COLORS['IP_HIGH_RISK']}` &bull; Tor exit / Bulletproof host / Darknet
        - 🟢 **Clustered Entity (Hexagon):** `{NODE_COLORS['ENTITY']}` &bull; Grouped via Common-Input-Ownership
        """)

    with col_ent:
        st.markdown("#### 👥 Common-Input-Ownership Heuristic (CIOH) Summary")
        multi_wallet_clusters = {k: v for k, v in graph_engine.entity_metadata.items() if v["wallet_count"] > 1}
        st.write(f"Identified **{len(multi_wallet_clusters)} multi-wallet entity clusters** controlling co-spent addresses:")
        for eid, edata in list(multi_wallet_clusters.items())[:3]:
            st.markdown(f"""
            <div style="background: #161b22; border-left: 3px solid #06d6a0; padding: 8px 12px; margin-bottom: 6px; border-radius: 4px; font-size: 0.85rem;">
                <b>{eid}</b>: Controls <b>{edata['wallet_count']} Wallets</b> &bull; Total Volume: <b>{edata['total_volume']:.2f} crypto</b> &bull; Associated IPs: {len(edata['associated_ips'])}
            </div>
            """, unsafe_allow_html=True)


# =========================================================
# TAB 3: EXPLAINABILITY & FORENSIC DOSSIER
# =========================================================

with tab_explain:
    st.subheader("Transaction Explainability & Compliance Reason Code Audit")
    st.caption("SHAP TreeExplainer local feature attributions, regulatory reason code mapping, and automated Suspicious Activity Report (SAR) dossier.")

    # Transaction Selector
    tx_list = list(filtered_df.sort_values(by="risk_score", ascending=False)["tx_id"])
    if not tx_list:
        st.warning("No transactions match the selected filters.")
        st.stop()

    selected_tx_id = st.selectbox(
        "Select Transaction for Deep Forensic Audit",
        options=tx_list,
        format_func=lambda x: f"{x[:18]}... | Risk: {int(scored_df[scored_df['tx_id']==x]['risk_score'].values[0])}/100 [{scored_df[scored_df['tx_id']==x]['risk_tier'].values[0]}]"
    )

    # Retrieve selected transaction record
    tx_row = scored_df[scored_df["tx_id"] == selected_tx_id].iloc[0]
    row_idx = scored_df.index.get_loc(tx_row.name)

    # Compute SHAP explanation
    X_scaled = risk_model.scaler.transform(X_features)
    shap_vals = exp_engine.compute_shap_values(X_scaled)
    target_shap = shap_vals[row_idx]

    # Map Reason Codes
    reason_codes = exp_engine.explain_transaction(tx_row, target_shap, risk_model.feature_names)

    # 1. Telemetry & Entity Overview
    t_col1, t_col2 = st.columns([1.2, 1])

    with t_col1:
        st.markdown(f"""
        <div class="telemetry-card">
            <h4 style="margin-top: 0; color: #58a6ff;">📡 Transaction & Network Telemetry</h4>
            <div style="font-size: 0.85rem; line-height: 1.8;">
                <div><b>TX Hash:</b> <span class="mono-hash">{tx_row['tx_id']}</span></div>
                <div><b>Timestamp:</b> {tx_row['timestamp']}</div>
                <div><b>Amount:</b> <span style="font-size: 1.1rem; font-weight: 700; color: #ffffff;">{tx_row['amount']:.4f} {tx_row['network_protocol']}</span> (Fee: {tx_row['fee']:.6f})</div>
                <div><b>Sender Wallet(s):</b> <span class="mono-hash">{tx_row['sender_address']}</span></div>
                <div><b>Receiver Wallet(s):</b> <span class="mono-hash">{tx_row['receiver_address']}</span></div>
                <div><b>Source IP / ASN:</b> {tx_row['src_ip']} &bull; {tx_row['src_asn']} (<b>{tx_row.get('src_org', 'Unknown')}</b>)</div>
                <div><b>Relay IP / ASN:</b> {tx_row['dst_ip']} &bull; {tx_row['dst_asn']} (<b>{tx_row.get('dst_org', 'Unknown')}</b>)</div>
                <div><b>P2P Payload Size:</b> {int(tx_row['payload_size_bytes'])} bytes (Z-Score: {tx_row['payload_zscore']:.2f})</div>
                <div><b>Typology Tag:</b> <span class="badge-critical">{tx_row['pattern_label']}</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with t_col2:
        entity_info = graph_engine.entity_metadata.get(tx_row['entity_id'], {})
        tier = tx_row['risk_tier']
        badge_cls = f"badge-{tier.lower()}"

        st.markdown(f"""
        <div class="telemetry-card">
            <h4 style="margin-top: 0; color: #06d6a0;">👥 CIOH Entity Profile & Risk Rating</h4>
            <div style="font-size: 0.85rem; line-height: 1.8;">
                <div><b>Calibrated Risk Score:</b> <span class="{badge_cls}" style="font-size: 0.95rem;">{int(tx_row['risk_score'])} / 100 ({tier})</span></div>
                <div><b>Clustered Entity ID:</b> <span style="color: #06d6a0; font-weight: 700;">{tx_row['entity_id']}</span></div>
                <div><b>Wallets Controlled in Cluster:</b> {entity_info.get('wallet_count', 1)} address(es)</div>
                <div><b>Total Clustered Volume:</b> {entity_info.get('total_volume', tx_row['amount']):.2f} crypto</div>
                <div><b>Inter-Transaction Velocity (Δt):</b> {tx_row['tx_velocity_seconds']:.2f} seconds</div>
                <div><b>Peel-Chain Heuristic Score:</b> {tx_row['peel_chain_heuristic_score']:.2f} / 1.00</div>
                <div><b>Network PageRank:</b> {tx_row['tx_pagerank']:.6f}</div>
                <div><b>Betweenness Centrality:</b> {tx_row['tx_betweenness']:.6f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 2. SHAP Feature Attribution Waterfall / Bar Chart
    st.markdown("---")
    st.markdown("### 📊 SHAP TreeExplainer Local Feature Attribution")
    st.caption("Illustrates which multi-modal features pushed this transaction into the anomalous zone (Red = Positive Anomaly Contribution, Blue = Normalizing Contribution).")

    fig_shap = exp_engine.plot_shap_waterfall_or_bar(tx_row, target_shap, risk_model.feature_names, top_n=9)
    st.pyplot(fig_shap)

    # 3. Compliance Reason Codes Table
    st.markdown("---")
    st.markdown("### 📋 Triggered Regulatory Compliance Reason Codes")

    for r in reason_codes:
        sev_color = "#ff0054" if r.severity == "CRITICAL" else ("#ff9f1c" if r.severity == "HIGH" else "#ffe600")
        st.markdown(f"""
        <div style="background: #161b22; border-left: 4px solid {sev_color}; border-radius: 8px; padding: 14px 18px; margin-bottom: 10px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                <div>
                    <span class="badge-code">{r.code}</span>
                    <b style="color: #f0f6fc; font-size: 0.95rem;">{r.title}</b>
                </div>
                <div>
                    <span style="color: {sev_color}; font-size: 0.75rem; font-weight: 700; border: 1px solid {sev_color}; padding: 2px 6px; border-radius: 4px;">
                        {r.severity}
                    </span>
                    <span style="color: #8b949e; font-size: 0.75rem; margin-left: 8px;">SHAP Weight: +{r.shap_weight:.4f}</span>
                </div>
            </div>
            <div style="color: #c9d1d9; font-size: 0.85rem; margin-bottom: 4px;">
                {r.description}
            </div>
            <div style="color: #8b949e; font-size: 0.75rem;">
                <b>Regulatory Reference:</b> {r.regulatory_ref}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 4. Automated Forensic SAR Narrative Generator
    st.markdown("---")
    st.markdown("### 📄 Automated Forensic Dossier & SAR Compliance Narrative")

    sar_narrative = f"""# SUSPICIOUS ACTIVITY REPORT (SAR) / FORENSIC INVESTIGATION DOSSIER
CONFIDENTIAL // FOR COMPLIANCE & LAW ENFORCEMENT AUDIT USE ONLY
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
Investigation Reference: CASE-TX-{tx_row['tx_id'][:12]}

1. EXECUTIVE SUMMARY:
On {tx_row['timestamp']}, automated forensic telemetry identified an anomalous transaction
(TXID: {tx_row['tx_id']}) exhibiting an elevated Risk Score of {int(tx_row['risk_score'])}/100 ({tx_row['risk_tier']} Priority).
The event was captured on the {tx_row['network_protocol']} protocol involving a transfer amount of {tx_row['amount']:.4f} crypto units.

2. SUBJECT & ENTITY ATTRIBUTION:
- Primary Sender: {tx_row['sender_address']}
- Receiver Destination: {tx_row['receiver_address']}
- Common-Input-Ownership Cluster: {tx_row['entity_id']} (Controls {entity_info.get('wallet_count', 1)} discrete address(es))
- Originating IP: {tx_row['src_ip']} (Autonomous System: {tx_row['src_asn']} - {tx_row.get('src_org', 'Unknown')})
- Relay/Destination IP: {tx_row['dst_ip']} (Autonomous System: {tx_row['dst_asn']} - {tx_row.get('dst_org', 'Unknown')})

3. TRIGGERED COMPLIANCE REASON CODES & SHAP ATTRIBUTION:
{chr(10).join([f"- [{r.code}] {r.title} ({r.severity}): {r.description} (Ref: {r.regulatory_ref})" for r in reason_codes])}

4. FORENSIC GRAPH & BEHAVIORAL INDICATORS:
- Transaction Inter-Arrival Velocity (Δt): {tx_row['tx_velocity_seconds']:.3f} seconds (automated sweep indicator).
- Peel-Chain Heuristic Factor: {tx_row['peel_chain_heuristic_score']:.3f}.
- Heterogeneous Graph PageRank: {tx_row['tx_pagerank']:.6f} | Betweenness Centrality: {tx_row['tx_betweenness']:.6f}.
- Network Payload Packet Size: {int(tx_row['payload_size_bytes'])} bytes (Deviation Z-Score: {tx_row['payload_zscore']:.2f}).

5. INVESTIGATOR RECOMMENDATION:
Flag entity {tx_row['entity_id']} for Enhanced Due Diligence (EDD). Place originating IP subnet {tx_row['src_ip']}/24
on watch list. Submit formal SAR filing under FATF Recommendation 16 and BSA smurfing/layering typologies.
"""

    st.text_area("Forensic Compliance Dossier", sar_narrative, height=260)
    st.download_button(
        label="💾 Download Full SAR Investigation Dossier (.md)",
        data=sar_narrative.encode("utf-8"),
        file_name=f"SAR_Dossier_{tx_row['tx_id'][:12]}.md",
        mime="text/markdown",
        use_container_width=True
    )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #8b949e; font-size: 0.8rem; padding-bottom: 20px;">
    Crypto & P2P Forensic Intelligence Platform &bull; Production-Grade Offline-First Architecture &bull; Built with Streamlit, NetworkX, Scikit-Learn & SHAP
</div>
""", unsafe_allow_html=True)

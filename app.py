"""
app.py
------
Crypto & P2P Forensic Intelligence Dashboard.
Offline-First, Production-Grade Streamlit Application.
Features:
- Dark cyber forensic theme with glassmorphism KPI cards and glowing risk badges.
- Tab 1: Alerts Feed (Interactive triage queue with multi-filter search, real-world asset USD valuation, and export).
- Tab 2: Graph View (PyVis interactive physics network with CIOH entity cluster explorer).
- Tab 3: Explainability Card (SHAP waterfall/bar feature attribution, asset market tiering, and automated SAR generator).
- Tab 4: Real-World Market Intelligence & Asset Risk (4,150 real cryptocurrencies, wash-trading radar, volatility scanner, and cross-reference).
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
from crypto_market_data import get_market_loader, CryptoMarketDataLoader

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
        padding: 18px 20px;
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
        font-size: 0.80rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 6px;
    }
    .metric-value {
        color: #ffffff;
        font-size: 1.75rem;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    .metric-sub {
        font-size: 0.76rem;
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
    .badge-token {
        background-color: rgba(56, 189, 248, 0.18);
        color: #38bdf8;
        border: 1px solid #0284c7;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 700;
        font-family: monospace;
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

@st.cache_resource(show_spinner="Initializing Real-World Crypto Asset Database...")
def load_crypto_market_data() -> CryptoMarketDataLoader:
    """Loads and cleans 4,150 real cryptocurrencies from CryptocurrencyData.csv."""
    return get_market_loader()


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
    market_loader = load_crypto_market_data()
    market_df = market_loader.dataframe
    market_summary = market_loader.get_market_summary()

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
            Offline-First Forensic Telemetry &bull; Real-World Market Grounding (4,150 Assets) &bull; CIOH Graph Correlation &bull; Isolation Forest &bull; SHAP Explainability
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

# Compute USD Value at Risk using real cryptocurrency prices
if "amount_usd" in scored_df.columns:
    total_usd_at_risk = float(scored_df[scored_df["risk_tier"].isin(["CRITICAL", "HIGH"])]["amount_usd"].sum())
else:
    total_usd_at_risk = float(scored_df[scored_df["risk_tier"].isin(["CRITICAL", "HIGH"])]["amount"].sum()) * 36456.94

total_crypto_units = float(scored_df[scored_df["risk_tier"].isin(["CRITICAL", "HIGH"])]["amount"].sum())
unique_entities = len(graph_engine.entity_metadata)
suspicious_asn_count = int((scored_df["is_bulletproof_asn"] == 1).sum())
total_tracked_assets = market_summary.get("total_tracked_assets", 4150)
total_tracked_mcap = market_summary.get("total_market_cap_usd", 1.44e12)

col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Transactions</div>
        <div class="metric-value">{total_tx:,}</div>
        <div class="metric-sub" style="color: #58a6ff;">Ingested & Enriched</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card" style="border-left: 4px solid #ff0054;">
        <div class="metric-title">Critical / High Alerts</div>
        <div class="metric-value" style="color: #ff0054;">{crit_alerts + high_alerts}</div>
        <div class="metric-sub" style="color: #ff9f1c;">{crit_alerts} Critical &bull; {high_alerts} High</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card" style="border-left: 4px solid #06d6a0;">
        <div class="metric-title">USD Value at Risk</div>
        <div class="metric-value" style="color: #06d6a0; font-size: 1.55rem;">${total_usd_at_risk:,.2f}</div>
        <div class="metric-sub" style="color: #8b949e;">{total_crypto_units:,.1f} native units</div>
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
        <div class="metric-title">High-Risk ASNs</div>
        <div class="metric-value" style="color: #ffb703;">{suspicious_asn_count}</div>
        <div class="metric-sub" style="color: #8b949e;">Bulletproof / Tor Relays</div>
    </div>
    """, unsafe_allow_html=True)

with col6:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Real Assets Tracked</div>
        <div class="metric-value" style="color: #38bdf8;">{total_tracked_assets:,}</div>
        <div class="metric-sub" style="color: #8b949e;">${total_tracked_mcap/1e12:.2f}T Real Market Cap</div>
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

# Real-World Asset Filter
all_assets = ["All Assets"]
if "coin_symbol" in scored_df.columns:
    all_assets += sorted([str(s) for s in scored_df["coin_symbol"].dropna().unique()])

selected_asset_filter = st.sidebar.selectbox(
    "Filter by Real-World Asset",
    options=all_assets,
    index=0
)

# Market Cap Tier Filter
if "market_cap_tier" in scored_df.columns:
    all_tiers = sorted([str(t) for t in scored_df["market_cap_tier"].dropna().unique()])
    tier_filter = st.sidebar.multiselect("Asset Market Cap Tier", options=all_tiers, default=all_tiers)
else:
    tier_filter = []

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
if selected_asset_filter != "All Assets" and "coin_symbol" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["coin_symbol"] == selected_asset_filter]
if tier_filter and "market_cap_tier" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["market_cap_tier"].isin(tier_filter)]
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
        filtered_df["entity_id"].str.lower().str.contains(q_lower) |
        (filtered_df["coin_name"].str.lower().str.contains(q_lower) if "coin_name" in filtered_df.columns else False) |
        (filtered_df["coin_symbol"].str.lower().str.contains(q_lower) if "coin_symbol" in filtered_df.columns else False)
    ]


# ---------------------------------------------------------
# Main Tabs UI
# ---------------------------------------------------------

tab_alerts, tab_graph, tab_explain, tab_market = st.tabs([
    "🚨 Tab 1: Alerts Feed (Triage Queue)",
    "🕸️ Tab 2: Graph View (Heterogeneous Topology)",
    "🧬 Tab 3: Explainability & Forensic Dossier",
    "🌐 Tab 4: Real-World Asset & Market Intelligence"
])


# =========================================================
# TAB 1: ALERTS FEED
# =========================================================

with tab_alerts:
    st.subheader(f"Triage Queue ({len(filtered_df)} Transactions Matching Filters)")

    # Action bar
    col_a1, col_a2, col_a3 = st.columns([3, 1, 1])
    with col_a1:
        st.caption("Sorted by calibrated Risk Score (0-100). Grounded in real-world crypto prices. Click any transaction hash to inspect in Tab 3.")
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

    # Display Styled Table with Real-World Asset Grounding
    table_cols = ["risk_score", "risk_tier", "tx_id", "timestamp"]
    if "coin_symbol" in filtered_df.columns:
        table_cols.extend(["coin_symbol", "amount", "amount_usd"])
    else:
        table_cols.append("amount")
    table_cols.extend(["network_protocol", "entity_id", "src_ip", "src_org", "pattern_label"])

    table_display = filtered_df[table_cols].sort_values(by="risk_score", ascending=False).copy()

    # Column configuration
    col_configs = {
        "risk_score": st.column_config.ProgressColumn(
            "Risk Score",
            help="Normalized 0-100 Anomaly Risk Score",
            format="%d",
            min_value=0,
            max_value=100
        ),
        "risk_tier": st.column_config.TextColumn("Priority", width="small"),
        "tx_id": st.column_config.TextColumn("Transaction Hash", width="medium"),
        "timestamp": st.column_config.TextColumn("Timestamp (UTC)", width="medium"),
        "amount": st.column_config.NumberColumn("Crypto Amount", format="%.4f"),
        "network_protocol": st.column_config.TextColumn("Protocol", width="small"),
        "entity_id": st.column_config.TextColumn("Entity", width="small"),
        "src_ip": st.column_config.TextColumn("Source IP", width="small"),
        "src_org": st.column_config.TextColumn("ISP / Autonomous System", width="medium"),
        "pattern_label": st.column_config.TextColumn("Typology", width="medium")
    }

    if "coin_symbol" in filtered_df.columns:
        col_configs["coin_symbol"] = st.column_config.TextColumn("Asset", width="small")
        col_configs["amount_usd"] = st.column_config.NumberColumn("USD Value", format="$%.2f")

    st.dataframe(
        table_display,
        use_container_width=True,
        height=520,
        column_config=col_configs
    )

    # Top Alerts Drilldown Card Preview
    st.markdown("---")
    st.markdown("### ⚡ Top 3 Critical Incidents Requiring Immediate Compliance Review")
    top_critical = filtered_df.sort_values(by="risk_score", ascending=False).head(3)

    for _, row in top_critical.iterrows():
        tier = row["risk_tier"]
        badge_cls = f"badge-{tier.lower()}"
        sym = row.get("coin_symbol", "BTC")
        name = row.get("coin_name", "Bitcoin")
        amt_usd = float(row.get("amount_usd", 0.0))
        mcap_tier = row.get("market_cap_tier", "Mega-Cap")

        st.markdown(f"""
        <div class="telemetry-card" style="border-left: 4px solid {'#ff0054' if tier=='CRITICAL' else '#ff9f1c'};">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <div>
                    <span class="{badge_cls}">{tier} PRIORITY &bull; RISK {row['risk_score']}/100</span>
                    <span class="badge-token" style="margin-left: 8px;">{sym} ({name})</span>
                    <span style="color: #8b949e; margin-left: 10px; font-size: 0.85rem;">Pattern: <b>{row['pattern_label']}</b> &bull; Tier: <b>{mcap_tier}</b></span>
                </div>
                <div style="color: #8b949e; font-size: 0.8rem;">{row['timestamp']}</div>
            </div>
            <div style="margin-bottom: 6px;">
                <span style="color: #8b949e;">TX Hash:</span> <span class="mono-hash">{row['tx_id']}</span>
            </div>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; font-size: 0.85rem; color: #c9d1d9;">
                <div><b>Amount:</b> {row['amount']:.4f} {sym} <span style="color: #06d6a0;">(${amt_usd:,.2f})</span></div>
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
    st.caption("SHAP TreeExplainer local feature attributions, regulatory reason code mapping, real asset market valuation, and automated Suspicious Activity Report (SAR) dossier.")

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

    sym = tx_row.get("coin_symbol", "BTC")
    coin_nm = tx_row.get("coin_name", "Bitcoin")
    token_price = float(tx_row.get("token_price_usd", 36456.94))
    amt_usd = float(tx_row.get("amount_usd", tx_row['amount'] * token_price))
    mcap_tier = tx_row.get("market_cap_tier", "Mega-Cap")

    with t_col1:
        st.markdown(f"""
        <div class="telemetry-card">
            <h4 style="margin-top: 0; color: #58a6ff;">📡 Transaction & Network Telemetry</h4>
            <div style="font-size: 0.85rem; line-height: 1.8;">
                <div><b>TX Hash:</b> <span class="mono-hash">{tx_row['tx_id']}</span></div>
                <div><b>Timestamp:</b> {tx_row['timestamp']}</div>
                <div><b>Asset Grounding:</b> <span class="badge-token">{sym}</span> <b>{coin_nm}</b> &bull; Price: <b>${token_price:,.2f} USD</b> ({mcap_tier})</div>
                <div><b>Amount:</b> <span style="font-size: 1.1rem; font-weight: 700; color: #ffffff;">{tx_row['amount']:.4f} {sym}</span> &bull; <span style="color: #06d6a0; font-weight: 700;">${amt_usd:,.2f} USD</span> (Fee: {tx_row['fee']:.6f})</div>
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
The transfer involved {tx_row['amount']:.4f} {sym} ({coin_nm}), representing an estimated ${amt_usd:,.2f} USD exposure.
Protocol: {tx_row['network_protocol']} | Asset Market Tier: {mcap_tier} (Current Market Price: ${token_price:,.2f} USD).

2. SUBJECT & ENTITY ATTRIBUTION:
- Primary Sender: {tx_row['sender_address']}
- Receiver Destination: {tx_row['receiver_address']}
- Common-Input-Ownership Cluster: {tx_row['entity_id']} (Controls {entity_info.get('wallet_count', 1)} discrete address(es))
- Originating IP: {tx_row['src_ip']} (Autonomous System: {tx_row['src_asn']} - {tx_row.get('src_org', 'Unknown')})
- Relay/Destination IP: {tx_row['dst_ip']} (Autonomous System: {tx_row['dst_asn']} - {tx_row.get('dst_org', 'Unknown')})

3. REAL-WORLD ASSET VALUATION & LIQUIDITY:
- Token Symbol & Name: {sym} ({coin_nm})
- Spot Benchmark Price: ${token_price:,.2f} USD
- Calculated USD Transfer Value: ${amt_usd:,.2f} USD
- Market Capitalization Classification: {mcap_tier}

4. TRIGGERED COMPLIANCE REASON CODES & SHAP ATTRIBUTION:
{chr(10).join([f"- [{r.code}] {r.title} ({r.severity}): {r.description} (Ref: {r.regulatory_ref})" for r in reason_codes])}

5. FORENSIC GRAPH & BEHAVIORAL INDICATORS:
- Transaction Inter-Arrival Velocity (Δt): {tx_row['tx_velocity_seconds']:.3f} seconds (automated sweep indicator).
- Peel-Chain Heuristic Factor: {tx_row['peel_chain_heuristic_score']:.3f}.
- Heterogeneous Graph PageRank: {tx_row['tx_pagerank']:.6f} | Betweenness Centrality: {tx_row['tx_betweenness']:.6f}.
- Network Payload Packet Size: {int(tx_row['payload_size_bytes'])} bytes (Deviation Z-Score: {tx_row['payload_zscore']:.2f}).

6. INVESTIGATOR RECOMMENDATION:
Flag entity {tx_row['entity_id']} for Enhanced Due Diligence (EDD). Place originating IP subnet {tx_row['src_ip']}/24
on watch list. Submit formal SAR filing under FATF Recommendation 16, FinCEN travel rule guidelines, and BSA smurfing typologies.
"""

    st.text_area("Forensic Compliance Dossier", sar_narrative, height=260)
    st.download_button(
        label="💾 Download Full SAR Investigation Dossier (.md)",
        data=sar_narrative.encode("utf-8"),
        file_name=f"SAR_Dossier_{tx_row['tx_id'][:12]}.md",
        mime="text/markdown",
        use_container_width=True
    )


# =========================================================
# TAB 4: REAL-WORLD ASSET & MARKET INTELLIGENCE
# =========================================================

with tab_market:
    st.subheader("🌐 Real-World Cryptocurrency Market Intelligence & Asset Risk Radar")
    st.caption("Active monitoring across 4,150 real cryptocurrencies from CryptocurrencyData.csv. Features institutional market cap tiering, wash-trading anomaly detection, and forensic cross-referencing.")

    # Top Global Market KPIs
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)

    with m_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Tracked Cryptocurrencies</div>
            <div class="metric-value" style="color: #38bdf8;">{len(market_df):,}</div>
            <div class="metric-sub" style="color: #8b949e;">Global Asset Universe</div>
        </div>
        """, unsafe_allow_html=True)

    with m_col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Market Cap</div>
            <div class="metric-value" style="color: #06d6a0;">${market_summary['total_market_cap_usd']/1e12:.2f}T</div>
            <div class="metric-sub" style="color: #8b949e;">USD Total Capitalization</div>
        </div>
        """, unsafe_allow_html=True)

    with m_col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">24h Trading Volume</div>
            <div class="metric-value" style="color: #f59e0b;">${market_summary['total_24h_volume_usd']/1e9:.2f}B</div>
            <div class="metric-sub" style="color: #8b949e;">Aggregate Daily Liquidity</div>
        </div>
        """, unsafe_allow_html=True)

    with m_col4:
        wash_count = market_summary.get("wash_trading_suspects", 0)
        vol_count = market_summary.get("extreme_volatility_tokens", 0)
        st.markdown(f"""
        <div class="metric-card" style="border-left: 4px solid #ff0054;">
            <div class="metric-title">Market Anomaly Flags</div>
            <div class="metric-value" style="color: #ff0054;">{wash_count + vol_count}</div>
            <div class="metric-sub" style="color: #ff9f1c;">{wash_count} Wash-Trading &bull; {vol_count} Volatility</div>
        </div>
        """, unsafe_allow_html=True)

    # Sub-Tabs for Market Exploration
    m_tab_radar, m_tab_scanner, m_tab_crossref = st.tabs([
        "🚨 Wash-Trading & Anomaly Radar",
        "📊 4,150-Asset Market Explorer",
        "🔎 Asset Forensic Cross-Reference"
    ])

    # -----------------------------------------------------
    # SUB-TAB 1: Wash-Trading & Anomaly Radar
    # -----------------------------------------------------
    with m_tab_radar:
        st.markdown("### 🚨 Market Anomaly & Wash-Trading Detection")
        st.caption("Identifies tokens exhibiting abnormal liquidity ratios (Daily Volume > Market Cap, indicating potential wash trading or synthetic circulation) and extreme pump-and-dump spikes.")

        radar_mode = st.radio(
            "Select Anomaly Heuristic Filter",
            [
                "Wash-Trading Suspects (Volume-to-Market-Cap Ratio > 1.0)",
                "Extreme Pump & Dump Volatility (|24h Change| >= 35% or |7d Change| >= 100%)",
                "Phantom Tokens (High Volume with Zero Reported Circulating Supply)",
                "All Flagged Anomalies"
            ],
            horizontal=True
        )

        radar_df = market_df.copy()
        if radar_mode == "Wash-Trading Suspects (Volume-to-Market-Cap Ratio > 1.0)":
            radar_df = radar_df[radar_df["is_wash_trading_suspect"]]
        elif radar_mode == "Extreme Pump & Dump Volatility (|24h Change| >= 35% or |7d Change| >= 100%)":
            radar_df = radar_df[radar_df["is_pump_dump_suspect"]]
        elif radar_mode == "Phantom Tokens (High Volume with Zero Reported Circulating Supply)":
            radar_df = radar_df[radar_df["is_phantom_token"]]
        else:
            radar_df = radar_df[radar_df["is_wash_trading_suspect"] | radar_df["is_pump_dump_suspect"] | radar_df["is_phantom_token"]]

        st.write(f"Displaying **{len(radar_df)} flagged tokens** matching criteria:")

        radar_display = radar_df[[
            "rank", "coin_name", "symbol", "price_usd", "change_24h", "change_7d",
            "volume_24h_usd", "market_cap_usd", "vol_to_mcap_ratio", "market_cap_tier"
        ]].sort_values(by="vol_to_mcap_ratio", ascending=False).copy()

        st.dataframe(
            radar_display,
            use_container_width=True,
            height=440,
            column_config={
                "rank": st.column_config.NumberColumn("Rank", width="small"),
                "coin_name": st.column_config.TextColumn("Coin Name", width="medium"),
                "symbol": st.column_config.TextColumn("Symbol", width="small"),
                "price_usd": st.column_config.NumberColumn("Price (USD)", format="$%.4f"),
                "change_24h": st.column_config.NumberColumn("24h Change", format="%.2f%%"),
                "change_7d": st.column_config.NumberColumn("7d Change", format="%.2f%%"),
                "volume_24h_usd": st.column_config.NumberColumn("24h Volume", format="$%.0f"),
                "market_cap_usd": st.column_config.NumberColumn("Market Cap", format="$%.0f"),
                "vol_to_mcap_ratio": st.column_config.ProgressColumn(
                    "Vol / MCap Ratio",
                    help="Values > 1.0 indicate potential wash trading or extreme turnover",
                    format="%.2fx",
                    min_value=0.0,
                    max_value=3.0
                ),
                "market_cap_tier": st.column_config.TextColumn("Tier", width="small")
            }
        )

    # -----------------------------------------------------
    # SUB-TAB 2: Full 4,150-Asset Market Explorer
    # -----------------------------------------------------
    with m_tab_scanner:
        st.markdown("### 📊 Comprehensive 4,150-Asset Market Scanner")

        col_s1, col_s2, col_s3, col_s4 = st.columns(4)

        with col_s1:
            token_search = st.text_input("Search Coin Name or Symbol", placeholder="Bitcoin, SOL, DOGE, etc.").strip().lower()

        with col_s2:
            tier_options = ["All Tiers", "Mega-Cap", "Large-Cap", "Mid-Cap", "Small-Cap", "Micro/Meme-Cap"]
            selected_m_tier = st.selectbox("Filter by Market Cap Tier", tier_options)

        with col_s3:
            min_vol_choice = st.selectbox(
                "Minimum 24h Volume",
                ["Any Volume", "> $100K USD", "> $1M USD", "> $10M USD", "> $100M USD"]
            )

        with col_s4:
            sort_choice = st.selectbox(
                "Sort Assets By",
                ["Market Cap (Rank)", "Highest 24h Volume", "Top 24h Gainers", "Top 24h Losers", "Highest Vol/MCap Ratio"]
            )

        scanner_df = market_df.copy()

        if token_search:
            scanner_df = scanner_df[
                scanner_df["coin_name"].str.lower().str.contains(token_search) |
                scanner_df["symbol"].str.lower().str.contains(token_search)
            ]

        if selected_m_tier != "All Tiers":
            scanner_df = scanner_df[scanner_df["market_cap_tier"] == selected_m_tier]

        if min_vol_choice == "> $100K USD":
            scanner_df = scanner_df[scanner_df["volume_24h_usd"] >= 100_000]
        elif min_vol_choice == "> $1M USD":
            scanner_df = scanner_df[scanner_df["volume_24h_usd"] >= 1_000_000]
        elif min_vol_choice == "> $10M USD":
            scanner_df = scanner_df[scanner_df["volume_24h_usd"] >= 10_000_000]
        elif min_vol_choice == "> $100M USD":
            scanner_df = scanner_df[scanner_df["volume_24h_usd"] >= 100_000_000]

        if sort_choice == "Highest 24h Volume":
            scanner_df = scanner_df.sort_values(by="volume_24h_usd", ascending=False)
        elif sort_choice == "Top 24h Gainers":
            scanner_df = scanner_df.sort_values(by="change_24h", ascending=False)
        elif sort_choice == "Top 24h Losers":
            scanner_df = scanner_df.sort_values(by="change_24h", ascending=True)
        elif sort_choice == "Highest Vol/MCap Ratio":
            scanner_df = scanner_df.sort_values(by="vol_to_mcap_ratio", ascending=False)
        else:
            scanner_df = scanner_df.sort_values(by="rank", ascending=True)

        st.caption(f"Showing {len(scanner_df):,} matching cryptocurrencies:")

        display_cols = [
            "rank", "coin_name", "symbol", "price_usd", "change_1h", "change_24h", "change_7d", "change_30d",
            "volume_24h_usd", "market_cap_usd", "circulating_supply", "market_cap_tier"
        ]

        st.dataframe(
            scanner_df[display_cols].head(300),
            use_container_width=True,
            height=500,
            column_config={
                "rank": st.column_config.NumberColumn("Rank", width="small"),
                "coin_name": st.column_config.TextColumn("Name", width="medium"),
                "symbol": st.column_config.TextColumn("Symbol", width="small"),
                "price_usd": st.column_config.NumberColumn("Price (USD)", format="$%.4f"),
                "change_1h": st.column_config.NumberColumn("1h %", format="%.2f%%"),
                "change_24h": st.column_config.NumberColumn("24h %", format="%.2f%%"),
                "change_7d": st.column_config.NumberColumn("7d %", format="%.2f%%"),
                "change_30d": st.column_config.NumberColumn("30d %", format="%.2f%%"),
                "volume_24h_usd": st.column_config.NumberColumn("24h Volume", format="$%.0f"),
                "market_cap_usd": st.column_config.NumberColumn("Market Cap", format="$%.0f"),
                "circulating_supply": st.column_config.NumberColumn("Circulating Supply", format="%.0f"),
                "market_cap_tier": st.column_config.TextColumn("Tier", width="small")
            }
        )

    # -----------------------------------------------------
    # SUB-TAB 3: Asset Forensic Cross-Reference
    # -----------------------------------------------------
    with m_tab_crossref:
        st.markdown("### 🔎 Forensic Cross-Reference: Asset to On-Chain Alerts")
        st.caption("Select any tracked cryptocurrency to view its real-world market liquidity profile and instantly cross-reference active on-chain forensic alerts involving this asset.")

        # Top 100 assets for quick selection
        top_symbols = list(market_df.sort_values(by="rank", ascending=True)["symbol"].head(100))
        # Ensure symbols present in scored_df are also readily accessible
        if "coin_symbol" in scored_df.columns:
            on_chain_symbols = list(scored_df["coin_symbol"].unique())
            for s in on_chain_symbols:
                if s not in top_symbols:
                    top_symbols.append(s)

        selected_coin_symbol = st.selectbox(
            "Select Cryptocurrency to Cross-Reference",
            options=top_symbols,
            index=0,
            format_func=lambda s: f"{s} - {market_loader.get_coin_by_symbol(s)['coin_name'] if market_loader.get_coin_by_symbol(s) else 'Unknown'}"
        )

        coin_data = market_loader.get_coin_by_symbol(selected_coin_symbol)
        if coin_data:
            c_col1, c_col2 = st.columns([1.2, 1])

            with c_col1:
                st.markdown(f"""
                <div class="telemetry-card">
                    <h4 style="margin-top: 0; color: #38bdf8;">🪙 Real-World Asset Dossier: {coin_data['coin_name']} ({coin_data['symbol']})</h4>
                    <div style="font-size: 0.88rem; line-height: 1.8;">
                        <div><b>Global Rank:</b> #{coin_data['rank']} &bull; <b>Market Cap Tier:</b> {coin_data['market_cap_tier']}</div>
                        <div><b>Spot Price:</b> <span style="font-size: 1.2rem; font-weight: 700; color: #ffffff;">${coin_data['price_usd']:,.4f} USD</span></div>
                        <div><b>Market Capitalization:</b> ${coin_data['market_cap_usd']:,.2f} USD</div>
                        <div><b>24h Trading Volume:</b> ${coin_data['volume_24h_usd']:,.2f} USD (Turnover: {coin_data['vol_to_mcap_ratio']:.2f}x)</div>
                        <div><b>Price Movement:</b> 1h: <b>{coin_data['change_1h']:.2f}%</b> &bull; 24h: <b>{coin_data['change_24h']:.2f}%</b> &bull; 7d: <b>{coin_data['change_7d']:.2f}%</b> &bull; 30d: <b>{coin_data['change_30d']:.2f}%</b></div>
                        <div><b>Circulating Supply:</b> {coin_data['circulating_supply']:,.0f} &bull; <b>Total Supply:</b> {coin_data['total_supply']:,.0f}</div>
                        <div><b>Wash-Trading Suspect Flag:</b> {'<span class="badge-critical">YES (High Turnover)</span>' if coin_data['is_wash_trading_suspect'] else '<span style="color: #06d6a0;">Normal Liquidity</span>'}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with c_col2:
                # Query on-chain alerts for this symbol
                if "coin_symbol" in scored_df.columns:
                    matching_txs = scored_df[scored_df["coin_symbol"] == selected_coin_symbol]
                    high_risk_txs = matching_txs[matching_txs["risk_tier"].isin(["CRITICAL", "HIGH"])]
                    total_exposed_usd = float(high_risk_txs["amount_usd"].sum()) if not high_risk_txs.empty else 0.0

                    if not matching_txs.empty:
                        st.markdown(f"""
                        <div class="telemetry-card" style="border-left: 4px solid {'#ff0054' if not high_risk_txs.empty else '#06d6a0'};">
                            <h4 style="margin-top: 0; color: #ff0054;">🛡️ Active Forensic Telemetry Correlated</h4>
                            <div style="font-size: 0.88rem; line-height: 1.8;">
                                <div><b>Total On-Chain Transactions Captured:</b> {len(matching_txs)}</div>
                                <div><b>Flagged Critical / High Incidents:</b> <span class="badge-critical">{len(high_risk_txs)}</span></div>
                                <div><b>Total USD Exposure at Risk:</b> <span style="font-size: 1.15rem; font-weight: 700; color: #06d6a0;">${total_exposed_usd:,.2f} USD</span></div>
                                <div><b>Associated Entities Identified:</b> {matching_txs['entity_id'].nunique()} distinct group(s)</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        if not high_risk_txs.empty:
                            st.markdown("##### ⚠️ Flagged Illicit Transactions Involving this Token")
                            sub_table = high_risk_txs[["risk_score", "risk_tier", "tx_id", "amount", "amount_usd", "pattern_label"]].head(5)
                            st.dataframe(sub_table, use_container_width=True)
                    else:
                        st.markdown(f"""
                        <div class="telemetry-card" style="border-left: 4px solid #06d6a0;">
                            <h4 style="margin-top: 0; color: #06d6a0;">🛡️ Forensic Telemetry Clear</h4>
                            <p style="font-size: 0.88rem; color: #8b949e; margin-bottom: 0;">
                                No transactions involving <b>{coin_data['coin_name']} ({coin_data['symbol']})</b> were flagged in the active on-chain telemetry buffer.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)


# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #8b949e; font-size: 0.8rem; padding-bottom: 20px;">
    Crypto & P2P Forensic Intelligence Platform &bull; Real-World Market Intelligence (4,150 Cryptocurrencies) &bull; Built with Streamlit, NetworkX, Scikit-Learn & SHAP
</div>
""", unsafe_allow_html=True)

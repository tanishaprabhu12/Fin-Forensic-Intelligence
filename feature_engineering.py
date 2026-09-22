"""
feature_engineering.py
-----------------------
Multi-Modal Forensic Feature Engineering Matrix.
Computes and fuses:
- Network Features: IP connection burst frequency, ASN prevalence & risk weights, payload anomalies.
- Blockchain Features: Fan-in/fan-out ratios, transaction velocity (delta-t), fee ratios,
  round-number structuring heuristics, peel-chain heuristic scores.
- Graph Topology Features: In/Out Degree, PageRank, Betweenness Centrality.
- Structural Graph Embeddings: 16-dimensional spectral/graph structural projection.
- Merges all signals into a clean, normalized, ML-ready feature matrix.
"""

import logging
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

from ingestion_pipeline import EnrichedTransaction
from graph_engine import HeterogeneousGraphEngine

logger = logging.getLogger("ForensicFeatureEngineering")


class FeatureEngineeringMatrix:
    """
    Feature Extraction and Engineering Engine that synthesizes raw blockchain,
    P2P telemetry, CIOH entity clusters, and graph centrality into an ML-ready matrix.
    """

    def __init__(self, graph_engine: HeterogeneousGraphEngine):
        self.graph_engine = graph_engine
        self.feature_columns: List[str] = []
        self.svd_transformer: Optional[TruncatedSVD] = None

    def is_round_number(self, amount: float) -> int:
        """Heuristic check for structured round amounts (e.g., 5.0, 10.0, 50.0, 100.0, 500.0)."""
        if amount <= 0:
            return 0
        # Check if amount is an exact integer or close to common structuring thresholds
        is_int = abs(amount - round(amount)) < 1e-4
        if is_int:
            val = int(round(amount))
            if val in [1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 5000, 10000]:
                return 1
            if val % 5 == 0 or val % 10 == 0:
                return 1
        # Common decimal thresholds (e.g. 0.5, 1.5, 9.5, 9.8)
        if abs(amount - round(amount, 1)) < 1e-4 and round(amount, 1) in [0.5, 1.5, 2.5, 5.0, 9.5, 9.8, 9.9]:
            return 1
        return 0

    def compute_peel_chain_heuristic(self, tx: EnrichedTransaction, delta_t_sec: float) -> float:
        """
        Calculates a peel chain heuristic score:
        - Rapid succession (delta_t < 30 sec)
        - 2 outputs (peel amount + change return address)
        - Protocol is Bitcoin or UTXO-like
        """
        score = 0.0
        if tx.receiver_count >= 2:
            score += 0.4
        if delta_t_sec < 15.0:
            score += 0.35
        elif delta_t_sec < 60.0:
            score += 0.15
        if tx.amount < 5.0 and tx.receiver_count >= 2:
            score += 0.25
        return min(score, 1.0)

    def extract_features(
        self,
        transactions: List[EnrichedTransaction],
        fit_embedder: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extracts comprehensive network, blockchain, graph, and embedding features.
        Returns:
          - X_features: DataFrame containing only numerical features for ML.
          - full_df: DataFrame containing metadata + engineered features for dashboard/reporting.
        """
        logger.info(f"Extracting forensic feature matrix for {len(transactions)} transactions...")

        # Ensure graph and metrics are populated
        if self.graph_engine.graph.number_of_nodes() == 0:
            self.graph_engine.build_heterogeneous_graph(transactions)
        graph_metrics = self.graph_engine.compute_graph_metrics()

        # Sort chronologically for velocity calculations
        sorted_txs = sorted(transactions, key=lambda t: t.timestamp)

        # Precompute IP frequencies and last seen timestamps
        ip_counts: Dict[str, int] = {}
        ip_last_seen: Dict[str, float] = {}
        sender_last_seen: Dict[str, float] = {}
        ip_wallets_set: Dict[str, set] = {}

        for tx in sorted_txs:
            ip = tx.src_ip
            ip_counts[ip] = ip_counts.get(ip, 0) + 1
            if ip not in ip_wallets_set:
                ip_wallets_set[ip] = set()
            for w in tx.input_wallets:
                ip_wallets_set[ip].add(w)

        # Compute payload size distribution stats
        payloads = [tx.payload_size_bytes for tx in sorted_txs]
        mean_payload = float(np.mean(payloads)) if payloads else 500.0
        std_payload = float(np.std(payloads)) if payloads and np.std(payloads) > 0 else 100.0

        records: List[Dict[str, Any]] = []

        for tx in sorted_txs:
            t_seconds = tx.timestamp.timestamp()
            src_ip = tx.src_ip
            primary_sender = tx.input_wallets[0] if tx.input_wallets else tx.sender_address

            # 1. Transaction Velocity (Delta-T in seconds)
            last_ip_t = ip_last_seen.get(src_ip, t_seconds - 3600.0)
            last_sender_t = sender_last_seen.get(primary_sender, t_seconds - 3600.0)
            delta_t_ip = max(0.001, t_seconds - last_ip_t)
            delta_t_sender = max(0.001, t_seconds - last_sender_t)
            delta_t = min(delta_t_ip, delta_t_sender)

            ip_last_seen[src_ip] = t_seconds
            sender_last_seen[primary_sender] = t_seconds

            # 2. Network Features
            conn_freq_src = ip_counts.get(src_ip, 1)
            ip_wallet_diversity = len(ip_wallets_set.get(src_ip, set()))
            payload_zscore = abs(tx.payload_size_bytes - mean_payload) / (std_payload + 1e-5)
            is_bulletproof = 1 if (tx.src_enrichment.is_bulletproof or tx.dst_enrichment.is_bulletproof) else 0
            is_tor = 1 if (tx.src_enrichment.is_tor or tx.dst_enrichment.is_tor) else 0
            src_asn_risk = tx.src_enrichment.asn_risk_weight
            dst_asn_risk = tx.dst_enrichment.asn_risk_weight

            # 3. Blockchain Features
            fee_to_amount = tx.fee_to_amount_ratio
            round_num_flag = self.is_round_number(tx.amount)
            fan_in = tx.sender_count
            fan_out = tx.receiver_count
            fan_ratio = fan_in / (fan_out + 1e-5)
            peel_score = self.compute_peel_chain_heuristic(tx, delta_t)

            # Entity cluster info from CIOH
            entity_id = self.graph_engine.wallet_to_entity.get(primary_sender, "Entity-Unclustered")
            entity_info = self.graph_engine.entity_metadata.get(entity_id, {})
            entity_wallet_count = entity_info.get("wallet_count", 1)

            # 4. Graph Topology Features
            tx_node = tx.tx_id
            src_ip_node = f"IP:{tx.src_ip}"
            sender_wallet_node = f"WALLET:{primary_sender}"

            tx_m = graph_metrics.get(tx_node, {})
            src_ip_m = graph_metrics.get(src_ip_node, {})
            wallet_m = graph_metrics.get(sender_wallet_node, {})

            tx_pagerank = tx_m.get("pagerank", 0.0)
            src_ip_pagerank = src_ip_m.get("pagerank", 0.0)
            wallet_pagerank = wallet_m.get("pagerank", 0.0)
            tx_betweenness = tx_m.get("betweenness", 0.0)
            wallet_betweenness = wallet_m.get("betweenness", 0.0)
            tx_in_deg = tx_m.get("in_degree", 1)
            tx_out_deg = tx_m.get("out_degree", 1)

            # Compile record
            rec = {
                # Metadata
                "tx_id": tx.tx_id,
                "timestamp": tx.timestamp,
                "sender_address": tx.sender_address,
                "receiver_address": tx.receiver_address,
                "network_protocol": tx.network_protocol,
                "src_ip": tx.src_ip,
                "dst_ip": tx.dst_ip,
                "src_asn": tx.src_asn,
                "dst_asn": tx.dst_asn,
                "pattern_label": tx.pattern_label,
                "entity_id": entity_id,
                "src_org": tx.src_enrichment.org,
                "dst_org": tx.dst_enrichment.org,
                "is_high_risk_network": int(tx.is_high_risk_network),

                # Real-world cryptocurrency metadata
                "coin_symbol": getattr(tx, "coin_symbol", "BTC"),
                "coin_name": getattr(tx, "coin_name", "Bitcoin"),
                "token_price_usd": float(getattr(tx, "token_price_usd", 36456.94) or 36456.94),
                "amount_usd": float(getattr(tx, "amount_usd", 0.0) or round(float(tx.amount) * 36456.94, 2)),
                "market_cap_tier": getattr(tx, "market_cap_tier", "Mega-Cap"),

                # Numerical ML Features
                "amount": float(tx.amount),
                "fee": float(tx.fee),
                "fee_to_amount_ratio": float(fee_to_amount),
                "log_amount": float(np.log1p(tx.amount)),
                "log_fee": float(np.log1p(tx.fee)),
                "payload_size_bytes": float(tx.payload_size_bytes),
                "payload_zscore": float(payload_zscore),
                "tx_velocity_seconds": float(delta_t),
                "log_velocity": float(np.log1p(delta_t)),
                "is_round_number_amount": int(round_num_flag),
                "fan_in_count": int(fan_in),
                "fan_out_count": int(fan_out),
                "fan_in_fan_out_ratio": float(fan_ratio),
                "peel_chain_heuristic_score": float(peel_score),
                "entity_wallet_cluster_size": int(entity_wallet_count),
                "ip_connection_freq_src": int(conn_freq_src),
                "ip_wallet_diversity": int(ip_wallet_diversity),
                "src_asn_risk_weight": float(src_asn_risk),
                "dst_asn_risk_weight": float(dst_asn_risk),
                "is_bulletproof_asn": int(is_bulletproof),
                "is_tor_relay": int(is_tor),
                "tx_pagerank": float(tx_pagerank),
                "src_ip_pagerank": float(src_ip_pagerank),
                "wallet_pagerank": float(wallet_pagerank),
                "tx_betweenness": float(tx_betweenness),
                "wallet_betweenness": float(wallet_betweenness),
                "tx_in_degree": int(tx_in_deg),
                "tx_out_degree": int(tx_out_deg)
            }
            records.append(rec)

        full_df = pd.DataFrame(records)

        # 5. Extract 16-Dimensional Structural Embeddings via Graph Spectral SVD
        embedding_df = self._generate_structural_embeddings(full_df, fit=fit_embedder)
        
        # Merge structural embeddings with full dataframe
        full_df = pd.concat([full_df, embedding_df], axis=1)

        # Define explicit ML feature column list
        self.feature_columns = [
            "amount", "fee", "fee_to_amount_ratio", "log_amount", "log_fee",
            "payload_size_bytes", "payload_zscore", "tx_velocity_seconds", "log_velocity",
            "is_round_number_amount", "fan_in_count", "fan_out_count", "fan_in_fan_out_ratio",
            "peel_chain_heuristic_score", "entity_wallet_cluster_size",
            "ip_connection_freq_src", "ip_wallet_diversity",
            "src_asn_risk_weight", "dst_asn_risk_weight", "is_bulletproof_asn", "is_tor_relay",
            "tx_pagerank", "src_ip_pagerank", "wallet_pagerank",
            "tx_betweenness", "wallet_betweenness", "tx_in_degree", "tx_out_degree"
        ] + list(embedding_df.columns)

        X_features = full_df[self.feature_columns].copy()
        X_features = X_features.fillna(0.0)

        logger.info(f"Engineered feature matrix shape: {X_features.shape} ({len(self.feature_columns)} features)")
        return X_features, full_df

    def _generate_structural_embeddings(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Extracts a 16-dimensional structural / topological embedding matrix
        projecting graph connectivity, entity clustering, and flow vectors.
        """
        n_components = 16
        n_samples = len(df)

        # Build structural seed matrix from topological and relational features
        seed_features = [
            "tx_pagerank", "src_ip_pagerank", "wallet_pagerank",
            "tx_betweenness", "wallet_betweenness", "tx_in_degree", "tx_out_degree",
            "fan_in_count", "fan_out_count", "entity_wallet_cluster_size",
            "ip_connection_freq_src", "ip_wallet_diversity", "peel_chain_heuristic_score"
        ]
        sub_mat = df[seed_features].to_numpy()

        # Expand feature space non-linearly to provide rich basis for SVD
        expanded = np.hstack([
            sub_mat,
            np.sin(sub_mat * 10),
            np.cos(sub_mat * 5),
            np.log1p(np.maximum(0, sub_mat))
        ])

        if fit or self.svd_transformer is None:
            self.svd_transformer = TruncatedSVD(n_components=n_components, random_state=42)
            embeddings = self.svd_transformer.fit_transform(expanded)
        else:
            embeddings = self.svd_transformer.transform(expanded)

        embed_cols = [f"embed_dim_{i:02d}" for i in range(n_components)]
        return pd.DataFrame(embeddings, columns=embed_cols)


if __name__ == "__main__":
    from ingestion_pipeline import IngestionPipeline
    pipeline = IngestionPipeline()
    txs = pipeline.ingest_file("data/transactions.csv")

    graph_eng = HeterogeneousGraphEngine()
    fe_matrix = FeatureEngineeringMatrix(graph_eng)
    X, df_full = fe_matrix.extract_features(txs)

    print(f"[+] Feature Matrix X shape: {X.shape}")
    print(f"[+] Full DataFrame shape: {df_full.shape}")
    print(f"[+] Feature names: {fe_matrix.feature_columns[:10]} ... + {len(fe_matrix.feature_columns)-10} more")

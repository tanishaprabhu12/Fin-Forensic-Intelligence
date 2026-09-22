"""
graph_engine.py
----------------
Heterogeneous Graph Correlation Engine & Common-Input-Ownership Heuristic.
Features:
- Constructs directed heterogeneous NetworkX graph [IP, Wallet, TXID, Entity]
- Edge types: BROADCASTED_FROM, INPUTS_TO, OUTPUTS_TO, RECEIVED_BY
- Implements Common-Input-Ownership Heuristic (CIOH) via connected components
- Computes graph metrics: Node In/Out Degree, PageRank (0.85 damping), Betweenness Centrality
- Subgraph / Ego-network extraction for forensic deep-dive visualization
- Seamless export to PyVis interactive physics network visualization.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple, Any
import networkx as nx
import pandas as pd
from pyvis.network import Network

from ingestion_pipeline import EnrichedTransaction

logger = logging.getLogger("ForensicGraphEngine")

# Color palette for PyVis & Dark Mode UI
NODE_COLORS = {
    "WALLET": "#00F0FF",       # Neon Cyan
    "TXID": "#9D4EDD",         # Electric Purple
    "IP": "#FFB703",           # Amber Yellow
    "IP_HIGH_RISK": "#FF0054",  # Crimson Red
    "ENTITY": "#06D6A0"        # Emerald Green
}


class HeterogeneousGraphEngine:
    """
    Core graph correlation engine managing heterogeneous multi-modal networks,
    co-spending entity clustering heuristics, and topological centrality calculations.
    """

    def __init__(self):
        self.graph: nx.DiGraph = nx.DiGraph()
        self.undirected_cospend_graph: nx.Graph = nx.Graph()
        self.wallet_to_entity: Dict[str, str] = {}
        self.entity_metadata: Dict[str, Dict[str, Any]] = {}
        self.pagerank_scores: Dict[str, float] = {}
        self.betweenness_scores: Dict[str, float] = {}

    def run_common_input_ownership_heuristic(self, transactions: List[EnrichedTransaction]) -> Dict[str, str]:
        """
        Implements the Common-Input-Ownership Heuristic (CIOH).
        If multiple input addresses co-sign a transaction, they are controlled by the same entity.
        Computes connected components over the co-spending graph to group wallets into single Entity IDs.
        """
        self.undirected_cospend_graph.clear()
        all_wallets: Set[str] = set()

        # Step 1: Add all wallets and co-spending edges
        for tx in transactions:
            inputs = tx.input_wallets
            outputs = tx.output_wallets
            for w in inputs + outputs:
                all_wallets.add(w)
                self.undirected_cospend_graph.add_node(w)

            # If multi-input, connect all pairs in the input set
            if len(inputs) > 1:
                for i in range(len(inputs)):
                    for j in range(i + 1, len(inputs)):
                        self.undirected_cospend_graph.add_edge(
                            inputs[i], inputs[j],
                            tx_id=tx.tx_id,
                            timestamp=str(tx.timestamp)
                        )

        # Step 2: Extract Connected Components
        connected_clusters = list(nx.connected_components(self.undirected_cospend_graph))
        self.wallet_to_entity.clear()
        self.entity_metadata.clear()

        # Sort clusters by size (largest first)
        connected_clusters.sort(key=lambda c: len(c), reverse=True)

        entity_idx = 1
        for cluster in connected_clusters:
            entity_id = f"Entity-{entity_idx:03d}"
            entity_idx += 1
            
            self.entity_metadata[entity_id] = {
                "entity_id": entity_id,
                "wallet_count": len(cluster),
                "wallets": list(cluster),
                "total_volume": 0.0,
                "tx_count": 0,
                "associated_ips": set()
            }
            for wallet in cluster:
                self.wallet_to_entity[wallet] = entity_id

        logger.info(
            f"CIOH clustering complete: Grouped {len(all_wallets)} unique wallets into "
            f"{len(connected_clusters)} distinct entities. "
            f"Multi-wallet entities: {sum(1 for c in connected_clusters if len(c) > 1)}"
        )
        return self.wallet_to_entity

    def build_heterogeneous_graph(self, transactions: List[EnrichedTransaction]) -> nx.DiGraph:
        """
        Constructs the full heterogeneous directed network:
        Nodes:
          - IP (attributes: ip, asn, country, org, is_high_risk, type='IP')
          - Wallet (attributes: address, entity_id, type='WALLET')
          - TXID (attributes: tx_id, amount, fee, protocol, timestamp, type='TXID')
        Edges:
          - (src_ip, tx_id): BROADCASTED_FROM
          - (sender_wallet, tx_id): INPUTS_TO
          - (tx_id, receiver_wallet): OUTPUTS_TO
          - (tx_id, dst_ip): RECEIVED_BY
        """
        self.graph.clear()
        # Ensure CIOH clustering is run first
        self.run_common_input_ownership_heuristic(transactions)

        for tx in transactions:
            tx_node = tx.tx_id
            src_ip_node = f"IP:{tx.src_ip}"
            dst_ip_node = f"IP:{tx.dst_ip}"

            # 1. Add Transaction Node
            self.graph.add_node(
                tx_node,
                node_type="TXID",
                label=tx.tx_id[:10] + "...",
                full_id=tx.tx_id,
                amount=tx.amount,
                fee=tx.fee,
                protocol=tx.network_protocol,
                timestamp=str(tx.timestamp),
                pattern_label=tx.pattern_label,
                is_high_risk=tx.is_high_risk_network,
                title=(
                    f"<b>Transaction:</b> {tx.tx_id}<br>"
                    f"<b>Amount:</b> {tx.amount:.4f} {tx.network_protocol}<br>"
                    f"<b>Fee:</b> {tx.fee:.6f}<br>"
                    f"<b>Protocol:</b> {tx.network_protocol}<br>"
                    f"<b>Timestamp:</b> {tx.timestamp}<br>"
                    f"<b>Pattern:</b> {tx.pattern_label}"
                ),
                color=NODE_COLORS["TXID"],
                size=16
            )

            # 2. Add Source IP Node
            src_color = NODE_COLORS["IP_HIGH_RISK"] if tx.src_enrichment.tier in ["HIGH_RISK", "CRITICAL_RISK"] else NODE_COLORS["IP"]
            self.graph.add_node(
                src_ip_node,
                node_type="IP",
                label=tx.src_ip,
                full_id=tx.src_ip,
                asn=tx.src_asn,
                org=tx.src_enrichment.org,
                country=tx.src_enrichment.country,
                tier=tx.src_enrichment.tier,
                is_high_risk=tx.src_enrichment.tier in ["HIGH_RISK", "CRITICAL_RISK"],
                title=(
                    f"<b>Source IP:</b> {tx.src_ip}<br>"
                    f"<b>ASN:</b> {tx.src_asn} ({tx.src_enrichment.org})<br>"
                    f"<b>Country:</b> {tx.src_enrichment.country}<br>"
                    f"<b>Risk Tier:</b> {tx.src_enrichment.tier}"
                ),
                color=src_color,
                size=18
            )

            # 3. Add Destination IP Node
            dst_color = NODE_COLORS["IP_HIGH_RISK"] if tx.dst_enrichment.tier in ["HIGH_RISK", "CRITICAL_RISK"] else NODE_COLORS["IP"]
            self.graph.add_node(
                dst_ip_node,
                node_type="IP",
                label=tx.dst_ip,
                full_id=tx.dst_ip,
                asn=tx.dst_asn,
                org=tx.dst_enrichment.org,
                country=tx.dst_enrichment.country,
                tier=tx.dst_enrichment.tier,
                is_high_risk=tx.dst_enrichment.tier in ["HIGH_RISK", "CRITICAL_RISK"],
                title=(
                    f"<b>Relay/Dest IP:</b> {tx.dst_ip}<br>"
                    f"<b>ASN:</b> {tx.dst_asn} ({tx.dst_enrichment.org})<br>"
                    f"<b>Country:</b> {tx.dst_enrichment.country}<br>"
                    f"<b>Risk Tier:</b> {tx.dst_enrichment.tier}"
                ),
                color=dst_color,
                size=18
            )

            # 4. Add Sender Wallet Nodes & Edges (INPUTS_TO)
            for sender_w in tx.input_wallets:
                sender_node = f"WALLET:{sender_w}"
                entity_id = self.wallet_to_entity.get(sender_w, "Entity-Unclustered")
                
                # Update entity metadata
                if entity_id in self.entity_metadata:
                    self.entity_metadata[entity_id]["total_volume"] += tx.amount
                    self.entity_metadata[entity_id]["tx_count"] += 1
                    self.entity_metadata[entity_id]["associated_ips"].add(tx.src_ip)

                self.graph.add_node(
                    sender_node,
                    node_type="WALLET",
                    label=sender_w[:8] + "...",
                    full_id=sender_w,
                    entity_id=entity_id,
                    title=(
                        f"<b>Wallet Address:</b> {sender_w}<br>"
                        f"<b>Clustered Entity:</b> {entity_id}<br>"
                        f"<b>Entity Wallet Count:</b> {self.entity_metadata.get(entity_id, {}).get('wallet_count', 1)}"
                    ),
                    color=NODE_COLORS["WALLET"],
                    size=14
                )
                self.graph.add_edge(
                    sender_node, tx_node,
                    relation="INPUTS_TO",
                    amount=tx.amount / len(tx.input_wallets),
                    title=f"INPUTS_TO: {tx.amount:.4f} crypto"
                )

            # 5. Add Receiver Wallet Nodes & Edges (OUTPUTS_TO)
            for receiver_w in tx.output_wallets:
                receiver_node = f"WALLET:{receiver_w}"
                entity_id = self.wallet_to_entity.get(receiver_w, "Entity-Unclustered")
                
                self.graph.add_node(
                    receiver_node,
                    node_type="WALLET",
                    label=receiver_w[:8] + "...",
                    full_id=receiver_w,
                    entity_id=entity_id,
                    title=(
                        f"<b>Wallet Address:</b> {receiver_w}<br>"
                        f"<b>Clustered Entity:</b> {entity_id}<br>"
                        f"<b>Entity Wallet Count:</b> {self.entity_metadata.get(entity_id, {}).get('wallet_count', 1)}"
                    ),
                    color=NODE_COLORS["WALLET"],
                    size=14
                )
                self.graph.add_edge(
                    tx_node, receiver_node,
                    relation="OUTPUTS_TO",
                    amount=tx.amount / len(tx.output_wallets),
                    title=f"OUTPUTS_TO: {tx.amount:.4f} crypto"
                )

            # 6. Add Network Broadcast and Relay Edges
            self.graph.add_edge(
                src_ip_node, tx_node,
                relation="BROADCASTED_FROM",
                protocol=tx.network_protocol,
                title=f"BROADCASTED_FROM IP: {tx.src_ip} ({tx.src_enrichment.org})"
            )
            self.graph.add_edge(
                tx_node, dst_ip_node,
                relation="RECEIVED_BY",
                protocol=tx.network_protocol,
                title=f"RECEIVED_BY IP: {tx.dst_ip} ({tx.dst_enrichment.org})"
            )

        logger.info(
            f"Heterogeneous Graph constructed: {self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} edges."
        )
        return self.graph

    def compute_graph_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Calculates Node In-Degree, Out-Degree, PageRank (damping 0.85),
        and Betweenness Centrality across the entire heterogeneous topology.
        """
        if self.graph.number_of_nodes() == 0:
            return {}

        logger.info("Computing PageRank and Centrality metrics...")
        # PageRank with standard damping = 0.85
        try:
            self.pagerank_scores = nx.pagerank(self.graph, alpha=0.85, max_iter=150, tol=1e-6)
        except Exception as e:
            logger.warning(f"PageRank fallback: {e}")
            self.pagerank_scores = {node: 1.0 / self.graph.number_of_nodes() for node in self.graph.nodes()}

        # Betweenness Centrality (sample-based if graph is large for fast interactive execution)
        k_samples = min(200, self.graph.number_of_nodes())
        try:
            self.betweenness_scores = nx.betweenness_centrality(self.graph, k=k_samples, normalized=True, seed=42)
        except Exception as e:
            logger.warning(f"Betweenness Centrality fallback: {e}")
            self.betweenness_scores = {node: 0.0 for node in self.graph.nodes()}

        metrics_by_node = {}
        for node in self.graph.nodes():
            metrics_by_node[node] = {
                "in_degree": self.graph.in_degree(node),
                "out_degree": self.graph.out_degree(node),
                "total_degree": self.graph.degree(node),
                "pagerank": self.pagerank_scores.get(node, 0.0),
                "betweenness": self.betweenness_scores.get(node, 0.0)
            }
        return metrics_by_node

    def extract_subgraph_for_entity_or_tx(self, query_id: str, hops: int = 2, max_nodes: int = 80) -> nx.DiGraph:
        """
        Extracts a localized ego-network / subgraph around a specific TXID, Wallet, or IP for deep forensic inspection.
        """
        target_node = query_id
        if not self.graph.has_node(target_node):
            # Check prefixes
            candidates = [f"WALLET:{query_id}", f"IP:{query_id}", query_id]
            for cand in candidates:
                if self.graph.has_node(cand):
                    target_node = cand
                    break

        if not self.graph.has_node(target_node):
            # Return high-degree sample if target not found
            top_nodes = sorted(self.graph.nodes(), key=lambda n: self.graph.degree(n), reverse=True)[:max_nodes]
            return self.graph.subgraph(top_nodes).copy()

        # Extract k-hop ego network (undirected view for neighborhood, directed subgraph)
        undirected_view = self.graph.to_undirected()
        sub_nodes = set([target_node])
        current_layer = set([target_node])

        for _ in range(hops):
            next_layer = set()
            for n in current_layer:
                next_layer.update(undirected_view.neighbors(n))
            sub_nodes.update(next_layer)
            current_layer = next_layer
            if len(sub_nodes) >= max_nodes:
                break

        if len(sub_nodes) > max_nodes:
            sub_nodes = set(list(sub_nodes)[:max_nodes])
            sub_nodes.add(target_node)

        return self.graph.subgraph(sub_nodes).copy()

    def export_to_pyvis(
        self,
        subgraph: Optional[nx.DiGraph] = None,
        height: str = "650px",
        width: str = "100%",
        physics: bool = True
    ) -> str:
        """
        Generates an interactive PyVis HTML network visualization with dark mode aesthetics,
        physics toggles, and customized node/edge styles.
        """
        g_to_render = subgraph if subgraph is not None else self.graph

        net = Network(height=height, width=width, directed=True, bgcolor="#0d1117", font_color="#c9d1d9")
        net.toggle_physics(physics)

        # Set physics options for clean layout
        net.set_options("""
        var options = {
          "nodes": {
            "borderWidth": 2,
            "borderWidthSelected": 4,
            "shadow": true,
            "font": { "size": 12, "face": "Courier New, monospace", "color": "#e6edf3" }
          },
          "edges": {
            "color": { "inherit": false, "color": "#484f58", "highlight": "#58a6ff", "hover": "#8b949e" },
            "smooth": { "type": "continuous" },
            "arrows": { "to": { "enabled": true, "scaleFactor": 0.6 } }
          },
          "physics": {
            "forceAtlas2Based": {
              "gravitationalConstant": -50,
              "centralGravity": 0.01,
              "springLength": 100,
              "springConstant": 0.08,
              "damping": 0.4
            },
            "solver": "forceAtlas2Based",
            "stabilization": { "iterations": 100 }
          },
          "interaction": {
            "hover": true,
            "tooltipDelay": 200,
            "navigationButtons": true,
            "keyboard": true
          }
        }
        """)

        # Add nodes with custom visual styles
        for node_id, data in g_to_render.nodes(data=True):
            node_type = data.get("node_type", "WALLET")
            label = data.get("label", str(node_id))
            color = data.get("color", NODE_COLORS.get(node_type, "#58a6ff"))
            title = data.get("title", str(node_id))
            size = data.get("size", 15)

            # Node shape based on type
            shape = "dot"
            if node_type == "TXID":
                shape = "diamond"
            elif node_type == "IP":
                shape = "square"
            elif node_type == "WALLET":
                shape = "dot"
            elif node_type == "ENTITY":
                shape = "hexagon"

            net.add_node(
                node_id,
                label=label,
                title=title,
                color=color,
                size=size,
                shape=shape
            )

        # Add edges
        for u, v, data in g_to_render.edges(data=True):
            rel = data.get("relation", "")
            title = data.get("title", rel)
            edge_color = "#30363d"
            if rel == "INPUTS_TO":
                edge_color = "#38bdf8"  # light cyan
            elif rel == "OUTPUTS_TO":
                edge_color = "#a855f7"  # purple
            elif rel == "BROADCASTED_FROM":
                edge_color = "#fbbf24"  # amber
            elif rel == "RECEIVED_BY":
                edge_color = "#f43f5e"  # rose

            net.add_edge(u, v, title=title, color=edge_color)

        return net.generate_html()


if __name__ == "__main__":
    from ingestion_pipeline import IngestionPipeline
    pipeline = IngestionPipeline()
    txs = pipeline.ingest_file("data/transactions.csv")

    graph_eng = HeterogeneousGraphEngine()
    G = graph_eng.build_heterogeneous_graph(txs)
    metrics = graph_eng.compute_graph_metrics()

    print(f"[+] Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    sample_sub = graph_eng.extract_subgraph_for_entity_or_tx(txs[0].tx_id, hops=2)
    print(f"[+] Sample 2-hop Subgraph size: {sample_sub.number_of_nodes()} nodes, {sample_sub.number_of_edges()} edges.")
    html_str = graph_eng.export_to_pyvis(sample_sub)
    print(f"[+] Generated PyVis HTML payload of length {len(html_str)} chars.")

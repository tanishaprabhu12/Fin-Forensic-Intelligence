"""
generate_synthetic_data.py
---------------------------
Realistic Synthetic Blockchain & P2P Transaction Generator.
Simulates normal network activity as well as sophisticated forensic anomaly patterns:
- Peel Chains (rapid sequential peeling with change addresses)
- Smurfing / Structuring Rings (fan-in consolidation & fan-out dispersal)
- High-Velocity Automated Draining (sub-second bursts)
- High-Risk ASN & Bulletproof / Tor Exit Node Broadcasts
- Multi-Input Co-Spending (for Common-Input-Ownership Heuristic verification)
- Round-Number Structuring & Fee Anomalies
"""

import json
import random
import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
from faker import Faker

fake = Faker()
Faker.seed(42)
random.seed(42)

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Protocol catalog
PROTOCOLS = ["Bitcoin", "Ethereum", "P2P-Gossip", "Libp2p", "BitTorrent-DHT", "Lightning", "Monero-P2P"]

# Realistic Mock GeoIP and ASN database
MOCK_ASN_DATABASE: Dict[str, Dict[str, Any]] = {
    "AS15169": {"org": "Google LLC", "country": "US", "city": "Mountain View", "tier": "LOW_RISK", "is_tor": False, "is_bulletproof": False},
    "AS16509": {"org": "Amazon.com, Inc.", "country": "US", "city": "Seattle", "tier": "LOW_RISK", "is_tor": False, "is_bulletproof": False},
    "AS13335": {"org": "Cloudflare, Inc.", "country": "US", "city": "San Francisco", "tier": "LOW_RISK", "is_tor": False, "is_bulletproof": False},
    "AS32934": {"org": "Meta Platforms, Inc.", "country": "US", "city": "Menlo Park", "tier": "LOW_RISK", "is_tor": False, "is_bulletproof": False},
    "AS24940": {"org": "Hetzner Online GmbH", "country": "DE", "city": "Gunzenhausen", "tier": "MEDIUM_RISK", "is_tor": False, "is_bulletproof": False},
    "AS16276": {"org": "OVH SAS", "country": "FR", "city": "Roubaix", "tier": "MEDIUM_RISK", "is_tor": False, "is_bulletproof": False},
    "AS9009":  {"org": "M247 Ltd (Bulletproof Host)", "country": "RO", "city": "Bucharest", "tier": "CRITICAL_RISK", "is_tor": False, "is_bulletproof": True},
    "AS60781": {"org": "LeaseWeb Netherlands (Proxy Hub)", "country": "NL", "city": "Amsterdam", "tier": "HIGH_RISK", "is_tor": True, "is_bulletproof": False},
    "AS200052": {"org": "Stark Industries VPN / Darknet", "country": "SC", "city": "Victoria", "tier": "CRITICAL_RISK", "is_tor": True, "is_bulletproof": True},
    "AS206980": {"org": "Flokinet Iceland Offshore", "country": "IS", "city": "Reykjavik", "tier": "HIGH_RISK", "is_tor": False, "is_bulletproof": True},
    "AS4837":  {"org": "China Unicom Beijing", "country": "CN", "city": "Beijing", "tier": "MEDIUM_RISK", "is_tor": False, "is_bulletproof": False},
    "AS209":   {"org": "CenturyLink Communications", "country": "US", "city": "Monroe", "tier": "LOW_RISK", "is_tor": False, "is_bulletproof": False},
    "AS5607":  {"org": "Sky UK Limited Residential", "country": "GB", "city": "London", "tier": "LOW_RISK", "is_tor": False, "is_bulletproof": False},
    "AS7922":  {"org": "Comcast Cable Communications", "country": "US", "city": "Philadelphia", "tier": "LOW_RISK", "is_tor": False, "is_bulletproof": False},
}

NORMAL_ASNS = ["AS15169", "AS16509", "AS13335", "AS32934", "AS24940", "AS16276", "AS209", "AS5607", "AS7922"]
SUSPICIOUS_ASNS = ["AS9009", "AS60781", "AS200052", "AS206980"]

def generate_txid(seed_str: str) -> str:
    """Generate deterministic 64-char hex transaction hash."""
    return "0x" + hashlib.sha256(seed_str.encode("utf-8")).hexdigest()

def generate_wallet_address(prefix: str = "0x") -> str:
    """Generate realistic cryptocurrency wallet address (ETH / BTC style)."""
    raw_hex = hashlib.sha256(f"{fake.uuid4()}-{random.random()}".encode("utf-8")).hexdigest()[:40]
    return f"{prefix}{raw_hex}"

def generate_ipv4_for_asn(asn: str) -> str:
    """Generate an IPv4 subnet mapped consistently to ASN for realistic network clustering."""
    asn_hash = int(hashlib.md5(asn.encode("utf-8")).hexdigest()[:4], 16) % 200 + 20
    b = (asn_hash * 7) % 250 + 1
    c = random.randint(1, 254)
    d = random.randint(1, 254)
    return f"{asn_hash}.{b}.{c}.{d}"

def generate_dataset(total_records: int = 1500) -> List[Dict[str, Any]]:
    """
    Generate synthetic transactions with realistic normal flows and specific forensic anomaly typologies.
    """
    transactions: List[Dict[str, Any]] = []
    base_time = datetime.now(timezone.utc) - timedelta(days=3)
    current_time = base_time

    # Wallet pools
    normal_wallets = [generate_wallet_address() for _ in range(80)]
    cluster_entity_a = [generate_wallet_address() for _ in range(6)]   # Entity Alpha
    cluster_entity_b = [generate_wallet_address() for _ in range(5)]   # Entity Beta
    mixer_pool_wallets = [generate_wallet_address() for _ in range(12)]

    tx_counter = 0

    # 1. Normal Retail & P2P Transactions (~70%)
    normal_count = int(total_records * 0.70)
    for _ in range(normal_count):
        tx_counter += 1
        current_time += timedelta(seconds=random.randint(15, 300))
        sender = random.choice(normal_wallets)
        receiver = random.choice(normal_wallets)
        while receiver == sender:
            receiver = generate_wallet_address()

        src_asn = random.choice(NORMAL_ASNS)
        dst_asn = random.choice(NORMAL_ASNS)
        src_ip = generate_ipv4_for_asn(src_asn)
        dst_ip = generate_ipv4_for_asn(dst_asn)

        amount = round(random.uniform(0.01, 15.5), 4)
        fee = round(amount * random.uniform(0.0005, 0.003), 6)
        protocol = random.choice(PROTOCOLS)
        payload_size = random.randint(220, 850)

        tx_id = generate_txid(f"norm-{tx_counter}-{current_time.isoformat()}")
        transactions.append({
            "tx_id": tx_id,
            "timestamp": current_time.isoformat(),
            "sender_address": sender,
            "receiver_address": receiver,
            "amount": amount,
            "fee": fee,
            "network_protocol": protocol,
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "src_asn": src_asn,
            "dst_asn": dst_asn,
            "payload_size_bytes": payload_size,
            "pattern_label": "NORMAL_TRANSFER"
        })

    # 2. Multi-Input Co-Spending Transactions (for Common-Input-Ownership Heuristic)
    # Transactions where multiple sender wallets from Entity A or Entity B co-spend
    for cluster_name, cluster_wallets in [("Entity-Alpha", cluster_entity_a), ("Entity-Beta", cluster_entity_b)]:
        for _ in range(12):
            tx_counter += 1
            current_time += timedelta(seconds=random.randint(40, 600))
            # Pick 2 or 3 wallets from the cluster as co-spenders
            co_spenders = random.sample(cluster_wallets, k=random.randint(2, min(3, len(cluster_wallets))))
            sender_str = ";".join(co_spenders)
            receiver = random.choice(normal_wallets)
            src_asn = random.choice(NORMAL_ASNS)
            dst_asn = random.choice(NORMAL_ASNS)
            src_ip = generate_ipv4_for_asn(src_asn)
            dst_ip = generate_ipv4_for_asn(dst_asn)
            amount = round(random.uniform(2.5, 45.0), 4)
            fee = round(amount * 0.0012, 6)

            tx_id = generate_txid(f"co-spend-{tx_counter}-{current_time.isoformat()}")
            transactions.append({
                "tx_id": tx_id,
                "timestamp": current_time.isoformat(),
                "sender_address": sender_str,
                "receiver_address": receiver,
                "amount": amount,
                "fee": fee,
                "network_protocol": "Bitcoin",
                "src_ip": src_ip,
                "dst_ip": dst_ip,
                "src_asn": src_asn,
                "dst_asn": dst_asn,
                "payload_size_bytes": random.randint(450, 950),
                "pattern_label": f"CO_SPEND_{cluster_name.upper()}"
            })

    # 3. Peel Chain Anomaly Pattern (~7%)
    # An initial large sum is peeled off sequentially to different cash-out addresses
    peel_chains_count = 5
    for chain_idx in range(peel_chains_count):
        current_wallet = generate_wallet_address()
        remaining_balance = round(random.uniform(40.0, 100.0), 2)
        chain_ip = generate_ipv4_for_asn(random.choice(SUSPICIOUS_ASNS))
        chain_asn = random.choice(SUSPICIOUS_ASNS)

        for step in range(random.randint(5, 9)):
            tx_counter += 1
            # Rapid succession: 3 to 12 seconds
            current_time += timedelta(seconds=random.randint(3, 12))
            peeled_amount = round(random.uniform(0.75, 2.5), 3)
            if remaining_balance <= peeled_amount:
                break
            remaining_balance = round(remaining_balance - peeled_amount - 0.005, 3)
            peel_destination = generate_wallet_address()
            new_change_wallet = generate_wallet_address()

            # Main peel output
            tx_id = generate_txid(f"peel-{chain_idx}-{step}-{tx_counter}")
            transactions.append({
                "tx_id": tx_id,
                "timestamp": current_time.isoformat(),
                "sender_address": current_wallet,
                "receiver_address": f"{peel_destination};{new_change_wallet}",
                "amount": peeled_amount,
                "fee": 0.005,
                "network_protocol": "Bitcoin",
                "src_ip": chain_ip,
                "dst_ip": generate_ipv4_for_asn("AS24940"),
                "src_asn": chain_asn,
                "dst_asn": "AS24940",
                "payload_size_bytes": random.randint(380, 520),
                "pattern_label": "PEEL_CHAIN_ANOMALY"
            })
            current_wallet = new_change_wallet

    # 4. Smurfing / Structuring Rings (~8%)
    # Multiple discrete wallets deposit round sums into a consolidation wallet, which fans out
    smurf_rings = 4
    for ring_idx in range(smurf_rings):
        collector_wallet = generate_wallet_address()
        collector_ip = generate_ipv4_for_asn("AS9009")
        collector_asn = "AS9009"
        smurf_senders = [generate_wallet_address() for _ in range(8)]

        # Fan-in stage: round structured amounts
        for smurf in smurf_senders:
            tx_counter += 1
            current_time += timedelta(seconds=random.randint(5, 25))
            round_amt = random.choice([5.0, 9.5, 10.0, 15.0, 20.0])
            s_ip = generate_ipv4_for_asn("AS60781")

            tx_id = generate_txid(f"smurf-in-{ring_idx}-{tx_counter}")
            transactions.append({
                "tx_id": tx_id,
                "timestamp": current_time.isoformat(),
                "sender_address": smurf,
                "receiver_address": collector_wallet,
                "amount": round_amt,
                "fee": 0.002,
                "network_protocol": "Ethereum",
                "src_ip": s_ip,
                "dst_ip": collector_ip,
                "src_asn": "AS60781",
                "dst_asn": collector_asn,
                "payload_size_bytes": random.randint(300, 480),
                "pattern_label": "SMURFING_FAN_IN"
            })

        # Fan-out stage: collector distributes to multiple mixer wallets
        for mixer_dest in random.sample(mixer_pool_wallets, k=4):
            tx_counter += 1
            current_time += timedelta(seconds=random.randint(10, 40))
            fanout_amt = round(random.uniform(15.0, 25.0), 2)
            tx_id = generate_txid(f"smurf-out-{ring_idx}-{tx_counter}")
            transactions.append({
                "tx_id": tx_id,
                "timestamp": current_time.isoformat(),
                "sender_address": collector_wallet,
                "receiver_address": mixer_dest,
                "amount": fanout_amt,
                "fee": 0.015,
                "network_protocol": "Ethereum",
                "src_ip": collector_ip,
                "dst_ip": generate_ipv4_for_asn("AS200052"),
                "src_asn": collector_asn,
                "dst_asn": "AS200052",
                "payload_size_bytes": random.randint(400, 650),
                "pattern_label": "SMURFING_FAN_OUT"
            })

    # 5. High-Velocity Automated Draining / Bot Sweeps (~5%)
    drain_events = 3
    for drain_idx in range(drain_events):
        compromised_wallet = generate_wallet_address()
        drainer_dest = generate_wallet_address()
        bot_ip = generate_ipv4_for_asn("AS200052")
        bot_asn = "AS200052"

        for burst_i in range(12):
            tx_counter += 1
            # Sub-second to 2-second velocity
            current_time += timedelta(milliseconds=random.randint(200, 1800))
            sweep_amount = round(random.uniform(1.2, 8.8), 4)
            high_fee = round(sweep_amount * 0.08, 4)  # High fee bribery / priority gas

            tx_id = generate_txid(f"drain-{drain_idx}-{burst_i}-{tx_counter}")
            transactions.append({
                "tx_id": tx_id,
                "timestamp": current_time.isoformat(),
                "sender_address": compromised_wallet,
                "receiver_address": drainer_dest,
                "amount": sweep_amount,
                "fee": high_fee,
                "network_protocol": "P2P-Gossip",
                "src_ip": bot_ip,
                "dst_ip": generate_ipv4_for_asn("AS206980"),
                "src_asn": bot_asn,
                "dst_asn": "AS206980",
                "payload_size_bytes": random.randint(1200, 4500), # Suspicious payload size
                "pattern_label": "HIGH_VELOCITY_DRAIN"
            })

    # 6. High-Risk ASN & Bulletproof / Tor Exit Node Broadcasts (~5%)
    for _ in range(45):
        tx_counter += 1
        current_time += timedelta(seconds=random.randint(20, 200))
        suspicious_asn = random.choice(SUSPICIOUS_ASNS)
        s_ip = generate_ipv4_for_asn(suspicious_asn)
        dest_ip = generate_ipv4_for_asn(random.choice(NORMAL_ASNS))
        amt = round(random.uniform(0.5, 30.0), 3)

        tx_id = generate_txid(f"tor-asn-{tx_counter}-{current_time.isoformat()}")
        transactions.append({
            "tx_id": tx_id,
            "timestamp": current_time.isoformat(),
            "sender_address": generate_wallet_address(),
            "receiver_address": generate_wallet_address(),
            "amount": amt,
            "fee": round(amt * 0.0025, 5),
            "network_protocol": random.choice(["Monero-P2P", "Libp2p", "BitTorrent-DHT"]),
            "src_ip": s_ip,
            "dst_ip": dest_ip,
            "src_asn": suspicious_asn,
            "dst_asn": random.choice(NORMAL_ASNS),
            "payload_size_bytes": random.randint(1800, 6000),
            "pattern_label": "BULLETPROOF_HOST_BROADCAST"
        })

    # Shuffle chronologically
    transactions.sort(key=lambda x: x["timestamp"])
    return transactions

def export_synthetic_data():
    """Generate and export data to CSV and JSON formats, along with mock GeoIP database."""
    print("[-] Generating realistic synthetic crypto & P2P transactions...")
    records = generate_dataset(total_records=1600)

    csv_path = DATA_DIR / "transactions.csv"
    json_path = DATA_DIR / "transactions.json"
    geoip_path = DATA_DIR / "geoip_asn_mock.json"

    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)
    print(f"[+] Saved {len(df)} transactions to {csv_path}")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"[+] Saved JSON records to {json_path}")

    with open(geoip_path, "w", encoding="utf-8") as f:
        json.dump(MOCK_ASN_DATABASE, f, indent=2)
    print(f"[+] Saved Mock GeoIP / ASN Database to {geoip_path}")

    # Summary statistics
    print("\n=== Synthetic Dataset Profile ===")
    print(df["pattern_label"].value_counts())
    print("=================================\n")

if __name__ == "__main__":
    export_synthetic_data()

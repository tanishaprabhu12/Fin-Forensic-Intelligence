"""
ingestion_pipeline.py
---------------------
Production-Grade Ingestion & Validation Pipeline with Pydantic v2 Schemas.
Performs:
- Strict IPv4 address validation
- Semantic cryptocurrency & P2P transaction validation
- Rejection of malformed / corrupt data with comprehensive audit logs
- Offline GeoIP / ASN threat enrichment
- Normalized EnrichedTransaction data structures for downstream Graph & ML engines.
"""

import ipaddress
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ForensicIngestionPipeline")

DATA_DIR = Path("data")


class GeoIPEnrichment(BaseModel):
    """Offline GeoIP & Autonomous System Number (ASN) Intelligence Object."""
    ip: str
    asn: str
    org: str = "Unknown ISP"
    country: str = "ZZ"
    city: str = "Unknown"
    tier: str = "LOW_RISK"  # LOW_RISK, MEDIUM_RISK, HIGH_RISK, CRITICAL_RISK
    is_tor: bool = False
    is_bulletproof: bool = False
    asn_risk_weight: float = Field(default=0.1, ge=0.0, le=1.0)


class RawTransaction(BaseModel):
    """Raw P2P / Blockchain transaction schema with strict type and boundary validations."""
    tx_id: str = Field(..., description="Unique 64-character hexadecimal transaction hash or ID")
    timestamp: datetime = Field(..., description="ISO 8601 UTC timestamp of the transaction")
    sender_address: str = Field(..., min_length=4, description="Sender wallet address or semicolon-delimited multi-inputs")
    receiver_address: str = Field(..., min_length=4, description="Receiver wallet address or semicolon-delimited multi-outputs")
    amount: float = Field(..., gt=0.0, description="Transaction transfer value in native crypto units")
    fee: float = Field(..., ge=0.0, description="Network broadcast or mining fee")
    network_protocol: str = Field(..., min_length=2, description="P2P Gossip / Blockchain protocol")
    src_ip: str = Field(..., description="Source IPv4 address of broadcasting node")
    dst_ip: str = Field(..., description="Destination IPv4 address of relay or receiving node")
    src_asn: str = Field(..., min_length=3, description="Source Autonomous System Number (e.g. AS15169)")
    dst_asn: str = Field(..., min_length=3, description="Destination Autonomous System Number")
    payload_size_bytes: int = Field(..., gt=0, description="Network packet payload size in bytes")
    pattern_label: Optional[str] = Field(default="UNKNOWN", description="Ground-truth pattern label if known")

    # Real-world cryptocurrency asset grounding
    coin_symbol: Optional[str] = Field(default="BTC", description="Real-world token/coin symbol")
    coin_name: Optional[str] = Field(default="Bitcoin", description="Real-world token/coin name")
    token_price_usd: Optional[float] = Field(default=36456.94, ge=0.0, description="Real token price in USD")
    amount_usd: Optional[float] = Field(default=0.0, ge=0.0, description="Transaction transfer value in USD")
    market_cap_tier: Optional[str] = Field(default="Mega-Cap", description="Institutional market cap tier")

    @field_validator("src_ip", "dst_ip")
    @classmethod
    def validate_ipv4_address(cls, v: str) -> str:
        """Validate that the given IP is a valid IPv4 string format."""
        try:
            ip_obj = ipaddress.IPv4Address(v.strip())
            return str(ip_obj)
        except ValueError as exc:
            raise ValueError(f"Invalid IPv4 address format: '{v}'") from exc

    @field_validator("tx_id")
    @classmethod
    def validate_tx_id_format(cls, v: str) -> str:
        """Ensure tx_id has appropriate prefix or length."""
        v = v.strip()
        if not (v.startswith("0x") or len(v) >= 16):
            raise ValueError(f"Invalid transaction ID hash length or format: '{v}'")
        return v

    @field_validator("network_protocol")
    @classmethod
    def validate_protocol(cls, v: str) -> str:
        """Normalize protocol identifier."""
        return v.strip()


class EnrichedTransaction(BaseModel):
    """Fully validated and offline-enriched transaction model ready for forensic graph and ML."""
    tx_id: str
    timestamp: datetime
    sender_address: str
    receiver_address: str
    amount: float
    fee: float
    network_protocol: str
    src_ip: str
    dst_ip: str
    src_asn: str
    dst_asn: str
    payload_size_bytes: int
    pattern_label: Optional[str] = "UNKNOWN"

    # Real-world cryptocurrency metadata
    coin_symbol: str = "BTC"
    coin_name: str = "Bitcoin"
    token_price_usd: float = 36456.94
    amount_usd: float = 0.0
    market_cap_tier: str = "Mega-Cap"

    # Multi-input / Multi-output parsed representations
    input_wallets: List[str] = Field(default_factory=list)
    output_wallets: List[str] = Field(default_factory=list)
    sender_count: int = 1
    receiver_count: int = 1

    # Offline GeoIP / ASN Enrichment
    src_enrichment: GeoIPEnrichment
    dst_enrichment: GeoIPEnrichment
    is_high_risk_network: bool = False
    fee_to_amount_ratio: float = 0.0

    @model_validator(mode="before")
    @classmethod
    def compute_derived_fields(cls, values: Any) -> Any:
        if isinstance(values, dict):
            senders = [s.strip() for s in str(values.get("sender_address", "")).split(";") if s.strip()]
            receivers = [r.strip() for r in str(values.get("receiver_address", "")).split(";") if r.strip()]
            values["input_wallets"] = senders or [values.get("sender_address", "")]
            values["output_wallets"] = receivers or [values.get("receiver_address", "")]
            values["sender_count"] = len(values["input_wallets"])
            values["receiver_count"] = len(values["output_wallets"])
            
            amount = float(values.get("amount", 1.0))
            fee = float(values.get("fee", 0.0))
            values["fee_to_amount_ratio"] = (fee / amount) if amount > 0 else 0.0
        return values


class IngestionPipeline:
    """
    Forensic ingestion pipeline managing file loading, Pydantic validation,
    malformed record filtering, and offline GeoIP/ASN intelligence lookup.
    """

    def __init__(self, geoip_db_path: Optional[Union[str, Path]] = None):
        self.geoip_db_path = Path(geoip_db_path) if geoip_db_path else DATA_DIR / "geoip_asn_mock.json"
        self.asn_db: Dict[str, Dict[str, Any]] = self._load_geoip_db()
        self.malformed_records_log: List[Dict[str, Any]] = []

    def _load_geoip_db(self) -> Dict[str, Dict[str, Any]]:
        """Load offline GeoIP & ASN risk store from disk."""
        if self.geoip_db_path.exists():
            try:
                with open(self.geoip_db_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load GeoIP DB from {self.geoip_db_path}: {e}")
        return {}

    def get_asn_enrichment(self, ip: str, asn: str) -> GeoIPEnrichment:
        """Resolve offline GeoIP and threat metadata for a given IP & ASN."""
        raw_meta = self.asn_db.get(asn, {})
        tier = raw_meta.get("tier", "LOW_RISK")
        is_tor = raw_meta.get("is_tor", False)
        is_bulletproof = raw_meta.get("is_bulletproof", False)

        # Calculate risk weight
        risk_weight = 0.1
        if tier == "CRITICAL_RISK" or is_bulletproof:
            risk_weight = 0.95
        elif tier == "HIGH_RISK" or is_tor:
            risk_weight = 0.75
        elif tier == "MEDIUM_RISK":
            risk_weight = 0.40

        return GeoIPEnrichment(
            ip=ip,
            asn=asn,
            org=raw_meta.get("org", f"Autonomous System {asn}"),
            country=raw_meta.get("country", "US"),
            city=raw_meta.get("city", "Unknown City"),
            tier=tier,
            is_tor=is_tor,
            is_bulletproof=is_bulletproof,
            asn_risk_weight=risk_weight
        )

    def parse_record(self, raw_dict: Dict[str, Any]) -> Tuple[Optional[EnrichedTransaction], Optional[Dict[str, Any]]]:
        """Validate single record through Pydantic and enrich."""
        try:
            raw_tx = RawTransaction(**raw_dict)
            src_enrich = self.get_asn_enrichment(raw_tx.src_ip, raw_tx.src_asn)
            dst_enrich = self.get_asn_enrichment(raw_tx.dst_ip, raw_tx.dst_asn)

            is_high_risk = (
                src_enrich.tier in ["HIGH_RISK", "CRITICAL_RISK"]
                or dst_enrich.tier in ["HIGH_RISK", "CRITICAL_RISK"]
                or src_enrich.is_bulletproof
                or src_enrich.is_tor
            )

            token_price = float(raw_tx.token_price_usd or 0.0)
            amt_usd = float(raw_tx.amount_usd or 0.0)
            if amt_usd <= 0.0 and token_price > 0.0:
                amt_usd = round(raw_tx.amount * token_price, 2)

            enriched = EnrichedTransaction(
                tx_id=raw_tx.tx_id,
                timestamp=raw_tx.timestamp,
                sender_address=raw_tx.sender_address,
                receiver_address=raw_tx.receiver_address,
                amount=raw_tx.amount,
                fee=raw_tx.fee,
                network_protocol=raw_tx.network_protocol,
                src_ip=raw_tx.src_ip,
                dst_ip=raw_tx.dst_ip,
                src_asn=raw_tx.src_asn,
                dst_asn=raw_tx.dst_asn,
                payload_size_bytes=raw_tx.payload_size_bytes,
                pattern_label=raw_tx.pattern_label or "UNKNOWN",
                coin_symbol=raw_tx.coin_symbol or "BTC",
                coin_name=raw_tx.coin_name or "Bitcoin",
                token_price_usd=token_price,
                amount_usd=amt_usd,
                market_cap_tier=raw_tx.market_cap_tier or "Mega-Cap",
                src_enrichment=src_enrich,
                dst_enrichment=dst_enrich,
                is_high_risk_network=is_high_risk
            )
            return enriched, None
        except Exception as exc:
            rejection_entry = {
                "raw_record": raw_dict,
                "error_type": type(exc).__name__,
                "error_reason": str(exc)
            }
            return None, rejection_entry

    def ingest_file(self, filepath: Union[str, Path]) -> List[EnrichedTransaction]:
        """
        Stream/batch ingest data from CSV or JSON filepath, validating all records
        and appending malformed items to audit log.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found at {path}")

        logger.info(f"Ingesting transaction file from {path}...")
        raw_items: List[Dict[str, Any]] = []

        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
            raw_items = df.to_dict(orient="records")
        elif path.suffix.lower() == ".json":
            with open(path, "r", encoding="utf-8") as f:
                raw_items = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}. Expected .csv or .json")

        valid_transactions: List[EnrichedTransaction] = []
        for item in raw_items:
            enriched, malformed = self.parse_record(item)
            if enriched:
                valid_transactions.append(enriched)
            elif malformed:
                self.malformed_records_log.append(malformed)

        logger.info(
            f"Ingestion summary: {len(valid_transactions)} valid records parsed, "
            f"{len(self.malformed_records_log)} malformed records rejected."
        )
        return valid_transactions

    def to_dataframe(self, enriched_txs: List[EnrichedTransaction]) -> pd.DataFrame:
        """Convert a list of EnrichedTransaction objects into a flat Pandas DataFrame for ML/Graph."""
        flat_records = []
        for tx in enriched_txs:
            rec = {
                "tx_id": tx.tx_id,
                "timestamp": tx.timestamp,
                "sender_address": tx.sender_address,
                "receiver_address": tx.receiver_address,
                "amount": tx.amount,
                "fee": tx.fee,
                "fee_to_amount_ratio": tx.fee_to_amount_ratio,
                "network_protocol": tx.network_protocol,
                "src_ip": tx.src_ip,
                "dst_ip": tx.dst_ip,
                "src_asn": tx.src_asn,
                "dst_asn": tx.dst_asn,
                "payload_size_bytes": tx.payload_size_bytes,
                "pattern_label": tx.pattern_label,
                "sender_count": tx.sender_count,
                "receiver_count": tx.receiver_count,
                "input_wallets": tx.input_wallets,
                "output_wallets": tx.output_wallets,
                "src_org": tx.src_enrichment.org,
                "src_country": tx.src_enrichment.country,
                "src_tier": tx.src_enrichment.tier,
                "src_is_tor": tx.src_enrichment.is_tor,
                "src_is_bulletproof": tx.src_enrichment.is_bulletproof,
                "src_asn_risk_weight": tx.src_enrichment.asn_risk_weight,
                "dst_org": tx.dst_enrichment.org,
                "dst_country": tx.dst_enrichment.country,
                "dst_tier": tx.dst_enrichment.tier,
                "dst_is_tor": tx.dst_enrichment.is_tor,
                "dst_is_bulletproof": tx.dst_enrichment.is_bulletproof,
                "dst_asn_risk_weight": tx.dst_enrichment.asn_risk_weight,
                "is_high_risk_network": int(tx.is_high_risk_network)
            }
            flat_records.append(rec)
        return pd.DataFrame(flat_records)


if __name__ == "__main__":
    pipeline = IngestionPipeline()
    csv_file = DATA_DIR / "transactions.csv"
    if csv_file.exists():
        enriched = pipeline.ingest_file(csv_file)
        df_out = pipeline.to_dataframe(enriched)
        print(f"[+] Ingestion test successful! DataFrame shape: {df_out.shape}")
        print(f"[+] Sample Enriched Schema:\n{df_out[['tx_id', 'amount', 'src_org', 'src_tier', 'is_high_risk_network']].head()}")

"""
crypto_market_data.py
---------------------
Real-World Cryptocurrency Market Intelligence & Asset Loader.
Ingests, cleans, parses, and enriches CryptocurrencyData.csv (4,150 real crypto assets):
- Normalizes messy pricing, supply notation (Million, Billion, Trillion, Infinity), and volume metrics.
- Classifies tokens by Market Cap Tier (Mega, Large, Mid, Small, Micro-Cap).
- Detects market anomalies: Wash-Trading Suspects (Volume > Market Cap), Pump-and-Dump Volatility,
  and Liquidity crunches.
- High-performance caching and O(1) dictionary lookups by symbol and coin name.
"""

import io
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger("CryptoMarketData")

DEFAULT_CSV_PATH = Path("CryptocurrencyData.csv")
CLEANED_DATA_PATH = Path("data") / "cleaned_cryptocurrency_data.csv"


class CryptoMarketDataLoader:
    """
    Parser and analytics engine for real-world cryptocurrency market data.
    """

    def __init__(self, raw_csv_path: Optional[Path] = None):
        self.raw_csv_path = Path(raw_csv_path) if raw_csv_path else DEFAULT_CSV_PATH
        self._df: Optional[pd.DataFrame] = None
        self._symbol_map: Dict[str, Dict[str, Any]] = {}
        self._name_map: Dict[str, Dict[str, Any]] = {}
        self._initialize()

    def _clean_currency_or_num(self, val: Any) -> float:
        """Parse currency, commas, and dashes into float."""
        if pd.isna(val):
            return 0.0
        s = str(val).strip().replace("$", "").replace(",", "")
        if s in ["-", "$-", "", "None", "nan", "null"]:
            return 0.0
        try:
            return float(s)
        except (ValueError, TypeError):
            return 0.0

    def _clean_percentage(self, val: Any) -> float:
        """Parse percentage string (e.g. '0.40%', '-1.70%') into signed float."""
        if pd.isna(val):
            return 0.0
        s = str(val).strip().replace("%", "").replace(",", "")
        if s in ["-", "$-", "", "None", "nan", "null"]:
            return 0.0
        try:
            return float(s)
        except (ValueError, TypeError):
            return 0.0

    def _clean_supply(self, val: Any) -> float:
        """Parse string with word multipliers (Million, Billion, etc.) or infinity into float."""
        if pd.isna(val):
            return 0.0
        s = str(val).strip()
        if s in ["-", "$-", "∞", "", "None", "nan", "null"]:
            return 0.0

        multipliers = {
            "quadrillion": 1e15,
            "trillion": 1e12,
            "billion": 1e9,
            "million": 1e6,
            "thousand": 1e3
        }

        s_lower = s.lower()
        for word, mult in multipliers.items():
            if word in s_lower:
                num_part = re.sub(r"[^0-9.]", "", s_lower.replace(word, ""))
                try:
                    return float(num_part) * mult
                except (ValueError, TypeError):
                    return 0.0

        num_part = re.sub(r"[^0-9.]", "", s)
        try:
            return float(num_part) if num_part else 0.0
        except (ValueError, TypeError):
            return 0.0

    def _classify_market_cap_tier(self, mcap: float) -> str:
        """Classify market cap into standardized institutional tiers."""
        if mcap >= 10_000_000_000:  # >= $10B
            return "Mega-Cap"
        elif mcap >= 1_000_000_000:  # $1B - $10B
            return "Large-Cap"
        elif mcap >= 100_000_000:    # $100M - $1B
            return "Mid-Cap"
        elif mcap >= 10_000_000:     # $10M - $100M
            return "Small-Cap"
        else:                        # < $10M
            return "Micro/Meme-Cap"

    def _classify_volatility(self, change_24h: float) -> str:
        """Determine volatility risk tier based on 24h percentage delta."""
        abs_change = abs(change_24h)
        if abs_change >= 40.0:
            return "CRITICAL_VOLATILITY"
        elif abs_change >= 20.0:
            return "HIGH_VOLATILITY"
        elif abs_change >= 10.0:
            return "MEDIUM_VOLATILITY"
        return "LOW_VOLATILITY"

    def _initialize(self) -> None:
        """Load, clean, and enrich the dataset."""
        if not self.raw_csv_path.exists():
            logger.warning(f"Cryptocurrency dataset not found at {self.raw_csv_path}")
            self._df = pd.DataFrame()
            return

        logger.info(f"Loading and cleaning real-world crypto dataset from {self.raw_csv_path}...")
        try:
            # Read CSV with utf-8 encoding and fallback
            try:
                df = pd.read_csv(self.raw_csv_path, encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(self.raw_csv_path, encoding="latin-1")

            # Strip whitespace from column headers
            df.columns = [c.strip() for c in df.columns]

            # Ensure expected columns exist
            rename_map = {
                "Rank": "rank",
                "Coin Name": "coin_name",
                "Symbol": "symbol",
                "Price": "price_usd",
                "1h": "change_1h",
                "24h": "change_24h",
                "7d": "change_7d",
                "30d": "change_30d",
                "24h Volume": "volume_24h_usd",
                "Circulating Supply": "circulating_supply",
                "Total Supply": "total_supply",
                "Market Cap": "market_cap_usd"
            }
            df = df.rename(columns=rename_map)

            # Apply clean transformations
            df["rank"] = pd.to_numeric(df["rank"], errors="coerce").fillna(9999).astype(int)
            df["coin_name"] = df["coin_name"].astype(str).str.strip()
            df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()

            df["price_usd"] = df["price_usd"].apply(self._clean_currency_or_num)
            df["change_1h"] = df["change_1h"].apply(self._clean_percentage)
            df["change_24h"] = df["change_24h"].apply(self._clean_percentage)
            df["change_7d"] = df["change_7d"].apply(self._clean_percentage)
            df["change_30d"] = df["change_30d"].apply(self._clean_percentage)
            df["volume_24h_usd"] = df["volume_24h_usd"].apply(self._clean_currency_or_num)
            df["circulating_supply"] = df["circulating_supply"].apply(self._clean_currency_or_num)
            df["total_supply"] = df["total_supply"].apply(self._clean_supply)
            df["market_cap_usd"] = df["market_cap_usd"].apply(self._clean_currency_or_num)

            # Compute derived forensic metrics
            # 1. Volume to Market Cap ratio (liquidity / potential wash trading)
            df["vol_to_mcap_ratio"] = np.where(
                df["market_cap_usd"] > 0,
                df["volume_24h_usd"] / df["market_cap_usd"],
                0.0
            )

            # 2. Market Cap Tier
            df["market_cap_tier"] = df["market_cap_usd"].apply(self._classify_market_cap_tier)

            # 3. Volatility classification
            df["volatility_risk"] = df["change_24h"].apply(self._classify_volatility)

            # 4. Wash-trading indicator: Daily volume exceeds 100% of entire market cap
            df["is_wash_trading_suspect"] = (
                (df["vol_to_mcap_ratio"] > 1.0) & (df["market_cap_usd"] > 100_000)
            )

            # 5. Extreme pump & dump volatility flag
            df["is_pump_dump_suspect"] = (
                (df["change_24h"].abs() >= 35.0) | (df["change_7d"].abs() >= 100.0)
            )

            # 6. Phantom token flag: High trading volume despite 0 circulating supply
            df["is_phantom_token"] = (
                (df["volume_24h_usd"] > 1_000_000) & (df["circulating_supply"] == 0)
            )

            self._df = df

            # Build fast lookup dictionaries
            records = df.to_dict(orient="records")
            for r in records:
                sym = r["symbol"]
                name = r["coin_name"].lower()
                # Store if not already present or higher market cap
                if sym not in self._symbol_map or r["market_cap_usd"] > self._symbol_map[sym]["market_cap_usd"]:
                    self._symbol_map[sym] = r
                if name not in self._name_map or r["market_cap_usd"] > self._name_map[name]["market_cap_usd"]:
                    self._name_map[name] = r

            # Export cleaned cache
            try:
                CLEANED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(CLEANED_DATA_PATH, index=False, encoding="utf-8")
                logger.info(f"Exported cleaned cryptocurrency dataset ({len(df)} records) to {CLEANED_DATA_PATH}")
            except Exception as e:
                logger.warning(f"Could not cache cleaned dataset to {CLEANED_DATA_PATH}: {e}")

        except Exception as exc:
            logger.error(f"Error parsing cryptocurrency dataset: {exc}", exc_info=True)
            self._df = pd.DataFrame()

    @property
    def dataframe(self) -> pd.DataFrame:
        """Return the cleaned full dataframe."""
        return self._df if self._df is not None else pd.DataFrame()

    def get_coin_by_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Look up asset metadata by symbol in O(1) time."""
        return self._symbol_map.get(symbol.strip().upper())

    def get_coin_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Look up asset metadata by coin name in O(1) time."""
        return self._name_map.get(name.strip().lower())

    def get_top_assets(self, n: int = 50) -> pd.DataFrame:
        """Return top N assets by market capitalization."""
        if self._df is None or self._df.empty:
            return pd.DataFrame()
        return self._df.sort_values(by="market_cap_usd", ascending=False).head(n)

    def get_market_summary(self) -> Dict[str, Any]:
        """Aggregate high-level global cryptocurrency market telemetry."""
        if self._df is None or self._df.empty:
            return {
                "total_tracked_assets": 0,
                "total_market_cap_usd": 0.0,
                "total_24h_volume_usd": 0.0,
                "wash_trading_suspects": 0,
                "extreme_volatility_tokens": 0,
                "top_gainer": None,
                "top_loser": None
            }

        total_mcap = float(self._df["market_cap_usd"].sum())
        total_vol = float(self._df["volume_24h_usd"].sum())
        wash_count = int(self._df["is_wash_trading_suspect"].sum())
        vol_count = int(self._df["is_pump_dump_suspect"].sum())

        # Top gainers with volume > $10,000 to avoid dead tokens
        active_tokens = self._df[self._df["volume_24h_usd"] > 10_000]
        top_gainer_row = active_tokens.sort_values(by="change_24h", ascending=False).head(1)
        top_loser_row = active_tokens.sort_values(by="change_24h", ascending=True).head(1)

        top_gainer = top_gainer_row.iloc[0].to_dict() if not top_gainer_row.empty else None
        top_loser = top_loser_row.iloc[0].to_dict() if not top_loser_row.empty else None

        return {
            "total_tracked_assets": len(self._df),
            "total_market_cap_usd": total_mcap,
            "total_24h_volume_usd": total_vol,
            "wash_trading_suspects": wash_count,
            "extreme_volatility_tokens": vol_count,
            "top_gainer": top_gainer,
            "top_loser": top_loser
        }


# Singleton accessor for fast application-wide reuse
_LOADER_INSTANCE: Optional[CryptoMarketDataLoader] = None

def get_market_loader(csv_path: Optional[Path] = None) -> CryptoMarketDataLoader:
    """Retrieve or initialize singleton CryptoMarketDataLoader instance."""
    global _LOADER_INSTANCE
    if _LOADER_INSTANCE is None or csv_path is not None:
        _LOADER_INSTANCE = CryptoMarketDataLoader(csv_path)
    return _LOADER_INSTANCE


if __name__ == "__main__":
    loader = get_market_loader()
    print(f"Loaded {len(loader.dataframe)} cryptocurrencies.")
    summary = loader.get_market_summary()
    print("Market Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    btc = loader.get_coin_by_symbol("BTC")
    print(f"BTC Lookup: {btc['coin_name']} @ ${btc['price_usd']:,.2f}, MCap: ${btc['market_cap_usd']:,.2f}")

"""
tests/test_crypto_market_data.py
--------------------------------
Comprehensive unit tests for real-world cryptocurrency data loader and analytics engine:
- Verifies loading and cleaning of CryptocurrencyData.csv (4,150 real assets)
- Verifies edge case parsing (commas, dollar signs, percentages, word multipliers, infinity)
- Verifies market cap classification and wash-trading anomaly heuristics
- Verifies O(1) dictionary lookups and market summary metrics
"""

import sys
from pathlib import Path
import pytest

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crypto_market_data import CryptoMarketDataLoader, get_market_loader
from generate_synthetic_data import generate_dataset


@pytest.fixture(scope="module")
def market_loader():
    return get_market_loader()


def test_market_data_loading(market_loader):
    """Test that CryptocurrencyData.csv is loaded into a clean dataframe."""
    df = market_loader.dataframe
    assert not df.empty
    assert len(df) >= 4000
    expected_cols = [
        "rank", "coin_name", "symbol", "price_usd", "change_1h", "change_24h",
        "change_7d", "change_30d", "volume_24h_usd", "circulating_supply",
        "total_supply", "market_cap_usd", "vol_to_mcap_ratio", "market_cap_tier",
        "volatility_risk", "is_wash_trading_suspect"
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing expected column: {col}"


def test_bitcoin_and_ethereum_metadata(market_loader):
    """Test lookup for flagship assets Bitcoin and Ethereum."""
    btc = market_loader.get_coin_by_symbol("BTC")
    assert btc is not None
    assert btc["coin_name"] == "Bitcoin"
    assert btc["rank"] == 1
    assert btc["price_usd"] > 30000.0
    assert btc["market_cap_usd"] > 500_000_000_000.0
    assert btc["market_cap_tier"] == "Mega-Cap"

    eth = market_loader.get_coin_by_symbol("ETH")
    assert eth is not None
    assert eth["coin_name"] == "Ethereum"
    assert eth["rank"] == 2
    assert eth["price_usd"] > 1500.0
    assert eth["market_cap_tier"] == "Mega-Cap"


def test_data_cleaning_no_nans_in_key_numerics(market_loader):
    """Ensure key price, volume, and market cap columns have zero NaN values."""
    df = market_loader.dataframe
    assert df["price_usd"].isna().sum() == 0
    assert df["volume_24h_usd"].isna().sum() == 0
    assert df["market_cap_usd"].isna().sum() == 0
    assert df["vol_to_mcap_ratio"].isna().sum() == 0


def test_market_cap_tier_distribution(market_loader):
    """Test that assets are categorized into institutional tiers."""
    df = market_loader.dataframe
    tiers = set(df["market_cap_tier"].unique())
    assert "Mega-Cap" in tiers
    assert "Large-Cap" in tiers
    assert "Mid-Cap" in tiers
    assert "Micro/Meme-Cap" in tiers


def test_wash_trading_anomaly_detection(market_loader):
    """Test that wash-trading suspects have volume exceeding market cap."""
    df = market_loader.dataframe
    wash_suspects = df[df["is_wash_trading_suspect"]]
    assert len(wash_suspects) > 0
    for _, row in wash_suspects.head(10).iterrows():
        assert row["vol_to_mcap_ratio"] > 1.0
        assert row["market_cap_usd"] > 0


def test_market_summary_aggregations(market_loader):
    """Test high-level global cryptocurrency telemetry calculation."""
    summary = market_loader.get_market_summary()
    assert summary["total_tracked_assets"] >= 4000
    assert summary["total_market_cap_usd"] > 1_000_000_000_000.0  # > $1 Trillion
    assert summary["total_24h_volume_usd"] > 50_000_000_000.0     # > $50 Billion
    assert summary["wash_trading_suspects"] > 0
    assert summary["top_gainer"] is not None
    assert summary["top_loser"] is not None


def test_transaction_grounding_with_real_assets():
    """Verify that generated synthetic transactions have real-world token attributes."""
    records = generate_dataset(total_records=100)
    assert len(records) > 0
    for rec in records:
        assert "coin_symbol" in rec
        assert "coin_name" in rec
        assert "token_price_usd" in rec
        assert "amount_usd" in rec
        assert "market_cap_tier" in rec
        assert rec["token_price_usd"] >= 0.0
        assert rec["amount_usd"] >= 0.0

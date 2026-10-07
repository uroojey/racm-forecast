"""Unit tests for the preprocessing module."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ──────────────────────────────────────────────
# Test MarketDataLoader, NewsDataLoader, etc.
# ──────────────────────────────────────────────

from preprocessing.loader import MarketDataLoader, NewsDataLoader, DataLoaderFactory
from preprocessing.alignment import TimeAligner
from preprocessing.features import TechnicalFeatureEngineer, Normalizer


# ── Fixtures ──────────────────────────────────

def make_market_df(n_days: int = 100) -> pd.DataFrame:
    """Create a synthetic market DataFrame."""
    dates = pd.date_range(start="2023-01-01", periods=n_days, freq="B")
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n_days) * 0.5)
    return pd.DataFrame({
        "open": close + np.random.randn(n_days) * 0.2,
        "high": close + abs(np.random.randn(n_days)) * 0.5,
        "low": close - abs(np.random.randn(n_days)) * 0.5,
        "close": close,
        "volume": np.random.randint(1_000_000, 10_000_000, n_days),
        "ticker": "AAPL",
    }, index=dates)


def make_macro_df(n_days: int = 100) -> pd.DataFrame:
    """Create a synthetic macro DataFrame."""
    dates = pd.date_range(start="2023-01-01", periods=n_days, freq="B")
    np.random.seed(42)
    return pd.DataFrame({
        "gdp": 3.0 + np.random.randn(n_days) * 0.1,
        "inflation": 2.5 + np.random.randn(n_days) * 0.05,
        "interest_rate": 5.0 + np.random.randn(n_days) * 0.02,
        "unemployment": 4.0 + np.random.randn(n_days) * 0.1,
    }, index=dates)


# ── Tests ─────────────────────────────────────

class TestTechnicalFeatureEngineer:
    """Tests for TechnicalFeatureEngineer."""

    def test_compute_adds_indicator_columns(self):
        df = make_market_df(100)
        fe = TechnicalFeatureEngineer()
        result = fe.compute(df, indicators=["rsi", "macd", "bb", "atr", "obv"])

        expected_cols = ["rsi_14", "macd", "macd_signal", "macd_diff",
                         "bb_high", "bb_mid", "bb_low", "atr_14", "obv"]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_compute_preserves_original_columns(self):
        df = make_market_df(50)
        fe = TechnicalFeatureEngineer()
        result = fe.compute(df, indicators=["rsi"])
        for col in ["open", "high", "low", "close", "volume"]:
            assert col in result.columns

    def test_compute_empty_df_returns_empty(self):
        fe = TechnicalFeatureEngineer()
        result = fe.compute(pd.DataFrame())
        assert result.empty


class TestNormalizer:
    """Tests for Normalizer."""

    def test_standard_normalization_roundtrip(self):
        df = make_market_df(50)[["open", "high", "low", "close", "volume"]]
        norm = Normalizer()
        norm.fit(df, method="standard")
        transformed = norm.transform(df)

        # Mean should be ~0, std ~1 for each column
        for col in transformed.columns:
            assert abs(transformed[col].mean()) < 0.01, f"Mean not ~0 for {col}"

        # Inverse should recover original
        recovered = norm.inverse_transform(transformed)
        pd.testing.assert_frame_equal(recovered, df, atol=1e-6, check_dtype=False)

    def test_minmax_normalization(self):
        df = make_market_df(50)[["close"]]
        norm = Normalizer()
        norm.fit(df, method="minmax")
        transformed = norm.transform(df)
        assert transformed["close"].min() >= -0.01
        assert transformed["close"].max() <= 1.01


class TestTimeAligner:
    """Tests for TimeAligner."""

    def test_align_produces_common_index(self):
        market = make_market_df(80)
        macro = make_macro_df(80)
        aligner = TimeAligner()
        result = aligner.align({"market": market, "macro": macro})

        assert "market" in result
        assert "macro" in result
        # Both should have the same index
        pd.testing.assert_index_equal(result["market"].index, result["macro"].index)

    def test_align_empty_input(self):
        aligner = TimeAligner()
        result = aligner.align({})
        assert result == {}


class TestDataLoaderFactory:
    """Tests for DataLoaderFactory."""

    def test_factory_creates_all_loaders(self):
        config = {
            "market_dir": "data/raw/market",
            "news_dir": "data/raw/news",
            "fundamental_dir": "data/raw/fundamentals",
            "macro_dir": "data/raw/macro",
        }
        loaders = DataLoaderFactory.create_loaders(config)
        assert "market" in loaders
        assert "news" in loaders
        assert "fundamental" in loaders
        assert "macro" in loaders


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

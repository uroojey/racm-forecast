import logging
import pandas as pd
import numpy as np
import ta
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class TechnicalFeatureEngineer:
    """Computes technical indicators for market data."""
    
    def __init__(self):
        pass

    def compute(self, df: pd.DataFrame, indicators: Optional[List[str]] = None) -> pd.DataFrame:
        """Computes technical indicators and appends them to the DataFrame.
        
        Args:
            df: Market DataFrame containing OHLCV columns ('open', 'high', 'low', 'close', 'volume').
            indicators: List of indicators to compute. Defaults to ['rsi', 'macd', 'bb', 'atr', 'obv'].
        
        Returns:
            DataFrame with additional indicator columns.
        """
        if df.empty:
            logger.warning("Empty DataFrame provided to TechnicalFeatureEngineer.")
            return df
            
        required_cols = {'open', 'high', 'low', 'close', 'volume'}
        if not required_cols.issubset(set(df.columns.str.lower())):
            logger.error(f"Missing required columns. Expected: {required_cols}")
            return df

        # Use lower case for column access mapping
        col_map = {col.lower(): col for col in df.columns}
        
        close_s = df[col_map['close']]
        high_s = df[col_map['high']]
        low_s = df[col_map['low']]
        volume_s = df[col_map['volume']]
        
        result_df = df.copy()
        
        if indicators is None:
            indicators = ['rsi', 'macd', 'bb', 'atr', 'obv']
            
        try:
            if 'rsi' in indicators:
                result_df['rsi_14'] = ta.momentum.RSIIndicator(close=close_s, window=14).rsi()
                
            if 'macd' in indicators:
                macd = ta.trend.MACD(close=close_s, window_slow=26, window_fast=12, window_sign=9)
                result_df['macd'] = macd.macd()
                result_df['macd_signal'] = macd.macd_signal()
                result_df['macd_diff'] = macd.macd_diff()
                
            if 'bb' in indicators:
                bb = ta.volatility.BollingerBands(close=close_s, window=20, window_dev=2)
                result_df['bb_high'] = bb.bollinger_hband()
                result_df['bb_mid'] = bb.bollinger_mavg()
                result_df['bb_low'] = bb.bollinger_lband()
                
            if 'atr' in indicators:
                result_df['atr_14'] = ta.volatility.AverageTrueRange(high=high_s, low=low_s, close=close_s, window=14).average_true_range()
                
            if 'obv' in indicators:
                result_df['obv'] = ta.volume.OnBalanceVolumeIndicator(close=close_s, volume=volume_s).on_balance_volume()
                
        except Exception as e:
            logger.error(f"Error computing technical features: {e}")
            
        return result_df


class Normalizer:
    """Normalizes features using standard or min-max scaling."""
    
    def __init__(self):
        self.method = 'standard'
        self.params: Dict[str, Dict[str, float]] = {}

    def fit(self, df: pd.DataFrame, method: str = 'standard') -> None:
        """Computes parameters for normalization."""
        self.method = method
        self.params = {}
        
        # Only normalize numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        if method == 'standard':
            for col in numeric_df.columns:
                self.params[col] = {
                    'mean': numeric_df[col].mean(),
                    'std': numeric_df[col].std() if numeric_df[col].std() != 0 else 1.0
                }
        elif method == 'minmax':
            for col in numeric_df.columns:
                min_val = numeric_df[col].min()
                max_val = numeric_df[col].max()
                self.params[col] = {
                    'min': min_val,
                    'max': max_val if max_val != min_val else min_val + 1.0
                }
        else:
            logger.error(f"Unknown normalization method: {method}")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies normalization to the DataFrame."""
        if not self.params:
            logger.warning("Normalizer not fitted. Call fit() first.")
            return df
            
        result_df = df.copy()
        for col, params in self.params.items():
            if col in result_df.columns:
                if self.method == 'standard':
                    result_df[col] = (result_df[col] - params['mean']) / params['std']
                elif self.method == 'minmax':
                    result_df[col] = (result_df[col] - params['min']) / (params['max'] - params['min'])
        return result_df

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reverses the normalization."""
        if not self.params:
            logger.warning("Normalizer not fitted. Call fit() first.")
            return df
            
        result_df = df.copy()
        for col, params in self.params.items():
            if col in result_df.columns:
                if self.method == 'standard':
                    result_df[col] = (result_df[col] * params['std']) + params['mean']
                elif self.method == 'minmax':
                    result_df[col] = (result_df[col] * (params['max'] - params['min'])) + params['min']
        return result_df

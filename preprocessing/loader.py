import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

class MarketDataLoader:
    """Loads market OHLCV data."""
    def __init__(self, directory: str):
        self.directory = Path(directory)

    def load(self) -> pd.DataFrame:
        """Loads all CSV files in the directory, concatenates them, and returns a DataFrame.
        Expected columns: date, open, high, low, close, volume, ticker.
        """
        dataframes = []
        try:
            if not self.directory.exists() or not self.directory.is_dir():
                logger.error(f"Directory {self.directory} does not exist or is not a directory.")
                return pd.DataFrame()

            for file_path in self.directory.glob("*.csv"):
                try:
                    df = pd.read_csv(file_path)
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                        df.set_index('date', inplace=True)
                    dataframes.append(df)
                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {e}")
            
            if not dataframes:
                logger.warning("No CSV files found or loaded.")
                return pd.DataFrame()
                
            combined_df = pd.concat(dataframes)
            # Optionally sort by index
            combined_df.sort_index(inplace=True)
            return combined_df
        except Exception as e:
            logger.error(f"Error loading market data: {e}")
            return pd.DataFrame()


class NewsDataLoader:
    """Loads news data."""
    def __init__(self, directory: str):
        self.directory = Path(directory)

    def load(self) -> pd.DataFrame:
        """Loads news CSV files. Expected columns: date, headline, source, ticker."""
        dataframes = []
        try:
            if not self.directory.exists() or not self.directory.is_dir():
                logger.error(f"Directory {self.directory} does not exist or is not a directory.")
                return pd.DataFrame()

            for file_path in self.directory.glob("*.csv"):
                try:
                    df = pd.read_csv(file_path)
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                    dataframes.append(df)
                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {e}")
            
            if not dataframes:
                return pd.DataFrame()
                
            return pd.concat(dataframes, ignore_index=True)
        except Exception as e:
            logger.error(f"Error loading news data: {e}")
            return pd.DataFrame()


class FundamentalDataLoader:
    """Loads fundamental data."""
    def __init__(self, directory: str):
        self.directory = Path(directory)

    def load(self) -> pd.DataFrame:
        """Loads fundamental CSV files. 
        Expected columns: date, ticker, revenue, net_income, eps, pe_ratio, debt_to_equity, roe, roa.
        """
        dataframes = []
        try:
            if not self.directory.exists() or not self.directory.is_dir():
                logger.error(f"Directory {self.directory} does not exist or is not a directory.")
                return pd.DataFrame()

            for file_path in self.directory.glob("*.csv"):
                try:
                    df = pd.read_csv(file_path)
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                    dataframes.append(df)
                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {e}")
            
            if not dataframes:
                return pd.DataFrame()
                
            return pd.concat(dataframes, ignore_index=True)
        except Exception as e:
            logger.error(f"Error loading fundamental data: {e}")
            return pd.DataFrame()


class MacroDataLoader:
    """Loads macroeconomic data."""
    def __init__(self, directory: str):
        self.directory = Path(directory)

    def load(self) -> pd.DataFrame:
        """Loads macro CSV files. Expected columns: date, gdp, inflation, interest_rate, unemployment."""
        dataframes = []
        try:
            if not self.directory.exists() or not self.directory.is_dir():
                logger.error(f"Directory {self.directory} does not exist or is not a directory.")
                return pd.DataFrame()

            for file_path in self.directory.glob("*.csv"):
                try:
                    df = pd.read_csv(file_path)
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                    dataframes.append(df)
                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {e}")
            
            if not dataframes:
                return pd.DataFrame()
                
            return pd.concat(dataframes, ignore_index=True)
        except Exception as e:
            logger.error(f"Error loading macro data: {e}")
            return pd.DataFrame()


class DataLoaderFactory:
    """Factory to instantiate data loaders from configuration."""
    @staticmethod
    def create_loaders(config: Dict[str, Any]) -> Dict[str, Any]:
        """Creates and returns all four data loaders based on config dictionary.
        Expected config keys: 'market_dir', 'news_dir', 'fundamental_dir', 'macro_dir'.
        """
        loaders = {
            'market': MarketDataLoader(config.get('market_dir', 'data/market')),
            'news': NewsDataLoader(config.get('news_dir', 'data/news')),
            'fundamental': FundamentalDataLoader(config.get('fundamental_dir', 'data/fundamental')),
            'macro': MacroDataLoader(config.get('macro_dir', 'data/macro'))
        }
        return loaders

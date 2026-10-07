import logging
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, Any, Optional, Tuple

from .loader import DataLoaderFactory
from .alignment import TimeAligner
from .features import TechnicalFeatureEngineer, Normalizer

logger = logging.getLogger(__name__)

class RACMDataModule:
    """Orchestrates data loading, alignment, feature engineering, and dataloader creation."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the data module.
        Args:
            config: Dictionary containing configuration paths, seq_length, etc.
        """
        self.config = config
        self.seq_len = config.get('sequence_length', 30)
        self.horizon = config.get('forecast_horizon', 1)
        
        self.train_ratio = config.get('train_ratio', 0.7)
        self.val_ratio = config.get('val_ratio', 0.15)
        # test_ratio is 1.0 - train_ratio - val_ratio
        
        self.processed_data: Optional[pd.DataFrame] = None
        self.normalizer = Normalizer()

    def prepare(self) -> None:
        """Loads, aligns, engineers features, normalizes, and stores the final DataFrame."""
        logger.info("Starting data preparation...")
        
        # 1. Load Data
        loaders = DataLoaderFactory.create_loaders(self.config)
        raw_data = {}
        for name, loader in loaders.items():
            df = loader.load()
            if not df.empty:
                raw_data[name] = df
            else:
                logger.warning(f"No data loaded for source: {name}")
                
        if 'market' not in raw_data:
            logger.error("Market data is required but missing.")
            return
            
        # 2. Add technical features to market data before alignment
        fe = TechnicalFeatureEngineer()
        raw_data['market'] = fe.compute(raw_data['market'])
        
        # 3. Align Data
        aligner = TimeAligner()
        aligned_dfs = aligner.align(raw_data, fill_method='ffill')
        
        if not aligned_dfs:
            logger.error("Failed to align data.")
            return
            
        # Combine into single DataFrame
        # Flatten columns using MultiIndex or prefixing. We'll use prefixing here.
        combined_df = pd.DataFrame(index=aligned_dfs[list(aligned_dfs.keys())[0]].index)
        for name, df in aligned_dfs.items():
            # Drop non-numeric for simplicity in combined DF, except if handled
            num_df = df.select_dtypes(include=[np.number])
            num_df.columns = [f"{name}_{c}" for c in num_df.columns]
            combined_df = combined_df.join(num_df, how='inner')
            
        # 4. Normalize Data
        # We should fit normalizer on train set only in practice, but API requested 
        # using Normalizer here. We will normalize all data here as per instructions.
        self.normalizer.fit(combined_df, method='standard')
        self.processed_data = self.normalizer.transform(combined_df)
        
        # We need a target column to predict close price returns.
        # Let's calculate the returns of the market_close if it exists.
        market_close_col = 'market_close'
        if market_close_col in combined_df.columns:
            # We add un-normalized return as target
            self.processed_data['target_return'] = combined_df[market_close_col].pct_change().shift(-self.horizon)
            
        self.processed_data.dropna(inplace=True)
        logger.info(f"Data preparation complete. Shape: {self.processed_data.shape}")

    def get_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Returns chronological train, val, test splits."""
        if self.processed_data is None or self.processed_data.empty:
            logger.error("No processed data available. Call prepare() first.")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            
        total_len = len(self.processed_data)
        train_end = int(total_len * self.train_ratio)
        val_end = train_end + int(total_len * self.val_ratio)
        
        train_df = self.processed_data.iloc[:train_end]
        val_df = self.processed_data.iloc[train_end:val_end]
        test_df = self.processed_data.iloc[val_end:]
        
        return train_df, val_df, test_df

    def create_sequences(self, df: pd.DataFrame, seq_len: int, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
        """Creates sliding window sequences. Target is 'target_return'."""
        if df.empty or 'target_return' not in df.columns:
            return np.array([]), np.array([])
            
        feature_cols = [c for c in df.columns if c != 'target_return']
        X_data = df[feature_cols].values
        y_data = df['target_return'].values
        
        X, y = [], []
        # Create sequences
        for i in range(len(df) - seq_len):
            X.append(X_data[i : i + seq_len])
            y.append(y_data[i + seq_len - 1]) # The target corresponding to the end of sequence
            
        return np.array(X), np.array(y).reshape(-1, 1)

    def get_dataloaders(self, batch_size: int = 32) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Returns PyTorch DataLoaders for train, val, and test splits."""
        train_df, val_df, test_df = self.get_splits()
        
        X_train, y_train = self.create_sequences(train_df, self.seq_len, self.horizon)
        X_val, y_val = self.create_sequences(val_df, self.seq_len, self.horizon)
        X_test, y_test = self.create_sequences(test_df, self.seq_len, self.horizon)
        
        def create_dl(X: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
            if X.size == 0:
                return None
            dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), 
                                    torch.tensor(y, dtype=torch.float32))
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
            
        train_dl = create_dl(X_train, y_train, shuffle=True)
        val_dl = create_dl(X_val, y_val, shuffle=False)
        test_dl = create_dl(X_test, y_test, shuffle=False)
        
        return train_dl, val_dl, test_dl
